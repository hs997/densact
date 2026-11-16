#!/usr/bin/env python3
"""Isolated localhost ROS1 integration test. Synthetic poses; NO chassis connected.

Creates its own temporary master port, private topics, and test-only confirmed
geometry. These simulated configuration confirmations are not hardware evidence.
"""
import argparse
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import xmlrpc.client
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def wait_for(fn,timeout=8):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=fn()
        if value: return value
        time.sleep(.02)
    raise TimeoutError('Loopback assertion timed out')


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    os.environ.update(ROS_MASTER_URI='http://127.0.0.1:%d'%port,ROS_HOSTNAME='127.0.0.1',
                      ROS_HOME=str(args.output/'ros_home'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    os.environ.pop('ROS_IP',None)
    processes=[]; handles=[]; shutdown=threading.Event(); pose_enabled=threading.Event()
    report=dict(passed=False,isolated_master=os.environ['ROS_MASTER_URI'],physical_hardware_connected=False)
    def launch(name,command):
        log=(args.output/(name+'.log')).open('x'); handles.append(log)
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
        processes.append(child); return child
    ros=None
    try:
        master=launch('master',['/opt/ros/noetic/bin/roscore','-p',str(port)])
        def master_ready():
            if master.poll() is not None: raise RuntimeError('Test master failed; see master.log')
            try: return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getUri('/ams_loopback')[0]==1
            except OSError: return False
        wait_for(master_ready)
        import rospy as ros
        from geometry_msgs.msg import PoseStamped,Twist
        from std_srvs.srv import Trigger
        ros.init_node('ams_loopback_harness',anonymous=True,disable_signals=True)
        config=json.loads((ROOT/'config/single_car_clean.json').read_text())
        for key in config['topics']: config['topics'][key]='/ams_loopback_test/'+key
        config['hardware'].update(command_mode='speed_yaw_rate',wheelbase_m=.25,protocol_confirmed=True,
            geometry_confirmed=True,base_command_timeout_verified=True)
        config['mocap']['transform_confirmed']=True; config['safety']['geofence_confirmed']=True
        config['control']['max_run_s']=30.
        cfg=args.output/'SYNTHETIC_TEST_ONLY.json'; cfg.write_text(json.dumps(config,indent=2)+'\n')
        commands=[]; mutex=threading.Lock()
        def on_command(msg):
            with mutex: commands.append((time.monotonic(),msg.linear.x,msg.angular.z))
        sub=ros.Subscriber(config['topics']['chassis'],Twist,on_command,queue_size=100)
        pub=ros.Publisher(config['topics']['mocap'],PoseStamped,queue_size=1)
        def poses():
            while not shutdown.is_set():
                if pose_enabled.is_set():
                    msg=PoseStamped(); msg.header.stamp=ros.Time.now(); msg.pose.orientation.w=1.
                    # MATLAB convention x=y=yaw=0 maps to trained leader start.
                    pub.publish(msg)
                shutdown.wait(.02)
        thread=threading.Thread(target=poses,daemon=True); thread.start(); pose_enabled.set()
        def node(role,execute=False):
            cmd=[sys.executable,str(ROOT/'run_node.py'),role,'--ros','1','--config',str(cfg),
                 '--model',str(ROOT/'models/meta_ams_s0_leader'),'--log-dir',str(args.output/role)]
            if execute: cmd.append('--execute')
            return launch(role,cmd)
        guard=node('guard',True); policy=node('policy')
        ros.wait_for_service('/ams/car05/arm',timeout=10)
        arm=ros.ServiceProxy('/ams/car05/arm',Trigger); stop=ros.ServiceProxy('/ams/car05/stop',Trigger)
        def nonzero_since(t):
            with mutex: return any(stamp>t and speed>0 for stamp,speed,_ in commands)
        def zero_since(t):
            with mutex:
                recent=[(speed,w) for stamp,speed,w in commands if stamp>t]
            return len(recent)>=3 and all(speed==0 and w==0 for speed,w in recent)
        time.sleep(.8)
        with mutex: assert commands and all(v==0 and w==0 for _,v,w in commands), 'Moved before arm'
        assert arm().success
        started=time.monotonic(); wait_for(lambda: nonzero_since(started))
        report['explicit_arming']=True
        os.kill(policy.pid,signal.SIGSTOP); frozen=time.monotonic()
        wait_for(lambda: zero_since(frozen+.25))
        os.kill(policy.pid,signal.SIGCONT)
        time.sleep(.2); assert zero_since(frozen+.25), 'Unexpected auto restart'
        report['hung_policy_stops_and_latches']=True
        assert arm().success
        resumed=time.monotonic(); wait_for(lambda: nonzero_since(resumed))
        pose_enabled.clear(); lost=time.monotonic()
        wait_for(lambda: zero_since(lost+.25))
        report['lost_mocap_stops']=True
        pose_enabled.set(); time.sleep(.2)
        assert zero_since(lost+.25), 'Unexpected auto restart after mocap recovery'
        # After a missing-frame fault, two valid samples rebuild velocity.
        assert arm().success
        resumed=time.monotonic(); wait_for(lambda: nonzero_since(resumed))
        assert stop().success; stopped=time.monotonic()
        wait_for(lambda: zero_since(stopped+.05))
        report['explicit_stop_stops_and_latches']=True
        with mutex:
            report['output_samples']=len(commands)
            report['max_mock_speed_mps']=max(abs(v) for _,v,_ in commands)
        assert policy.poll() is None and guard.poll() is None
        report['passed']=True
    finally:
        shutdown.set()
        for child in reversed(processes):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGCONT)
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=3)
        if ros is not None: ros.signal_shutdown('Isolated test complete')
        for f in handles:f.close()
        (args.output/'loopback_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
