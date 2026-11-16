"""Fault-injection tests with synthetic clocks/poses. No ROS or vehicle connection."""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import (Guard, NumpyActor, PoseTracker, chassis_command, decode_pose, leader_action,
                  leader_observation, load_config, pose_fresh, spatial_reference)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT/'config/single_car_clean.json')
        self.p = json.loads((ROOT/'models/meta_ams_s0_leader/metadata.json').read_text())['profile']
        self.g = Guard(self.c, self.p, 'test')

    def state(self, t=100., x=-8.75, y=-.065, yaw=0.):
        return dict(x=x, y=y, yaw=yaw, stamp=t, receive_mono=t, ready=True,
                    vx_body=0., vy_body=0., vx_world=0., vy_world=0., yaw_rate=0.)

    def ready(self):
        self.c['hardware'].update(command_mode='speed_yaw_rate',wheelbase_m=.25,protocol_confirmed=True,
                                  geometry_confirmed=True,base_command_timeout_verified=True)
        self.c['mocap']['transform_confirmed'] = True
        self.c['safety']['geofence_confirmed'] = True
        self.g.update_pose(self.state())
        self.assertTrue(self.g.arm(100.,100.)[0])

    def feed(self, t=100.02, v=.4, delta=.1, state=None):
        self.g.update_pose(state or self.state(t))
        self.g.receive_candidate(v,delta,t,t,t,'test')
        return self.g.tick(t,t)

    def test_default_cannot_arm(self):
        self.g.update_pose(self.state())
        self.assertFalse(self.g.arm(100,100)[0])
        self.assertEqual(self.g.tick(100,100)['speed'],0)

    def test_matlab_z_means_yaw(self):
        x,y,h=decode_pose((1,2,.3),(0,0,0,1),self.c['mocap'])
        self.assertAlmostEqual(x,-7.75); self.assertAlmostEqual(y,1.935); self.assertAlmostEqual(h,.3)

    def test_marker_and_frame_transform(self):
        m=self.c['mocap']
        m.update(origin_world_xy=[2.,3.],path_heading_world_rad=math.pi/2,body_to_marker_xy_m=[.2,0.])
        x,y,h=decode_pose((2.,3.2,math.pi/2),(0,0,0,0),m)
        self.assertAlmostEqual(x,-8.75); self.assertAlmostEqual(y,-.065); self.assertAlmostEqual(h,0)

    def test_standard_quaternion_separate_encoding(self):
        self.c['mocap']['encoding']='quaternion'
        _,_,h=decode_pose((0,0,99),(0,0,math.sin(.25),math.cos(.25)),self.c['mocap'])
        self.assertAlmostEqual(h,.5)
        with self.assertRaises(ValueError): decode_pose((0,0,0),(0,0,0,0),self.c['mocap'])

    def test_bad_pose_rejected(self):
        for pose in [(float('nan'),0,0),(0,float('inf'),0),(0,0,float('nan'))]:
            with self.assertRaises(ValueError): decode_pose(pose,(0,0,0,1),self.c['mocap'])

    def test_velocity_and_yaw_wrap(self):
        tracker=PoseTracker(self.c)
        tracker.push(0,0,math.pi-.005,100,100,100)
        s=tracker.push(-.002,0,-math.pi+.005,100.02,100.02,100.02)
        self.assertAlmostEqual(s['yaw_rate'],.5)
        self.assertAlmostEqual(s['vx_body'],.1*math.cos(.005))
        self.assertTrue(s['ready'])

    def test_mocap_timestamps_and_jumps(self):
        for stamp in [0,99.,101.]:
            with self.assertRaises(ValueError): PoseTracker(self.c).push(0,0,0,stamp,100,100)
        for dt,dx in [(0,0),(-.01,0),(.2,0),(.02,1)]:
            tracker=PoseTracker(self.c); tracker.push(0,0,0,100,100,100)
            with self.assertRaises(ValueError): tracker.push(dx,0,0,100+dt,100+dt,100+dt)

    def test_freshness_checks_receipt_and_source(self):
        s=self.state()
        self.assertFalse(pose_fresh(s,self.c,100.16,100.01))
        self.assertFalse(pose_fresh(s,self.c,100.01,100.16))
        self.assertFalse(pose_fresh(s,self.c,99.99,100))

    def test_leader_layout(self):
        s=self.state(x=-4.75,y=-.6125,yaw=0)
        s.update(vx_body=.2,vy_body=.01,yaw_rate=-.03)
        obs=leader_observation(s,self.p)
        self.assertEqual(obs.shape,(11,))
        np.testing.assert_allclose(obs[:7],[0,0,0,0,.2,.01,-.03])
        self.assertAlmostEqual(float(obs[7]),0,places=6)
        self.assertGreater(obs[8],0); self.assertEqual(list(obs[-2:]),[0,0])

    def test_action_units_and_centerline(self):
        s=self.state()
        physical,_,correction=leader_action(np.array([10.,10.]),s,self.p)
        np.testing.assert_allclose(physical,[.48,.128],atol=1e-7)
        self.assertEqual(correction,0)
        s['y']+=.1
        physical,_,_=leader_action(np.array([.2,0.]),s,self.p)
        self.assertAlmostEqual(physical[1],-.028*.32,places=7)

    def test_output_modes(self):
        h=dict(command_mode='speed_yaw_rate',wheelbase_m=.25,steering_sign=1)
        self.assertAlmostEqual(chassis_command(.1,.1,h)[1],.1*math.tan(.1)/.25)
        self.assertEqual(chassis_command(0,.1,h),(0,0))
        h['command_mode']='speed_steering_angle'
        self.assertEqual(chassis_command(.1,.1,h),(.1,.1))

    def test_wait_for_new_candidate_after_arm(self):
        self.ready()
        self.g.receive_candidate(.4,.1,100,100,100,'test')
        self.assertEqual(self.g.tick(100.01,100.01)['speed'],0)
        self.assertGreater(self.feed()['speed'],0)

    def test_operator_stop_is_latched(self):
        self.ready(); self.assertGreater(self.feed()['speed'],0)
        self.g.stop('operator_estop')
        self.assertEqual(self.feed(100.04)['speed'],0)
        self.assertEqual(self.g.mode,'STOPPED')

    def test_candidate_identity_nan_and_bounds(self):
        for v,delta,identity in [(float('nan'),0,'test'),(.2,0,'other'),(.49,0,'test'),(.1,.129,'test')]:
            self.g=Guard(self.c,self.p,'test'); self.ready()
            self.g.receive_candidate(v,delta,100.02,100.02,100.02,identity)
            self.assertEqual(self.g.tick(100.02,100.02)['speed'],0)
            self.assertEqual(self.g.mode,'STOPPED')

    def test_stale_inference_not_accepted_as_new(self):
        self.ready()
        self.g.receive_candidate(.2,0,99.8,100.02,100.02,'test')
        self.assertEqual(self.g.reason,'stale_candidate_source')

    def test_duplicate_candidate_stops(self):
        self.ready(); self.feed()
        self.g.receive_candidate(.2,0,100.02,100.03,100.03,'test')
        self.assertEqual(self.g.reason,'candidate_timestamp_order')

    def test_policy_failure_timeout(self):
        self.ready(); self.feed()
        for t in [100.06,100.10,100.15]:
            self.g.update_pose(self.state(t)); result=self.g.tick(t,t)
        self.assertEqual(result['reason'],'policy_timeout'); self.assertEqual(result['speed'],0)

    def test_mocap_failure_timeout(self):
        self.ready(); self.feed()
        for t in [100.06,100.10,100.14,100.18]:
            self.g.receive_candidate(.2,0,t,t,t,'test'); result=self.g.tick(t,t)
        self.assertEqual(result['reason'],'mocap_timeout'); self.assertEqual(result['speed'],0)

    def test_guard_deadline(self):
        self.ready(); self.feed()
        self.assertEqual(self.feed(100.2)['reason'],'guard_loop_deadline')

    def test_footprint_and_tracking_stop(self):
        self.ready()
        state=self.state(100.02,x=-9.45)
        self.assertEqual(self.feed(state=state)['reason'],'vehicle_footprint_outside_geofence')
        self.g=Guard(self.c,self.p,'test'); self.ready()
        self.assertEqual(self.feed(state=self.state(100.02,y=-.55))['reason'],'tracking_error')

    def test_slew_forward_only_and_immediate_brake(self):
        self.ready(); r=self.feed()
        self.assertAlmostEqual(r['speed'],.004); self.assertAlmostEqual(r['steering'],.004)
        r=self.feed(100.04,v=-.1,delta=-.1)
        self.assertEqual(r['speed'],0); self.assertAlmostEqual(r['steering'],0)

    def test_limited_yaw_rate_at_wheelbase(self):
        self.ready()
        self.c['hardware']['wheelbase_m']=.03
        self.c['safety']['max_yaw_rate_rps']=.05
        for i in range(1,100): r=self.feed(100+i*.02,delta=.128)
        self.assertLessEqual(abs(r['angular_z']),.05+1e-10)
        self.assertLessEqual(r['speed'],.15)

    def test_completion_requires_dwell_and_remains_latched(self):
        self.ready()
        for i in range(1,103):
            t=100+i*.02
            r=self.feed(t,v=.1,delta=0,state=self.state(t,x=-1.70,y=-1.16))
            if i<100: self.assertFalse(r['completed'])
        self.assertTrue(r['completed']); self.assertEqual(r['speed'],0)
        self.assertTrue(self.g.tick(103,103)['completed'])

    def test_missing_motion_or_heading_prevents_arm(self):
        self.ready(); self.g.stop()
        for change in [dict(x=-8.4),dict(yaw=.2),dict(vx_body=.1)]:
            s=self.state(); s.update(change); self.g.update_pose(s)
            self.assertFalse(self.g.arm(100,100)[0])

    def test_numpy_actor_rejects_nonfinite(self):
        actor=NumpyActor(ROOT/'models/meta_ams_s0_leader')
        with self.assertRaises(ValueError): actor(np.full(11,float('nan')))
        self.assertEqual(actor(np.zeros(11)).shape,(2,))


if __name__ == '__main__': unittest.main(verbosity=2)
