#!/usr/bin/env python3
"""Run either the shadow/clean actor or the independent latched command guard.

Neither role publishes to the chassis without guard --execute. No motion is
possible until explicit arming plus verified commissioning settings.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import fcntl
import json
import signal
import sys
import threading
import time
from pathlib import Path

from core import (Guard, NumpyActor, PoseTracker, decode_pose, leader_action,
                  leader_observation, load_config, pose_fresh, runtime_identity)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('role', choices=['policy', 'guard'])
    parser.add_argument('--ros', type=int, choices=[1, 2], default=1)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True, help='Must not already exist')
    parser.add_argument('--execute', action='store_true', help='Guard only: create the physical chassis publisher')
    args = parser.parse_args()
    if args.role == 'policy' and args.execute:
        parser.error('Only the independent guard may execute chassis commands')
    config = load_config(args.config)
    actor = NumpyActor(args.model)
    profile, identity = actor.meta['profile'], runtime_identity(config, actor.meta)
    if args.execute:
        h = config['hardware']
        if h['command_mode'] == 'unconfirmed' or not all(h[k] is True for k in
                ['protocol_confirmed', 'geometry_confirmed', 'base_command_timeout_verified']):
            parser.error('Confirm protocol, measured geometry and independent base command timeout first')
        if not config['mocap']['transform_confirmed'] or not config['safety']['geofence_confirmed']:
            parser.error('Confirm mocap transform and measured geofence first')
    # Avoid duplicate same-host publishers. Other hosts/controllers must also be stopped.
    lockfile = None
    if args.role == 'guard':
        import hashlib
        key = hashlib.sha256(config['topics']['chassis'].encode()).hexdigest()[:16]
        lockfile = open('/tmp/ams_guard_'+key+'.lock', 'a')
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            parser.error('A guard for this chassis topic is already running on this host')
    args.log_dir.mkdir(parents=True, exist_ok=False)
    (args.log_dir/'session.json').write_text(json.dumps(dict(config=config, model=actor.meta,
        role=args.role, ros=args.ros, execute=args.execute, runtime_identity=identity,
        started_wall_time=time.time(), argv=sys.argv, status='started'), indent=2)+'\n')
    from ros_io import ROSBus
    bus = ROSBus(args.ros, 'ams_car05_'+args.role)
    stop_event, lock = threading.Event(), threading.RLock()
    tracker = PoseTracker(config)
    guard = Guard(config, profile, identity)
    last_error = ['']
    publisher = bus.publisher(config['topics']['candidate'], bus.TwistStamped) if args.role == 'policy' else None
    chassis = bus.publisher(config['topics']['chassis'], bus.Twist) if args.execute else None
    status_pub = bus.publisher(config['topics']['status'], bus.String) if args.role == 'guard' else None
    output = open(str(args.log_dir/'samples.csv'), 'x', newline='')
    fields = ['wall_time','monotonic_time','role','source_stamp','mode','reason','x','y','yaw',
              'vx_body','vy_body','yaw_rate','observation','raw_action','semantic_action',
              'centerline_correction','speed','steering','angular_z','limited','completed']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader(); output.flush()

    def write_row(values):
        row = dict(wall_time=time.time(), monotonic_time=time.monotonic(), role=args.role)
        row.update(values)
        writer.writerow({k: row.get(k, '') for k in fields})
        output.flush()  # failed/stopped runs are retained, not discarded

    def on_pose(message):
        with lock:
            try:
                expected = config['mocap']['expected_frame_id']
                if expected and message.header.frame_id != expected:
                    raise ValueError('Unexpected mocap frame_id')
                p, q = message.pose.position, message.pose.orientation
                xyz = decode_pose((p.x,p.y,p.z), (q.x,q.y,q.z,q.w), config['mocap'])
                state = tracker.push(*xyz, stamp=bus.stamp(message), receive_mono=time.monotonic(), ros_now=bus.now())
                guard.update_pose(state)
                last_error[0] = ''
            except Exception as exc:
                guard.stop('invalid_mocap: '+str(exc))
                tracker.state = None
                guard.state = None
                last_error[0] = str(exc)

    def on_candidate(message):
        with lock:
            t = message.twist
            if any(v != 0 for v in [t.linear.y,t.linear.z,t.angular.x,t.angular.y]):
                guard.stop('unexpected_candidate_fields'); return
            guard.receive_candidate(t.linear.x,t.angular.z,bus.stamp(message),time.monotonic(),bus.now(),message.header.frame_id)

    def arm():
        with lock:
            if not args.execute: return False, 'guard is in shadow mode; no physical publisher'
            return guard.arm(time.monotonic(), bus.now())

    def emergency_stop():
        with lock:
            guard.stop('operator_estop')
            if chassis is not None: chassis.publish(bus.twist())
        return True, 'latched stop; explicit arm required to resume'

    def shutdown(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    bus.subscribe(config['topics']['mocap'], bus.PoseStamped, on_pose, sensor=True)
    if args.role == 'guard':
        bus.subscribe(config['topics']['candidate'], bus.TwistStamped, on_candidate)
        bus.subscribe(config['topics']['estop'], bus.Bool, lambda msg: emergency_stop() if msg.data else None)
        bus.trigger('/ams/car05/arm', arm)
        bus.trigger('/ams/car05/stop', emergency_stop)
    print('role=%s ros=%d execute=%s identity=%s' % (args.role,args.ros,args.execute,identity), flush=True)
    print('Hardware output only from guard --execute, initially DISARMED. Logs: '+str(args.log_dir), flush=True)
    period, last_stamp, status_time = 1.0/config['control']['rate_hz'], None, 0.0
    failure = None
    try:
        while bus.ok() and not stop_event.is_set():
            start = time.monotonic()
            if args.role == 'policy':
                with lock: state = dict(tracker.state) if tracker.state else None
                if pose_fresh(state, config, start, bus.now()) and state['stamp'] != last_stamp:
                    observation = leader_observation(state, profile)
                    raw = actor(observation)
                    physical, semantic, correction = leader_action(raw, state, profile)
                    # Stamp denotes the input observation, not completion time of potentially slow inference.
                    if pose_fresh(state, config, time.monotonic(), bus.now()):
                        publisher.publish(bus.candidate_message(physical[0],physical[1],state['stamp'],identity))
                        write_row(dict(state, source_stamp=state['stamp'], mode='CANDIDATE_ONLY',
                            observation=json.dumps(observation.tolist()), raw_action=json.dumps(raw.tolist()),
                            semantic_action=json.dumps(semantic.tolist()), centerline_correction=correction,
                            speed=physical[0], steering=physical[1]))
                    last_stamp = state['stamp']
                elif start-status_time >= 1:
                    write_row(dict(mode='WAITING',reason=last_error[0] or 'waiting_for_fresh_pose'))
                    status_time = start
            else:
                # Publish while holding the state lock: a concurrent stop cannot be overwritten by an old nonzero tick.
                with lock:
                    result = guard.tick(start, bus.now())
                    if chassis is not None: chassis.publish(bus.twist(result['speed'],result['angular_z']))
                    state = dict(guard.state) if guard.state else {}
                    row = dict(state, source_stamp=state.get('stamp',''))
                    row.update(result)
                    write_row(row)
                    if start-status_time >= 0.2:
                        msg = bus.String(); msg.data = json.dumps(dict(result, execute=args.execute, identity=identity))
                        status_pub.publish(msg); status_time = start
            stop_event.wait(max(0, period-(time.monotonic()-start)))
    except Exception as exc:
        failure = repr(exc)
        raise
    finally:
        # Does not cover SIGKILL/power loss: a base watchdog is a separate commissioning requirement.
        try:
            with lock:
                final_mode, final_reason, completed = guard.mode, guard.reason, guard.completed
                guard.stop('process_shutdown')
                if chassis is not None:
                    for _ in range(3):
                        chassis.publish(bus.twist()); time.sleep(0.02)
            (args.log_dir/'result.json').write_text(json.dumps(dict(ended_wall_time=time.time(),
                status='error' if failure else 'stopped', error=failure, role=args.role,
                execute=args.execute, final_guard_mode=final_mode, final_guard_reason=final_reason,
                single_vehicle_completed=completed, partial_data_preserved=True),indent=2)+'\n')
        finally:
            output.close(); bus.close()
            if lockfile is not None: lockfile.close()


if __name__ == '__main__':
    main()
