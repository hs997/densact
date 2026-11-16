"""ROS/Isaac/PyTorch-free single-leader clean deployment. Python 3.6+.

Internal candidates always carry physical [m/s, equivalent front angle rad].
Only the independent guard translates these into the confirmed chassis protocol.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(*values):
    return all(math.isfinite(float(v)) for v in values)


def clip(value, low, high):
    return max(low, min(high, float(value)))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 'ams_single_leader_clean_v1' or config.get('attack') != 'off':
        raise ValueError('This executable only supports single-leader CLEAN; attacks are not enabled')
    for section, keys in {
        'control': ['rate_hz', 'max_run_s'],
        'mocap': ['timeout_s', 'max_source_age_s', 'max_future_s', 'velocity_filter_tau_s',
                  'max_speed_mps', 'max_yaw_rate_rps', 'max_sample_gap_s'],
        'safety': ['command_timeout_s', 'max_speed_mps', 'max_steering_rad', 'max_yaw_rate_rps',
                   'max_acceleration_mps2', 'max_steering_rate_rps', 'max_tracking_error_m',
                   'max_heading_error_rad', 'vehicle_length_m', 'vehicle_width_m',
                   'arm_position_tolerance_m', 'arm_yaw_tolerance_rad', 'arm_speed_tolerance_mps'],
    }.items():
        for key in keys:
            v = config[section][key]
            if not finite(v) or float(v) <= 0:
                raise ValueError('Invalid positive setting: %s.%s' % (section, key))
    if config['control']['rate_hz'] > 200:
        raise ValueError('Unexpected control rate')
    m = config['mocap']
    if m['encoding'] not in ('xy_yaw_position_z', 'quaternion'):
        raise ValueError('Unknown mocap encoding')
    if m['yaw_unit'] not in ('rad', 'deg'):
        raise ValueError('Unknown yaw unit')
    for key in ('origin_world_xy', 'origin_model_xy', 'body_to_marker_xy_m'):
        if len(m[key]) != 2 or not finite(*m[key]):
            raise ValueError('Invalid mocap transform: '+key)
    if not finite(m['path_heading_world_rad'], m['marker_to_body_yaw_rad']):
        raise ValueError('Invalid heading transform')
    if config['hardware']['command_mode'] not in ('unconfirmed', 'speed_yaw_rate', 'speed_steering_angle'):
        raise ValueError('Unknown output command mode')
    if config['hardware']['steering_sign'] not in (-1, 1):
        raise ValueError('steering_sign must be +1 or -1')
    if config['safety']['max_steering_rad'] >= math.pi/2:
        raise ValueError('Steering limit must be below pi/2')
    length = config['hardware']['wheelbase_m']
    if length is not None and (not finite(length) or length <= 0):
        raise ValueError('Measured wheelbase must be positive or unconfirmed (null)')
    for section, keys in [('hardware', ['protocol_confirmed', 'geometry_confirmed', 'base_command_timeout_verified']),
                          ('mocap', ['transform_confirmed']), ('safety', ['geofence_confirmed'])]:
        if any(type(config[section][key]) is not bool for key in keys):
            raise ValueError('Confirmation fields must be JSON booleans')
    bounds = config['safety']['model_geofence_xyxy']
    if len(bounds) != 4 or not finite(*bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError('Invalid geofence')
    topics = config['topics']
    if len(set(topics.values())) != len(topics) or any(not x.startswith('/') for x in topics.values()):
        raise ValueError('Topics must be distinct absolute ROS names')
    return config


class NumpyActor:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / 'metadata.json').read_text())
        if self.meta['schema'] != 'ams_leader_numpy_v1':
            raise ValueError('Unsupported actor bundle')
        if digest(model_dir / 'actor.npz') != self.meta['weights_sha256']:
            raise ValueError('Actor hash mismatch')
        if self.meta['obs_dim'] != 11 or self.meta['act_dim'] != 2 or self.meta['agent_index'] != 0:
            raise ValueError('Expected the 11D leader actor, not a follower/14D model')
        self.layers = []
        with np.load(str(model_dir / 'actor.npz'), allow_pickle=False) as arrays:
            for i, (n, m) in enumerate(zip(self.meta['sizes'][:-1], self.meta['sizes'][1:])):
                w, b = arrays['w%d' % i], arrays['b%d' % i]
                if w.shape != (m, n) or b.shape != (m,) or not np.isfinite(w).all() or not np.isfinite(b).all():
                    raise ValueError('Invalid actor tensor')
                self.layers.append((w.copy(), b.copy()))

    def __call__(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape[-1:] != (11,) or not np.isfinite(x).all():
            raise ValueError('Actor input must be finite and 11-dimensional')
        for i, (w, b) in enumerate(self.layers):
            x = np.matmul(x, w.T) + b
            if i < len(self.layers)-1:
                x = np.where(x >= 0, x, np.expm1(np.minimum(x, 0)))
        if not np.isfinite(x).all():
            raise ValueError('Non-finite actor output')
        return x


def runtime_identity(config, meta):
    data = json.dumps({'config': config, 'model': meta}, sort_keys=True, allow_nan=False).encode('utf-8')
    return 'ams_clean_' + hashlib.sha256(data).hexdigest()


def spatial_reference(x, profile):
    p = profile['path']
    u = clip((x-p['start_x']) / p['length_m'], 0, 1)
    dy = p['end_y']-p['start_y']
    smooth = 10*u**3-15*u**4+6*u**5
    slope = dy*(30*u**2-60*u**3+30*u**4)/p['length_m']
    return u, p['start_y']+dy*smooth, math.atan(slope)


def leader_observation(state, profile):
    _, y_ref, heading_ref = spatial_reference(state['x'], profile)
    # Exact trained leader layout; relative/pair terms and role_id are zero.
    return np.asarray([0, 0, 0, 0, state['vx_body'], state['vy_body'], state['yaw_rate'],
                       clip(state['y']-y_ref, -2, 2), math.sin(state['yaw']-heading_ref), 0, 0],
                      dtype=np.float32)


def leader_action(raw, state, profile):
    """Actual clean leader chain: initial clip -> centerline shield -> final clip."""
    if np.asarray(raw).shape != (2,) or not np.isfinite(raw).all():
        raise ValueError('Invalid raw actor output')
    limit = profile['action_clip']
    action = np.clip(raw, -limit, limit).astype(np.float32)
    before = action.copy()
    _, ref_y, _ = spatial_reference(state['x'], profile)
    s = profile['shield']
    error = state['y']-ref_y
    if s['enabled'] and abs(error) > s['lateral_tol']:
        action[1] -= clip(s['centerline_turn_sign']*s['centerline_turn_gain']*error,
                          -s['centerline_turn_clip'], s['centerline_turn_clip'])
    action = np.clip(action, -limit, limit)
    physical = np.array([action[0]*profile['max_speed_mps'],
                         action[1]*profile['max_steering_rad']], dtype=np.float64)
    return physical, action, float(action[1]-before[1])


def decode_pose(position, quaternion, mocap):
    x, y = float(position[0]), float(position[1])
    if mocap['encoding'] == 'xy_yaw_position_z':
        yaw = float(position[2])
        if mocap['yaw_unit'] == 'deg':
            yaw = math.radians(yaw)
    else:
        q = np.asarray(quaternion, dtype=float)  # ROS x,y,z,w
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
            raise ValueError('Invalid quaternion')
        q /= np.linalg.norm(q)
        qx, qy, qz, qw = q
        yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    if not finite(x, y, yaw):
        raise ValueError('Non-finite mocap pose')
    yaw = wrap(yaw + mocap['marker_to_body_yaw_rad'])
    dx, dy = mocap['body_to_marker_xy_m']
    x -= math.cos(yaw)*dx-math.sin(yaw)*dy
    y -= math.sin(yaw)*dx+math.cos(yaw)*dy
    x -= mocap['origin_world_xy'][0]
    y -= mocap['origin_world_xy'][1]
    theta = mocap['path_heading_world_rad']
    xm = math.cos(theta)*x+math.sin(theta)*y+mocap['origin_model_xy'][0]
    ym = -math.sin(theta)*x+math.cos(theta)*y+mocap['origin_model_xy'][1]
    return xm, ym, wrap(yaw-theta)


class PoseTracker:
    def __init__(self, config):
        self.cfg = config['mocap']
        self.state = None

    def push(self, x, y, yaw, stamp, receive_mono, ros_now):
        c = self.cfg
        if not finite(x, y, yaw, stamp, receive_mono, ros_now) or stamp <= 0:
            raise ValueError('Missing/non-finite source timestamp or pose')
        age = ros_now-stamp
        if age > c['max_source_age_s'] or age < -c['max_future_s']:
            raise ValueError('Mocap source clock is stale/future; synchronize clocks')
        old = self.state
        ready = False
        vx = vy = omega = 0.0
        if old is not None:
            dt = stamp-old['stamp']
            if dt <= 0 or dt > c['max_sample_gap_s']:
                raise ValueError('Mocap timestamp repeated/reversed or sample gap too large')
            vx = (x-old['x'])/dt
            vy = (y-old['y'])/dt
            omega = wrap(yaw-old['yaw'])/dt
            if math.hypot(vx, vy) > c['max_speed_mps'] or abs(omega) > c['max_yaw_rate_rps']:
                raise ValueError('Mocap jump or excessive measured motion')
            alpha = dt/(c['velocity_filter_tau_s']+dt)
            if old['ready']:
                vx = alpha*vx+(1-alpha)*old['vx_world']
                vy = alpha*vy+(1-alpha)*old['vy_world']
                omega = alpha*omega+(1-alpha)*old['yaw_rate']
            ready = True
        self.state = dict(x=x, y=y, yaw=wrap(yaw), stamp=stamp, receive_mono=receive_mono,
                          vx_world=vx, vy_world=vy, yaw_rate=omega, ready=ready,
                          vx_body=math.cos(yaw)*vx+math.sin(yaw)*vy,
                          vy_body=-math.sin(yaw)*vx+math.cos(yaw)*vy)
        return dict(self.state)


def pose_fresh(state, config, now, ros_now):
    if not state or not state['ready']:
        return False
    return (0 <= now-state['receive_mono'] <= config['mocap']['timeout_s'] and
            -config['mocap']['max_future_s'] <= ros_now-state['stamp'] <= config['mocap']['max_source_age_s'])


def chassis_command(v, delta, hardware):
    mode = hardware['command_mode']
    if mode == 'speed_yaw_rate':
        length = hardware['wheelbase_m']
        if length is None or not finite(length) or length <= 0:
            raise ValueError('Measured wheelbase is required for yaw-rate output')
        second = v*math.tan(delta)/length
    elif mode == 'speed_steering_angle':
        second = delta
    else:
        raise ValueError('Confirm chassis angular.z semantics before execution')
    return float(v), float(second)*hardware['steering_sign']


class Guard:
    """Independent process state machine. Faults latch; no automatic restart."""
    def __init__(self, config, profile, identity):
        self.cfg, self.profile, self.identity = config, profile, identity
        self.state = None
        self.candidate = None
        self.mode = 'DISARMED'
        self.reason = 'startup'
        self.armed_at = None
        self.arm_stamp = None
        self.last_tick = None
        self.done_since = None
        self.previous = (0.0, 0.0)
        self.completed = False

    def stop(self, reason='operator_stop'):
        self.mode, self.reason = 'STOPPED', str(reason)
        self.candidate = None
        self.previous = (0.0, 0.0)
        self.done_since = None

    def update_pose(self, state):
        self.state = dict(state)

    def receive_candidate(self, v, delta, stamp, received, ros_now, identity):
        if identity != self.identity or not finite(v, delta, stamp, received, ros_now):
            self.stop('candidate_identity_or_nonfinite'); return
        p = self.profile
        if abs(v) > p['max_speed_mps']*p['action_clip']+1e-5 or abs(delta) > p['max_steering_rad']*p['action_clip']+1e-5:
            self.stop('candidate_outside_training_action_bounds'); return
        if stamp <= 0 or not -self.cfg['mocap']['max_future_s'] <= ros_now-stamp <= self.cfg['mocap']['max_source_age_s']:
            self.stop('stale_candidate_source'); return
        if self.candidate and stamp <= self.candidate['stamp']:
            self.stop('candidate_timestamp_order'); return
        if self.mode != 'ARMED' or stamp <= self.arm_stamp:
            return
        self.candidate = dict(v=v, delta=delta, stamp=stamp, received=received)

    def arming_problem(self, now, ros_now):
        h, s = self.cfg['hardware'], self.cfg['safety']
        required = ('protocol_confirmed', 'geometry_confirmed', 'base_command_timeout_verified')
        if not all(h[k] is True for k in required) or not self.cfg['mocap']['transform_confirmed'] or not s['geofence_confirmed']:
            return 'commissioning_fields_not_confirmed'
        if h['command_mode'] == 'unconfirmed' or h['wheelbase_m'] is None or not finite(h['wheelbase_m']) or h['wheelbase_m'] <= 0:
            return 'missing_chassis_mode_or_wheelbase'
        if s['vehicle_length_m'] < h['wheelbase_m']:
            return 'vehicle_length_smaller_than_wheelbase'
        if not pose_fresh(self.state, self.cfg, now, ros_now):
            return 'mocap_not_ready_or_stale'
        start = self.cfg['mocap']['origin_model_xy']
        if math.hypot(self.state['x']-start[0], self.state['y']-start[1]) > s['arm_position_tolerance_m']:
            return 'not_at_start_position'
        if abs(self.state['yaw']) > s['arm_yaw_tolerance_rad']:
            return 'not_at_start_heading'
        if math.hypot(self.state['vx_body'], self.state['vy_body']) > s['arm_speed_tolerance_mps']:
            return 'vehicle_not_stationary'
        return self.pose_problem()

    def arm(self, now, ros_now):
        if self.mode == 'ARMED':
            return False, 'already_armed'
        problem = self.arming_problem(now, ros_now)
        if problem:
            return False, problem
        self.mode, self.reason = 'ARMED', 'waiting_for_new_candidate'
        self.armed_at, self.arm_stamp = now, ros_now
        self.candidate = None
        self.previous = (0.0, 0.0)
        self.done_since = None
        self.last_tick = now
        self.completed = False
        return True, 'armed'

    def pose_problem(self):
        state, s = self.state, self.cfg['safety']
        if state is None:
            return 'no_pose'
        xlo, ylo, xhi, yhi = s['model_geofence_xyxy']
        c, t = math.cos(state['yaw']), math.sin(state['yaw'])
        for dx in (-s['vehicle_length_m']/2, s['vehicle_length_m']/2):
            for dy in (-s['vehicle_width_m']/2, s['vehicle_width_m']/2):
                x = state['x']+c*dx-t*dy; y = state['y']+t*dx+c*dy
                if not xlo <= x <= xhi or not ylo <= y <= yhi:
                    return 'vehicle_footprint_outside_geofence'
        _, yref, href = spatial_reference(state['x'], self.profile)
        if abs(state['y']-yref) > s['max_tracking_error_m']:
            return 'tracking_error'
        if abs(wrap(state['yaw']-href)) > s['max_heading_error_rad']:
            return 'heading_error'
        return None

    def tick(self, now, ros_now):
        result = dict(mode=self.mode, reason=self.reason, speed=0.0, steering=0.0, angular_z=0.0,
                      limited=False, completed=self.completed)
        if self.mode != 'ARMED':
            return result
        problem = None
        if self.last_tick is not None and (now < self.last_tick or now-self.last_tick > self.cfg['safety']['command_timeout_s']):
            problem = 'guard_loop_deadline'
        elif not pose_fresh(self.state, self.cfg, now, ros_now):
            problem = 'mocap_timeout'
        elif self.pose_problem():
            problem = self.pose_problem()
        elif now-self.armed_at > self.cfg['control']['max_run_s']:
            problem = 'run_timeout'
        self.last_tick, previous_tick = now, self.last_tick
        elapsed = now-self.armed_at if not self.candidate else now-self.candidate['received']
        if not problem and elapsed > self.cfg['safety']['command_timeout_s']:
            problem = 'policy_timeout'
        if not problem and self.candidate and ros_now-self.candidate['stamp'] > self.cfg['mocap']['max_source_age_s']:
            problem = 'candidate_observation_timeout'
        if problem:
            self.stop(problem)
            result.update(mode=self.mode, reason=self.reason)
            return result
        progress, yref, href = spatial_reference(self.state['x'], self.profile)
        completion = self.profile['completion']
        done = (progress >= completion['progress'] and abs(self.state['y']-yref) <= completion['lateral_m']
                and abs(wrap(self.state['yaw']-href)) <= completion['heading_rad'])
        if done:
            if self.done_since is None: self.done_since = now
            if now-self.done_since >= completion['dwell_s']:
                self.stop('single_vehicle_complete')
                self.completed = True
                result.update(mode=self.mode, reason=self.reason, completed=True)
                return result
        else:
            self.done_since = None
        if not self.candidate:
            result.update(mode=self.mode, reason='waiting_for_new_candidate'); return result
        v, delta = self.candidate['v'], self.candidate['delta']
        s = self.cfg['safety']
        target_v = clip(v, 0, s['max_speed_mps'])  # first clean commissioning is forward-only
        target_delta = clip(delta, -s['max_steering_rad'], s['max_steering_rad'])
        length = self.cfg['hardware']['wheelbase_m']
        if abs(target_delta) > 1e-9:
            target_v = min(target_v, s['max_yaw_rate_rps']*length/abs(math.tan(target_delta)))
        dt = 0 if previous_tick is None else max(0, now-previous_tick)
        old_v, old_delta = self.previous
        # Decreases in speed are never delayed by an acceleration limit.
        v_out = min(target_v, old_v+s['max_acceleration_mps2']*dt)
        step = s['max_steering_rate_rps']*dt
        delta_out = clip(target_delta, old_delta-step, old_delta+step)
        if abs(delta_out) > 1e-9:
            v_out = min(v_out, s['max_yaw_rate_rps']*length/abs(math.tan(delta_out)))
        self.previous = (v_out, delta_out)
        speed, second = chassis_command(v_out, delta_out, self.cfg['hardware'])
        self.reason = 'running'
        result.update(mode=self.mode, reason=self.reason, speed=speed, steering=delta_out, angular_z=second,
                      limited=abs(v_out-v)>1e-7 or abs(delta_out-delta)>1e-7)
        return result
