#!/usr/bin/env python3
"""CPU parity against hash-matched real source definitions; never launch Isaac/ROS.

AST extraction avoids importing simulator modules. Only environment tensors and
planar quaternion utilities are supplied by a small test harness. The original
observation/reference/shield/action function bodies are executed unchanged.
"""
import argparse
import ast
import csv
import dataclasses
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import NumpyActor, digest, leader_action, leader_observation


def extract(path, names, ns):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if len(nodes)!=len(names): raise ValueError('Source definitions missing: '+str(names))
    future=ast.parse('from __future__ import annotations').body
    exec(compile(ast.Module(body=future+nodes,type_ignores=[]),str(path),'exec'),ns)


def rotate(q,v,inverse=False):
    yaw=2*torch.atan2(q[:,3],q[:,0])
    if inverse: yaw=-yaw
    c,s=torch.cos(yaw),torch.sin(yaw)
    return torch.stack((c*v[:,0]-s*v[:,1],s*v[:,0]+c*v[:,1],v[:,2]),dim=-1)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--queue-root',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new validation directory')
    torch.set_num_threads(1)
    files=['source/my_exts/marl_platoon/'+x for x in
           ['tasks/platoon/config.py','trajectory.py','algorithms/shield.py','ackermann_action.py',
            'algorithms/happo/actor.py','wrappers.py']]
    manifest=json.loads((args.queue_root/'queue_manifest.json').read_text())['sources']
    evidence={}
    for rel in files:
        current,snapshot=args.source_root/rel,args.queue_root/'source_snapshot'/rel
        hashes=[digest(current),digest(snapshot),manifest[rel]]
        if len(set(hashes))!=1: raise ValueError('Source/snapshot/manifest mismatch: '+rel)
        evidence[rel]=dict(sha256=hashes[0],current=str(current.resolve()),snapshot=str(snapshot.resolve()))
    actor=NumpyActor(args.model); profile=actor.meta['profile']
    env_file=Path(actor.meta['checkpoint']).parent/'params/env.yaml'
    if digest(env_file)!=actor.meta['env_yaml_sha256']: raise ValueError('Saved env changed')
    cfg=yaml.load(env_file.read_text(),Loader=yaml.BaseLoader)
    with args.trajectory.open() as f:
        rows=[r for r in csv.DictReader(f) if int(r['robot_id'])==1 and int(r['step'])<3500]
    # Recorded trajectory is a historical hard-b state dataset. This is a clean
    # counterfactual function check, NOT a new clean rollout or physical result.
    states=[]
    for r in rows:
        yaw=float(r['yaw']); vx,vy=float(r['vx']),float(r['vy'])
        states.append(dict(x=float(r['x']),y=float(r['y']),yaw=yaw,
            vx_body=math.cos(yaw)*vx+math.sin(yaw)*vy,
            vy_body=-math.sin(yaw)*vx+math.cos(yaw)*vy,yaw_rate=float(r['yaw_rate'])))
    recorded_count=len(states)
    rng=np.random.RandomState(20261116)
    for i in range(2048):
        states.append(dict(x=float(rng.uniform(-9,0)),y=float(rng.uniform(-1.7,.4)),
            yaw=float(rng.uniform(-.5,.5)),vx_body=float(rng.uniform(-.1,.5)),
            vy_body=float(rng.uniform(-.1,.1)),yaw_rate=float(rng.uniform(-.4,.4))))
    n=len(states)
    tensor=lambda key: torch.tensor([s[key] for s in states],dtype=torch.float32)
    z=torch.zeros(n); yaw=tensor('yaw')
    data=types.SimpleNamespace(root_pos_w=torch.stack((tensor('x'),tensor('y'),z),dim=-1),
        root_quat_w=torch.stack((torch.cos(yaw/2),z,z,torch.sin(yaw/2)),dim=-1),
        root_lin_vel_b=torch.stack((tensor('vx_body'),tensor('vy_body'),z),dim=-1),
        root_ang_vel_b=torch.stack((z,z,tensor('yaw_rate')),dim=-1))
    names=('robot','robot_2','robot_3','robot_4','robot_5')
    path=profile['path']
    term=types.SimpleNamespace(cfg=types.SimpleNamespace(lane_change_enabled=True,lane_change_spatial=True,
        lane_change_start_x=path['start_x'],lane_change_length_m=path['length_m'],lane_start_y=path['start_y'],
        lane_end_y=path['end_y'],enforce_road_boundaries=True),nominal_speed_w=torch.full((n,),.35))
    env=types.SimpleNamespace(device='cpu',num_envs=n,scene={k:types.SimpleNamespace(data=data) for k in names},
        cfg=types.SimpleNamespace(algorithm=types.SimpleNamespace(spatial_path_features=False)),
        command_manager=types.SimpleNamespace(_terms={'base_velocity':term}))
    module=types.ModuleType('ams_source_parity'); sys.modules[module.__name__]=module
    ns=module.__dict__
    ns.update(torch=torch,dataclass=dataclasses.dataclass,ShieldModule=object,
              PLATOON_ASSET_NAMES=names,PLATOON_ROBOTS=names,quat_apply=rotate,
              quat_apply_inverse=lambda q,v:rotate(q,v,True))
    extract(args.source_root/files[1],['_env_local_position','_lane_change_term','quintic_lane_change_spatial_profile',
        'get_platoon_lane_change_reference','reference_heading_vectors','road_boundary_enforced'],ns)
    extract(args.source_root/files[0],['_env_local_y','obs_platoon_chain'],ns)
    expected_obs=ns['obs_platoon_chain'](env)[:,:11].numpy()
    actual_obs=np.stack([leader_observation(s,profile) for s in states])
    np.testing.assert_allclose(actual_obs,expected_obs,atol=5e-6,rtol=1e-5)
    extract(args.source_root/files[2],['BadHeadingShieldCfg','BadHeadingShieldModule'],ns)
    shield_cfg=ns['BadHeadingShieldCfg'](enabled=True,expert_path_tracking_enabled=False,expert_blend_ratio=0.)
    for k,v in profile['shield'].items(): setattr(shield_cfg,k,v)
    shield=ns['BadHeadingShieldModule'](env,shield_cfg)
    raw=actor(actual_obs)
    clipped=torch.from_numpy(raw).clamp(-profile['action_clip'],profile['action_clip']).unsqueeze(1)
    expected_semantic=shield.project_action(None,clipped).clamp(-profile['action_clip'],profile['action_clip'])[:,0,:]
    actual_results=[leader_action(raw[i],s,profile) for i,s in enumerate(states)]
    actual_semantic=np.stack([r[1] for r in actual_results])
    np.testing.assert_allclose(actual_semantic,expected_semantic.numpy(),atol=2e-6,rtol=1e-5)
    # Run unchanged actual ActionTerm.process_actions with tensor buffers.
    action_tree=ast.parse((args.source_root/files[3]).read_text())
    cls=next(c for c in action_tree.body if isinstance(c,ast.ClassDef) and c.name=='AckermannAction')
    method=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='process_actions')
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[method],type_ignores=[]),
                 str(args.source_root/files[3]),'exec'),ns)
    a=cfg['actions']['ackermann_1']
    numeric=['max_speed_mps','max_steering_rad','wheelbase_m','track_width_m','wheel_radius_m',
             'rear_left_axis_sign','rear_right_axis_sign','front_joint_limit_rad','front_left_axis_sign','front_right_axis_sign']
    adapter=types.SimpleNamespace(cfg=types.SimpleNamespace(asset_name='robot',**{k:float(a[k]) for k in numeric}),
        _raw_actions=torch.zeros(n,2),_processed_actions=torch.zeros(n,2),_rear_targets=torch.zeros(n,2),
        _steer_targets=torch.zeros(n,2),_env=types.SimpleNamespace())
    ns['process_actions'](adapter,expected_semantic)
    expected_physical=adapter._processed_actions.numpy()
    actual_physical=np.stack([r[0] for r in actual_results])
    np.testing.assert_allclose(actual_physical,expected_physical,atol=1e-6,rtol=1e-5)
    args.output.mkdir(parents=True)
    with (args.output/'offline_candidates.csv').open('x',newline='') as f:
        fields=['dataset','x','y','yaw','vx_body','vy_body','yaw_rate','raw_speed','raw_turn','candidate_speed_mps','candidate_steering_rad']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for i,s in enumerate(states):
            writer.writerow(dict(s,dataset='historical_hard_b_state_clean_inference' if i<recorded_count else 'synthetic',
                raw_speed=float(raw[i,0]),raw_turn=float(raw[i,1]),candidate_speed_mps=float(actual_physical[i,0]),
                candidate_steering_rad=float(actual_physical[i,1])))
    report=dict(passed=True,recorded_states=recorded_count,synthetic_states=n-recorded_count,
        obs_max_absolute_error=float(np.max(np.abs(expected_obs-actual_obs))),
        semantic_action_max_absolute_error=float(np.max(np.abs(expected_semantic.numpy()-actual_semantic))),
        physical_action_max_absolute_error=float(np.max(np.abs(expected_physical-actual_physical))),
        source_hash_matches=evidence,queue_manifest_sha256=digest(args.queue_root/'queue_manifest.json'),
        env_yaml_sha256=digest(env_file),trajectory_sha256=digest(args.trajectory),model=actor.meta,
        limits=['No ROS, physics, hardware or closed-loop rollout executed.',
                'Historical hard-b states are inputs for clean function parity, not clean success evidence.',
                'Harness supplies planar quaternion rotation and environment tensors; original function bodies unchanged.',
                'Additional physical commissioning guard limits intentionally differ from simulator execution.'])
    (args.output/'source_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_hash_matches','model']},indent=2))


if __name__=='__main__': main()
