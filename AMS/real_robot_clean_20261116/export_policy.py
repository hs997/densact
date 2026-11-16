#!/usr/bin/env python3
"""Export the verified s0 leader as finite NumPy weights; run on the training server."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from core import NumpyActor, digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if digest(args.checkpoint) != args.expected_sha256:
        raise ValueError('Checkpoint does not match the explicitly pinned SHA256')
    if args.output.exists():
        raise FileExistsError('Export into a new directory; never overwrite a validated bundle')
    env_file = args.checkpoint.parent/'params/env.yaml'
    env = yaml.load(env_file.read_text(), Loader=yaml.BaseLoader)  # no Python-object construction
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['platoon_happo_state']
    if state['observation_schema'] != 'legacy_v1_11d' or state['num_agents'] != 5:
        raise ValueError('Only the validated 11D five-vehicle policy is supported')
    runner = state['runner']
    cfg = runner['cfg']
    if (cfg['obs_dim'], cfg['act_dim']) != (11, 2):
        raise ValueError('Unexpected actor dimensions')
    actor_source = args.source_root/'source/my_exts/marl_platoon/algorithms/happo/actor.py'
    spec = importlib.util.spec_from_file_location('ams_export_actor', actor_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    network = module.GaussianActorNet(module.HAPPOActorCfg(obs_dim=11, act_dim=2,
                                                         hidden_dims=tuple(cfg['hidden_dims'])))
    network.load_state_dict(runner['actors'][0], strict=True)
    network.eval()
    torch.set_num_threads(1)
    arrays = {}
    layers = [layer for layer in network.mean_net if isinstance(layer, torch.nn.Linear)]
    for i, layer in enumerate(layers):
        arrays['w%d' % i] = layer.weight.detach().cpu().numpy().copy()
        arrays['b%d' % i] = layer.bias.detach().cpu().numpy().copy()
    a = env['actions']['ackermann_1']
    path = env['commands']['base_velocity']
    shield = env['safety_shield']
    if env['algorithm']['enable_shield'] != 'true' or shield['enable_shield'] != 'true':
        raise ValueError('Expected shield-enabled validated profile')
    if any(float(shield[k]) != 0 for k in ['forward_bias_gain', 'boundary_turn_gain', 'expert_blend_ratio']):
        raise ValueError('Unsupported additional leader shield branch')
    if shield['expert_path_tracking_enabled'] != 'false' or env['algorithm']['spatial_path_features'] != 'false':
        raise ValueError('Unsupported runtime features')
    if path['lane_change_spatial'] != 'true':
        raise ValueError('Only spatial references are exported')
    profile = {
        'action_clip': float(env['algorithm']['happo_action_clip']),
        'max_speed_mps': float(a['max_speed_mps']),
        'max_steering_rad': float(a['max_steering_rad']),
        'simulation_wheelbase_m': float(a['wheelbase_m']),
        'nominal_control_dt_s': float(env['sim']['dt'])*int(env['decimation']),
        'path': {k: float(path[v]) for k, v in [('start_x','lane_change_start_x'), ('length_m','lane_change_length_m'),
                                                ('start_y','lane_start_y'), ('end_y','lane_end_y')]},
        'shield': dict(enabled=True, **{k: float(shield[k]) for k in ['lateral_tol','centerline_turn_gain',
                                                                   'centerline_turn_clip','centerline_turn_sign']}),
        'completion': {k: float(path[v]) for k, v in [('progress','completion_progress'),
            ('lateral_m','completion_lateral_tolerance_m'), ('heading_rad','completion_heading_tolerance_rad'),
            ('dwell_s','completion_dwell_s')]},
    }
    args.output.mkdir(parents=True)
    np.savez(args.output/'actor.npz', **arrays)
    meta = dict(schema='ams_leader_numpy_v1', obs_dim=11, act_dim=2, agent_index=0,
                sizes=[11]+list(cfg['hidden_dims'])+[2], activation='ELU(alpha=1)',
                inference='deterministic Gaussian mean; no sampling, teacher, critic or online updates',
                weights_sha256=digest(args.output/'actor.npz'),
                checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=args.expected_sha256,
                env_yaml_sha256=digest(env_file), actor_source_sha256=digest(actor_source),
                profile=profile)
    (args.output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    portable = NumpyActor(args.output)
    rng = np.random.RandomState(20261116)
    observations = rng.normal(0, 1, size=(4096, 11)).astype(np.float32)
    with torch.no_grad():
        expected = network(torch.from_numpy(observations), deterministic=True)[0].numpy()
    actual = portable(observations)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    report = dict(samples=len(observations), max_absolute_error=float(np.max(np.abs(expected-actual))),
                  tolerance_atol=1e-5, tolerance_rtol=1e-5, passed=True,
                  runtime_torch_required=False, checkpoint_sha256=args.expected_sha256)
    (args.output/'export_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
