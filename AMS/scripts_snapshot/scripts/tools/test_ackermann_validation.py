"""CPU-only protocol regression tests; no simulator or project artifacts modified."""
import csv
import tempfile
import unittest
from pathlib import Path

from ackermann_validation import inspect, digest, write_json, gate, load_verified, SEEDS


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'model_final.pt'
        self.checkpoint.write_bytes(b'fixture checkpoint')

    def fixture(self, name='clean_env42', enabled=False, env_seed=42, attack_seed=3407, completion=1):
        output = self.root / 'validation' / name
        output.mkdir(parents=True, exist_ok=True)
        spec = dict(checkpoint=str(self.checkpoint), checkpoint_sha256=digest(self.checkpoint),
                    steps=2, num_envs=16, env_seed=env_seed, attack_seed=attack_seed,
                    enabled=enabled, level='hard' if enabled else 'off',
                    pos=2.65 if enabled else 0, acc=.71 if enabled else 0, dos=.115 if enabled else 0)
        row = dict(num_envs=16, num_agents=5, attack_enabled=int(enabled), attack_level=spec['level'],
                   attack_max_fdi_pos=spec['pos'], attack_max_fdi_acc=spec['acc'], attack_max_dos_rate=spec['dos'],
                   shield_expert_blend_ratio=0, attack_strength_scale=1, attack_clean_env_fraction=0,
                   attack_action_dim=2, attack_action_fdi_longitudinal_only=1,
                   platoon_completion_ever_rate=completion, platoon_completion_instant_rate=completion,
                   platoon_completion_dwell_rate=completion, collision_rate=0, road_boundary_any_rate=0,
                   termination_road_boundary_exceeded=0, eval_total_reward_mean=1, eval_step=1)
        row.update({f'trajectory_progress_robot_{i}_mean': 1 for i in range(1, 6)})
        rows = [row, dict(row, eval_step=2)]
        summary = dict(checkpoint_path=str(self.checkpoint), eval_rows=2, summary_rows=2,
                       platoon_completion_ever_rate_final=completion,
                       platoon_completion_instant_rate_peak=completion, platoon_completion_dwell_rate_peak=completion)
        self.csv(output / 'eval_steps_model_final.csv', rows)
        self.csv(output / 'eval_summary.csv', [summary])
        write_json(output / 'verified.json', dict(spec=spec, artifacts={p.name: digest(p) for p in output.glob('*.csv')}))
        return output, spec, rows

    @staticmethod
    def csv(path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_gate_and_seed_identity(self):
        self.fixture()
        for env, attack in SEEDS:
            self.fixture(f'attack_env{env}_seed{attack}', True, env, attack, .5)
        self.assertTrue(gate(self.root, .5)['passed'])
        self.assertFalse(gate(self.root, .55)['passed'])
        self.fixture('attack_env777_seed17011', True, 42, 3407, .5)
        with self.assertRaisesRegex(ValueError, 'seeds'):
            gate(self.root, .5)

    def test_clean_failure_stops_attack_gate(self):
        self.fixture(completion=.9375)
        self.assertFalse(gate(self.root, .5)['passed'])

    def test_protocol_corruptions(self):
        for key, value in [('attack_enabled', 0), ('attack_strength_scale', .25),
                           ('attack_action_dim', 1), ('attack_clean_env_fraction', .35),
                           ('shield_expert_blend_ratio', .1), ('num_agents', 4),
                           ('attack_max_fdi_acc', .5), ('eval_total_reward_mean', float('nan')),
                           ('eval_step', 3)]:
            with self.subTest(key=key):
                output, spec, rows = self.fixture('attack_env42_seed3407', True)
                rows[-1][key] = value
                self.csv(output / 'eval_steps_model_final.csv', rows)
                with self.assertRaises(ValueError):
                    inspect(output, spec)

    def test_truncation_and_hash_change(self):
        output, spec, rows = self.fixture()
        self.csv(output / 'eval_steps_model_final.csv', rows[:1])
        with self.assertRaises(ValueError):
            inspect(output, spec)
        with self.assertRaisesRegex(ValueError, 'changed artifact'):
            load_verified(output)

    def test_single_unsafe_step_cannot_average_away(self):
        self.fixture()
        for env, attack in SEEDS:
            output, spec, rows = self.fixture(f'attack_env{env}_seed{attack}', True, env, attack, .5)
        rows[0]['road_boundary_any_rate'] = .0625
        self.csv(output / 'eval_steps_model_final.csv', rows)
        write_json(output / 'verified.json', dict(spec=spec, artifacts={p.name: digest(p) for p in output.glob('*.csv')}))
        self.assertFalse(gate(self.root, .5)['safe'])


if __name__ == '__main__':
    unittest.main()
