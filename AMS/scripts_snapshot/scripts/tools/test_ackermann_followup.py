#!/usr/bin/env python3
"""CPU-only orchestration regressions; no simulator or training is started."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_ackermann_verified_followup as flow
import ackermann_validation as validation


def clean(ok=True):
    return dict(completion=float(ok), instant=float(ok), dwell=float(ok),
                collision=0, road_any=0, road_termination=0, attack_enabled=0)


class FollowupTests(unittest.TestCase):
    def runner(self, root):
        runner = object.__new__(flow.Followup)
        runner.output = root
        runner.args = argparse.Namespace(eval_envs=16, train_envs=32, stage_updates=40, rounds=3,
            sim=Path("/sim"), host_python="python3", bc_checkpoint=root / "bc.pt", actor_lr=5e-7, reuse_root=None)
        runner.common = []
        runner.environment = {}
        runner.current = {}
        runner.sources = {}
        runner.evaluation_sources = {}
        runner.render_sources = {}
        runner.args.bc_checkpoint.write_bytes(b"bc")
        return runner

    def test_clean_failure_locks_attacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self.runner(root)
            calls = []
            runner.evaluate = lambda *args: calls.append(args) or clean(False)
            report = runner.validate("mappo", root / "bc.pt", flow.LEVELS[-1], root)
            self.assertFalse(report["passed"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][2][0], "off")

    def test_evaluation_budget_literals_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self.runner(root)
            commands = []
            runner.run = lambda command, *args: commands.append(command)
            with patch.object(flow, "load_verified", return_value=(clean(), {})):
                runner.evaluate("mappo", root / "bc.pt", flow.LEVELS[0], root / "eval")
            command = commands[0]
            self.assertIn("env.algorithm.max_fdi_pos=0.0", command)
            self.assertIn("env.algorithm.max_fdi_acc=0.0", command)
            self.assertIn("env.algorithm.max_dos_rate=0.0", command)
            self.assertIn("--fresh_env_per_checkpoint", command)
            self.assertNotIn("--enable_attack_eval", command)
            self.assertIn("env.algorithm.happo_ams_enabled=false", command)

    def test_rejected_candidate_is_not_parent_or_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self.runner(root)
            candidate = root / "candidate.pt"
            candidate.write_bytes(b"bad")
            parents, exported, trained_levels = [], [], []
            runner.evaluate = lambda *args: clean()
            def train(label, parent, level, stage, *args):
                stage.mkdir(parents=True)
                parents.append(parent)
                trained_levels.append(level[0])
                return candidate
            runner.train = train
            runner.validate = lambda *args: dict(passed=False, clean_ok=True, attack_completion_min=0.0)
            runner.run = lambda *args: None
            runner.evidence = lambda *args, **kwargs: None
            runner.export = lambda label, checkpoint: exported.append(checkpoint)
            with patch.object(flow, "note"):
                runner.matrix()
            self.assertEqual(set(trained_levels), {"light"})
            self.assertEqual(len(parents), 18)
            self.assertTrue(all(parent == runner.args.bc_checkpoint for parent in parents + exported))
            outcomes = json.loads((root / "method_outcomes.json").read_text())
            self.assertEqual(len(outcomes), 6)
            self.assertTrue(all(row["status"] == "stopped_at_gate" for row in outcomes.values()))

    def test_clean_rejections_continue_all_retries_and_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self.runner(root)
            candidate = root / "failed.pt"
            candidate.write_bytes(b"failed")
            parents, diagnostics = [], []
            runner.evaluate = lambda *args: clean()
            def train(label, parent, level, stage, *args):
                stage.mkdir(parents=True)
                parents.append(parent)
                return candidate
            runner.train = train
            runner.validate = lambda *args: dict(passed=False, clean_ok=False, attack_completion_min=0.0)
            runner.run = lambda *args: None
            def evidence(label, checkpoint, level, destination, verified_dir, **kwargs):
                if kwargs.get("diagnostic_clean"):
                    self.assertEqual(level, flow.LEVELS[0])
                    diagnostics.append(checkpoint)
            runner.evidence = evidence
            runner.export = lambda label, checkpoint: self.assertEqual(checkpoint, runner.args.bc_checkpoint)
            with patch.object(flow, "note"):
                runner.matrix()
            self.assertEqual(len(parents), 18)
            self.assertEqual(len(diagnostics), 18)
            self.assertTrue(all(parent == runner.args.bc_checkpoint for parent in parents))

    def test_diagnostic_flag_never_enables_attack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self.runner(root)
            envs = []
            def run(command, log, env):
                envs.append(env)
                directory = Path(env["OUTPUT_ROOT"]) / "off"
                (directory / "video").mkdir(parents=True)
                for name in ("trajectory_env0.csv", "topdown_true_xy.png", "reward_attack_safety.png", "video/test.mp4"):
                    (directory / name).write_bytes(b"dummy")
            runner.run = run
            with patch.object(flow, "load_verified", return_value=(clean(False), {})):
                with patch.object(flow.subprocess, "run", return_value=argparse.Namespace(stdout='{"streams":[{"nb_frames":"3500","duration":"70"}]}')):
                    runner.evidence("mappo", runner.args.bc_checkpoint, flow.LEVELS[0], root / "diagnostic", root / "eval",
                                    diagnostic_clean=True)
            self.assertEqual(envs[0]["DIAGNOSTIC_CLEAN_ONLY"], "1")
            self.assertEqual(envs[0]["SINGLE_BASE_LEVEL"], "off")
            self.assertEqual(envs[0]["SINGLE_FDI_POS"], "0.0")
            with self.assertRaises(ValueError):
                runner.evidence("mappo", runner.args.bc_checkpoint, flow.LEVELS[-1], root / "forbidden", root / "eval",
                                diagnostic_clean=True)

    def test_actual_shell_gate_rejects_normal_export_but_allows_clean_diagnostic(self):
        script = (flow.ROOT / "scripts/tools/run_platoon_spatial_attack_ladder.sh").read_text()
        section = script.split("# Check the exact evaluation protocol", 1)[1]
        code = section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["gate", str(root / "evaluation"), str(root / "model.pt"), "3500", "16", "42", "3407",
                    "false", "off", "0.0", "0.0", "0.0", "0", str(root)]
            spec = dict(checkpoint=str(root / "model.pt"), steps=3500, num_envs=16, env_seed=42,
                        attack_seed=3407, enabled=False, level="off", pos=0.0, acc=0.0, dos=0.0)
            with patch.object(validation, "load_verified", return_value=(clean(False), spec)), patch("sys.argv", args):
                with self.assertRaises(SystemExit):
                    exec(code, {})
                args[12] = "1"
                exec(code, {})
                report = json.loads((root / "diagnostic_only.json").read_text())
                self.assertFalse(report["attacks_unlocked"])
                self.assertFalse(report["gate_passed"])
                args[7], args[8] = "true", "hard"
                spec.update(enabled=True, level="hard")
                with self.assertRaises(ValueError):
                    exec(code, {})

    def test_cached_evaluation_requires_same_semantics_sources_and_seeds(self):
        spec = dict(command=["sim", "--output_dir", "/old", "--ros_replay_path", "/old/replay.csv", "budget=4.0"],
                    sources={"runner.py": "sha"}, attack_seed=3407)
        expected = dict(spec, command=["sim", "--output_dir", "/new", "--ros_replay_path", "/new/replay.csv", "budget=4.0"])
        self.assertTrue(flow.same_evaluation(spec, expected))
        self.assertFalse(flow.same_evaluation(spec, dict(expected, attack_seed=9917)))
        self.assertFalse(flow.same_evaluation(spec, dict(expected, sources={"runner.py": "changed"})))
        self.assertFalse(flow.same_evaluation(spec, dict(expected, command=[*expected["command"][:-1], "budget=2.0"])))

    def test_manifest_migration_cannot_change_training_or_gates(self):
        old = dict(stage_updates=40, thresholds=[0.75, 0.50], sources={
            "scripts/tools/run_ackermann_verified_followup.py": "old",
            "source/my_exts/marl_platoon/algorithms/happo/actor.py": "unchanged"})
        new = json.loads(json.dumps(old))
        new["sources"]["scripts/tools/run_ackermann_verified_followup.py"] = "new"
        self.assertTrue(flow.coordinator_only_update(old, new))
        self.assertFalse(flow.coordinator_only_update(old, dict(new, stage_updates=600)))
        self.assertFalse(flow.coordinator_only_update(old, dict(new, thresholds=[0.25, 0.25])))
        new["sources"]["source/my_exts/marl_platoon/algorithms/happo/actor.py"] = "changed"
        self.assertFalse(flow.coordinator_only_update(old, new))

    def test_waits_for_custom_external_isaac_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self.runner(root)
            runner.current = dict(state="training", algorithm="mappo")
            child = unittest.mock.MagicMock()
            child.__enter__.return_value.wait.return_value = 0
            conflicts = [[dict(pid=123, command="kit/python custom_collect.py")], []]
            with patch.object(flow, "external_isaac_processes", side_effect=conflicts), \
                 patch.object(flow, "note"), patch.object(flow.time, "sleep") as sleep, \
                 patch.object(flow.subprocess, "run", return_value=argparse.Namespace(returncode=1, stdout="")), \
                 patch.object(flow.subprocess, "Popen", return_value=child) as launch:
                runner.run(["bash", "train.sh"], root / "training.log")
            sleep.assert_called_once_with(15)
            launch.assert_called_once()
            self.assertEqual(runner.current["state"], "training")

    def test_training_overrides_apply_to_meta_method_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self.runner(root)
            runner.common = ["env.algorithm.happo_actor_lr=1.0e-6"]
            stage = root / "stage"
            captured = []
            def run(command, log, env, **kwargs):
                captured.append(env)
                (stage / "training").mkdir(parents=True)
                actual = root / "actual_run"
                actual.mkdir()
                (actual / "model_final.pt").write_bytes(b"trained")
                (stage / "training/runs.txt").write_text(f"happo_meta_ams={actual}\n")
            runner.run = run
            with patch.object(flow, "ROOT", root):
                runner.train("happo_meta_ams", runner.args.bc_checkpoint, flow.LEVELS[1], stage, 40, 1)
            self.assertEqual(captured[0]["MAX_ITERATIONS"], "40")
            for field in ("COMMON_EXTRA_OVERRIDES", "HAPPO_META_TRAIN_EXTRA_OVERRIDES"):
                overrides = dict(item.split("=", 1) for item in captured[0][field].split())
                self.assertEqual(overrides["env.algorithm.happo_actor_lr"], "5e-07")
                self.assertEqual(overrides["env.algorithm.attack_clean_env_fraction"], "0.50")
                self.assertEqual(overrides["env.algorithm.attack_strength_ramp_updates"], "40")
                self.assertEqual(overrides["env.algorithm.attack_seed"], "9917")

    def test_promotion_requires_all_three_attack_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = self.runner(root)
            calls = []
            runner.evaluate = lambda *args: calls.append(args) or clean()
            with patch.object(flow, "gate", return_value={"passed": True}) as checker:
                runner.validate("happo_meta_ams", root / "bc.pt", flow.LEVELS[-1], root)
            self.assertEqual(len(calls), 4)
            self.assertEqual([(c[-2], c[-1]) for c in calls[1:]], list(flow.SEEDS))
            checker.assert_called_once_with(root, 0.5)


if __name__ == "__main__":
    unittest.main()
