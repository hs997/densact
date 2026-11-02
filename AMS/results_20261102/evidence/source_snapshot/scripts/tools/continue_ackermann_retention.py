#!/usr/bin/env python3
"""Continue repaired, verified retention models through the remaining attack levels."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil

import torch
import yaml

from ackermann_validation import SEEDS, clean_passed, digest, load_verified, write_json
from revalidate_ackermann_retention import stage_inputs
from run_ackermann_verified_followup import Followup, LEVELS, ROOT, note


def checkpoint_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["platoon_happo_state"]["runner"]


def actor_learning_rate(state):
    rates = [float(group["lr"]) for optimizer in state["actor_optimizers"]
             for group in optimizer["param_groups"]]
    if not rates or any(not math.isfinite(v) or v <= 0 for v in rates) or len(set(rates)) != 1:
        raise ValueError("Expected one finite, positive actor learning rate across restored optimizers")
    return rates[0]


def training_overrides(common, algorithm, level, lr, attack_seed):
    """Keep the saved algorithm; isolate adaptation from the expert curriculum."""
    values = {f"env.algorithm.{key}": value for key, value in algorithm.items()}
    values.update(dict(item.split("=", 1) for item in common))
    _, base, pos, acc, dos, _ = level
    values.update({
        "env.algorithm.happo_actor_lr": str(lr),
        "env.algorithm.enable_attack": "true",
        "env.algorithm.attack_level": base,
        "env.algorithm.attack_mode": "profile",
        "env.algorithm.attack_seed": str(attack_seed),
        "env.algorithm.max_fdi_pos": str(pos),
        "env.algorithm.max_fdi_acc": str(acc),
        "env.algorithm.max_dos_rate": str(dos),
        # The external checkpoint-bound clean gate authorizes this stage.
        # No resumed global counter is used as a new attack ramp clock.
        "env.algorithm.attack_curriculum_warmup_updates": "0",
        "env.algorithm.attack_strength_start_scale": "1.0",
        "env.algorithm.attack_strength_ramp_updates": "0",
        "env.algorithm.attack_clean_env_fraction": "0.5",
        "env.algorithm.retention_curriculum_enabled": "false",
        "env.algorithm.retention_expert_start_ratio": "0.0",
        "env.algorithm.retention_expert_end_ratio": "0.0",
        "env.algorithm.happo_retention_enabled": "true",
        "env.algorithm.happo_retention_schedule_enabled": "false",
        # Continue with the previous schedule's terminal coefficients, without
        # accidentally restarting expert acquisition or using absolute counters.
        "env.algorithm.happo_retention_bc_coef": algorithm["happo_retention_bc_end_coef"],
        "env.algorithm.happo_retention_kl_coef": algorithm["happo_retention_kl_end_coef"],
        "env.safety_shield.expert_path_tracking_enabled": "false",
        "env.safety_shield.expert_blend_ratio": "0.0",
        "env.safety_shield.expert_blend_end_ratio": "0.0",
        "env.safety_shield.expert_blend_decay_steps": "0",
        "env.safety_shield.catchup_action": "0.4",
    })
    return [f"{k}={v}" for k, v in values.items()]


def audit_training(run_dir, parent, level, updates, expected_lr):
    """Require actual policy updates and effective attacks, not just a checkpoint."""
    candidate = run_dir / "model_final.pt"
    before, after = checkpoint_state(parent), checkpoint_state(candidate)
    if int(after["update_count"]) - int(before["update_count"]) != updates:
        raise ValueError("Wrong HAPPO update count; refusing to repeat or promote ambiguous training")
    if actor_learning_rate(after) != expected_lr or float(after["cfg"]["actor_lr"]) != expected_lr:
        raise ValueError("Configured and restored actor learning rates disagree")
    changed = any(not torch.equal(old[key], new[key])
                  for old, new in zip(before["actors"], after["actors"]) for key in old)
    if not changed:
        raise ValueError("Actor parameters did not change")
    with (run_dir / "platoon_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != updates:
        raise ValueError("Incomplete training metrics")
    _, base, pos, acc, dos, _ = level
    expected = dict(attack_enabled=1, shield_expert_blend_ratio=0, attack_strength_scale=1,
                    attack_clean_env_fraction=.5, attack_max_fdi_pos=pos,
                    attack_max_fdi_acc=acc, attack_max_dos_rate=dos, ams_enabled=1)
    for row in rows:
        for key, value in expected.items():
            if not math.isclose(float(row[key]), value, abs_tol=1e-6):
                raise ValueError(f"Training protocol mismatch: {key}")
        if row["attack_level"] != base or row["attack_mode"] != "profile":
            raise ValueError("Training attack profile mismatch")
        for key in ("bc_loss", "reference_kl", "retention_loss", "ams_q_loss", "reward_env_mean"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite training metric: {key}")
    if max(float(row["action_attack_delta_abs_mean"]) for row in rows) <= 0:
        raise ValueError("No observed action perturbation during attack training")
    return dict(checkpoint=str(candidate), checkpoint_sha256=digest(candidate),
                parent_sha256=digest(parent), updates_before=int(before["update_count"]),
                updates_after=int(after["update_count"]), actual_actor_lr=expected_lr,
                actor_parameters_changed=True, protocol_verified=True,
                metrics_sha256=digest(run_dir / "platoon_metrics.csv"),
                action_attack_delta_abs_mean_max=max(float(x["action_attack_delta_abs_mean"]) for x in rows))


class RetentionContinuation(Followup):
    def __init__(self, args):
        self.args, self.output, self.current = args, args.output.resolve(), {}
        self.output.mkdir(parents=True, exist_ok=True)
        self.environment = dict(os.environ)
        for key in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
            self.environment.pop(key, None)
        self.environment["PYTHONPATH"] = ":".join(str(ROOT / p) for p in (
            "source/isaaclab", "source/isaaclab_rl", "source/isaaclab_tasks", "source/my_exts"))
        self.environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        files = sorted((ROOT / "source/my_exts/marl_platoon").rglob("*.py")) + [ROOT / p for p in (
            "scripts/tools/continue_ackermann_retention.py", "scripts/tools/revalidate_ackermann_retention.py",
            "scripts/tools/ackermann_validation.py", "scripts/tools/run_ackermann_verified_followup.py",
            "scripts/tools/run_medium_algorithm_comparison.sh", "scripts/tools/run_platoon_spatial_attack_ladder.sh",
            "scripts/reinforcement_learning/rsl_rl/train.py", "scripts/reinforcement_learning/rsl_rl/play.py",
            "scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py")]
        self.sources = {str(p.relative_to(ROOT)): digest(p) for p in files}
        self.evaluation_sources = {k: v for k, v in self.sources.items()
            if k.startswith("source/my_exts/marl_platoon/") or k in (
                "scripts/tools/ackermann_validation.py", "scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py")}
        self.render_sources = self.sources
        self.items, self.algorithms, self.provenance = [], [], []
        for index in range(3):
            reference = args.reference.resolve() / f"candidate_{index}"
            parent_info = json.loads((reference / "clean/validation_gate.json").read_text())["parent"]
            stage = Path(parent_info["stage"])
            checkpoint, common, provenance = stage_inputs(stage, .4)
            checked = []
            for name in ["clean_env42", *[f"attack_env{e}_seed{a}" for e, a in SEEDS]]:
                row, spec = load_verified(reference / "med_a/validation" / name)
                if spec["checkpoint_sha256"] != digest(checkpoint) or spec["sources"] != self.evaluation_sources:
                    raise ValueError("Parent checkpoint/source differs from verified med_a")
                if name == "clean_env42":
                    if not clean_passed(row):
                        raise ValueError("Parent clean gate failed")
                elif ((spec["level"], spec["pos"], spec["acc"], spec["dos"]) != ("medium", 1.75, .45, .09)
                      or row["completion"] < .65
                      or any(row[k] != 0 for k in ("collision", "road_any", "road_termination"))):
                    raise ValueError("Parent med_a gate failed")
                checked.append(dict(path=str(reference / "med_a/validation" / name), metrics=row))
            algorithm = yaml.load((stage / "training/happo_meta_ams/params/env.yaml").read_text(),
                                  Loader=yaml.BaseLoader)["algorithm"]
            if any(not isinstance(v, str) for v in algorithm.values()):
                raise ValueError("Unexpected non-scalar algorithm configuration")
            state = checkpoint_state(checkpoint)
            lr = actor_learning_rate(state)
            self.items.append((checkpoint, common, lr))
            self.algorithms.append(algorithm)
            self.provenance.append(dict(provenance, verified_parent=checked, actual_actor_lr=lr,
                                       saved_config_actor_lr=state["cfg"]["actor_lr"]))
        manifest = dict(reference=str(args.reference.resolve()), parents=self.provenance, sources=self.sources,
                        levels=[list(x) for x in LEVELS if x[0] in ("med_b", "hard_a", "hard_b")],
                        updates=args.updates, train_envs=32, eval_envs=16, eval_steps=3500,
                        train_seeds=[42, 3407, 777], attack_seeds=SEEDS,
                        training_expert_ratio=0, training_clean_fraction=.5, training_strength=1,
                        training_performed=True, render=args.render)
        manifest = json.loads(json.dumps(manifest))
        path = self.output / "continuation_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Continuation identity mismatch; use a new output")
        write_json(path, manifest)
        for source in files:
            target = self.output / "source_snapshot" / source.relative_to(ROOT)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def train_candidate(self, index, parent, level, stage):
        lr = actor_learning_rate(checkpoint_state(parent))
        extra = training_overrides(self.common, self.algorithms[index], level, lr, SEEDS[index][1])
        tag = f"{self.output.name}_s{index}_{level[0]}"
        signature = dict(parent_sha256=digest(parent), level=list(level), updates=self.args.updates,
                         seed=SEEDS[index][0], common=extra, sources=self.sources, actor_lr=lr)
        requested = stage / "training_request.json"
        if requested.exists() and json.loads(requested.read_text()) != signature:
            raise ValueError("Training request changed; refusing to overwrite")
        marker = stage / "training_complete.json"
        if marker.exists():
            recorded = json.loads(marker.read_text())
            if recorded["signature"] != signature:
                raise ValueError("Completed training identity mismatch")
            checkpoint = Path(recorded["audit"]["checkpoint"])
            current = audit_training(checkpoint.parent, parent, level, self.args.updates, lr)
            if current != recorded["audit"]:
                raise ValueError("Completed training artifacts changed")
            return checkpoint
        candidates = list((ROOT / "logs/rsl_rl/platoon_happo").glob(f"*_{tag}_happo_meta_ams/model_final.pt"))
        if requested.exists() and len(candidates) == 1:
            # A simulator teardown or later packaging error must not cause a
            # finished, auditable training run to be repeated on resume.
            checkpoint = candidates[0]
            audit = audit_training(checkpoint.parent, parent, level, self.args.updates, lr)
            (stage / "training").mkdir(exist_ok=True)
            (stage / "training/runs.txt").write_text(f"happo_meta_ams={checkpoint.parent}\n")
            write_json(marker, dict(signature=signature, audit=audit, adopted_after_restart=True))
            return checkpoint
        if requested.exists():
            raise RuntimeError(f"Partial training retained for diagnosis; refusing blind repeat: {stage}")
        stage.mkdir(parents=True, exist_ok=True)
        write_json(requested, signature)
        seed_dir = ROOT / "logs/rsl_rl/platoon_happo" / f"{tag}_parent"
        seed_dir.mkdir(parents=True, exist_ok=True)
        link = seed_dir / "model_final.pt"
        if link.exists() or link.is_symlink():
            if link.resolve() != parent.resolve():
                raise ValueError("Parent alias already names another model")
        else:
            link.symlink_to(parent.resolve())
        resume = stage / "resume_runs.txt"
        resume.write_text(f"happo_meta_ams={seed_dir}\n")
        _, base, pos, acc, dos, _ = level
        environment = dict(self.environment, TAG=tag, PIPELINE_LOG=str(stage / "training_driver.log"),
            RESULT_ROOT=str(stage), RESULT_TAR=str(stage.with_suffix(".tar.gz")),
            PROGRESS_NOTES_PATH=str(ROOT.parent / "ams_debug_notes.md"),
            NUM_ENVS="32", EVAL_NUM_ENVS="16", SEED=str(SEEDS[index][0]),
            MAX_ITERATIONS=str(self.args.updates), EVAL_EVERY=str(self.args.updates), EVAL_STEPS="3500",
            SKIP_EVAL="1", FRESH_ENV_PER_CHECKPOINT="1", PARALLEL_TRAIN_JOBS="1",
            COMPARE_ALGOS="happo_meta_ams", RESUME_RUNS_FILE=str(resume), ENABLE_ATTACK="true",
            ATTACK_PROFILE_LABEL=level[0], ATTACK_LEVEL=base, ATTACK_MODE="profile",
            ATTACK_MAX_FDI_POS=str(pos), ATTACK_MAX_FDI_ACC=str(acc), ATTACK_MAX_DOS_RATE=str(dos),
            SAFETY_CATCHUP_ACTION="0.4", COMMON_EXTRA_OVERRIDES=" ".join(extra),
            HAPPO_META_EXTRA_OVERRIDES="", HAPPO_META_TRAIN_EXTRA_OVERRIDES=" ".join(extra),
            HAPPO_META_EVAL_EXTRA_OVERRIDES="", PLOT_PYTHON=self.args.host_python)
        self.status("training", candidate=index, level=level[0], parent=str(parent), updates=self.args.updates,
                    actual_actor_lr=lr, output=str(stage))
        self.run(["bash", "scripts/tools/run_medium_algorithm_comparison.sh"],
                 self.output / "training_process.log", environment, timeout=14400)
        runs = dict(line.split("=", 1) for line in (stage / "training/runs.txt").read_text().splitlines())
        run_dir = Path(runs["happo_meta_ams"])
        audit = audit_training(run_dir, parent, level, self.args.updates, lr)
        write_json(marker, dict(signature=signature, audit=audit))
        note(f"- retention s{index}/{level[0]}真实续训完成，updates={self.args.updates}，"
             f"实际actor_lr={lr}，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`{stage}`。")
        return run_dir / "model_final.pt"

    def perform(self):
        parents = [item[0] for item in self.items]
        outcomes = {}
        for level in [x for x in LEVELS if x[0] in ("med_b", "hard_a", "hard_b")]:
            all_passed = True
            for index, (_, common, _) in enumerate(self.items):
                self.common = common
                stage = self.output / f"candidate_{index}" / level[0]
                candidate = self.train_candidate(index, parents[index], level, stage)
                report = self.validate("happo_meta_ams", candidate, level, stage)
                outcomes[f"s{index}/{level[0]}"] = report
                write_json(self.output / "outcomes.json", outcomes)
                self.run([self.args.host_python, "scripts/tools/plot_ackermann_recovery_stage.py", str(stage)],
                         stage / "plotting.log")
                if self.args.render:
                    self.evidence("happo_meta_ams", candidate, LEVELS[0], stage / "evidence",
                                  stage / "validation/clean_env42", diagnostic_clean=not report["clean_ok"])
                    if report["clean_ok"]:
                        self.evidence("happo_meta_ams", candidate, level, stage / "evidence",
                                      stage / "validation/attack_env42_seed3407")
                if report["passed"]:
                    parents[index] = candidate
                    write_json(stage / "promotion.json", dict(checkpoint=str(candidate),
                               checkpoint_sha256=digest(candidate), gate_sha256=digest(stage / "validation_gate.json")))
                all_passed &= report["passed"]
                note(f"- retention s{index}/{level[0]}续训后门禁passed={report['passed']}，"
                     f"clean={report['clean_ok']}，attack_min={report.get('attack_completion_min')}；`{stage}`。")
            if not all_passed:
                self.status("stopped_at_gate", level=level[0], outcomes=outcomes,
                            safe_parents=[str(p) for p in parents])
                return
        self.status("complete", outcomes=outcomes, checkpoints=[str(p) for p in parents])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    args.sim, args.host_python = ROOT.parent / "isaac-sim", "/home/cnc/Downloads/ENTER/bin/python3"
    args.eval_envs, args.reuse_root = 16, None
    with (ROOT / "logs/ams_ackermann_gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = RetentionContinuation(args)
        if args.preflight_only:
            runner.status("preflight_passed")
            return
        note(f"- 启动修复分支真实续训med_b→hard_a→hard_b，三种子逐级门禁；`{runner.output}`。")
        try:
            runner.perform()
        except BaseException as exc:
            runner.status("error", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
