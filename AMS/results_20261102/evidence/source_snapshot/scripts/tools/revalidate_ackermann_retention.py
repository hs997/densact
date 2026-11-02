#!/usr/bin/env python3
"""Revalidate existing retention checkpoints; never train or overwrite old results."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil

import yaml

from ackermann_validation import SEEDS, clean_passed, digest, write_json
from run_ackermann_verified_followup import Followup, LEVELS, ROOT, note


STRICT_EVAL = {
    "env.algorithm.retention_curriculum_enabled": "false",
    "env.algorithm.happo_retention_schedule_enabled": "false",
    "env.algorithm.retention_expert_start_ratio": "0.0",
    "env.algorithm.retention_expert_end_ratio": "0.0",
    "env.algorithm.attack_strength_start_scale": "1.0",
    "env.algorithm.attack_strength_ramp_updates": "0",
    "env.algorithm.attack_clean_env_fraction": "0.0",
    "env.algorithm.attack_curriculum_warmup_updates": "0",
    "env.algorithm.action_fdi_longitudinal_only": "true",
    "env.safety_shield.expert_path_tracking_enabled": "false",
    "env.safety_shield.expert_blend_ratio": "0.0",
    "env.safety_shield.expert_blend_end_ratio": "0.0",
    "env.safety_shield.expert_blend_hold_steps": "0",
    "env.safety_shield.expert_blend_decay_steps": "0",
}


def stage_inputs(stage, catchup_action=None):
    """Preserve recorded settings, with an explicit, recorded catch-up repair."""
    stage = stage.resolve()
    manifest = dict(line.split("=", 1) for line in (stage / "manifest.txt").read_text().splitlines()
                    if "=" in line)
    runs = dict(line.split("=", 1) for line in (stage / "training/runs.txt").read_text().splitlines()
                if "=" in line)
    checkpoint = (Path(runs["happo_meta_ams"]) / "model_final.pt").resolve(strict=True)
    config = stage / "training/happo_meta_ams/params/env.yaml"
    # BaseLoader does not instantiate Python objects from Isaac's YAML tags.
    saved = yaml.load(config.read_text(), Loader=yaml.BaseLoader)
    values = dict(token.split("=", 1) for token in shlex.split(manifest["common_extra_overrides"]))
    for key, value in saved["safety_shield"].items():
        if not isinstance(value, str):
            raise ValueError(f"Unsupported non-scalar safety setting {key}")
        values[f"env.safety_shield.{key}"] = value
    for key, value in saved["commands"]["base_velocity"].items():
        if key.startswith("completion_"):
            values[f"env.commands.base_velocity.{key}"] = value
    for key in ("spatial_path_features", "happo_action_clip"):
        values[f"env.algorithm.{key}"] = saved["algorithm"][key]
    values.update(STRICT_EVAL)
    repairs = {}
    if catchup_action is not None:
        limit = float(saved["algorithm"]["happo_action_clip"])
        if not math.isfinite(catchup_action) or not 0 < catchup_action <= limit:
            raise ValueError("catchup_action must be positive and within the saved action clip")
        key = "env.safety_shield.catchup_action"
        repairs[key] = {"saved": values[key], "evaluation": str(catchup_action)}
        values[key] = str(catchup_action)
    return checkpoint, [f"{k}={v}" for k, v in values.items()], {
        "stage": str(stage), "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(stage / "manifest.txt"), "env_yaml_sha256": digest(config),
        "recorded_training_seed": saved["seed"],
        "explicit_repairs": repairs,
    }


class RetentionValidation(Followup):
    def __init__(self, args):
        # Reuse the established bounded validator/rendering machinery, without
        # the six-method constructor's unrelated BC and hard-b prerequisites.
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.current = {}
        self.environment = dict(os.environ)
        for key in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
            self.environment.pop(key, None)
        self.environment["PYTHONPATH"] = ":".join(str(ROOT / p) for p in (
            "source/isaaclab", "source/isaaclab_rl", "source/isaaclab_tasks", "source/my_exts"))
        self.environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        files = sorted((ROOT / "source/my_exts/marl_platoon").rglob("*.py"))
        files += [ROOT / p for p in (
            "scripts/tools/revalidate_ackermann_retention.py", "scripts/tools/ackermann_validation.py",
            "scripts/tools/run_ackermann_verified_followup.py", "scripts/tools/run_platoon_spatial_attack_ladder.sh",
            "scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py",
            "scripts/reinforcement_learning/rsl_rl/play.py")]
        self.sources = {str(p.relative_to(ROOT)): digest(p) for p in files}
        self.evaluation_sources = {k: v for k, v in self.sources.items()
            if k.startswith("source/my_exts/marl_platoon/") or k in (
                "scripts/tools/ackermann_validation.py", "scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py")}
        self.render_sources = self.sources
        self.items = [stage_inputs(p, args.catchup_action) for p in args.stage_root]
        manifest = dict(revision=1, stages=[dict(info, common=common) for _, common, info in self.items],
                        levels=[list(level) for level in LEVELS if level[0] in args.levels],
                        eval_envs=16, eval_steps=3500, attack_seeds=SEEDS, sources=self.sources,
                        render=args.render, training_performed=False)
        manifest = json.loads(json.dumps(manifest))
        path = self.output / "revalidation_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Resume identity mismatch; use a new output directory")
        write_json(path, manifest)
        for source in files:
            copy = self.output / "source_snapshot" / source.relative_to(ROOT)
            if not copy.exists():
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, copy)

    def perform(self):
        rows = {}
        # Complete all three clean regressions before unlocking any attacks.
        for idx, (checkpoint, common, info) in enumerate(self.items):
            self.common = common
            stage = self.output / f"candidate_{idx}" / "clean"
            verified = stage / "validation/clean_env42"
            clean = self.evaluate("happo_meta_ams", checkpoint, LEVELS[0], verified)
            passed = clean_passed(clean)
            report = dict(passed=passed, clean_ok=passed, safe=passed, rows=[clean], parent=info,
                          protocol_verified=True, training_performed=False)
            write_json(stage / "validation_gate.json", report)
            rows[f"candidate_{idx}"] = report
            write_json(self.output / "outcomes.json", rows)
            self.run([self.args.host_python, "scripts/tools/plot_ackermann_recovery_stage.py", str(stage)],
                     stage / "plotting.log")
            note(f"- retention 严格无专家 clean 复验 candidate_{idx}: passed={passed}, "
                 f"completion={clean['completion']}, collision={clean['collision']}, road_any={clean['road_any']}；"
                 f"已有模型未重训，结果 `{stage}`。")
        if not all(r["passed"] for r in rows.values()):
            self.status("clean_rejected", attacks_unlocked=False, outcomes=rows)
            # Diagnostic-only exports cannot approve a model or unlock attacks.
            if self.args.render:
                for idx, (checkpoint, common, _) in enumerate(self.items):
                    self.common = common
                    stage = self.output / f"candidate_{idx}" / "clean"
                    self.evidence("happo_meta_ams", checkpoint, LEVELS[0], stage / "evidence",
                                  stage / "validation/clean_env42",
                                  diagnostic_clean=not rows[f"candidate_{idx}"]["passed"])
            self.status("complete_clean_rejected", attacks_unlocked=False, outcomes=rows)
            return
        # Retain a clean demonstration even when subsequent attack gates fail.
        if self.args.render:
            for idx, (checkpoint, common, _) in enumerate(self.items):
                self.common = common
                stage = self.output / f"candidate_{idx}" / "clean"
                self.evidence("happo_meta_ams", checkpoint, LEVELS[0], stage / "evidence",
                              stage / "validation/clean_env42")
        for level in (l for l in LEVELS[1:] if l[0] in self.args.levels):
            passed_all = True
            for idx, (checkpoint, common, _) in enumerate(self.items):
                self.common = common
                stage = self.output / f"candidate_{idx}" / level[0]
                stage.mkdir(parents=True, exist_ok=True)
                # evaluate() reuses only hash/command-identical verified clean.
                (stage / "validation").mkdir(exist_ok=True)
                clean_link = stage / "validation/clean_env42"
                if not clean_link.exists():
                    clean_link.symlink_to(self.output / f"candidate_{idx}/clean/validation/clean_env42",
                                          target_is_directory=True)
                report = self.validate("happo_meta_ams", checkpoint, level, stage)
                rows[f"candidate_{idx}"][level[0]] = report
                write_json(self.output / "outcomes.json", rows)
                passed_all &= report["passed"]
                self.run([self.args.host_python, "scripts/tools/plot_ackermann_recovery_stage.py", str(stage)],
                         stage / "plotting.log")
                if self.args.render:
                    self.evidence("happo_meta_ams", checkpoint, level, stage / "evidence",
                                  stage / "validation/attack_env42_seed3407")
                note(f"- retention candidate_{idx}/{level[0]} 严格复验 passed={report['passed']}，"
                     f"attack_min={report['attack_completion_min']}，结果 `{stage}`。")
            if not passed_all:
                self.status("complete_attack_rejected", stopped_level=level[0], outcomes=rows)
                return
        self.status("complete_revalidated", outcomes=rows, training_performed=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", choices=[l[0] for l in LEVELS[1:]],
                        default=["light", "easy_0", "easy_a", "easy_b", "med_a"])
    parser.add_argument("--sim", type=Path, default=ROOT.parent / "isaac-sim")
    parser.add_argument("--host-python", default="/home/cnc/Downloads/ENTER/bin/python3")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--catchup-action", type=float,
                        help="Explicit normalized catch-up repair; recorded in the new manifest, never saved over training config")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.eval_envs = 16
    args.reuse_root = None
    with (ROOT / "logs/ams_ackermann_gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = RetentionValidation(args)
        if args.preflight_only:
            runner.status("preflight_passed", training_performed=False)
            return
        note(f"- 启动 retention 已有 checkpoint 严格复验（不训练），新目录 `{runner.output}`；"
             "无专家 clean 全部通过才解锁低到高攻击；全过程保持原始模型与旧产物。")
        try:
            runner.perform()
        except BaseException as exc:
            runner.status("error", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
