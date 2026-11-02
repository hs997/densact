#!/usr/bin/env python3
"""Resume verified AMS evidence, then run an isolated six-method curriculum.

Each promotion uses raw, hash-verified clean plus three attack-seed evaluations.
Failed candidates are retained as evidence; stronger training never follows a
failed gate. All methods start from the same real-trajectory BC checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from ackermann_validation import SEEDS, clean_passed, digest, gate, load_verified, write_json

ROOT = Path(__file__).resolve().parents[2]
# All subprocesses launched by the verification matrix inherit these paths,
# including direct invocations that do not pass through isaaclab.sh.
os.environ.setdefault("ISAACLAB_EXPERIMENT_ROOT", str(ROOT))
os.environ.setdefault("ISAACLAB_RUNTIME_ROOT", "/home/cnc/SSD_1T/xzw/ams_runtime")
os.environ.setdefault("ISAACLAB_TMP_ROOT", os.path.join(os.environ["ISAACLAB_RUNTIME_ROOT"], "IsaacLab"))
os.environ.setdefault("ISAACLAB_TMPDIR", os.path.join(os.environ["ISAACLAB_RUNTIME_ROOT"], "tmp"))
os.environ["TMPDIR"] = os.environ["ISAACLAB_TMPDIR"]
os.environ["TEMP"] = os.environ["ISAACLAB_TMPDIR"]
os.environ["TMP"] = os.environ["ISAACLAB_TMPDIR"]
NOTES = ROOT.parent / "ams_debug_notes.md"
LEVELS = (
    ("off", "off", 0.0, 0.0, 0.0, 1.0),
    ("light", "light", 0.75, 0.10, 0.02, 0.75),
    ("easy_0", "easy", 1.00, 0.18, 0.035, 0.75),
    ("easy_a", "easy", 1.25, 0.25, 0.05, 0.75),
    ("easy_b", "easy", 1.50, 0.35, 0.07, 0.75),
    ("med_a", "medium", 1.75, 0.45, 0.09, 0.65),
    ("med_b", "medium", 2.00, 0.50, 0.10, 0.55),
    ("hard_a", "hard", 2.75, 0.75, 0.12, 0.50),
    ("hard_b", "hard", 4.00, 1.30, 0.18, 0.50),
)
METHODS = {
    "mappo": ("mappo", False, False, False, "ppo", False),
    "happo_no_meta": ("happo", False, True, False, "ppo", False),
    "happo_meta_ams": ("happo", True, True, True, "ppo", True),
    "harl_mappo_shared": ("mappo", False, False, True, "ppo", False),
    "harl_haa2c": ("haa2c", False, True, True, "a2c", False),
    "harl_hatrpo": ("hatrpo", False, True, True, "trpo", False),
}
ORTHOGONAL_METHODS = {
    # Exact 2x2 factorial cells use the same shared HAPPO actor architecture.
    # Keep them outside the historical default set so that matrix remains
    # unchanged; the existing happo_meta_ams arm is the fourth cell.
    "happo_base_shared": ("happo", False, True, True, "ppo", False),
    "happo_meta": ("happo", True, True, True, "ppo", False),
    "happo_ams_only": ("happo", False, True, True, "ppo", True),
}
ALL_METHODS = {**METHODS, **ORTHOGONAL_METHODS}
# These files were revised together as one legacy-compatible AMS protocol
# migration (metrics/export paths, explicit Ackermann overrides, and timeout
# bookkeeping).  The explicit resume path must not invalidate completed
# training/evaluations; unrelated task/policy sources remain strict below.
RESUME_COMPAT_SOURCE_KEYS = {
    "scripts/tools/ackermann_validation.py",
    "scripts/tools/run_ackermann_verified_followup.py",
    "scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py",
    "scripts/reinforcement_learning/rsl_rl/play.py",
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "scripts/tools/run_medium_algorithm_comparison.sh",
    "scripts/tools/run_platoon_spatial_attack_ladder.sh",
    "source/my_exts/marl_platoon/algorithms/router.py",
}


def note(message):
    with NOTES.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"\n### {dt.datetime.now().isoformat(timespec='seconds')} AMS后续实验\n\n{message}\n")


def method_overrides(label):
    algo, teacher, factor, shared, mode, ams = ALL_METHODS[label]
    values = dict(algorithm=algo, method_variant="legacy", enable_teacher=teacher,
                  happo_use_factor=factor, happo_share_actor=shared,
                  happo_actor_update_mode=mode, happo_ams_enabled=ams)
    return [f"env.algorithm.{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in values.items()]


def semantic_command(command):
    """Only output file locations may differ when importing a recorded run."""
    result = list(command)
    for flag in ("--output_dir", "--ros_replay_path"):
        if flag in result:
            result[result.index(flag) + 1] = "<output>"
    return result


def same_evaluation(spec, expected):
    for key, value in expected.items():
        if key == "command":
            continue
        if key == "sources":
            actual_sources = spec.get("sources", {})
            # Permit only the explicit orchestration/cleanup revision.  Any
            # task, policy, evaluator or rendering source change invalidates
            # reuse and forces a fresh isolated evaluation.
            if any(actual_sources.get(name) != sha for name, sha in value.items()
                   if name not in RESUME_COMPAT_SOURCE_KEYS):
                return False
            if any(name not in value for name in actual_sources
                   if name not in RESUME_COMPAT_SOURCE_KEYS):
                return False
            continue
        if spec.get(key) != value:
            return False
    return semantic_command(spec.get("command", [])) == semantic_command(expected["command"])


def coordinator_only_update(old, new):
    """Allow infrastructure-only driver revisions, never silent protocol changes."""
    old = json.loads(json.dumps(old))
    new = json.loads(json.dumps(new))
    for key in RESUME_COMPAT_SOURCE_KEYS:
        old.get("sources", {}).pop(key, None)
        new.get("sources", {}).pop(key, None)
    return old == new


def training_signature_compatible(old, new):
    """Allow reuse after launch/timeout cleanup changes only."""
    old = json.loads(json.dumps(old))
    new = json.loads(json.dumps(new))
    for signature in (old, new):
        for key in RESUME_COMPAT_SOURCE_KEYS:
            signature.get("sources", {}).pop(key, None)
    return old == new


def external_isaac_processes(sim, ignored=()):
    """Detect custom Kit entrypoints and their active diagnostic coordinator."""
    conflicts = []
    sim = sim.resolve()
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) in ignored:
            continue
        try:
            argv = (directory / "cmdline").read_bytes().split(b"\0")
            words = [word.decode(errors="replace") for word in argv if word]
            executable = (directory / "exe").resolve(strict=True)
            if executable.is_relative_to(sim / "kit/python") or (
                "python" in executable.name and any(Path(word).name == "launch_diagnostic.py" for word in words[1:])):
                conflicts.append(dict(pid=int(directory.name), command=" ".join(words)))
        except (OSError, ValueError):
            continue  # Processes may exit during the snapshot.
    return conflicts


def protocol_from_reference(reference):
    """Keep the passing path, completion, safety and retention settings."""
    _, spec = load_verified(reference / "validation/clean_env42")
    drop = {"algorithm", "enable_teacher", "happo_use_factor", "happo_share_actor",
            "happo_actor_update_mode", "happo_ams_enabled", "enable_attack",
            "attack_level", "attack_seed", "max_fdi_pos", "max_fdi_acc", "max_dos_rate"}
    overrides = []
    for item in spec["command"]:
        if not item.startswith("env."):
            continue
        if item.startswith("env.algorithm.attack_"):
            continue  # Stage-specific train/eval schedules are set explicitly.
        if item.startswith("env.algorithm.") and item.split("=", 1)[0].rsplit(".", 1)[-1] in drop:
            continue
        overrides.append(item)
    overrides += ["env.algorithm.enable_shield=true", "env.algorithm.happo_action_clip=0.4",
                  "env.algorithm.spatial_path_features=false",
                  "env.algorithm.retention_curriculum_enabled=false",
                  "env.algorithm.happo_retention_schedule_enabled=false",
                  "env.algorithm.action_fdi_longitudinal_only=true"]
    # Shell training/rendering helpers otherwise supply different shield defaults.
    overrides += [f"env.safety_shield.{key}={value}" for key, value in {
        "lateral_tol": 0.035, "lateral_turn_gain": 0.30, "lateral_turn_clip": 0.08,
        "lateral_velocity_gain": 0.10, "centerline_turn_gain": 0.25,
        "centerline_turn_clip": 0.08, "pair3_lateral_gain_scale": 1.0,
        "pair4_lateral_gain_scale": 1.0, "first_follower_centerline_gain": 0.0,
        "first_follower_centerline_clip": 0.0, "forward_bias_gain": 0.0,
        "forward_bias_clip": 0.0, "forward_bias_speed_margin": 0.02,
        "forward_bias_min_command": 0.0, "forward_bias_min_gap": 0.75,
        "forward_bias_leader_gain_scale": 1.0, "forward_bias_leader_clip_scale": 1.0,
    }.items()]
    return overrides


def verify_hard_b(reference):
    report = json.loads((reference / "validation_gate.json").read_text())
    if report.get("threshold", 0) < 0.5 or not report.get("passed"):
        raise ValueError("Reference hard-b gate has not passed the original threshold")
    evaluations = [load_verified(reference / "validation" / name) for name in
                   ["clean_env42", *[f"attack_env{e}_seed{a}" for e, a in SEEDS]]]
    if not clean_passed(evaluations[0][0]):
        raise ValueError("Reference clean gate failed")
    expected_sha = evaluations[0][1]["checkpoint_sha256"]
    for row, spec in evaluations[1:]:
        if spec["checkpoint_sha256"] != expected_sha or spec["sources"] != evaluations[0][1]["sources"]:
            raise ValueError("Reference checkpoint/source identity mismatch")
        if (spec["level"], spec["pos"], spec["acc"], spec["dos"]) != ("hard", 4.0, 1.3, 0.18):
            raise ValueError("Reference is not a full-budget hard-b evaluation")
        if row["completion"] < 0.5 or any(row[k] != 0 for k in ("collision", "road_any", "road_termination")):
            raise ValueError("Reference attack gate failed")
    return Path(evaluations[0][1]["checkpoint"])


class Followup:
    def __init__(self, args):
        self.args = args
        self.method_labels = tuple(args.methods) if getattr(args, "methods", None) else tuple(METHODS)
        unknown = [label for label in self.method_labels if label not in ALL_METHODS]
        if unknown:
            raise ValueError(f"Unknown method labels: {unknown}")
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.environment = dict(os.environ)
        for key in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
            self.environment.pop(key, None)
        self.environment["PYTHONPATH"] = ":".join(str(ROOT / p) for p in
            ("source/isaaclab", "source/isaaclab_rl", "source/isaaclab_tasks", "source/my_exts"))
        self.environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.common = protocol_from_reference(args.reference)
        self.checkpoint = verify_hard_b(args.reference)
        bc_report = json.loads(args.bc_checkpoint.with_suffix(".json").read_text())
        if Path(bc_report["output"]).resolve() != args.bc_checkpoint.resolve() or bc_report["samples"] <= 0:
            raise ValueError("BC checkpoint provenance report is invalid")
        self.bc_provenance = dict(report=bc_report, trajectory_sha256=digest(bc_report["source"]))
        sources = sorted((ROOT / "source/my_exts/marl_platoon").rglob("*.py"))
        sources += [ROOT / name for name in (
            "scripts/reinforcement_learning/rsl_rl/train.py",
            "scripts/reinforcement_learning/rsl_rl/play.py",
            "scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py",
            "scripts/tools/ackermann_validation.py",
            "scripts/tools/run_medium_algorithm_comparison.sh",
            "scripts/tools/run_platoon_spatial_attack_ladder.sh",
            "scripts/tools/run_ackermann_verified_followup.py")]
        self.sources = {str(p.relative_to(ROOT)): digest(p) for p in sources}
        self.evaluation_sources = {name: sha for name, sha in self.sources.items()
            if name.startswith("source/my_exts/marl_platoon/") or name in (
                "scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py",
                "scripts/tools/ackermann_validation.py")}
        self.render_sources = {name: sha for name, sha in self.sources.items()
            if name.startswith("source/my_exts/marl_platoon/") or name in (
                "scripts/reinforcement_learning/rsl_rl/play.py",
                "scripts/tools/run_platoon_spatial_attack_ladder.sh")}
        self.current = {}

    def status(self, state, **details):
        self.current = dict(time=dt.datetime.now().isoformat(), state=state, **details)
        write_json(self.output / "followup_status.json", self.current)
        print(json.dumps(self.current), flush=True)

    def run(self, command, log, env=None, timeout=7200):
        # The shared lock coordinates AMS launchers. The process check also
        # detects independent workloads that do not participate in that lock.
        previous = dict(self.current)
        waiting = False
        while True:
            conflicts = external_isaac_processes(self.args.sim, ignored={os.getpid()})
            if not conflicts:
                if waiting:
                    self.status(previous.get("state", "resuming"), **{
                        key: value for key, value in previous.items() if key not in ("state", "time")})
                    note("- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。")
                break
            self.status("waiting_for_exclusive_gpu", pending=previous, conflicts=conflicts, log=str(log))
            if not waiting:
                note(f"- 发现外部GPU任务，排队而非并发启动；等待PID={[item['pid'] for item in conflicts]}，"
                     f"待运行命令 `{command[0]}`，日志 `{log}`。未终止任何外部进程。")
            waiting = True
            time.sleep(15)
        busy = subprocess.run(["pgrep", "-af", r"scripts/reinforcement_learning/.*/(train|play|eval_happo_platoon)\.py"],
                              capture_output=True, text=True)
        if busy.returncode not in (0, 1):
            raise RuntimeError("Cannot inspect competing Isaac processes")
        if busy.returncode == 0:
            raise RuntimeError("Another Isaac workload is active: " + busy.stdout)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write("\nCOMMAND_JSON=" + json.dumps(command) + "\n")
            stream.flush()
            with subprocess.Popen(command, cwd=ROOT, env=env or self.environment,
                                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True) as child:
                try:
                    result = child.wait(timeout=timeout)
                except BaseException:
                    # Resolve the actual process group.  Isaac Sim's wrapper
                    # can leave the evaluator in a group whose id differs
                    # from the wrapper PID; signalling ``child.pid`` then
                    # fails to clean up a timed-out GPU job.
                    try:
                        pgid = os.getpgid(child.pid)
                    except ProcessLookupError:
                        pgid = None
                    if pgid is not None:
                        try:
                            os.killpg(pgid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        if pgid is not None:
                            try:
                                os.killpg(pgid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        child.wait()
                    raise
        if result:
            raise RuntimeError(f"Command exit={result}; log={log}")

    def evaluate(self, label, checkpoint, level, output, env_seed=42, attack_seed=3407):
        name, base, pos, acc, dos, _ = level
        enabled = name != "off"
        self.status("evaluating", algorithm=label, level=name, checkpoint=str(checkpoint),
                    env_seed=env_seed, attack_seed=attack_seed, output=str(output))
        attack = [f"env.algorithm.enable_attack={str(enabled).lower()}",
                  f"env.algorithm.attack_level={base}", "env.algorithm.attack_mode=profile",
                  f"env.algorithm.attack_seed={attack_seed}",
                  "env.algorithm.attack_curriculum_warmup_updates=0",
                  "env.algorithm.attack_strength_start_scale=1.0",
                  "env.algorithm.attack_strength_ramp_updates=0",
                  "env.algorithm.attack_clean_env_fraction=0.0",
                  f"env.algorithm.max_fdi_pos={pos}", f"env.algorithm.max_fdi_acc={acc}",
                  f"env.algorithm.max_dos_rate={dos}"]
        evaluator = [str(self.args.sim / "python.sh"), "scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py",
                     "--task", "Isaac-Marl-Platoon-HAPPO-v0", "--num_envs", str(self.args.eval_envs),
                     "--seed", str(env_seed), "--headless", "--fresh_env_per_checkpoint",
                     "--eval_checkpoints", str(checkpoint), "--eval_steps", "3500", "--warmup_steps", "0",
                     "--output_dir", str(output), "--export_ros_replay",
                     "--ros_replay_path", str(output / "trajectory_env0.csv"),
                     "--kit_args=--/rtx/verifyDriverVersion/enabled=false --/app/viewport/grid/enabled=false",
                     *(["--enable_attack_eval"] if enabled else []),
                     *self.common, *method_overrides(label), *attack]
        expected = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                        steps=3500, num_envs=self.args.eval_envs, env_seed=env_seed, attack_seed=attack_seed,
                        enabled=enabled, level=base, pos=pos, acc=acc, dos=dos, command=evaluator,
                        sources=self.evaluation_sources)
        candidates = [output]
        if getattr(self.args, "reuse_root", None):
            candidates.append(self.args.reuse_root.resolve() / output.relative_to(self.output))
        for cached in candidates:
            try:
                row, spec = load_verified(cached)
                if not same_evaluation(spec, expected):
                    continue
                if cached != output:
                    if output.exists() and any(output.iterdir()):
                        output.rename(output.with_name(output.name + f".partial_{time.time_ns()}"))
                    output.mkdir(parents=True, exist_ok=True)
                    stamp = json.loads((cached / "verified.json").read_text())
                    for name in [*stamp["artifacts"], "verified.json", "request.json", "evaluation.log"]:
                        if (cached / name).is_file():
                            shutil.copy2(cached / name, output / name)
                    write_json(output / "imported_from.json", dict(source=str(cached),
                               source_verified_sha256=digest(cached / "verified.json"),
                               semantic_command_matched=True))
                    note(f"- 复用同checkpoint/源码/协议的原始评估 `{cached}` → `{output}`；没有重跑仿真。")
                return row
            except (OSError, ValueError, KeyError):
                continue
        command = [self.args.host_python, "scripts/tools/ackermann_validation.py", "run",
                   "--output", str(output), "--checkpoint", str(checkpoint), "--steps", "3500",
                   "--num-envs", str(self.args.eval_envs), "--env-seed", str(env_seed),
                   "--attack-seed", str(attack_seed), "--enabled", str(enabled).lower(),
                   "--level", base, "--pos", str(pos), "--acc", str(acc), "--dos", str(dos), "--", *evaluator]
        self.run(command, output.parent / (output.name + "_validator.log"))
        return load_verified(output)[0]

    def validate(self, label, checkpoint, level, stage):
        clean_dir = stage / "validation/clean_env42"
        clean = self.evaluate(label, checkpoint, LEVELS[0], clean_dir)
        if not clean_passed(clean):
            report = dict(passed=False, clean_ok=False, safe=False, rows=[clean],
                          threshold=level[-1], attack_completion_min=0.0)
            write_json(stage / "validation_gate.json", report)
            return report
        for env_seed, attack_seed in SEEDS:
            self.evaluate(label, checkpoint, level,
                          stage / f"validation/attack_env{env_seed}_seed{attack_seed}", env_seed, attack_seed)
        return gate(stage, level[-1])

    def evidence(self, label, checkpoint, level, destination, verified_dir, *, diagnostic_clean=False):
        name, base, pos, acc, dos, _ = level
        if diagnostic_clean and level != LEVELS[0]:
            raise ValueError("Failed-clean diagnostics cannot enable any attack")
        row, spec = load_verified(verified_dir)
        directory = destination / name
        marker = directory / "render_complete.json"
        signature = dict(checkpoint_sha256=digest(checkpoint), level=list(level), method=label,
                         common=self.common, diagnostic_clean=diagnostic_clean,
                         sources=self.render_sources, evaluation_spec=spec)
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous.get("signature") == signature and previous.get("artifacts") and all(
                (directory / item).is_file() and digest(directory / item) == sha
                for item, sha in previous["artifacts"].items()):
                return
        # Import only the already passing, unchanged BC clean demonstration.
        reuse = getattr(self.args, "reuse_root", None)
        if reuse and clean_passed(row) and name == "off" and "clean_seed" in destination.parts and not directory.exists():
            prior = reuse.resolve() / destination.relative_to(self.output)
            manifest = prior / "manifest.txt"
            prior_eval = reuse.resolve() / verified_dir.relative_to(self.output)
            if manifest.exists() and (prior / name / "trajectory_env0.csv").exists():
                metadata = dict(line.split("=", 1) for line in manifest.read_text().splitlines() if "=" in line)
                _, old_spec = load_verified(prior_eval)
                if same_evaluation(old_spec, spec) and metadata.get("checkpoint") == str(checkpoint.resolve()) \
                        and metadata.get("algorithm") == label and metadata.get("actual_levels") == "off" \
                        and metadata.get("video_steps") == "3500":
                    with (prior / name / "trajectory_env0.csv").open() as handle:
                        counts = {}
                        for record in csv.DictReader(handle):
                            counts[record["robot_id"]] = counts.get(record["robot_id"], 0) + 1
                    if counts == {str(i): 3500 for i in range(1, 6)} and any((prior / name / "video").glob("*.mp4")):
                        shutil.copytree(prior / name, directory)
                        write_json(directory / "imported_evidence.json", dict(source=str(prior / name)))
        if not directory.exists():
            directory.mkdir(parents=True)
        adopted = False
        diagnostic_stamp = directory / "diagnostic_only.json"
        if diagnostic_clean and getattr(self.args, "resume_coordinator_update", False) and diagnostic_stamp.exists():
            stamp = json.loads(diagnostic_stamp.read_text())
            manifest = destination / "manifest.txt"
            metadata = dict(line.split("=", 1) for line in manifest.read_text().splitlines() if "=" in line)
            adopted = (stamp.get("diagnostic_only") is True and stamp.get("attacks_unlocked") is False
                       and stamp.get("gate_passed") is False and stamp["metrics"]["checkpoint_sha256"] == digest(checkpoint)
                       and metadata.get("checkpoint") == str(checkpoint.resolve())
                       and metadata.get("actual_levels") == "off" and metadata.get("algorithm") == label
                       and all((directory / name).is_file() for name in (
                           "trajectory_env0.csv", "topdown_true_xy.png", "reward_attack_safety.png"))
                       and len(list((directory / "video").glob("*.mp4"))) == 1)
            # The manifest migration below already checks that every physics,
            # evaluator and renderer source hash is unchanged.
        if not (directory / "imported_evidence.json").exists() and not adopted:
            video = directory / "video"
            if video.exists():
                video.rename(directory / f"video.partial_{time.time_ns()}")
            self.status("rendering_diagnostic" if diagnostic_clean else "rendering",
                        algorithm=label, level=name, output=str(destination))
            environment = dict(self.environment, CHECKPOINT=str(checkpoint), ALGORITHM_LABEL=label,
                               OUTPUT_ROOT=str(destination), SINGLE_LEVEL=name, SINGLE_BASE_LEVEL=base,
                               SINGLE_FDI_POS=str(pos), SINGLE_FDI_ACC=str(acc), SINGLE_DOS_RATE=str(dos),
                               VERIFIED_EVAL_DIR=str(verified_dir), COMMON_EXTRA_OVERRIDES=" ".join(self.common),
                               EVAL_NUM_ENVS=str(self.args.eval_envs), EVAL_STEPS="3500", VIDEO_STEPS="3500",
                               DIAGNOSTIC_CLEAN_ONLY="1" if diagnostic_clean else "0",
                               RESUME_EXISTING="0", PLOT_PYTHON=self.args.host_python)
            self.run(["bash", "scripts/tools/run_platoon_spatial_attack_ladder.sh"],
                     destination / "evidence_export.log", environment)
        artifacts = {str(p.relative_to(directory)): digest(p) for p in directory.rglob("*")
                     if p.is_file() and (p.suffix in (".png", ".mp4") or p.name == "trajectory_env0.csv")
                     and not any(".partial_" in part for part in p.parts)}
        for filename in ("trajectory_env0.csv", "topdown_true_xy.png", "reward_attack_safety.png"):
            if filename not in artifacts or not (directory / filename).stat().st_size:
                raise RuntimeError(f"Missing rendered evidence: {directory / filename}")
        videos = list((directory / "video").glob("*.mp4"))
        if len(videos) != 1:
            raise RuntimeError("Expected exactly one recorded video: " + str(directory))
        metadata = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration,nb_frames", "-of", "json", str(videos[0])],
            check=True, capture_output=True, text=True, timeout=30).stdout)["streams"]
        if len(metadata) != 1 or int(metadata[0].get("nb_frames", 0)) != 3500 \
                or abs(float(metadata[0].get("duration", 0)) - 70) > 0.05:
            raise RuntimeError("Recorded video must contain all 3500 frames / 70 seconds")
        write_json(marker, dict(signature=signature, artifacts=artifacts,
                              diagnostic_only=diagnostic_clean, imported=(directory / "imported_evidence.json").exists(),
                              adopted_after_coordinator_restart=adopted))

    def export(self, label, checkpoint, root=None):
        root = root or self.output / "final_ladders" / label
        clean_dir = root / "off/validation/clean_env42"
        clean = self.evaluate(label, checkpoint, LEVELS[0], clean_dir)
        if not clean_passed(clean):
            raise RuntimeError(f"Strict clean export gate failed for {label}; attacks remain locked")
        for level in LEVELS:
            name = level[0]
            directory = root / name
            if name == "off":
                report = dict(passed=True, clean_ok=True, safe=True, rows=[clean])
                write_json(directory / "validation_gate.json", report)
                verified = clean_dir
            else:
                link = directory / "validation/clean_env42"
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(clean_dir, target_is_directory=True)
                for env_seed, attack_seed in SEEDS:
                    self.evaluate(label, checkpoint, level,
                        directory / f"validation/attack_env{env_seed}_seed{attack_seed}", env_seed, attack_seed)
                report = gate(directory, level[-1])
                verified = directory / "validation/attack_env42_seed3407"
            # Failed attacks remain visible in the matrix; this fixed-model
            # assessment does not promote a model or unlock stronger training.
            write_json(directory / "assessment.json", dict(report, assessment_only=True))
            link = directory / "evaluation"
            if not link.exists():
                link.symlink_to(verified, target_is_directory=True)
            marker = directory / "evidence_complete.json"
            signature = dict(checkpoint_sha256=digest(checkpoint), spec=load_verified(verified)[1])
            reusable = False
            if marker.exists():
                previous = json.loads(marker.read_text())
                reusable = previous.get("signature") == signature and all(
                    (directory / name).is_file() and digest(directory / name) == sha
                    for name, sha in previous.get("artifacts", {}).items()) and bool(previous.get("artifacts"))
            if not reusable:
                video = directory / "video"
                if video.exists():
                    video.rename(directory / f"video.partial_{time.time_ns()}")
                self.evidence(label, checkpoint, level, root, verified)
                write_json(marker, dict(signature=signature, artifacts={
                    str(p.relative_to(directory)): digest(p) for p in directory.rglob("*")
                    if p.is_file() and (p.suffix in (".png", ".mp4") or p.name == "trajectory_env0.csv")
                    and not any(".partial_" in part or ".superseded_" in part for part in p.parts)
                }))
            note(f"- {label} 固定模型全强度证据完成级别 `{name}`，门禁passed={report['passed']}；"
                 f"原始验证/视频/轨迹/绘图 `{directory}`。")
        write_json(root / "ladder_complete.json", dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint)))

    def train(self, label, parent, level, stage, updates, round_index):
        runs = stage / "training/runs.txt"
        seed_index = (int(getattr(self.args, "seed_index", 0)) + round_index) % 3
        train_seeds = (42, 3407, 777)
        extra = [f"env.algorithm.happo_actor_lr={self.args.actor_lr}",
                 "env.algorithm.attack_clean_env_fraction=0.50",
                 "env.algorithm.attack_strength_start_scale=0.25",
                 f"env.algorithm.attack_strength_ramp_updates={updates}",
                 f"env.algorithm.attack_seed={SEEDS[seed_index][1]}"]
        signature = dict(parent_sha256=digest(parent), method=label, level=list(level), updates=updates,
                         seed=train_seeds[seed_index], seed_index=seed_index, common=self.common, sources=self.sources,
                         train_envs=self.args.train_envs, train_extra=extra)
        complete = stage / "training_complete.json"
        if complete.exists():
            old = json.loads(complete.read_text())
            if not training_signature_compatible(old["signature"], signature) \
                    or digest(Path(old["checkpoint"])) != old["checkpoint_sha256"]:
                raise RuntimeError("Training reuse identity mismatch: " + str(stage))
            return Path(old["checkpoint"])
        if stage.exists() and any(stage.iterdir()):
            stage.rename(stage.with_name(stage.name + f".partial_{time.time_ns()}"))
        seed_dir = ROOT / "logs/rsl_rl/platoon_happo" / f"{self.output.name}_seed_{label}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_link = seed_dir / "model_final.pt"
        if seed_link.is_symlink():
            seed_link.unlink()
        elif seed_link.exists():
            raise RuntimeError("Refuse to replace an existing real seed checkpoint")
        seed_link.symlink_to(parent)
        resume = self.output / f"resume_{label}.txt"
        resume.write_text(f"{label}={seed_dir}\n")
        name, base, pos, acc, dos, _ = level
        environment = dict(self.environment, TAG=f"{self.output.name}_{label}_{stage.name}",
            PIPELINE_LOG=str(stage.with_suffix(".log")), RESULT_ROOT=str(stage),
            RESULT_TAR=str(stage.with_suffix(".tar.gz")), PROGRESS_NOTES_PATH=str(NOTES),
            NUM_ENVS=str(self.args.train_envs), EVAL_NUM_ENVS=str(self.args.eval_envs),
            MAX_ITERATIONS=str(updates), EVAL_EVERY=str(updates), EVAL_STEPS="3500",
            FRESH_ENV_PER_CHECKPOINT="1", SKIP_EVAL="1", SEED=str(signature["seed"]),
            COMPARE_ALGOS=label, PARALLEL_TRAIN_JOBS="1", RESUME_RUNS_FILE=str(resume),
            ATTACK_PROFILE_LABEL=name, ENABLE_ATTACK=str(name != "off").lower(), ATTACK_LEVEL=base,
            ATTACK_MODE="profile", ATTACK_MAX_FDI_POS=str(pos), ATTACK_MAX_FDI_ACC=str(acc),
            ATTACK_MAX_DOS_RATE=str(dos), SAFETY_D_DROP="1.50", SAFETY_CATCHUP_ACTION="0.40",
            COMMON_EXTRA_OVERRIDES=" ".join([*self.common, *extra]),
            HAPPO_META_EXTRA_OVERRIDES=" ".join(self.common),
            HAPPO_META_TRAIN_EXTRA_OVERRIDES=" ".join(extra), HAPPO_META_EVAL_EXTRA_OVERRIDES="",
            HAPPO_AMS_ADVANTAGE_WEIGHT="0.02", HAPPO_AMS_WARMUP_UPDATES="5", HAPPO_AMS_RAMP_UPDATES="15",
            PLOT_PYTHON=self.args.host_python, METHOD_VARIANT="legacy")
        self.status("training", algorithm=label, level=name, parent=str(parent), updates=updates)
        self.run(["bash", "scripts/tools/run_medium_algorithm_comparison.sh"],
                 self.output / "training_driver.log", environment, timeout=14400)
        run_dir = Path(dict(line.split("=", 1) for line in runs.read_text().splitlines())[label])
        checkpoint = run_dir / "model_final.pt"
        write_json(complete, dict(signature=signature, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint)))
        return checkpoint

    def matrix(self):
        seed = self.args.bc_checkpoint.resolve()
        # All six receive the BC policy, never the already attack-trained AMS
        # policy. Its trajectory and warm-start report remain in the manifest.
        outcomes = {}
        labels = getattr(self, "method_labels", tuple(METHODS))
        for label in labels:
            parent = seed
            clean_stage = self.output / "stages" / label / "clean_seed"
            row = self.evaluate(label, parent, LEVELS[0], clean_stage / "validation/clean_env42")
            if clean_passed(row):
                self.evidence(label, parent, LEVELS[0], clean_stage / "evidence", clean_stage / "validation/clean_env42")
            if not clean_passed(row):
                for round_index in range(self.args.rounds):
                    stage = self.output / "stages" / label / f"clean_r{round_index + 1}"
                    candidate = self.train(label, parent, LEVELS[0], stage, 240, round_index)
                    row = self.evaluate(label, candidate, LEVELS[0], stage / "validation/clean_env42")
                    write_json(stage / "validation_gate.json", dict(passed=clean_passed(row),
                               clean_ok=clean_passed(row), safe=clean_passed(row), rows=[row]))
                    self.run([self.args.host_python, "scripts/tools/plot_ackermann_recovery_stage.py", str(stage)],
                             stage / "plotting.log")
                    if clean_passed(row):
                        self.evidence(label, candidate, LEVELS[0], stage / "evidence", stage / "validation/clean_env42")
                    if clean_passed(row):
                        parent = candidate
                        break
            if not clean_passed(row):
                outcomes[label] = dict(status="clean_gate_failed", checkpoint=str(parent))
                note(f"- 六算法 `{label}` 严格clean未通过，未解锁攻击；数据保留于 `{clean_stage.parent}`。")
                write_json(self.output / "method_outcomes.json", outcomes)
                continue
            stopped = None
            for level in LEVELS[1:]:
                promoted = False
                for round_index in range(self.args.rounds):
                    stage = self.output / "stages" / label / f"{level[0]}_r{round_index + 1}"
                    candidate = self.train(label, parent, level, stage, self.args.stage_updates, round_index)
                    report = self.validate(label, candidate, level, stage)
                    self.run([self.args.host_python, "scripts/tools/plot_ackermann_recovery_stage.py", str(stage)],
                             stage / "plotting.log")
                    verified = stage / "validation" / ("attack_env42_seed3407" if report["clean_ok"] else "clean_env42")
                    self.evidence(label, candidate, level if report["clean_ok"] else LEVELS[0],
                                  stage / ("evidence" if report["clean_ok"] else "failed_clean_diagnostics"),
                                  verified, diagnostic_clean=not report["clean_ok"])
                    note(f"- 六算法 `{label}` 阶段 `{stage.name}`，passed={report['passed']}，"
                         f"clean_ok={report['clean_ok']}，攻击最低完成率={report.get('attack_completion_min')}；数据 `{stage}`。")
                    if report["passed"]:
                        parent = candidate
                        promoted = True
                        break
                    # Reject and retain the last passing model; no failed clean
                    # or unsafe candidate becomes the next round's parent.
                if not promoted:
                    stopped = level[0]
                    break
            outcomes[label] = dict(status="passed_hard_b" if stopped is None else "stopped_at_gate",
                                   stopped_level=stopped, checkpoint=str(parent), checkpoint_sha256=digest(parent))
            write_json(self.output / "method_outcomes.json", outcomes)
            # Assess every difficulty using the last accepted model, and expose
            # its stopped training level. Assessment never implies promotion.
            self.export(label, parent)
        if len(outcomes) != len(labels) or any(x["status"] == "clean_gate_failed" for x in outcomes.values()):
            raise RuntimeError("Method matrix incomplete: at least one clean acquisition failed")
        self.run([self.args.host_python, "scripts/tools/plot_ackermann_six_attack_matrix.py",
                  str(self.output / "final_ladders")], self.output / "matrix_plotting.log")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "logs/rsl_rl/platoon_happo/ams_bridge_refined_20260922/stages/hard_b_r2")
    parser.add_argument("--sim", type=Path, default=ROOT.parent / "isaac-sim")
    parser.add_argument("--host-python", default="/home/cnc/Downloads/ENTER/bin/python3")
    parser.add_argument("--bc-checkpoint", type=Path, default=ROOT / "logs/ackermann_bc_warmstart/real_trajectory_bc/model_bc.pt")
    parser.add_argument("--train-envs", type=int, default=32)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--stage-updates", type=int, default=40)
    parser.add_argument("--actor-lr", type=float, default=5e-7)
    parser.add_argument("--reuse-root", type=Path,
                        help="Import only hash-verified, semantically identical evaluations and BC evidence")
    parser.add_argument("--diagnose-prior-failure", type=Path,
                        help="Record an attack-free diagnostic video for a rejected MAPPO stage before resuming")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2), default=0,
                        help="Independent training-seed index: 0->42, 1->3407, 2->777")
    parser.add_argument("--methods", nargs="+", choices=tuple(ALL_METHODS),
                        help="Method subset; default is the historical six-method matrix")
    parser.add_argument("--mode", choices=("all", "export", "matrix"), default="all")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-coordinator-update", action="store_true",
                        help="Archive an old manifest only if the coordinator alone changed; reuse completed clean diagnostics")
    args = parser.parse_args()
    if args.eval_envs != 16 or args.train_envs <= 0 or args.stage_updates <= 0 or not 1 <= args.rounds <= 3 \
            or not 0 < args.actor_lr <= 1e-6:
        parser.error("Use 16 evaluation environments, positive training budgets, and 1..3 rounds")
    runner = Followup(args)
    prior_failure = None
    if args.diagnose_prior_failure:
        prior_row, prior_spec = load_verified(args.diagnose_prior_failure / "validation/clean_env42")
        if clean_passed(prior_row) or prior_spec["enabled"] or prior_spec["sources"] != runner.evaluation_sources \
                or (prior_spec["level"], prior_spec["pos"], prior_spec["acc"], prior_spec["dos"]) != ("off", 0.0, 0.0, 0.0):
            raise ValueError("Prior failure must be a verified, attack-free rejection under unchanged evaluation sources")
        required_method = method_overrides("mappo")
        if any(item not in prior_spec["command"] for item in required_method):
            raise ValueError("Prior diagnostic checkpoint was not evaluated as MAPPO")
        prior_failure = dict(stage=str(args.diagnose_prior_failure.resolve()),
                             checkpoint_sha256=prior_spec["checkpoint_sha256"], metrics=prior_row)
    protocol = dict(protocol_revision=2, levels=LEVELS,
                    methods={label: ALL_METHODS[label] for label in runner.method_labels},
                    train_envs=args.train_envs, eval_envs=args.eval_envs,
                    stage_updates=args.stage_updates, rounds=args.rounds, common=runner.common,
                    actor_lr=args.actor_lr, train_clean_fraction=0.50, seed_index=args.seed_index,
                    reuse_root=str(args.reuse_root.resolve()) if args.reuse_root else None,
                    prior_failure=prior_failure,
                    sources=runner.sources,
                    bc_provenance=runner.bc_provenance,
                    reference_checkpoint=str(runner.checkpoint), reference_sha256=digest(runner.checkpoint),
                    bc_checkpoint=str(args.bc_checkpoint.resolve()), bc_sha256=digest(args.bc_checkpoint))
    manifest = runner.output / "followup_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(protocol)):
        if not args.resume_coordinator_update or not coordinator_only_update(json.loads(manifest.read_text()), protocol):
            raise RuntimeError("Existing followup manifest differs; choose a new output directory")
        manifest.rename(runner.output / f"followup_manifest.coordinator_previous_{time.time_ns()}.json")
    write_json(manifest, protocol)
    if args.preflight_only:
        runner.status("preflight_passed", mode=args.mode, protocol=str(manifest))
        return
    lock = (ROOT / "logs/ams_ackermann_gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    note(f"- 开始后续服务mode={args.mode}，hard-b参考SHA={protocol['reference_sha256']}；"
         f"{args.train_envs}训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×{args.stage_updates}更新、"
         f"actor LR={args.actor_lr}、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。"
         f"协议 `{manifest}`，状态 `{runner.output / 'followup_status.json'}`。")
    try:
        if args.mode in ("all", "export"):
            runner.export("happo_meta_ams", runner.checkpoint, runner.output / "reference_ams_ladder")
        if args.mode in ("all", "matrix"):
            if prior_failure:
                runner.evidence("mappo", Path(prior_spec["checkpoint"]), LEVELS[0],
                                runner.output / "prior_failure_diagnostics/mappo_light_r1",
                                args.diagnose_prior_failure / "validation/clean_env42", diagnostic_clean=True)
                note(f"- 已保存MAPPO旧失败候选的70秒clean-only诊断视频；completion={prior_row['completion']}，"
                     f"仍拒绝晋级，续训父模型为通过clean的BC，不使用该失败模型。"
                     f"诊断 `{runner.output / 'prior_failure_diagnostics/mappo_light_r1'}`。")
            runner.matrix()
        runner.status("complete", mode=args.mode)
        note(f"- 后续实验mode={args.mode}已完成，产物 `{runner.output}`。")
    except BaseException as exc:
        runner.status("stopped_at_error", error=str(exc), previous=runner.current)
        note(f"- 后续服务中止：{type(exc).__name__}: {exc}；状态/日志 `{runner.output}`。已完成验证保留，未宣告后续矩阵成功。")
        raise


if __name__ == "__main__":
    main()
