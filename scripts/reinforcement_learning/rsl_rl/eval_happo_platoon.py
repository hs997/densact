# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Finite-step deterministic evaluation for task-local HAPPO platoon checkpoints."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_IAL_ROOT = _THIS_FILE.parents[3]
for _p in [
    _IAL_ROOT / "source" / "isaaclab",
    _IAL_ROOT / "source" / "isaaclab_rl",
    _IAL_ROOT / "source" / "isaaclab_tasks",
    _IAL_ROOT / "source" / "my_exts",
]:
    sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate task-local HAPPO platoon checkpoints.")
parser.add_argument("--task", type=str, default="Isaac-Marl-Platoon-HAPPO-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--eval_checkpoints",
    type=str,
    required=True,
    help="Comma-separated checkpoint paths to evaluate.",
)
parser.add_argument("--eval_steps", type=int, default=600)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument(
    "--fresh_env_per_checkpoint",
    action="store_true",
    help="Recreate the IsaacLab environment for each checkpoint so eval order cannot leak state.",
)
parser.add_argument(
    "--enable_attack_eval",
    action="store_true",
    help="Keep the task attack configuration active during evaluation. By default eval disables attacks.",
)
parser.add_argument(
    "--export_ros_replay",
    action="store_true",
    help="Export env_id=0, per-robot executable control trajectory CSV for ROS replay.",
)
parser.add_argument(
    "--ros_replay_path",
    type=str,
    default=None,
    help="Output CSV path used with --export_ros_replay. Defaults to <output_dir>/ros_replay.csv.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import marl_platoon.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config
from ros_replay_exporter import RosReplayExporter


def _find_happo_wrapper(env):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if all(hasattr(current, name) for name in ("state_dict", "load_state_dict", "set_inference_mode")):
            if hasattr(current, "algorithm_router"):
                return current
        current = getattr(current, "env", None)
    return None


def _load_task_happo_state(happo_wrapper, checkpoint_path: Path, disable_attack: bool = True) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "platoon_happo_state" not in checkpoint:
        raise RuntimeError(f"Checkpoint has no platoon_happo_state: {checkpoint_path}")
    happo_wrapper.load_state_dict(checkpoint["platoon_happo_state"], strict=True)
    happo_wrapper.set_inference_mode(
        deterministic=True,
        disable_updates=True,
        force_happo_actions=True,
        disable_attack=disable_attack,
    )


def _action_dim(env) -> int:
    action_manager = getattr(env.unwrapped, "action_manager", None)
    if action_manager is not None and getattr(action_manager, "total_action_dim", None) is not None:
        return int(action_manager.total_action_dim)
    space = getattr(env.unwrapped, "single_action_space", None) or getattr(env.unwrapped, "action_space", None)
    if space is None or getattr(space, "shape", None) is None:
        raise RuntimeError("Cannot infer action dimension for evaluation.")
    return int(space.shape[-1])


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        out = float(value)
        if out == out and abs(out) != float("inf"):
            return out
    return None


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _max(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else 0.0


def _collect_eval_metrics(router, rew, dones) -> dict[str, Any]:
    rewards_tensor = router.to_agent_rewards(rew)
    dones_tensor = router.to_agent_dones(dones)
    row: dict[str, Any] = {}
    row.update(router._collect_platoon_metrics(rewards_tensor, dones_tensor))
    row.update(router._collect_reward_term_metrics())
    row.update(router._collect_termination_metrics())
    row.update(router._collect_attack_metrics(None))
    row.update(router._collect_teacher_metrics(None))
    row.update(router._collect_shield_metrics())
    row["eval_total_reward_mean"] = _reward_term_total(row)
    return row


def _reward_term_total(row: dict[str, Any]) -> float:
    """Approximate task total reward from logged reward-manager terms.

    In task-local HAPPO evaluation, the outer env reward can be zero while the
    reward manager still exposes the weighted task terms used for diagnostics.
    """

    total = 0.0
    found = False
    for key, value in row.items():
        if key == "reward_env_mean" or not key.startswith("reward_"):
            continue
        numeric = _to_float(value)
        if numeric is None:
            continue
        total += numeric
        found = True
    if found:
        return total
    fallback = _to_float(row.get("reward_env_mean"))
    return float(fallback) if fallback is not None else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(checkpoint: Path, rows: list[dict[str, Any]], warmup_steps: int) -> dict[str, Any]:
    usable = rows[min(max(warmup_steps, 0), len(rows)) :] or rows
    step_rewards = [_to_float(row.get("eval_total_reward_mean")) for row in usable]
    step_rewards = [value for value in step_rewards if value is not None]
    worst_counter = Counter()
    for row in usable:
        worst = _to_float(row.get("lateral_worst_pair"))
        if worst is not None:
            worst_counter[int(worst)] += 1

    summary = {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "eval_rows": len(rows),
        "summary_rows": len(usable),
        "episode_return_mean": sum(step_rewards),
        "step_reward_mean": (sum(step_rewards) / len(step_rewards)) if step_rewards else 0.0,
        "outer_reward_env_mean": _mean(usable, "reward_env_mean"),
        "command_speed_mean": _mean(usable, "command_speed_mean"),
        "leader_speed_mean": _mean(usable, "leader_speed_mean"),
        "platoon_speed_mean": _mean(usable, "platoon_speed_mean"),
        "leader_speed_error_mean": _mean(usable, "leader_speed_error_mean"),
        "speed_error_abs_mean": _mean(usable, "speed_error_abs_mean"),
        "centerline_error_abs_mean": _mean(usable, "centerline_error_abs_mean"),
        "centerline_error_abs_max_mean": _mean(usable, "centerline_error_abs_max"),
        "centerline_error_abs_max_peak": _max(usable, "centerline_error_abs_max"),
        "lateral_error_abs_mean": _mean(usable, "lateral_error_abs_mean"),
        "lateral_error_abs_max_mean": _mean(usable, "lateral_error_abs_max"),
        "lateral_error_abs_max_peak": _max(usable, "lateral_error_abs_max"),
        "gap_error_abs_mean": _mean(usable, "gap_error_abs_mean"),
        "min_pair_gap_mean": _mean(usable, "min_pair_gap_mean"),
        "collision_rate": _mean(usable, "collision_rate"),
        "heading_error_abs_mean": _mean(usable, "heading_error_abs_mean"),
        "pair_heading_error_abs_mean": _mean(usable, "pair_heading_error_abs_mean"),
        "done_rate": _mean(usable, "done_rate"),
        "termination_reset_on_bad_ori": _mean(usable, "termination_reset_on_bad_ori"),
        "termination_time_out": _mean(usable, "termination_time_out"),
        "shield_lateral_turn_mean": _mean(usable, "shield_lateral_turn_mean"),
        "shield_lateral_rate": _mean(usable, "shield_lateral_rate"),
        "shield_lateral_critical_rate": _mean(usable, "shield_lateral_critical_rate"),
        "attack_enabled": _mean(usable, "attack_enabled"),
        "attack_max_fdi_acc": _mean(usable, "attack_max_fdi_acc"),
        "attack_max_dos_rate": _mean(usable, "attack_max_dos_rate"),
        "worst_pair_counts": dict(sorted(worst_counter.items())),
    }
    for idx in range(1, 6):
        summary[f"centerline_robot_{idx}_abs_mean"] = _mean(usable, f"centerline_robot_{idx}_abs_mean")
        summary[f"centerline_robot_{idx}_signed_mean"] = _mean(usable, f"centerline_robot_{idx}_signed_mean")
    for idx in range(1, 5):
        summary[f"lateral_pair_{idx}_abs_mean"] = _mean(usable, f"lateral_pair_{idx}_abs_mean")
        summary[f"lateral_pair_{idx}_signed_mean"] = _mean(usable, f"lateral_pair_{idx}_signed_mean")
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "[EVAL] "
        f"{summary['checkpoint']}: "
        f"return={summary['episode_return_mean']:.3f}, "
        f"cmd={summary['command_speed_mean']:.3f}, "
        f"leader={summary['leader_speed_mean']:.3f}, "
        f"platoon={summary['platoon_speed_mean']:.3f}, "
        f"speed_err={summary['speed_error_abs_mean']:.3f}, "
        f"center={summary['centerline_error_abs_mean']:.3f}, "
        f"lat={summary['lateral_error_abs_mean']:.3f}, "
        f"min_gap={summary['min_pair_gap_mean']:.3f}, "
        f"collision={summary['collision_rate']:.3f}, "
        f"p=[{summary['lateral_pair_1_abs_mean']:.3f},"
        f"{summary['lateral_pair_2_abs_mean']:.3f},"
        f"{summary['lateral_pair_3_abs_mean']:.3f},"
        f"{summary['lateral_pair_4_abs_mean']:.3f}], "
        f"reset_bad={summary['termination_reset_on_bad_ori']:.3f}, "
        f"shield_lat_turn={summary['shield_lateral_turn_mean']:.3f}, "
        f"attack_enabled={summary['attack_enabled']:.0f}, "
        f"worst={summary['worst_pair_counts']}"
    )


def _make_eval_env(env_cfg):
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    happo_wrapper = _find_happo_wrapper(env)
    if happo_wrapper is None:
        env.close()
        raise RuntimeError("Could not find task-local HAPPO wrapper.")

    act_dim = _action_dim(env)
    device = env.unwrapped.device
    zero_actions = torch.zeros((env.unwrapped.num_envs, act_dim), device=device)
    return env, happo_wrapper, zero_actions


def _evaluate_checkpoint(
    env,
    happo_wrapper,
    zero_actions,
    checkpoint_path: Path,
    ros_replay_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env.reset()
    _load_task_happo_state(
        happo_wrapper,
        checkpoint_path,
        disable_attack=not bool(args_cli.enable_attack_eval),
    )
    router = happo_wrapper.algorithm_router
    ros_exporter = None
    if ros_replay_path is not None:
        ros_exporter = RosReplayExporter(
            env=env,
            happo_wrapper=happo_wrapper,
            csv_path=ros_replay_path,
            checkpoint=str(checkpoint_path),
            task=args_cli.task,
        )
    rows: list[dict[str, Any]] = []
    try:
        for step_idx in range(int(args_cli.eval_steps)):
            # Do not use torch.inference_mode() around IsaacLab env.step().
            # Reset writes into internal tensors after each checkpoint; inference tensors
            # can make those in-place updates fail.
            with torch.no_grad():
                _, rew, terminated, truncated, _ = env.step(zero_actions)
            if ros_exporter is not None:
                ros_exporter.export_step()
            dones = torch.logical_or(terminated, truncated)
            row = _collect_eval_metrics(router, rew, dones)
            row["eval_step"] = step_idx + 1
            row["checkpoint"] = checkpoint_path.name
            rows.append(row)
            if not simulation_app.is_running():
                break
    finally:
        if ros_exporter is not None:
            ros_exporter.close()

    summary = _summarize(checkpoint_path, rows, int(args_cli.warmup_steps))
    return rows, summary


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    checkpoint_paths = [Path(item).expanduser().resolve() for item in args_cli.eval_checkpoints.split(",") if item.strip()]
    missing = [str(path) for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    if args_cli.export_ros_replay and len(checkpoint_paths) != 1:
        raise ValueError("--export_ros_replay expects exactly one checkpoint in --eval_checkpoints.")

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.output_dir is None:
        output_dir = checkpoint_paths[0].parent / f"eval_noattack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(output_dir)
    ros_replay_path = None
    if args_cli.export_ros_replay:
        ros_replay_path = (
            Path(args_cli.ros_replay_path).expanduser().resolve()
            if args_cli.ros_replay_path
            else output_dir / "ros_replay.csv"
        )

    all_summaries: list[dict[str, Any]] = []

    if args_cli.fresh_env_per_checkpoint:
        for checkpoint_path in checkpoint_paths:
            env = None
            try:
                env, happo_wrapper, zero_actions = _make_eval_env(env_cfg)
                rows, summary = _evaluate_checkpoint(
                    env,
                    happo_wrapper,
                    zero_actions,
                    checkpoint_path,
                    ros_replay_path=ros_replay_path,
                )
            finally:
                if env is not None:
                    env.close()

            step_csv = output_dir / f"eval_steps_{checkpoint_path.stem}.csv"
            _write_csv(step_csv, rows)
            summary["step_csv"] = str(step_csv)
            all_summaries.append(summary)
            _print_summary(summary)
    else:
        env = None
        try:
            env, happo_wrapper, zero_actions = _make_eval_env(env_cfg)
            for checkpoint_path in checkpoint_paths:
                rows, summary = _evaluate_checkpoint(
                    env,
                    happo_wrapper,
                    zero_actions,
                    checkpoint_path,
                    ros_replay_path=ros_replay_path,
                )

                step_csv = output_dir / f"eval_steps_{checkpoint_path.stem}.csv"
                _write_csv(step_csv, rows)
                summary["step_csv"] = str(step_csv)
                all_summaries.append(summary)
                _print_summary(summary)
        finally:
            if env is not None:
                env.close()

    summary_csv = output_dir / "eval_summary.csv"
    _write_csv(summary_csv, all_summaries)
    print(f"[EVAL] wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()
