"""Utilities for saving best checkpoints during platoon training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class BestCheckpointState:
    """Tracks the best platoon checkpoint score seen during a training run."""

    best_score: float = float("-inf")
    best_iter: int = -1


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.float().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_episode_metrics(ep_infos: list[dict[str, Any]]) -> dict[str, float]:
    if not ep_infos:
        return {}
    tracked_reward_keys = {
        "Episode_Reward/alive",
        "Episode_Reward/formation",
        "Episode_Reward/forward_drive",
        "Episode_Reward/leader_motion",
        "Episode_Reward/lateral_correct",
        "Episode_Reward/leader_progress",
        "Episode_Reward/no_backward",
        "Episode_Reward/stall_penalty",
        "Episode_Reward/true_fail",
        "Episode_Reward/true_success",
    }
    values: dict[str, list[float]] = {}
    for ep_info in ep_infos:
        for key, value in ep_info.items():
            if key.startswith("Episode_Termination/") or key in tracked_reward_keys:
                values.setdefault(key, []).append(_to_float(value))
    return {key: sum(items) / max(len(items), 1) for key, items in values.items()}


def platoon_best_score(ep_infos: list[dict[str, Any]], lenbuffer: Any) -> tuple[float | None, dict[str, float]]:
    """Compute a stable best-checkpoint score from recent platoon episode metrics.

    Higher is better. The score still requires survival, but it now explicitly
    prefers policies that make forward progress instead of merely standing still.
    """

    metrics = _collect_episode_metrics(ep_infos)
    if not metrics:
        return None, {}

    alive = metrics.get("Episode_Reward/alive", 0.0)
    timeout = metrics.get("Episode_Termination/time_out", 0.0)
    bad_ori = metrics.get("Episode_Termination/reset_on_bad_ori", 0.0)
    true_success = metrics.get("Episode_Reward/true_success", 0.0)
    true_fail = metrics.get("Episode_Reward/true_fail", 0.0)
    forward_drive = metrics.get("Episode_Reward/forward_drive", 0.0)
    leader_motion = metrics.get("Episode_Reward/leader_motion", 0.0)
    leader_progress = metrics.get("Episode_Reward/leader_progress", 0.0)
    formation = metrics.get("Episode_Reward/formation", 0.0)
    no_backward = metrics.get("Episode_Reward/no_backward", 0.0)
    lateral_correct = metrics.get("Episode_Reward/lateral_correct", 0.0)
    stall_penalty = metrics.get("Episode_Reward/stall_penalty", 0.0)
    mean_len = 0.0
    if len(lenbuffer) > 0:
        mean_len = float(sum(lenbuffer) / len(lenbuffer))

    score = (
        80.0 * timeout
        + 60.0 * alive
        + 80.0 * forward_drive
        + 30.0 * leader_motion
        + 30.0 * leader_progress
        + 20.0 * true_success
        + 10.0 * formation
        - 120.0 * bad_ori
        - 20.0 * true_fail
        - 20.0 * abs(no_backward)
        - 10.0 * abs(lateral_correct)
        - 20.0 * abs(stall_penalty)
    )
    metrics["BestCheckpoint/score"] = score
    metrics["BestCheckpoint/mean_episode_length"] = mean_len
    return score, metrics


def maybe_save_best_checkpoint(runner: Any, locs: dict[str, Any], state: BestCheckpointState) -> None:
    """Save ``model_best.pt`` when recent platoon metrics improve."""

    if runner.log_dir is None or getattr(runner, "disable_logs", False):
        return
    score, metrics = platoon_best_score(locs.get("ep_infos", []), locs.get("lenbuffer", []))
    if score is None or score <= state.best_score:
        return

    state.best_score = score
    state.best_iter = int(locs.get("it", getattr(runner, "current_learning_iteration", -1)))
    best_path = os.path.join(runner.log_dir, "model_best.pt")
    runner.save(best_path, infos={"best_score": state.best_score, "best_iter": state.best_iter, "metrics": metrics})
    print(f"[INFO]: Saved best model checkpoint to: {best_path} (score={state.best_score:.4f}, iter={state.best_iter})")
