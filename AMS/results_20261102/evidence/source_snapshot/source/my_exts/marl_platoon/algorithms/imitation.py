"""Real-replay dataset reconstruction for task-local policy retention."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch


def _finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {row[key]}")
    return value


def load_ackermann_imitation_dataset(
    path: str | Path,
    *,
    max_speed: float = 1.2,
    max_steer: float = 0.32,
    include_spatial_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reconstructed policy observations, normalized actions, agent ids."""
    by_step: dict[int, list[dict[str, str]]] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_step.setdefault(int(row["step"]), []).append(row)
    steps = sorted(by_step)
    if len(steps) < 2:
        raise ValueError(f"imitation replay has fewer than two steps: {path}")
    observations: list[list[float]] = []
    targets: list[list[float]] = []
    agent_ids: list[int] = []
    previous = {int(row["robot_id"]): row for row in by_step[steps[0]]}
    for step in steps[1:]:
        current = {int(row["robot_id"]): row for row in by_step[step]}
        if len(current) != 5 or len(previous) != 5:
            previous = current
            continue
        body_vel: dict[int, tuple[float, float]] = {}
        for robot_id, row in previous.items():
            yaw, vx, vy = (_finite(row, "yaw"), _finite(row, "vx"), _finite(row, "vy"))
            body_vel[robot_id] = (
                math.cos(yaw) * vx + math.sin(yaw) * vy,
                -math.sin(yaw) * vx + math.cos(yaw) * vy,
            )
        for robot_id in range(1, 6):
            row = previous[robot_id]
            target_row = current[robot_id]
            yaw = _finite(row, "yaw")
            own_vx, own_vy = body_vel[robot_id]
            if robot_id == 1:
                rel_x = rel_y = rel_vx = rel_vy = pair_heading = 0.0
            else:
                pred = previous[robot_id - 1]
                dx = _finite(pred, "x") - _finite(row, "x")
                dy = _finite(pred, "y") - _finite(row, "y")
                rel_x = math.cos(yaw) * dx + math.sin(yaw) * dy
                rel_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
                pred_vx, pred_vy = body_vel[robot_id - 1]
                rel_vx, rel_vy = pred_vx - own_vx, pred_vy - own_vy
                pair_heading = math.sin(yaw - _finite(pred, "yaw"))
            observation = [
                rel_x,
                rel_y,
                rel_vx,
                rel_vy,
                own_vx,
                own_vy,
                _finite(row, "yaw_rate"),
                max(-2.0, min(2.0, _finite(row, "y") - _finite(row, "reference_y"))),
                math.sin(yaw - _finite(row, "reference_heading")),
                pair_heading,
                (robot_id - 1) / 4.0,
            ]
            if include_spatial_features:
                # Never silently fabricate new observation features from a
                # legacy 11-D replay. Export these reference fields first.
                reference_progress = _finite(row, "reference_progress")
                reference_yaw_rate = _finite(row, "reference_yaw_rate")
                reference_speed = max(abs(_finite(row, "reference_speed")), 0.10)
                cth_gap_error = 0.0 if robot_id == 1 else rel_x - (1.5 + 0.6 * max(own_vx, 0.0))
                observation.extend([
                    max(0.0, min(1.0, reference_progress)),
                    max(-10.0, min(10.0, reference_yaw_rate / reference_speed)),
                    max(-3.0, min(3.0, cth_gap_error)),
                ])
            observations.append(observation)
            targets.append([
                _finite(target_row, "executed_action_0") / max_speed,
                _finite(target_row, "executed_action_1") / max_steer,
            ])
            agent_ids.append(robot_id - 1)
        previous = current
    return (
        torch.tensor(observations, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(agent_ids, dtype=torch.long),
    )
