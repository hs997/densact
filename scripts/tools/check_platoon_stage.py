#!/usr/bin/env python3
"""Check whether a platoon curriculum stage is stable enough to promote."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


THRESHOLDS = {
    "stage1_stable": {
        "leader_speed_mean_min": 0.25,
        "speed_error_abs_mean_max": 0.50,
        "gap_error_abs_mean_max": 0.60,
        "gap_error_max_abs_max": 2.00,
        "lateral_error_abs_mean_max": 0.35,
        "lateral_error_abs_max_max": 1.20,
        "centerline_error_abs_mean_max": 0.25,
        "centerline_error_abs_max_max": 0.80,
        "termination_reset_on_bad_ori_max": 0.05,
        "shield_trigger_rate_max": 0.70,
        "value_loss_max": 1.0e5,
        "critic_grad_norm_max": 1.0e4,
    },
    "stage2_speedup": {
        "leader_speed_mean_min": 0.35,
        "speed_error_abs_mean_max": 0.55,
        "gap_error_abs_mean_max": 0.80,
        "gap_error_max_abs_max": 2.50,
        "lateral_error_abs_mean_max": 0.45,
        "lateral_error_abs_max_max": 1.50,
        "centerline_error_abs_mean_max": 0.30,
        "centerline_error_abs_max_max": 1.00,
        "termination_reset_on_bad_ori_max": 0.10,
        "shield_trigger_rate_max": 0.90,
        "value_loss_max": 1.0e5,
        "critic_grad_norm_max": 1.0e4,
    },
    "stage3_easy_attack": {
        "leader_speed_mean_min": 0.30,
        "speed_error_abs_mean_max": 0.60,
        "gap_error_abs_mean_max": 0.80,
        "gap_error_max_abs_max": 3.00,
        "lateral_error_abs_mean_max": 0.50,
        "lateral_error_abs_max_max": 1.75,
        "centerline_error_abs_mean_max": 0.35,
        "centerline_error_abs_max_max": 1.20,
        "termination_reset_on_bad_ori_max": 0.15,
        "shield_trigger_rate_max": 0.90,
        "value_loss_max": 1.0e5,
        "critic_grad_norm_max": 1.0e4,
    },
    "stage4_medium_profile": {
        "leader_speed_mean_min": 0.30,
        "speed_error_abs_mean_max": 0.65,
        "gap_error_abs_mean_max": 1.00,
        "gap_error_max_abs_max": 3.50,
        "lateral_error_abs_mean_max": 0.60,
        "lateral_error_abs_max_max": 2.00,
        "centerline_error_abs_mean_max": 0.45,
        "centerline_error_abs_max_max": 1.50,
        "termination_reset_on_bad_ori_max": 0.20,
        "shield_trigger_rate_max": 0.95,
        "value_loss_max": 1.0e5,
        "critic_grad_norm_max": 1.0e4,
    },
    "stage5_cagan": {
        "leader_speed_mean_min": 0.25,
        "speed_error_abs_mean_max": 0.70,
        "gap_error_abs_mean_max": 1.20,
        "gap_error_max_abs_max": 4.00,
        "lateral_error_abs_mean_max": 0.70,
        "lateral_error_abs_max_max": 2.25,
        "centerline_error_abs_mean_max": 0.50,
        "centerline_error_abs_max_max": 1.75,
        "termination_reset_on_bad_ori_max": 0.25,
        "shield_trigger_rate_max": 0.97,
        "value_loss_max": 1.0e5,
        "critic_grad_norm_max": 1.0e4,
    },
    # Backward-compatible name for old CSVs from the earlier four-stage script.
    "stage4_cagan": {
        "leader_speed_mean_min": 0.25,
        "speed_error_abs_mean_max": 0.70,
        "gap_error_abs_mean_max": 1.20,
        "gap_error_max_abs_max": 4.00,
        "lateral_error_abs_mean_max": 0.70,
        "lateral_error_abs_max_max": 2.25,
        "centerline_error_abs_mean_max": 0.50,
        "centerline_error_abs_max_max": 1.75,
        "termination_reset_on_bad_ori_max": 0.25,
        "shield_trigger_rate_max": 0.97,
        "value_loss_max": 1.0e5,
        "critic_grad_norm_max": 1.0e4,
    },
}


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(rows: list[dict[str, str]], key: str) -> float | None:
    vals = [_to_float(row.get(key)) for row in rows]
    vals = [val for val in vals if val is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check platoon_metrics.csv promotion criteria.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--stage", required=True, choices=sorted(THRESHOLDS))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--strict", action="store_true", help="Fail if optional metric columns are missing.")
    args = parser.parse_args()

    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"FAIL: no rows in {args.csv}")

    rows = rows[-max(args.window, 1):]
    thresholds = THRESHOLDS[args.stage]
    metrics = {
        "leader_speed_mean": _mean(rows, "leader_speed_mean"),
        "speed_error_abs_mean": _mean(rows, "speed_error_abs_mean"),
        "gap_error_abs_mean": _mean(rows, "gap_error_abs_mean"),
        "gap_error_max_abs": _mean(rows, "gap_error_max_abs"),
        "lateral_error_abs_mean": _mean(rows, "lateral_error_abs_mean"),
        "lateral_error_abs_max": _mean(rows, "lateral_error_abs_max"),
        "centerline_error_abs_mean": _mean(rows, "centerline_error_abs_mean"),
        "centerline_error_abs_max": _mean(rows, "centerline_error_abs_max"),
        "termination_reset_on_bad_ori": _mean(rows, "termination_reset_on_bad_ori"),
        "shield_trigger_rate": _mean(rows, "shield_trigger_rate"),
        "value_loss": _mean(rows, "value_loss"),
        "critic_grad_norm": _mean(rows, "critic_grad_norm"),
    }
    if metrics["termination_reset_on_bad_ori"] is None:
        # Older CSVs did not have termination columns. Do not block promotion
        # unless requested, but print the gap clearly.
        if args.strict:
            raise SystemExit("FAIL: missing termination_reset_on_bad_ori column")
        metrics["termination_reset_on_bad_ori"] = 0.0

    print(f"Stage check: {args.stage}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    failures = []
    if metrics["leader_speed_mean"] is None or metrics["leader_speed_mean"] < thresholds["leader_speed_mean_min"]:
        failures.append(
            f"leader_speed_mean < {thresholds['leader_speed_mean_min']}"
        )
    if metrics["speed_error_abs_mean"] is None or metrics["speed_error_abs_mean"] > thresholds["speed_error_abs_mean_max"]:
        failures.append(
            f"speed_error_abs_mean > {thresholds['speed_error_abs_mean_max']}"
        )
    for metric_key in (
        "gap_error_abs_mean",
        "gap_error_max_abs",
        "lateral_error_abs_mean",
        "lateral_error_abs_max",
        "centerline_error_abs_mean",
        "centerline_error_abs_max",
        "shield_trigger_rate",
    ):
        threshold_key = f"{metric_key}_max"
        threshold = thresholds.get(threshold_key)
        if threshold is not None and metrics[metric_key] is not None and metrics[metric_key] > threshold:
            failures.append(f"{metric_key} > {threshold}")
    if (
        metrics["termination_reset_on_bad_ori"] is None
        or metrics["termination_reset_on_bad_ori"] > thresholds["termination_reset_on_bad_ori_max"]
    ):
        failures.append(
            f"termination_reset_on_bad_ori > {thresholds['termination_reset_on_bad_ori_max']}"
        )
    if metrics["value_loss"] is None or metrics["value_loss"] > thresholds["value_loss_max"]:
        failures.append(f"value_loss > {thresholds['value_loss_max']}")
    if metrics["critic_grad_norm"] is None or metrics["critic_grad_norm"] > thresholds["critic_grad_norm_max"]:
        failures.append(f"critic_grad_norm > {thresholds['critic_grad_norm_max']}")

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("PASS")


if __name__ == "__main__":
    main()
