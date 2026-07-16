#!/usr/bin/env python3
"""Generate reproducible AMS versus Meta evaluation comparison artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    ("episode_return_mean", "1000-step return", "higher"),
    ("speed_error_abs_mean", "Speed absolute error", "lower"),
    ("gap_error_abs_mean", "Gap absolute error", "lower"),
    ("centerline_error_abs_mean", "Centerline error", "lower"),
    ("lateral_error_abs_mean", "Lateral error", "lower"),
    ("heading_error_abs_mean", "Heading error", "lower"),
    ("pair_heading_error_abs_mean", "Pair heading error", "lower"),
    ("min_pair_gap_mean", "Minimum pair gap", "higher"),
    ("collision_rate", "Collision rate", "lower"),
    ("termination_reset_on_bad_ori", "Bad-orientation reset rate", "lower"),
    ("shield_lateral_rate", "Lateral Shield rate", "lower"),
    ("shield_lateral_critical_rate", "Critical lateral Shield rate", "lower"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-summary", type=Path, required=True)
    parser.add_argument("--ams-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-iteration", type=int, default=3000)
    parser.add_argument("--late-start", type=int, default=1200)
    return parser.parse_args()


def checkpoint_iteration(checkpoint: str, final_iteration: int) -> int:
    if "final" in checkpoint.lower():
        return final_iteration
    match = re.search(r"(\d+)", checkpoint)
    if not match:
        raise ValueError(f"Cannot infer iteration from checkpoint: {checkpoint}")
    return int(match.group(1))


def read_summary(path: Path, final_iteration: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "checkpoint" not in frame or "episode_return_mean" not in frame:
        raise ValueError(f"Missing required columns in {path}")
    frame = frame.copy()
    frame["iteration"] = frame["checkpoint"].map(
        lambda value: checkpoint_iteration(str(value), final_iteration)
    )
    if frame["iteration"].duplicated().any():
        raise ValueError(f"Duplicate checkpoint iterations in {path}")
    return frame.sort_values("iteration").reset_index(drop=True)


def percent_improvement(meta: float, ams: float, direction: str) -> float:
    if np.isclose(meta, 0.0):
        return 0.0 if np.isclose(ams, 0.0) else np.nan
    signed_delta = ams - meta if direction == "higher" else meta - ams
    return 100.0 * signed_delta / abs(meta)


def build_checkpoint_table(meta: pd.DataFrame, ams: pd.DataFrame) -> pd.DataFrame:
    left = meta[["iteration", "checkpoint", "episode_return_mean"]].rename(
        columns={"checkpoint": "meta_checkpoint", "episode_return_mean": "meta_return"}
    )
    right = ams[["iteration", "checkpoint", "episode_return_mean"]].rename(
        columns={"checkpoint": "ams_checkpoint", "episode_return_mean": "ams_return"}
    )
    result = left.merge(right, on="iteration", how="inner", validate="one_to_one")
    if len(result) != len(left) or len(result) != len(right):
        raise ValueError("Meta and AMS summaries do not contain the same checkpoint iterations")
    result["ams_minus_meta"] = result["ams_return"] - result["meta_return"]
    result["ams_percent_change"] = 100.0 * result["ams_minus_meta"] / result["meta_return"].abs()
    return result


def build_final_metric_table(meta: pd.DataFrame, ams: pd.DataFrame, final_iteration: int) -> pd.DataFrame:
    meta_final = meta.loc[meta["iteration"] == final_iteration]
    ams_final = ams.loc[ams["iteration"] == final_iteration]
    if len(meta_final) != 1 or len(ams_final) != 1:
        raise ValueError(f"Expected exactly one final row at iteration {final_iteration}")
    meta_row = meta_final.iloc[0]
    ams_row = ams_final.iloc[0]

    rows = []
    for column, label, direction in METRICS:
        if column not in meta_row or column not in ams_row:
            raise ValueError(f"Required comparison metric is missing: {column}")
        meta_value = float(meta_row[column])
        ams_value = float(ams_row[column])
        rows.append(
            {
                "metric": column,
                "label": label,
                "better_direction": direction,
                "meta": meta_value,
                "ams": ams_value,
                "ams_minus_meta": ams_value - meta_value,
                "ams_improvement_percent": percent_improvement(meta_value, ams_value, direction),
            }
        )
    return pd.DataFrame(rows)


def plot_returns(checkpoints: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(checkpoints["iteration"], checkpoints["meta_return"], "o-", label="Meta")
    ax.plot(checkpoints["iteration"], checkpoints["ams_return"], "s-", label="Meta + AMS")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("1000-step cumulative evaluation return")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_final_improvements(metrics: pd.DataFrame, output: Path) -> None:
    plotted = metrics.loc[metrics["metric"] != "episode_return_mean"].copy()
    values = plotted["ams_improvement_percent"].fillna(0.0)
    colors = np.where(values >= 0.0, "#177245", "#b33a3a")
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    positions = np.arange(len(plotted))
    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions, plotted["label"])
    ax.invert_yaxis()
    ax.axvline(0.0, color="#333333", linewidth=0.9)
    ax.set_xlabel("AMS improvement over Meta (%)")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    checkpoints: pd.DataFrame,
    metrics: pd.DataFrame,
    late_start: int,
) -> None:
    meta_peak = checkpoints.loc[checkpoints["meta_return"].idxmax()]
    ams_peak = checkpoints.loc[checkpoints["ams_return"].idxmax()]
    final = checkpoints.iloc[-1]
    early = checkpoints.loc[checkpoints["iteration"] == 300]
    late = checkpoints.loc[checkpoints["iteration"] >= late_start]

    meta_late_mean = float(late["meta_return"].mean())
    ams_late_mean = float(late["ams_return"].mean())
    meta_late_std = float(late["meta_return"].std(ddof=0))
    ams_late_std = float(late["ams_return"].std(ddof=0))
    improved = metrics.loc[metrics["ams_improvement_percent"] > 0.0, "label"].tolist()
    regressed = metrics.loc[metrics["ams_improvement_percent"] < 0.0, "label"].tolist()

    lines = [
        "# AMS vs Meta Comparison",
        "",
        "Both methods were evaluated in the same current Isaac runtime with the same ordered "
        "11-checkpoint, seed-42, 32-environment, 1000-step latpair-v3 protocol.",
        "",
        "## Return Summary",
        "",
        f"- Meta peak: `{meta_peak['meta_return']:.3f}` at iteration `{int(meta_peak['iteration'])}`.",
        f"- AMS peak: `{ams_peak['ams_return']:.3f}` at iteration `{int(ams_peak['iteration'])}` "
        f"(`{ams_peak['ams_return'] - meta_peak['meta_return']:+.3f}` versus the Meta peak).",
        f"- Meta final: `{final['meta_return']:.3f}`; AMS final: `{final['ams_return']:.3f}` "
        f"(`{final['ams_minus_meta']:+.3f}`, `{final['ams_percent_change']:+.3f}%`).",
        f"- Late mean from iteration {late_start}: Meta `{meta_late_mean:.3f}`, AMS `{ams_late_mean:.3f}` "
        f"(`{ams_late_mean - meta_late_mean:+.3f}`).",
        f"- Late population standard deviation: Meta `{meta_late_std:.3f}`, AMS `{ams_late_std:.3f}`.",
    ]
    if len(early) == 1:
        row = early.iloc[0]
        lines.append(
            f"- Iteration 300: Meta `{row['meta_return']:.3f}`, AMS `{row['ams_return']:.3f}` "
            f"(`{row['ams_minus_meta']:+.3f}`)."
        )
    lines.extend(
        [
            "",
            "## Final Metrics",
            "",
            "Positive percentages mean AMS is better under each metric's stated direction.",
            "",
            "| Metric | Better | Meta | AMS | AMS improvement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics.itertuples(index=False):
        improvement = "n/a" if np.isnan(row.ams_improvement_percent) else f"{row.ams_improvement_percent:+.3f}%"
        lines.append(
            f"| {row.label} | {row.better_direction} | {row.meta:.6f} | {row.ams:.6f} | {improvement} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "AMS is an overall return win only if its peak, final, and late mean exceed Meta; otherwise "
            "the result is reported as a tradeoff even when individual physical metrics improve.",
            f"Improved final metrics: {', '.join(improved) if improved else 'none'}.",
            f"Regressed final metrics: {', '.join(regressed) if regressed else 'none'}.",
            "This is a fixed-seed comparison. Multiple independent training seeds are required for a "
            "statistical superiority claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    meta = read_summary(args.meta_summary, args.final_iteration)
    ams = read_summary(args.ams_summary, args.final_iteration)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = build_checkpoint_table(meta, ams)
    metrics = build_final_metric_table(meta, ams, args.final_iteration)
    checkpoints.to_csv(output_dir / "checkpoint_comparison.csv", index=False)
    metrics.to_csv(output_dir / "final_metric_comparison.csv", index=False)
    plot_returns(checkpoints, output_dir / "return_comparison.png")
    plot_final_improvements(metrics, output_dir / "final_metric_improvements.png")
    write_report(output_dir / "comparison_report.md", checkpoints, metrics, args.late_start)

    final = checkpoints.iloc[-1]
    print(
        f"Wrote comparison artifacts to {output_dir}; "
        f"final AMS-Meta={final['ams_minus_meta']:+.3f} "
        f"({final['ams_percent_change']:+.3f}%)"
    )


if __name__ == "__main__":
    main()
