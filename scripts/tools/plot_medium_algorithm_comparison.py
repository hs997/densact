#!/usr/bin/env python3
"""Plot fixed-medium evaluation curves for MAPPO/HAPPO ablations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "mappo": "MAPPO",
    "happo_no_meta": "HAPPO w/o meta",
    "happo_meta": "HAPPO + meta",
    "happo_meta_ams": "HAPPO + meta + AMS",
    "harl_mappo_shared": "HARL MAPPO shared",
    "harl_haa2c": "HARL HAA2C",
    "harl_hatrpo": "HARL HATRPO",
}


def manifest_value(result_root: Path, key: str, default: int) -> int:
    path = result_root / "manifest.txt"
    if not path.exists():
        return default
    for line in path.read_text().splitlines():
        if not line.startswith(f"{key}="):
            continue
        try:
            return int(line.split("=", 1)[1].strip())
        except ValueError:
            return default
    return default


def checkpoint_iteration(name: str, fallback: int) -> int:
    match = re.search(r"model_(\d+)", str(name))
    if match:
        return int(match.group(1))
    return int(fallback)


def load_eval(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    final_iteration = manifest_value(result_root, "max_iterations", -1)
    eval_root = result_root / "evaluation"
    labels = [label for label in LABELS if (eval_root / label / "eval_summary.csv").exists()]
    labels.extend(
        sorted(
            path.parent.name
            for path in eval_root.glob("*/eval_summary.csv")
            if path.parent.name not in labels
        )
    )
    for label in labels:
        summary_path = eval_root / label / "eval_summary.csv"
        if not summary_path.exists():
            print(f"[WARN] missing eval summary: {summary_path}")
            continue
        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        numeric_iters = [
            checkpoint_iteration(str(row.get("checkpoint", "")), -1)
            for _, row in df.iterrows()
            if checkpoint_iteration(str(row.get("checkpoint", "")), -1) >= 0
        ]
        fallback = final_iteration if final_iteration > 0 else (max(numeric_iters) if numeric_iters else len(df))
        df["algorithm"] = label
        df["algorithm_name"] = LABELS[label]
        df["iteration"] = [
            checkpoint_iteration(str(row.get("checkpoint", "")), fallback)
            for _, row in df.iterrows()
        ]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["algorithm", "iteration", "checkpoint"]).reset_index(drop=True)
    return out


def plot_return(df: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "fig_01_fixed_medium_episode_return.png"
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    for label, group in df.groupby("algorithm", sort=False):
        group = group.sort_values("iteration")
        ax.plot(
            group["iteration"],
            pd.to_numeric(group["episode_return_mean"], errors="coerce"),
            marker="o",
            linewidth=2.0,
            label=LABELS.get(label, label),
        )
    ax.set_title("Fixed-medium evaluation return across checkpoints")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("episode return, mean reward sum")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metrics(df: pd.DataFrame, out_dir: Path) -> Path:
    metrics = [
        ("speed_error_abs_mean", "Speed error"),
        ("gap_error_abs_mean", "Gap error"),
        ("lateral_error_abs_mean", "Pair lateral error"),
        ("centerline_error_abs_mean", "Centerline error"),
        ("min_pair_gap_mean", "Minimum pair gap"),
        ("collision_rate", "Collision rate"),
    ]
    path = out_dir / "fig_02_fixed_medium_eval_metrics.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=180)
    for ax, (col, title) in zip(axes.flat, metrics):
        if col not in df.columns:
            ax.axis("off")
            continue
        for label, group in df.groupby("algorithm", sort=False):
            group = group.sort_values("iteration")
            ax.plot(
                group["iteration"],
                pd.to_numeric(group[col], errors="coerce"),
                marker="o",
                linewidth=1.8,
                label=LABELS.get(label, label),
            )
        ax.set_title(title)
        ax.set_xlabel("training iteration")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Fixed-medium evaluation metrics across checkpoints")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_final_bars(df: pd.DataFrame, out_dir: Path) -> Path:
    final_rows = []
    for label, group in df.groupby("algorithm", sort=False):
        group = group.sort_values("iteration")
        final_rows.append(group.iloc[-1])
    final_df = pd.DataFrame(final_rows)
    metrics = [
        ("episode_return_mean", "Return"),
        ("speed_error_abs_mean", "Speed error"),
        ("lateral_error_abs_mean", "Lateral error"),
        ("centerline_error_abs_mean", "Centerline error"),
        ("min_pair_gap_mean", "Min gap"),
        ("collision_rate", "Collision"),
    ]
    path = out_dir / "fig_03_fixed_medium_final_bars.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=180)
    names = final_df["algorithm_name"].tolist()
    for ax, (col, title) in zip(axes.flat, metrics):
        if col not in final_df.columns:
            ax.axis("off")
            continue
        ax.bar(names, pd.to_numeric(final_df[col], errors="coerce"))
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Final checkpoint comparison under fixed medium attack")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    out_dir = result_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_eval(result_root)
    if df.empty:
        raise SystemExit(f"No eval summaries found under {result_root}")
    df.to_csv(out_dir / "combined_eval_summary.csv", index=False)
    saved = [
        plot_return(df, out_dir),
        plot_metrics(df, out_dir),
        plot_final_bars(df, out_dir),
    ]
    print("[COMPARE] saved:")
    for path in saved:
        print(f"  {path}")
    print(f"[COMPARE] combined summary: {out_dir / 'combined_eval_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
