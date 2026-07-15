#!/usr/bin/env python3
"""Plot updated HAPPO+meta against the other five fixed-medium algorithms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


OLD_ALGORITHMS = [
    "mappo",
    "happo_no_meta",
    "harl_mappo_shared",
    "harl_haa2c",
    "harl_hatrpo",
]

LABELS = {
    "mappo": "MAPPO",
    "happo_no_meta": "HAPPO w/o meta",
    "happo_meta_tuned": "HAPPO + meta (tuned)",
    "harl_mappo_shared": "HARL MAPPO shared",
    "harl_haa2c": "HARL HAA2C",
    "harl_hatrpo": "HARL HATRPO",
}

COLORS = {
    "mappo": "#4C78A8",
    "happo_no_meta": "#F58518",
    "happo_meta_tuned": "#54A24B",
    "harl_mappo_shared": "#B279A2",
    "harl_haa2c": "#E45756",
    "harl_hatrpo": "#72B7B2",
}


def load_old_five(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["algorithm"].isin(OLD_ALGORITHMS)].copy()
    return df


def load_tuned(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "algorithm" not in df.columns:
        df["algorithm"] = "happo_meta_tuned"
    if "algorithm_name" not in df.columns:
        df["algorithm_name"] = LABELS["happo_meta_tuned"]
    df["algorithm"] = "happo_meta_tuned"
    df["algorithm_name"] = LABELS["happo_meta_tuned"]
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["iteration"] = pd.to_numeric(out["iteration"], errors="coerce")
    out["episode_return_mean"] = pd.to_numeric(out["episode_return_mean"], errors="coerce")
    out = out.dropna(subset=["iteration", "episode_return_mean"])
    order = {name: idx for idx, name in enumerate([*OLD_ALGORITHMS[:2], "happo_meta_tuned", *OLD_ALGORITHMS[2:]])}
    out["_order"] = out["algorithm"].map(order).fillna(999)
    return out.sort_values(["_order", "iteration"]).reset_index(drop=True)


def plot(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig_04_updated_happo_meta_vs_five_reward_curves.png"
    pdf = out_dir / "fig_04_updated_happo_meta_vs_five_reward_curves.pdf"

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), dpi=180, sharex=True)
    for algorithm, group in df.groupby("algorithm", sort=False):
        group = group.sort_values("iteration")
        label = LABELS.get(algorithm, algorithm)
        kwargs = {
            "marker": "o",
            "linewidth": 2.0,
            "markersize": 4.5,
            "label": label,
            "color": COLORS.get(algorithm),
        }
        if algorithm == "happo_meta_tuned":
            kwargs.update({"linewidth": 2.8, "markersize": 5.5, "zorder": 5})
        for ax in axes:
            ax.plot(group["iteration"], group["episode_return_mean"], **kwargs)

    axes[0].set_title("Fixed-medium attack evaluation return")
    axes[0].set_ylabel("Episode return")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9, frameon=False)

    high = df[df["episode_return_mean"] > 10000]["episode_return_mean"]
    if not high.empty:
        lower = max(15000.0, float(high.min()) - 500.0)
        upper = float(high.max()) + 250.0
        axes[1].set_ylim(lower, upper)
    axes[1].set_title("Zoomed high-return region")
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Episode return")
    axes[1].grid(True, alpha=0.25)

    best = df.loc[df["episode_return_mean"].idxmax()]
    axes[1].scatter(
        [best["iteration"]],
        [best["episode_return_mean"]],
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.6,
        zorder=10,
    )
    axes[1].annotate(
        f"best: {LABELS.get(best['algorithm'], best['algorithm'])}\n"
        f"{best['episode_return_mean']:.1f} @ {int(best['iteration'])}",
        xy=(best["iteration"], best["episode_return_mean"]),
        xytext=(24, -36),
        textcoords="offset points",
        fontsize=8.5,
        va="top",
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )

    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-combined", required=True, type=Path)
    parser.add_argument("--tuned-combined", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    old_df = load_old_five(args.old_combined.expanduser().resolve())
    tuned_df = load_tuned(args.tuned_combined.expanduser().resolve())
    df = prepare(pd.concat([old_df, tuned_df], ignore_index=True, sort=False))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "combined_eval_summary_updated_happo_meta_vs_five.csv"
    df.drop(columns=["_order"], errors="ignore").to_csv(csv_path, index=False)
    saved = plot(df, args.out_dir)
    print("[UPDATED_HAPPO_META_PLOT] saved:")
    for path in saved:
        print(f"  {path}")
    print(f"[UPDATED_HAPPO_META_PLOT] combined csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
