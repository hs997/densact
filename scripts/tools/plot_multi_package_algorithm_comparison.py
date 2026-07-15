#!/usr/bin/env python3
"""Combine and plot evaluation summaries from multiple comparison packages."""

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
    "harl_mappo_shared": "HARL MAPPO shared",
    "harl_haa2c": "HARL HAA2C",
    "harl_hatrpo": "HARL HATRPO",
}

COLORS = {
    "mappo": "#4C78A8",
    "happo_no_meta": "#F58518",
    "happo_meta": "#54A24B",
    "harl_mappo_shared": "#B279A2",
    "harl_haa2c": "#E45756",
    "harl_hatrpo": "#72B7B2",
}

METRICS = [
    ("episode_return_mean", "max", "Episode return"),
    ("speed_error_abs_mean", "min", "Speed error"),
    ("gap_error_abs_mean", "min", "Gap error"),
    ("centerline_error_abs_mean", "min", "Centerline error"),
    ("lateral_error_abs_mean", "min", "Pair lateral error"),
    ("min_pair_gap_mean", "max", "Minimum pair gap"),
    ("collision_rate", "min", "Collision rate"),
    ("termination_reset_on_bad_ori", "min", "Bad-orientation reset"),
]


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


def eval_labels(eval_root: Path) -> list[str]:
    known = [label for label in LABELS if (eval_root / label / "eval_summary.csv").exists()]
    extra = sorted(
        path.parent.name
        for path in eval_root.glob("*/eval_summary.csv")
        if path.parent.name not in known
    )
    return [*known, *extra]


def load_eval_root(result_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    eval_root = result_root / "evaluation"
    final_iteration = manifest_value(result_root, "max_iterations", -1)

    for algorithm in eval_labels(eval_root):
        summary_path = eval_root / algorithm / "eval_summary.csv"
        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        numeric_iters = [
            checkpoint_iteration(str(row.get("checkpoint", "")), -1)
            for _, row in df.iterrows()
            if checkpoint_iteration(str(row.get("checkpoint", "")), -1) >= 0
        ]
        fallback = final_iteration if final_iteration > 0 else (max(numeric_iters) if numeric_iters else len(df))
        df["algorithm"] = algorithm
        df["algorithm_name"] = LABELS.get(algorithm, algorithm)
        df["iteration"] = [
            checkpoint_iteration(str(row.get("checkpoint", "")), fallback)
            for _, row in df.iterrows()
        ]
        df["source_package"] = str(result_root)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_all(result_roots: list[Path]) -> pd.DataFrame:
    frames = [load_eval_root(root) for root in result_roots]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df = df.dropna(subset=["iteration"]).copy()
    df["iteration"] = df["iteration"].astype(int)
    return df.sort_values(["algorithm", "iteration", "checkpoint"]).reset_index(drop=True)


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby("algorithm", sort=False):
        group = group.sort_values(["iteration", "checkpoint"])
        rows.append(group.iloc[-1])
    out = pd.DataFrame(rows)
    order = {name: idx for idx, name in enumerate(LABELS)}
    out["_order"] = out["algorithm"].map(order).fillna(999)
    return out.sort_values(["_order", "algorithm"]).drop(columns=["_order"], errors="ignore")


def comparison_table(final_df: pd.DataFrame, highlight_algorithm: str) -> pd.DataFrame:
    highlight = final_df[final_df["algorithm"] == highlight_algorithm]
    others = final_df[final_df["algorithm"] != highlight_algorithm]
    if highlight.empty or others.empty:
        return pd.DataFrame()
    hrow = highlight.iloc[-1]
    records = []
    for metric, direction, label in METRICS:
        if metric not in final_df.columns:
            continue
        hval = pd.to_numeric(pd.Series([hrow.get(metric)]), errors="coerce").iloc[0]
        oval = pd.to_numeric(others[metric], errors="coerce")
        if oval.dropna().empty or pd.isna(hval):
            continue
        best_other = oval.max() if direction == "max" else oval.min()
        margin = hval - best_other if direction == "max" else best_other - hval
        records.append(
            {
                "metric": metric,
                "label": label,
                "direction": direction,
                "happo_meta_final": hval,
                "best_other_final": best_other,
                "margin_positive_is_better": margin,
                "beats_or_ties": bool(margin >= -1e-9),
            }
        )
    return pd.DataFrame(records)


def plot_reward(df: pd.DataFrame, out_dir: Path, title: str, highlight_algorithm: str) -> list[Path]:
    png = out_dir / "fig_01_multi_package_reward_curves.png"
    pdf = out_dir / "fig_01_multi_package_reward_curves.pdf"

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), dpi=180, sharex=True)
    order = {name: idx for idx, name in enumerate(LABELS)}
    for algorithm, group in sorted(df.groupby("algorithm"), key=lambda item: order.get(item[0], 999)):
        if "episode_return_mean" not in group:
            continue
        group = group.sort_values("iteration")
        y = pd.to_numeric(group["episode_return_mean"], errors="coerce")
        kwargs = {
            "marker": "o",
            "linewidth": 2.0,
            "markersize": 4.2,
            "label": LABELS.get(algorithm, algorithm),
            "color": COLORS.get(algorithm),
        }
        if algorithm == highlight_algorithm:
            kwargs.update({"linewidth": 3.0, "markersize": 5.4, "zorder": 5})
        for ax in axes:
            ax.plot(group["iteration"], y, **kwargs)

    axes[0].set_title(title)
    axes[0].set_ylabel("Episode return")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9, frameon=False)

    valid = pd.to_numeric(df.get("episode_return_mean"), errors="coerce").dropna()
    high = valid[valid > 10000]
    if not high.empty:
        axes[1].set_ylim(max(0.0, float(high.min()) - 800.0), float(high.max()) + 300.0)
    axes[1].set_title("Zoomed high-return region")
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Episode return")
    axes[1].grid(True, alpha=0.25)

    if not valid.empty:
        best_idx = valid.idxmax()
        best = df.loc[best_idx]
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
            f"{float(best['episode_return_mean']):.1f} @ {int(best['iteration'])}",
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


def plot_final_metrics(final_df: pd.DataFrame, out_dir: Path) -> Path:
    metrics = [
        ("episode_return_mean", "Return"),
        ("speed_error_abs_mean", "Speed error"),
        ("gap_error_abs_mean", "Gap error"),
        ("lateral_error_abs_mean", "Lateral error"),
        ("centerline_error_abs_mean", "Centerline error"),
        ("min_pair_gap_mean", "Min gap"),
    ]
    path = out_dir / "fig_02_multi_package_final_metrics.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=180)
    names = final_df["algorithm_name"].tolist()
    for ax, (col, title) in zip(axes.flat, metrics):
        if col not in final_df.columns:
            ax.axis("off")
            continue
        ax.bar(names, pd.to_numeric(final_df[col], errors="coerce"), color=[
            COLORS.get(alg, "#777777") for alg in final_df["algorithm"].tolist()
        ])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Final checkpoint comparison")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", default="Multi-package evaluation return")
    parser.add_argument("--highlight-algorithm", default="happo_meta")
    args = parser.parse_args()

    roots = [path.expanduser().resolve() for path in args.result_root]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(roots)
    if df.empty:
        raise SystemExit(f"No eval summaries found under: {', '.join(map(str, roots))}")

    combined_path = out_dir / "combined_eval_summary.csv"
    final_path = out_dir / "final_rows.csv"
    comparison_path = out_dir / "final_happo_meta_vs_others.csv"

    final_df = final_rows(df)
    comparison_df = comparison_table(final_df, args.highlight_algorithm)

    df.to_csv(combined_path, index=False)
    final_df.to_csv(final_path, index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(comparison_path, index=False)

    saved = [
        *plot_reward(df, out_dir, args.title, args.highlight_algorithm),
        plot_final_metrics(final_df, out_dir),
    ]

    print("[MULTI_COMPARE] saved:")
    for path in saved:
        print(f"  {path}")
    print(f"[MULTI_COMPARE] combined summary: {combined_path}")
    print(f"[MULTI_COMPARE] final rows: {final_path}")
    if not comparison_df.empty:
        print(f"[MULTI_COMPARE] final comparison: {comparison_path}")
        print(comparison_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
