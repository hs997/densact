#!/usr/bin/env python3
"""Create paper-oriented figures from multi-stage platoon curriculum runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


TRACKING_COLS = [
    "speed_error_abs_mean",
    "gap_error_abs_mean",
    "lateral_error_abs_mean",
    "centerline_error_abs_mean",
    "heading_error_abs_mean",
]

BAR_COLS = [
    "speed_error_abs_mean",
    "gap_error_abs_mean",
    "lateral_error_abs_mean",
    "centerline_error_abs_mean",
    "physical_cost",
    "collision_rate",
    "termination_reset_on_bad_ori",
    "termination_time_out",
]

ATTACK_COLS = [
    "attack_max_fdi_pos",
    "attack_max_fdi_acc",
    "attack_max_dos_rate",
    "fdi_abs_mean",
    "dos_rate",
    "obs_fdi_abs_mean",
    "obs_dos_rate",
    "act_fdi_abs_mean",
    "act_dos_rate",
]

SHIELD_COLS = [
    "min_pair_gap_mean",
    "shield_trigger_rate",
    "shield_lateral_rate",
    "shield_lateral_critical_rate",
    "shield_lateral_turn_mean",
]

TEACHER_COLS = [
    "teacher_shaping_mean",
    "teacher_loss",
    "teacher_outer_loss",
    "teacher_advantage_corr",
    "local_reward_shaping_mean",
]


def _parse_item(item: str) -> tuple[str, Path]:
    if "=" not in item:
        path = Path(item).expanduser().resolve()
        return path.name, path
    label, path = item.split("=", 1)
    return label.strip(), Path(path).expanduser().resolve()


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in {"attack_level", "attack_mode", "attack_target_mode", "checkpoint", "checkpoint_path", "step_csv"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_training_runs(items: list[str]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    offset = 0

    for item in items:
        label, run_dir = _parse_item(item)
        csv_path = run_dir / "platoon_metrics.csv"
        if not csv_path.exists():
            print(f"[WARN] missing training metrics: {csv_path}")
            continue
        df = _numeric(pd.read_csv(csv_path))
        if df.empty:
            print(f"[WARN] empty training metrics: {csv_path}")
            continue
        df["stage"] = label
        df["stage_step"] = range(1, len(df) + 1)
        df["global_step"] = df["stage_step"] + offset
        offset += len(df)
        frames.append(df)

        tail = df.tail(min(500, len(df)))
        row: dict[str, object] = {
            "stage": label,
            "run_dir": str(run_dir),
            "rows": len(df),
        }
        for col in sorted(set(BAR_COLS + TRACKING_COLS + ATTACK_COLS + SHIELD_COLS + TEACHER_COLS)):
            if col in tail.columns:
                row[col] = float(pd.to_numeric(tail[col], errors="coerce").mean())
        for idx in range(1, 6):
            col = f"centerline_robot_{idx}_abs_mean"
            if col in tail.columns:
                row[col] = float(pd.to_numeric(tail[col], errors="coerce").mean())
        for idx in range(1, 5):
            col = f"lateral_pair_{idx}_abs_mean"
            if col in tail.columns:
                row[col] = float(pd.to_numeric(tail[col], errors="coerce").mean())
        summaries.append(row)

    if not frames:
        return pd.DataFrame(), summaries
    return pd.concat(frames, ignore_index=True), summaries


def _load_eval_dirs(items: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for item in items:
        label, eval_dir = _parse_item(item)
        summary_path = eval_dir / "eval_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            if "step_csv" in summary.columns and len(summary) > 0:
                step_path = Path(str(summary["step_csv"].iloc[0]))
                if not step_path.exists():
                    step_path = eval_dir / step_path.name
            else:
                step_path = None
        else:
            step_path = None
        if step_path is None:
            candidates = sorted(eval_dir.glob("eval_steps_*.csv"))
            step_path = candidates[0] if candidates else None
        if step_path is None or not step_path.exists():
            print(f"[WARN] missing eval step CSV in {eval_dir}")
            continue
        df = _numeric(pd.read_csv(step_path))
        if df.empty:
            print(f"[WARN] empty eval step CSV: {step_path}")
            continue
        df["eval_label"] = label
        out[label] = df
    return out


def _available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns and df[col].notna().any()]


def _plot_lines(
    df: pd.DataFrame,
    columns: list[str],
    output_path: Path,
    title: str,
    x_col: str = "global_step",
    smooth: int = 50,
    ylabel: str = "value",
) -> bool:
    available = _available(df, columns)
    if not available:
        return False
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    for col in available:
        y = pd.to_numeric(df[col], errors="coerce")
        if smooth > 1:
            y = y.rolling(smooth, min_periods=1).mean()
        ax.plot(df[x_col], y, linewidth=1.6, label=col)
    _add_stage_boundaries(ax, df, x_col)
    ax.set_title(title)
    ax.set_xlabel("curriculum update")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def _add_stage_boundaries(ax, df: pd.DataFrame, x_col: str) -> None:
    if "stage" not in df.columns or x_col not in df.columns:
        return
    seen = set()
    for stage, group in df.groupby("stage", sort=False):
        x0 = float(group[x_col].min())
        x1 = float(group[x_col].max())
        if stage not in seen:
            ax.axvspan(x0, x1, alpha=0.035)
            ax.text((x0 + x1) * 0.5, 0.98, str(stage), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
            seen.add(stage)
        ax.axvline(x1, color="black", alpha=0.10, linewidth=0.8)


def _plot_training_overview(df: pd.DataFrame, out_dir: Path, smooth: int) -> list[Path]:
    saved: list[Path] = []
    specs = [
        ("fig_01_reward_physical.png", ["reward_env_mean", "physical_cost", "policy_loss", "value_loss"], "Training reward and losses"),
        ("fig_02_tracking_errors.png", TRACKING_COLS, "Tracking and formation errors"),
        ("fig_03_attack_strength.png", ATTACK_COLS, "Attack budgets and measured corruption"),
        ("fig_04_safety_shield.png", SHIELD_COLS, "Safety and shield response"),
        ("fig_05_teacher_meta.png", TEACHER_COLS, "Teacher, meta, and local reward signals"),
    ]
    for filename, cols, title in specs:
        path = out_dir / filename
        if _plot_lines(df, cols, path, title, smooth=smooth):
            saved.append(path)
    return saved


def _plot_stage_bars(summary_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if summary_df.empty:
        return saved
    stages = summary_df["stage"].astype(str).tolist()

    cols = _available(summary_df, BAR_COLS)
    if cols:
        n = len(cols)
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=160)
        for ax, col in zip(axes.flat, cols):
            values = pd.to_numeric(summary_df[col], errors="coerce")
            ax.bar(stages, values)
            ax.set_title(col, fontsize=9)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, axis="y", alpha=0.20)
        for ax in axes.flat[len(cols) :]:
            ax.axis("off")
        fig.suptitle("Stage comparison over the final training window")
        fig.tight_layout()
        path = out_dir / "fig_06_attack_level_bars.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    center_cols = _available(summary_df, [f"centerline_robot_{i}_abs_mean" for i in range(1, 6)])
    pair_cols = _available(summary_df, [f"lateral_pair_{i}_abs_mean" for i in range(1, 5)])
    if center_cols or pair_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
        if center_cols:
            x = range(len(stages))
            width = 0.15
            for idx, col in enumerate(center_cols):
                values = pd.to_numeric(summary_df[col], errors="coerce")
                axes[0].bar([v + (idx - 2) * width for v in x], values, width=width, label=col.replace("_abs_mean", ""))
            axes[0].set_xticks(list(x), stages, rotation=30)
            axes[0].set_title("Per-vehicle centerline error")
            axes[0].legend(fontsize=7)
            axes[0].grid(True, axis="y", alpha=0.20)
        else:
            axes[0].axis("off")
        if pair_cols:
            x = range(len(stages))
            width = 0.18
            for idx, col in enumerate(pair_cols):
                values = pd.to_numeric(summary_df[col], errors="coerce")
                axes[1].bar([v + (idx - 1.5) * width for v in x], values, width=width, label=col.replace("_abs_mean", ""))
            axes[1].set_xticks(list(x), stages, rotation=30)
            axes[1].set_title("Per-pair lateral error")
            axes[1].legend(fontsize=7)
            axes[1].grid(True, axis="y", alpha=0.20)
        else:
            axes[1].axis("off")
        fig.tight_layout()
        path = out_dir / "fig_07_pair_centerline_bars.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
    return saved


def _plot_eval(eval_frames: dict[str, pd.DataFrame], out_dir: Path, smooth: int) -> list[Path]:
    saved: list[Path] = []
    if not eval_frames:
        return saved

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=160)
    eval_cols = [
        ("min_pair_gap_mean", "Minimum pair gap"),
        ("speed_error_abs_mean", "Speed error"),
        ("lateral_error_abs_mean", "Pair lateral error"),
        ("centerline_error_abs_mean", "Centerline error"),
    ]
    for ax, (col, title) in zip(axes.flat, eval_cols):
        for label, df in eval_frames.items():
            if col not in df.columns:
                continue
            x = pd.to_numeric(df.get("eval_step", pd.Series(range(1, len(df) + 1))), errors="coerce")
            y = pd.to_numeric(df[col], errors="coerce")
            if smooth > 1:
                y = y.rolling(smooth, min_periods=1).mean()
            ax.plot(x, y, linewidth=1.5, label=label)
        if col == "min_pair_gap_mean":
            ax.axhline(1.45, color="red", linestyle="--", linewidth=1.0, label="d_drop=1.45")
        ax.set_title(title)
        ax.set_xlabel("eval step")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("Rollout safety and tracking under attack levels")
    fig.tight_layout()
    path = out_dir / "fig_08_eval_safety_tracking.png"
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    hard_label = "hard" if "hard" in eval_frames else list(eval_frames.keys())[-1]
    hard_df = eval_frames[hard_label]
    cols = _available(hard_df, ["fdi_abs_mean", "dos_rate", "speed_error_abs_mean", "lateral_error_abs_mean", "gap_error_abs_mean"])
    if cols:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
        x = pd.to_numeric(hard_df.get("eval_step", pd.Series(range(1, len(hard_df) + 1))), errors="coerce")
        for col in cols:
            y = pd.to_numeric(hard_df[col], errors="coerce")
            if smooth > 1:
                y = y.rolling(smooth, min_periods=1).mean()
            ax.plot(x, y, linewidth=1.5, label=col)
        ax.set_title(f"Attack response time series ({hard_label})")
        ax.set_xlabel("eval step")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "fig_09_eval_attack_response.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
    return saved


def _write_summary(path: Path, rows: list[dict[str, object]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("stage,rows\n", encoding="utf-8")
        return pd.DataFrame()
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper figures from staged platoon runs.")
    parser.add_argument("--run", action="append", default=[], help="Training run as label=/path/to/run. Repeatable.")
    parser.add_argument("--eval", action="append", default=[], help="Eval dir as label=/path/to/eval_dir. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth", type=int, default=50)
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    training_df, summary_rows = _load_training_runs(args.run)
    summary_df = _write_summary(out_dir / "summary_by_stage.csv", summary_rows)

    saved: list[Path] = []
    if not training_df.empty:
        training_df.to_csv(out_dir / "combined_training_metrics.csv", index=False)
        saved.extend(_plot_training_overview(training_df, out_dir, max(args.smooth, 1)))
        saved.extend(_plot_stage_bars(summary_df, out_dir))

    eval_frames = _load_eval_dirs(args.eval)
    if eval_frames:
        eval_concat = pd.concat(eval_frames.values(), ignore_index=True)
        eval_concat.to_csv(out_dir / "combined_eval_steps.csv", index=False)
        saved.extend(_plot_eval(eval_frames, out_dir, max(args.smooth // 4, 1)))

    print("[PAPER FIGURES] saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
