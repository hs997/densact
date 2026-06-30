#!/usr/bin/env python3
"""Plot platoon HAPPO/MGRS training metrics from platoon_metrics.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing Python package: {missing}. Install plotting dependencies with: "
        "python3 -m pip install pandas matplotlib"
    ) from exc


PLOT_GROUPS = {
    "speed_tracking": [
        "command_speed_mean",
        "leader_speed_mean",
        "platoon_speed_mean",
        "speed_error_abs_mean",
    ],
    "formation_errors": [
        "gap_error_abs_mean",
        "gap_error_max_abs",
        "lateral_error_abs_mean",
        "lateral_error_abs_max",
        "centerline_error_abs_mean",
        "centerline_error_abs_max",
        "min_pair_gap_mean",
        "collision_rate",
    ],
    "lateral_pairs": [
        "lateral_pair_1_abs_mean",
        "lateral_pair_2_abs_mean",
        "lateral_pair_3_abs_mean",
        "lateral_pair_4_abs_mean",
        "lateral_worst_pair",
    ],
    "lateral_pairs_signed": [
        "lateral_pair_1_signed_mean",
        "lateral_pair_2_signed_mean",
        "lateral_pair_3_signed_mean",
        "lateral_pair_4_signed_mean",
    ],
    "centerline_robots": [
        "centerline_robot_1_abs_mean",
        "centerline_robot_2_abs_mean",
        "centerline_robot_3_abs_mean",
        "centerline_robot_4_abs_mean",
        "centerline_robot_5_abs_mean",
    ],
    "centerline_robots_signed": [
        "centerline_robot_1_signed_mean",
        "centerline_robot_2_signed_mean",
        "centerline_robot_3_signed_mean",
        "centerline_robot_4_signed_mean",
        "centerline_robot_5_signed_mean",
    ],
    "action_turn_bias": [
        "raw_action_turn_agent_1",
        "raw_action_turn_agent_2",
        "raw_action_turn_agent_3",
        "raw_action_turn_agent_4",
        "raw_action_turn_agent_5",
        "executed_action_turn_agent_1",
        "executed_action_turn_agent_2",
        "executed_action_turn_agent_3",
        "executed_action_turn_agent_4",
        "executed_action_turn_agent_5",
    ],
    "local_reward_shaping": [
        "local_reward_shaping_mean",
        "local_reward_centerline_penalty_mean",
        "local_reward_pair_lateral_penalty_mean",
        "local_reward_heading_penalty_mean",
        "local_reward_turn_penalty_mean",
    ],
    "heading_errors": [
        "heading_error_abs_mean",
        "heading_error_abs_max",
        "pair_heading_error_abs_mean",
        "pair_heading_error_abs_max",
    ],
    "rewards": [
        "reward_env_mean",
        "reward_leader_motion",
        "reward_leader_progress",
        "reward_formation",
        "reward_forward_drive",
        "reward_stall_penalty",
        "reward_lateral_correct",
        "reward_lateral_velocity",
        "reward_centerline_lateral",
        "reward_heading_align",
        "reward_true_success",
        "reward_true_fail",
    ],
    "attack": [
        "fdi_abs_mean",
        "dos_rate",
        "obs_fdi_abs_mean",
        "obs_dos_rate",
        "act_fdi_abs_mean",
        "act_dos_rate",
        "attack_objective",
        "attack_phy_ctx_norm",
    ],
    "teacher": [
        "teacher_shaping_mean",
        "teacher_loss",
        "teacher_delta_j",
        "teacher_norm_delta_j",
        "teacher_outer_loss",
        "teacher_advantage_corr",
        "teacher_grad_norm",
    ],
    "happo_losses": [
        "policy_loss",
        "entropy",
        "value_loss",
        "ratio",
        "actor_grad_norm",
        "critic_grad_norm",
    ],
    "physical_costs": [
        "physical_cost",
        "physical_spacing_cost",
        "physical_velocity_cost",
        "physical_acceleration_cost",
        "physical_jerk_cost",
        "physical_lateral_cost",
        "physical_forward_deficit_cost",
        "physical_collision_cost",
    ],
    "shield": [
        "shield_warn_rate",
        "shield_critical_rate",
        "shield_trigger_rate",
        "shield_scale_mean",
        "shield_lateral_rate",
        "shield_lateral_critical_rate",
        "shield_lateral_turn_mean",
    ],
}


def _numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in {"attack_level", "attack_mode", "attack_target_mode", "attack_ref_template_id"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _plot_group(df: pd.DataFrame, x_col: str, columns: list[str], title: str, output_path: Path, smooth: int) -> bool:
    available = [col for col in columns if col in df.columns and df[col].notna().any()]
    if not available:
        return False
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    for col in available:
        y = df[col]
        if smooth > 1:
            y = y.rolling(window=smooth, min_periods=1).mean()
        ax.plot(df[x_col], y, label=col, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def _plot_attack_modes(df: pd.DataFrame, x_col: str, output_path: Path) -> bool:
    if "attack_mode" not in df.columns:
        return False
    modes = df["attack_mode"].fillna("").astype(str)
    if modes.nunique() <= 1:
        return False
    categories = {name: idx for idx, name in enumerate(sorted(modes.unique()))}
    y = modes.map(categories)
    fig, ax = plt.subplots(figsize=(11, 3.5), dpi=140)
    ax.step(df[x_col], y, where="post")
    ax.set_yticks(list(categories.values()), list(categories.keys()))
    ax.set_title("attack mode schedule")
    ax.set_xlabel(x_col)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot platoon_metrics.csv produced by task-local HAPPO/MGRS.")
    parser.add_argument("csv", type=Path, help="Path to platoon_metrics.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for PNG figures")
    parser.add_argument("--x", choices=("update", "sim_time_s", "env_step"), default="update")
    parser.add_argument("--smooth", type=int, default=5, help="Rolling mean window. Use 1 to disable.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window after saving.")
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else csv_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"No rows found in {csv_path}")
    df = _numeric_columns(df)
    x_col = args.x if args.x in df.columns else "update"
    df = df.sort_values(x_col)

    saved = []
    for name, columns in PLOT_GROUPS.items():
        path = out_dir / f"{name}.png"
        if _plot_group(df, x_col, columns, name.replace("_", " "), path, smooth=max(args.smooth, 1)):
            saved.append(path)
    mode_path = out_dir / "attack_mode_schedule.png"
    if _plot_attack_modes(df, x_col, mode_path):
        saved.append(mode_path)

    summary_path = out_dir / "summary.csv"
    numeric = df.select_dtypes(include="number")
    numeric.describe().transpose().to_csv(summary_path)
    saved.append(summary_path)

    print("Saved:")
    for path in saved:
        print(f"  {path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
