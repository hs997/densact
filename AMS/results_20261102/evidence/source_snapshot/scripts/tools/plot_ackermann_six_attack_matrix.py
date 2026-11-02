"""Build cross-algorithm, cross-attack CSVs and heatmaps from final ladders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ackermann_validation import SEEDS, load_verified


LEVELS = ["off", "light", "easy_0", "easy_a", "easy_b", "med_a", "med_b", "hard_a", "hard_b"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    algos = sorted(path.name for path in args.root.iterdir() if path.is_dir())
    rows = []
    for algo in algos:
        for level in LEVELS:
            summaries = list((args.root / algo / level / "evaluation").glob("eval_summary.csv"))
            if not summaries:
                raise FileNotFoundError(f"missing summary for {algo}/{level}")
            data = list(csv.DictReader(summaries[0].open(newline="")))
            if not data:
                raise ValueError(f"empty summary for {algo}/{level}")
            row = data[-1]
            directory = args.root / algo / level
            assessment = directory / "assessment.json"
            verified = []
            if assessment.exists():
                report = json.loads(assessment.read_text())
                names = (["clean_env42"] if level == "off" else
                         [f"attack_env{e}_seed{a}" for e, a in SEEDS])
                verified = [load_verified(directory / "validation" / name) for name in names]
                if len({spec["checkpoint_sha256"] for _, spec in verified}) != 1:
                    raise ValueError(f"checkpoint mismatch in {algo}/{level}")
                if any(spec["sources"] != verified[0][1]["sources"] for _, spec in verified):
                    raise ValueError(f"source mismatch in {algo}/{level}")
                returns = []
                for name in names:
                    with (directory / "validation" / name / "eval_summary.csv").open(newline="") as stream:
                        returns.append(float(next(csv.DictReader(stream))["episode_return_mean"]))
            else:
                report = {}
                returns = [float(row.get("episode_return_mean", 0.0))]
            completions = [r["completion"] for r, _ in verified] or [float(row.get("platoon_completion_ever_rate_final", 0.0))]
            outcomes_path = args.root.parent / "method_outcomes.json"
            outcome = json.loads(outcomes_path.read_text()).get(algo, {}) if outcomes_path.exists() else {}
            rows.append({
                "algorithm": algo,
                "attack_level": level,
                "episode_return_mean": float(np.mean(returns)),
                "whole_platoon_completion_rate": float(np.mean(completions)),
                "completion_seed_min": min(completions),
                "completion_seed_std": float(np.std(completions)),
                "collision_rate": max((r["collision"] for r, _ in verified), default=float(row.get("collision_rate", 0.0))),
                "road_boundary_violation_rate": max((r["road_any"] for r, _ in verified), default=float(row.get("road_boundary_violation_rate", 0.0))),
                "road_termination_rate_max": max((r["road_termination"] for r, _ in verified), default=float("nan")),
                "verified_seed_count": len(verified),
                "gate_passed": report.get("passed", "unverified"),
                "training_status": outcome.get("status", "unknown"),
                "stopped_training_level": outcome.get("stopped_level"),
                "checkpoint_sha256": verified[0][1]["checkpoint_sha256"] if verified else "unverified",
            })
    output = args.root.parent / "six_algorithm_attack_matrix.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for field, title, filename, fmt in (
        ("whole_platoon_completion_rate", "Whole-platoon ever/dwell completion (attack: three-seed mean)", "completion_heatmap.png", ".2f"),
        ("completion_seed_min", "Whole-platoon completion (worst seed)", "completion_min_heatmap.png", ".2f"),
        ("episode_return_mean", "Episode return", "return_heatmap.png", ".0f"),
    ):
        matrix = np.array([[next(r[field] for r in rows if r["algorithm"] == a and r["attack_level"] == l) for l in LEVELS] for a in algos])
        fig, ax = plt.subplots(figsize=(12, 5.2))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(LEVELS)), LEVELS, rotation=30, ha="right")
        ax.set_yticks(range(len(algos)), algos)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white" if matrix[i, j] < np.nanmean(matrix) else "black", fontsize=8)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(args.root.parent / filename, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
