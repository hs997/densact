#!/usr/bin/env python3
"""Plot recorded stage data; never infer true XY trajectories from aggregate means."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_stage(root):
    output = root / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.glob('validation/*/eval_steps_*.csv')):
        if '.superseded_' in str(path):
            continue
        data = pd.read_csv(path)
        t = data.eval_step * .02
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for i in range(1, 6):
            axes[0].plot(t, data[f'trajectory_progress_robot_{i}_mean'], label=f'vehicle {i}')
        axes[0].axhline(.99, color='k', linestyle=':')
        axes[0].set_ylabel('Path progress (env mean)')
        axes[0].legend(ncol=5)
        axes[1].plot(t, data.platoon_completion_max_gap_error, label='Worst pair gap error (env mean)')
        axes[1].axhline(.35, color='r', linestyle='--', label='Per-env gap gate 0.35 m')
        axes[1].set_ylabel('Gap error (m)')
        axes[1].legend()
        for key, label in [('platoon_completion_ever_rate', 'ever dwell complete'),
                           ('platoon_completion_dwell_rate', 'current dwell complete'),
                           ('collision_rate', 'collision'), ('road_boundary_any_rate', 'road boundary')]:
            axes[2].plot(t, data[key], label=label)
        axes[2].set_ylabel('Environment fraction')
        axes[2].set_xlabel('Evaluation time (s); terminal reset retained')
        axes[2].legend(ncol=2)
        for ax in axes:
            ax.grid(alpha=.25)
        fig.suptitle(f'{root.name} / {path.parent.name}\nAggregate diagnostics, not a true XY or single-run success proof')
        fig.tight_layout()
        fig.savefig(output / f'{path.parent.name}_progress_gap_safety.png', dpi=160)
        plt.close(fig)
    runs = root / 'training/runs.txt'
    if runs.exists():
        for line in runs.read_text().splitlines():
            if '=' not in line:
                continue
            label = line.split('=', 1)[0]
            metrics = Path(line.split('=', 1)[1]) / 'platoon_metrics.csv'
            if not metrics.exists():
                continue
            data = pd.read_csv(metrics)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(data.sim_time_s, data.reward_env_mean, alpha=.25, label='Recorded step reward')
            ax.plot(data.sim_time_s, data.reward_env_mean.rolling(160, min_periods=1).mean(), label='160-step rolling mean')
            ax.set(xlabel='Recorded simulation time (s)', ylabel='Environment mean reward', title=root.name)
            ax.legend()
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(output / f'{label}_training_reward.png', dpi=160)
            plt.close(fig)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stages', type=Path, nargs='+')
    for stage in parser.parse_args().stages:
        if stage.is_dir():
            plot_stage(stage)
