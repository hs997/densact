"""Smoke test for the IsaacLab -> HARL adapter.

This script is intentionally minimal.
It verifies that the platoon task can be created, wrapped, reset, and stepped
without wiring any HAPPO training logic yet.

Run from the IsaacLab repo root with the usual stable launch command.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Keep the same stable IsaacLab launch style as the known-good platoon setup.
_THIS_FILE = Path(__file__).resolve()
_IAL_ROOT = _THIS_FILE.parents[4]
for _p in [
    _IAL_ROOT / "source" / "isaaclab",
    _IAL_ROOT / "source" / "isaaclab_rl",
    _IAL_ROOT / "source" / "isaaclab_tasks",
    _IAL_ROOT / "source" / "my_exts",
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test HARL adapter for platoon.")
parser.add_argument("--task", type=str, default="Isaac-Marl-Platoon-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_agents", type=int, default=5)
parser.add_argument("--obs_dim", type=int, default=18)
parser.add_argument("--act_dim", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import marl_platoon  # noqa: F401,E402
from marl_platoon.harl_adapter import IsaacLabHAPPOAdapter  # noqa: E402
from marl_platoon.tasks.platoon.config import PlatoonEnvCfg  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402


def main():
    env_cfg = PlatoonEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    adapter = IsaacLabHAPPOAdapter(
        env=env,
        num_agents=args_cli.num_agents,
        obs_dim=args_cli.obs_dim,
        act_dim=args_cli.act_dim,
    )

    obs, share_obs, masks = adapter.reset()
    print("[HARL-SMOKE] reset ok")
    print(f"[HARL-SMOKE] obs shape: {tuple(obs.shape)}")
    print(f"[HARL-SMOKE] share_obs shape: {tuple(share_obs.shape)}")
    print(f"[HARL-SMOKE] masks shape: {tuple(masks.shape)}")

    action = torch.zeros((args_cli.num_envs, args_cli.num_agents, args_cli.act_dim), dtype=torch.float32)
    step_out = adapter.step(action)
    next_obs, next_share_obs, next_rewards, next_dones, next_infos, next_masks = step_out
    print("[HARL-SMOKE] step ok")
    print(f"[HARL-SMOKE] next_obs shape: {tuple(next_obs.shape)}")
    print(f"[HARL-SMOKE] next_share_obs shape: {tuple(next_share_obs.shape)}")
    print(f"[HARL-SMOKE] next_rewards shape: {tuple(next_rewards.shape)}")
    print(f"[HARL-SMOKE] next_dones shape: {tuple(next_dones.shape)}")
    print(f"[HARL-SMOKE] next_masks shape: {tuple(next_masks.shape)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
