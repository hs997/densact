"""Run deterministic straight/turn commands through the real Ackermann action term."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--straight_steps", type=int, default=150)
parser.add_argument("--turn_steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import marl_platoon  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _snapshot(env, step: int) -> dict:
    asset = env.scene["robot"]
    names = list(asset.joint_names)
    joint_pos = asset.data.joint_pos[0].detach().cpu()
    joint_vel = asset.data.joint_vel[0].detach().cpu()
    return {
        "step": step,
        "root_pos": asset.data.root_pos_w[0].detach().cpu().tolist(),
        "root_lin_vel_b": asset.data.root_lin_vel_b[0].detach().cpu().tolist(),
        "joint_pos": {name: float(joint_pos[idx]) for idx, name in enumerate(names)},
        "joint_vel": {name: float(joint_vel[idx]) for idx, name in enumerate(names)},
    }


def main() -> None:
    cfg = parse_env_cfg("Isaac-Marl-Platoon-HAPPO-v0", device=args_cli.device, num_envs=1)
    cfg.episode_length_s = 120.0
    cfg.commands.base_velocity.resampling_time_range = (120.0, 120.0)
    cfg.commands.base_velocity.lane_change_enabled = False
    cfg.algorithm.use_happo_actions = False
    cfg.algorithm.enable_attack = False
    cfg.algorithm.enable_shield = False
    for idx, name in enumerate(("robot", "robot_2", "robot_3", "robot_4", "robot_5")):
        getattr(cfg.scene, name).init_state.pos = (-1.5 * idx, -0.065, 0.20)

    wrapped = gym.make("Isaac-Marl-Platoon-HAPPO-v0", cfg=cfg)
    env = wrapped.unwrapped
    wrapped.reset()
    samples = [_snapshot(env, 0)]

    action = torch.zeros((1, 10), device=env.device)
    action[:, 0::2] = 0.55
    for step in range(1, args_cli.straight_steps + 1):
        wrapped.step(action)
    samples.append(_snapshot(env, args_cli.straight_steps))

    action[:, 1::2] = 0.65
    final_step = args_cli.straight_steps + args_cli.turn_steps
    for step in range(args_cli.straight_steps + 1, final_step + 1):
        wrapped.step(action)
    samples.append(_snapshot(env, final_step))

    term = env.action_manager._terms["ackermann_1"]
    result = {
        "action_shape": list(env.action_manager.action.shape),
        "rear_joint_names": list(term._rear_names),
        "front_steer_joint_names": list(term._steer_names),
        "rear_targets": term.rear_wheel_velocity_targets[0].detach().cpu().tolist(),
        "front_steering_targets": term.front_steering_targets[0].detach().cpu().tolist(),
        "samples": samples,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
