# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (with optional Reward Teacher wrapper)."""

import sys
import os
import argparse
from pathlib import Path

# ------------------------------------------------------------
# Cross-platform path injection
# ------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_IAL_ROOT = _THIS_FILE.parents[3]
for _p in [
    _IAL_ROOT / "source" / "isaaclab",
    _IAL_ROOT / "source" / "isaaclab_rl",
    _IAL_ROOT / "source" / "isaaclab_tasks",
    _IAL_ROOT / "source" / "my_exts",
]:
    sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ensure custom task is registered
import marl_platoon  # noqa: F401

# ------------------------------------------------------------
# version check
# ------------------------------------------------------------
import importlib.metadata as metadata
import platform
from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

# ------------------------------------------------------------
# rest imports
# ------------------------------------------------------------
import gymnasium as gym
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from marl_platoon.utils.best_checkpoint import BestCheckpointState, maybe_save_best_checkpoint

# paper modules
from marl_platoon.tasks.platoon.attacks import sample_hybrid_attack
from marl_platoon.tasks.platoon.teacher import RewardTeacher as PaperRewardTeacher
from marl_platoon.tasks.platoon.config import PlatoonEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _find_happo_wrapper(env):
    """Find the task-local HAPPO gym wrapper under RSL/video wrappers."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if all(hasattr(current, name) for name in ("state_dict", "load_state_dict", "set_inference_mode")):
            if hasattr(current, "algorithm_router"):
                return current
        current = getattr(current, "env", None)
    return None


def _attach_happo_state_to_checkpoint(env, checkpoint_path: str) -> None:
    """Augment an RSL-RL checkpoint with task-local HAPPO weights when present."""
    happo_wrapper = _find_happo_wrapper(env)
    if happo_wrapper is None:
        return
    happo_state = happo_wrapper.state_dict()
    if not happo_state:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        print(f"[WARN] Cannot attach HAPPO state to non-dict checkpoint: {checkpoint_path}")
        return
    checkpoint["platoon_happo_state"] = happo_state
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO]: Attached task-local HAPPO state to: {checkpoint_path}")


def _load_happo_state_from_checkpoint(env, checkpoint_path: str) -> None:
    """Restore task-local HAPPO weights from an augmented RSL-RL checkpoint."""
    happo_wrapper = _find_happo_wrapper(env)
    if happo_wrapper is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        print(f"[WARN] Cannot load HAPPO state from non-dict checkpoint: {checkpoint_path}")
        return
    happo_state = checkpoint.get("platoon_happo_state")
    if not happo_state:
        print(f"[WARN] Checkpoint has no task-local HAPPO state: {checkpoint_path}")
        return
    happo_wrapper.load_state_dict(happo_state, strict=True)
    print(f"[INFO]: Loaded task-local HAPPO state from: {checkpoint_path}")

# ============================================================
# ✅ Reward Teacher 配置（可开关）
# ============================================================
TEACHER_ENABLED = True

TEACHER_HIDDEN_DIMS = (256, 256)
TEACHER_LR = 3e-4
TEACHER_WEIGHT_DECAY = 0.0
TEACHER_GRAD_CLIP = 1.0

TEACHER_UPDATES_PER_ITER = 2
TEACHER_BATCH_SIZE = 131072

TEACHER_WARMUP_TIMESTEPS = 200 * 4096 * 256
TEACHER_AFTER_WARMUP_USE_TEACHER_ONLY = False


# =========================
# ✅ 关键：兼容 TensorDict 的 policy obs 提取
# =========================
def _unwrap_obs(x):
    # gymnasium reset: (obs, info)
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return x[0]
    return x


def _extract_policy_tensor(obs_any):
    """
    支持：
    - TensorDict (torchrl.tensordict): obs_any["policy"]
    - dict: obs_any["policy"]
    - 直接 tensor
    """
    obs_any = _unwrap_obs(obs_any)

    # TensorDict / dict-like：优先拿 "policy"
    try:
        if hasattr(obs_any, "keys") and ("policy" in obs_any.keys()):
            return obs_any["policy"]
    except Exception:
        pass

    if isinstance(obs_any, dict) and ("policy" in obs_any):
        return obs_any["policy"]

    return obs_any


def extract_policy_obs_tensor(obs_any, num_envs: int, device: torch.device) -> torch.Tensor:
    x = _extract_policy_tensor(obs_any)
    if x is None:
        raise RuntimeError("[Teacher] policy obs is None")

    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device, dtype=torch.float32)
    else:
        x = x.to(device=device, dtype=torch.float32)

    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() > 2:
        x = x.view(num_envs, -1)

    # 强制 batch 对齐
    if x.shape[0] != num_envs:
        x = x.view(num_envs, -1)

    # ✅ 这一步就是你之前炸点的根因：dim=0
    if x.shape[-1] == 0:
        raise RuntimeError(f"[Teacher] policy obs has zero dim: shape={tuple(x.shape)}")

    return x


class RewardTeacher(nn.Module):
    """MLP: (obs, act) -> r_phi"""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256), activation="elu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}.get(str(activation).lower(), nn.ELU)
        layers = []
        in_dim = obs_dim + act_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)  # [N]


class TeacherRewardVecWrapper:
    """
    Wraps RslRlVecEnvWrapper:
    - r_phy = env reward
    - r_phi = teacher(obs, act)
    - r_used = shaped reward used by PPO
    - Teacher supervised-fits r_phy using (obs_t, act_t)
    - attack sampler injects paper-style DoS/FDI metadata for logging
    """
    def __init__(self, env, teacher: RewardTeacher, optim, device: torch.device, num_steps_per_env: int, obs_dim: int, act_dim: int, attack_cfg):
        self.env = env
        self.teacher = teacher
        self.optim = optim
        self.device = device
        self.num_steps_per_env = int(num_steps_per_env)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.attack_cfg = attack_cfg

        self._last_policy_obs = None
        self._buf_obs = []
        self._buf_act = []
        self._buf_rphy = []
        self._step_in_rollout = 0
        self._global_timesteps = 0
        self._last_teacher_loss = 0.0
        self._last_attack_stats = {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        out = self.env.reset(*args, **kwargs)
        pol = extract_policy_obs_tensor(out, int(self.num_envs), self.device)
        if pol.shape[-1] != self.obs_dim:
            raise RuntimeError(f"[Teacher] reset obs_dim mismatch: got {pol.shape[-1]} expected {self.obs_dim}")
        self._last_policy_obs = pol
        self._step_in_rollout = 0
        return out

    def get_observations(self):
        out = self.env.get_observations()
        pol = extract_policy_obs_tensor(out, int(self.num_envs), self.device)
        self._last_policy_obs = pol
        return out

    @torch.no_grad()
    def _compute_rphi(self, obs_policy: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        return self.teacher(obs_policy, actions)

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.device, dtype=torch.float32)

        if self._last_policy_obs is None:
            _ = self.get_observations()

        obs_t_policy = self._last_policy_obs  # (num_envs, obs_dim)
        obs_t_policy = obs_t_policy.view(int(self.num_envs), -1)
        actions = actions.view(int(self.num_envs), -1)

        if actions.shape[-1] != self.act_dim:
            raise RuntimeError(f"[Teacher] actions dim wrong: {tuple(actions.shape)} expected (*,{self.act_dim})")

        # Legacy non-HAPPO wrapper attack metadata. HAPPO/MGRS tasks use the
        # task-local CA-GAN attacker instead of this wrapper path.
        if getattr(self.attack_cfg, "enable_attack", False):
            beta_a, beta_p, f_a, f_p = sample_hybrid_attack(
                num_envs=int(self.num_envs),
                device=self.device,
                max_fdi_pos=float(getattr(self.attack_cfg, "max_fdi_pos", 8.0)),
                max_fdi_acc=float(getattr(self.attack_cfg, "max_fdi_acc", 3.0)),
                max_dos_rate=float(getattr(self.attack_cfg, "max_dos_rate", 0.3)),
            )
            self._last_attack_stats = {
                "attack/beta_a_mean": float(beta_a.mean().item()),
                "attack/beta_p_mean": float(beta_p.mean().item()),
                "attack/f_a_abs_mean": float(f_a.abs().mean().item()),
                "attack/f_p_abs_mean": float(f_p.abs().mean().item()),
            }

        # step underlying env (handle 4 or 5 returns)
        step_out = self.env.step(actions)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            obs_next, r_phy, terminated, truncated, infos = step_out
            dones = terminated | truncated
        else:
            obs_next, r_phy, dones, infos = step_out

        if not torch.is_tensor(r_phy):
            r_phy = torch.as_tensor(r_phy, device=self.device, dtype=torch.float32)
        r_phy = r_phy.view(-1)  # (num_envs,)

        r_phi = self._compute_rphi(obs_t_policy, actions)  # (num_envs,)
        r_phi = 5.0 * torch.tanh(r_phi / 5.0)

        # warmup mixing
        self._global_timesteps += int(self.num_envs)
        lam = 1.0
        if TEACHER_WARMUP_TIMESTEPS > 0:
            lam = min(1.0, float(self._global_timesteps) / float(TEACHER_WARMUP_TIMESTEPS))

        if (lam >= 1.0) and TEACHER_AFTER_WARMUP_USE_TEACHER_ONLY:
            r_used = r_phi
        else:
            r_used = (1.0 - lam) * r_phy + lam * r_phi

        # buffer for supervised teacher update
        self._buf_obs.append(obs_t_policy.detach())
        self._buf_act.append(actions.detach())
        self._buf_rphy.append(r_phy.detach())

        self._step_in_rollout += 1
        if self._step_in_rollout % self.num_steps_per_env == 0:
            self._update_teacher()

        # update cache for next step
        self._last_policy_obs = extract_policy_obs_tensor(obs_next, int(self.num_envs), self.device)

        # attach logs
        if infos is None:
            infos = {}
        if isinstance(infos, dict):
            extras = infos.get("extras", {})
            if not isinstance(extras, dict):
                extras = {}
            extras["teacher/r_phy_mean"] = float(r_phy.mean().item())
            extras["teacher/r_phi_mean"] = float(r_phi.mean().item())
            extras["teacher/lam"] = float(lam)
            extras["teacher/loss"] = float(self._last_teacher_loss)
            extras.update(self._last_attack_stats)
            infos["extras"] = extras

        # return same signature as underlying vec env
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            return obs_next, r_used, terminated, truncated, infos
        return obs_next, r_used, dones, infos

    def _update_teacher(self):
        obs = torch.cat(self._buf_obs, dim=0)
        act = torch.cat(self._buf_act, dim=0)
        tgt = torch.cat(self._buf_rphy, dim=0)

        self._buf_obs.clear()
        self._buf_act.clear()
        self._buf_rphy.clear()

        n = obs.shape[0]
        if n < 1024:
            return

        bs = min(int(TEACHER_BATCH_SIZE), n)
        updates = int(TEACHER_UPDATES_PER_ITER)

        with torch.inference_mode(False):
            with torch.enable_grad():
                self.teacher.train()
                last_loss = None
                for _ in range(updates):
                    idx = torch.randint(0, n, (bs,), device=obs.device)
                    pred = self.teacher(obs[idx], act[idx])
                    loss = F.smooth_l1_loss(pred, tgt[idx])
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.teacher.parameters(), float(TEACHER_GRAD_CLIP))
                    self.optim.step()
                    last_loss = loss

        self._last_teacher_loss = float(last_loss.item()) if last_loss is not None else 0.0
        self.teacher.eval()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported on CPU. Use --device cuda.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors only supported for manager based envs.")

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap env for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    algorithm_cfg = getattr(env_cfg, "algorithm", None)
    task_local_teacher = (
        str(getattr(algorithm_cfg, "algorithm", "")).lower() == "happo"
        and bool(getattr(algorithm_cfg, "enable_teacher", False))
    )

    # ✅ insert teacher wrapper (infer obs_dim from REAL reset output, not from spaces)
    # HAPPO platoon tasks use the task-local MGRS Teacher.  Wrapping here as well
    # would create a second, unrelated reward-teacher path for the frozen outer PPO.
    if TEACHER_ENABLED and not task_local_teacher:
        device = torch.device(str(agent_cfg.device))
        reset_out = env.reset()
        pol0 = extract_policy_obs_tensor(reset_out, int(env.num_envs), device)
        obs_dim = int(pol0.shape[-1])
        act_dim = int(env.action_space.shape[-1])

        teacher = RewardTeacher(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=TEACHER_HIDDEN_DIMS, activation="elu").to(device)
        optim = torch.optim.Adam(teacher.parameters(), lr=float(TEACHER_LR), weight_decay=float(TEACHER_WEIGHT_DECAY))

        env = TeacherRewardVecWrapper(
            env=env,
            teacher=teacher,
            optim=optim,
            device=device,
            num_steps_per_env=int(getattr(agent_cfg, "num_steps_per_env", 64)),
            obs_dim=obs_dim,
            act_dim=act_dim,
            attack_cfg=getattr(env_cfg, "attack", PlatoonEnvCfg.AttackCfg()),
        )
        print(f"[INFO] RewardTeacher enabled. obs_dim={obs_dim}, act_dim={act_dim}, device={device}")
    elif TEACHER_ENABLED and task_local_teacher:
        print("[INFO] RewardTeacher wrapper skipped: task-local HAPPO/MGRS teacher is enabled.")

    # runner
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.add_git_repo_to_log(__file__)

    original_save = runner.save

    def save_with_happo_state(path: str, *args, **kwargs):
        result = original_save(path, *args, **kwargs)
        _attach_happo_state_to_checkpoint(env, path)
        return result

    runner.save = save_with_happo_state

    best_state = BestCheckpointState()
    original_log = runner.log

    def log_and_save_best(locs: dict, width: int = 80, pad: int = 35):
        original_log(locs, width=width, pad=pad)
        maybe_save_best_checkpoint(runner, locs, best_state)

    runner.log = log_and_save_best

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        _load_happo_state_from_checkpoint(env, resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    final_checkpoint_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_checkpoint_path)
    print(f"[INFO]: Saved final model checkpoint to: {final_checkpoint_path}")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
