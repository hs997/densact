"""Task-local algorithm router for marl_platoon.

This module keeps the official IsaacLab/RSL-RL entrypoint intact. PPO remains a
no-op here because it is still handled by the stable RSL-RL runner. HAPPO is
built as an internal task component and can be stepped explicitly by task-side
code once the rollout loop is connected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from marl_platoon.harl_adapter import flatten_joint_action, merge_agent_obs, split_agent_obs
from marl_platoon.algorithms.happo import PlatoonHAPPOCfg, PlatoonHAPPORunner
from marl_platoon.algorithms.student import HAPPOStudentModule
from marl_platoon.algorithms.teacher import RewardTeacherCfg, RewardTeacherModule
from marl_platoon.algorithms.attacker import AttackModuleCfg, NoOpAttackModule, RandomFDIDoSAttackModule
from marl_platoon.algorithms.shield import BadHeadingShieldCfg, BadHeadingShieldModule
from marl_platoon.algorithms.pipeline import PipelineFlags, PipelineScheduleCfg, PlatoonTrainingPipeline, StepBatch


@dataclass
class PlatoonAlgorithmRuntimeInfo:
    algorithm: str
    num_agents: int
    obs_dim: int
    act_dim: int
    share_obs_dim: int
    num_envs: int
    device: str


class PlatoonAlgorithmRouter:
    """Runtime bridge between platoon env tensors and task-local algorithms."""

    def __init__(self, env: Any, cfg: Any, runner_cfg: Any | None = None):
        self.env = env
        self.cfg = cfg
        self.runner_cfg = runner_cfg
        self.algorithm = str(getattr(cfg, "algorithm", "ppo")).lower()
        self.num_agents = int(getattr(cfg, "num_agents", 4))
        self.device = torch.device(getattr(env, "device", "cpu"))
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.runner: PlatoonHAPPORunner | None = None
        self.info: PlatoonAlgorithmRuntimeInfo | None = None
        self.rollout_steps = 0
        self.happo_action_steps = 0
        self.happo_update_count = 0
        self._printed_build = False
        self._printed_action_override = False
        self._rollout_log_interval = int(getattr(cfg, "happo_log_interval", 32))
        self._log_level = str(getattr(cfg, "happo_log_level", "basic")).lower()
        self.pipeline: PlatoonTrainingPipeline | None = None
        self._last_next_obs: torch.Tensor | None = None

    def build_from_sample(self, obs: Any) -> PlatoonHAPPORunner | None:
        """Build the selected algorithm from a reset observation sample.

        PPO returns `None` so the existing RSL-RL PPO path remains untouched.
        HAPPO creates an internal runner but does not start a separate training
        entrypoint.
        """
        agent_obs = self.to_agent_obs(obs)
        obs_dim = int(agent_obs.shape[-1])
        act_dim = self._infer_agent_action_dim()
        share_obs = merge_agent_obs(agent_obs)
        share_obs_dim = int(share_obs.shape[-1])
        self.info = PlatoonAlgorithmRuntimeInfo(
            algorithm=self.algorithm,
            num_agents=self.num_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            share_obs_dim=share_obs_dim,
            num_envs=self.num_envs,
            device=str(self.device),
        )

        if self.algorithm == "ppo":
            return None
        if self.algorithm != "happo":
            raise ValueError(f"Unsupported platoon algorithm: {self.algorithm}")

        episode_length = int(getattr(self.runner_cfg, "num_steps_per_env", 32)) if self.runner_cfg is not None else 32
        happo_cfg = PlatoonHAPPOCfg(
            num_agents=self.num_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            share_obs_dim=share_obs_dim,
            num_envs=self.num_envs,
            episode_length=episode_length,
            actor_lr=float(getattr(self.cfg, "happo_actor_lr", 1.0e-4)),
            critic_lr=float(getattr(self.cfg, "happo_critic_lr", 1.0e-4)),
            clip_param=float(getattr(self.cfg, "happo_clip_param", 0.1)),
            ppo_epoch=int(getattr(self.cfg, "happo_ppo_epoch", 3)),
            max_grad_norm=float(getattr(self.cfg, "happo_max_grad_norm", 0.5)),
            device=str(self.device),
        )
        self.runner = PlatoonHAPPORunner(happo_cfg)
        self.runner.buffer.set_initial_obs(agent_obs, share_obs)

        # Non-invasive pipeline wiring: student goes through scheduler, while
        # attacker/teacher/shield remain NoOp unless enabled later.
        flags = PipelineFlags(
            enable_teacher=bool(getattr(self.cfg, "enable_teacher", False)),
            enable_attack=bool(getattr(self.cfg, "enable_attack", False)),
            enable_shield=bool(getattr(self.cfg, "enable_shield", False)),
        )
        schedule = PipelineScheduleCfg(student_every_steps=episode_length)
        teacher = RewardTeacherModule(
            RewardTeacherCfg(
                obs_dim=obs_dim,
                act_dim=act_dim * self.num_agents,
                shaping_coef=float(getattr(self.cfg, "teacher_shaping_coef", 0.0)),
                lr=float(getattr(self.cfg, "teacher_lr", 1.0e-4)),
                update_interval=int(getattr(self.cfg, "teacher_update_interval", 5)),
                shaping_clip=float(getattr(self.cfg, "teacher_shaping_clip", 0.05)),
                action_penalty_coef=float(getattr(self.cfg, "teacher_action_penalty_coef", 0.0)),
                device=str(self.device),
            )
        )
        attack_level = str(getattr(self.cfg, "attack_level", "medium")).lower()
        attack_presets = {
            "off": {"enabled": False, "max_fdi_pos": 0.0, "max_fdi_acc": 0.0, "max_dos_rate": 0.0},
            "easy": {"enabled": True, "max_fdi_pos": 2.0, "max_fdi_acc": 0.5, "max_dos_rate": 0.05},
            "medium": {"enabled": True, "max_fdi_pos": 5.0, "max_fdi_acc": 1.5, "max_dos_rate": 0.15},
            "hard": {"enabled": True, "max_fdi_pos": 8.0, "max_fdi_acc": 3.0, "max_dos_rate": 0.30},
        }
        preset = attack_presets.get(attack_level, attack_presets["medium"])

        # Override priority:
        # 1) explicit cfg values if present and not None
        # 2) level preset defaults
        cfg_dict = getattr(self.cfg, "__dict__", {}) if self.cfg is not None else {}
        max_fdi_pos = cfg_dict.get("max_fdi_pos", None)
        max_fdi_acc = cfg_dict.get("max_fdi_acc", None)
        max_dos_rate = cfg_dict.get("max_dos_rate", None)

        attack_cfg = AttackModuleCfg(
            max_fdi_pos=float(preset["max_fdi_pos"] if max_fdi_pos is None else max_fdi_pos),
            max_fdi_acc=float(preset["max_fdi_acc"] if max_fdi_acc is None else max_fdi_acc),
            max_dos_rate=float(preset["max_dos_rate"] if max_dos_rate is None else max_dos_rate),
            enabled=bool(flags.enable_attack and preset["enabled"]),
            target_mode=str(getattr(self.cfg, "attack_target_mode", "all")),
            obs_dim=obs_dim,
            seed=int(getattr(self.cfg, "attack_seed", 3407)),
        )
        attacker = RandomFDIDoSAttackModule(attack_cfg) if attack_cfg.enabled else NoOpAttackModule(attack_cfg)
        self._attack_config_text = (
            f"level={attack_level}, enabled={int(attack_cfg.enabled)}, "
            f"max_fdi_pos={attack_cfg.max_fdi_pos:.3f}, "
            f"max_fdi_acc={attack_cfg.max_fdi_acc:.3f}, "
            f"max_dos_rate={attack_cfg.max_dos_rate:.3f}, "
            f"target_mode={attack_cfg.target_mode}, seed={attack_cfg.seed}"
        )
        shield = BadHeadingShieldModule(
            self.env,
            BadHeadingShieldCfg(
                enabled=True,
                action_scale=float(getattr(self.cfg, "happo_bad_heading_action_scale", 0.2)),
            ),
        )
        self.pipeline = PlatoonTrainingPipeline(
            student=HAPPOStudentModule(self.runner),
            teacher=teacher,
            attacker=attacker,
            shield=shield,
            schedule=schedule,
            flags=flags,
        )

        if not self._printed_build and self._should_log("basic"):
            print(
                "[Platoon HAPPO] runner built: "
                f"num_envs={self.num_envs}, num_agents={self.num_agents}, "
                f"obs_dim={obs_dim}, act_dim={act_dim}, share_obs_dim={share_obs_dim}, "
                f"episode_length={episode_length}, device={self.device}"
            )
            self._printed_build = True
        return self.runner

    def act(self, obs: Any, deterministic: bool = False) -> torch.Tensor:
        """Produce flat IsaacLab actions from HAPPO for one rollout step."""
        if self.runner is None:
            raise RuntimeError("HAPPO runner has not been built. Call build_from_sample() first.")
        agent_obs = self.to_agent_obs(obs)
        if self.pipeline is not None:
            student_obs = self.pipeline.preprocess_obs_for_student(agent_obs)
            joint_actions = self.pipeline.student.act(student_obs)
        else:
            share_obs = merge_agent_obs(agent_obs)
            joint_actions, joint_log_probs, values = self.runner.act(agent_obs, share_obs, deterministic=deterministic)
            self._last_joint_log_probs = joint_log_probs
            self._last_values = values
        self._last_agent_obs = agent_obs
        self.happo_action_steps += 1
        if not self._printed_action_override and self._should_log("basic"):
            flat_dim = joint_actions.shape[1] * joint_actions.shape[2]
            print(
                "[Platoon HAPPO] action override active: "
                f"joint_actions={tuple(joint_actions.shape)}, flat_action_dim={flat_dim}"
            )
            self._printed_action_override = True
        return flatten_joint_action(joint_actions)

    def observe_step(self, next_obs: Any, rewards: Any, dones: Any) -> None:
        """Insert one transition into the HAPPO rollout buffer."""
        if self.runner is None:
            return
        if self.pipeline is None:
            required = ("_last_joint_actions", "_last_joint_log_probs", "_last_values")
            if not all(hasattr(self, name) for name in required):
                raise RuntimeError("No pending HAPPO action to insert. Call act() before observe_step().")
        elif not getattr(self.pipeline.student, "has_pending_action", False):
            raise RuntimeError("No pending HAPPO action to insert. Call act() before observe_step().")
        next_agent_obs = self.to_agent_obs(next_obs)
        self._last_next_obs = next_agent_obs
        rewards_tensor = self.to_agent_rewards(rewards)
        dones_tensor = self.to_agent_dones(dones)

        if self.pipeline is not None:
            batch = StepBatch(
                obs=self._last_agent_obs.detach().clone(),
                actions=self.pipeline.student.pending_actions.detach().clone(),
                rewards=rewards_tensor.detach().clone(),
                dones=dones_tensor.detach().clone(),
                next_obs=next_agent_obs.detach().clone(),
            )
            with torch.inference_mode(False), torch.enable_grad():
                logs = self.pipeline.process_transition(batch)
            update_info = logs.get("student_update") if isinstance(logs, dict) else None
            if update_info is not None:
                self.happo_update_count += 1
                if self._should_log("basic"):
                    print(
                        "[Platoon HAPPO] update: "
                        f"count={self.happo_update_count}, "
                        f"policy_loss={update_info.get('policy_loss', 0.0):.4f}, "
                        f"entropy={update_info.get('dist_entropy', 0.0):.4f}, "
                        f"value_loss={update_info.get('value_loss', 0.0):.4f}, "
                        f"ratio={update_info.get('ratio', 0.0):.4f}, "
                        f"actor_grad_norm={update_info.get('actor_grad_norm', 0.0):.4f}, "
                        f"critic_grad_norm={update_info.get('critic_grad_norm', 0.0):.4f}"
                    )
                atk_interval = int(getattr(self.cfg, "attack_log_interval_updates", 20))
                if isinstance(logs, dict) and atk_interval > 0 and self.happo_update_count % atk_interval == 0:
                    atk_info = logs.get("attacker_update")
                    if isinstance(atk_info, dict):
                        print(
                            "[Platoon Attack] stats: "
                            f"update={self.happo_update_count}, "
                            f"enabled={int(atk_info.get('attack_enabled', 0.0))}, "
                            f"fdi_abs_mean={float(atk_info.get('fdi_abs_mean', 0.0)):.4f}, "
                            f"dos_rate={float(atk_info.get('dos_rate', 0.0)):.4f}, "
                            f"config=({getattr(self, '_attack_config_text', 'n/a')})"
                        )
                    tea_info = logs.get("teacher_update")
                    if isinstance(tea_info, dict):
                        print(
                            "[Platoon Teacher] stats: "
                            f"update={self.happo_update_count}, "
                            f"shaping_mean={float(tea_info.get('teacher_shaping_mean', 0.0)):.4f}, "
                            f"teacher_loss={float(tea_info.get('teacher_loss', 0.0)):.4f}, "
                            f"teacher_grad_norm={float(tea_info.get('teacher_grad_norm', 0.0)):.4f}"
                        )
        else:
            next_share_obs = merge_agent_obs(next_agent_obs)
            self.runner.buffer.insert(
                next_agent_obs,
                next_share_obs,
                self._last_joint_actions,
                self._last_joint_log_probs,
                rewards_tensor,
                dones_tensor,
                self._last_values,
            )
        self.rollout_steps += 1
        if self._should_log("debug") and (
            self.rollout_steps == 1 or self.rollout_steps % self._rollout_log_interval == 0
        ):
            reward_mean = float(rewards_tensor.mean().item())
            done_count = int(dones_tensor.sum().item())
            print(
                "[Platoon HAPPO] rollout: "
                f"steps={self.rollout_steps}, buffer_step={self.runner.buffer.step}, "
                f"reward_mean={reward_mean:.4f}, done_count={done_count}"
            )

    def ready_to_train(self) -> bool:
        return self.runner is not None and self.runner.buffer.step == 0

    def train_if_ready(self, next_obs: Any) -> tuple[list[dict[str, float]], dict[str, float]] | None:
        """Run one HAPPO update when a rollout segment is full."""
        if self.runner is None or not self.ready_to_train():
            return None
        with torch.inference_mode(False), torch.enable_grad():
            next_share_obs = merge_agent_obs(self.to_agent_obs(next_obs)).detach().clone()
            actor_infos, critic_info = self.runner.train(next_share_obs)
        self.happo_update_count += 1
        policy_loss = 0.0
        actor_entropy = 0.0
        if actor_infos:
            policy_loss = sum(info.get("policy_loss", 0.0) for info in actor_infos) / len(actor_infos)
            actor_entropy = sum(info.get("dist_entropy", 0.0) for info in actor_infos) / len(actor_infos)
        ratio_mean = 0.0
        actor_grad_norm_mean = 0.0
        critic_grad_norm = float(critic_info.get("critic_grad_norm", 0.0))
        if actor_infos:
            ratio_mean = sum(info.get("ratio", 0.0) for info in actor_infos) / len(actor_infos)
            actor_grad_norm_mean = sum(info.get("actor_grad_norm", 0.0) for info in actor_infos) / len(actor_infos)

        if self._should_log("basic"):
            print(
                "[Platoon HAPPO] update: "
                f"count={self.happo_update_count}, "
                f"policy_loss={policy_loss:.4f}, entropy={actor_entropy:.4f}, "
                f"value_loss={critic_info.get('value_loss', 0.0):.4f}, "
                f"ratio={ratio_mean:.4f}, actor_grad_norm={actor_grad_norm_mean:.4f}, "
                f"critic_grad_norm={critic_grad_norm:.4f}"
            )
        return actor_infos, critic_info

    def to_agent_obs(self, obs: Any) -> torch.Tensor:
        obs_tensor = self._extract_tensor(obs).to(self.device, dtype=torch.float32)
        return split_agent_obs(obs_tensor, self.num_agents)

    def to_agent_rewards(self, rewards: Any) -> torch.Tensor:
        rewards_tensor = self._extract_tensor(rewards).to(self.device, dtype=torch.float32)
        if rewards_tensor.dim() == 0:
            rewards_tensor = rewards_tensor.view(1, 1, 1).repeat(self.num_envs, self.num_agents, 1)
        elif rewards_tensor.dim() == 1:
            rewards_tensor = rewards_tensor.view(self.num_envs, 1, 1).repeat(1, self.num_agents, 1)
        elif rewards_tensor.dim() == 2:
            if rewards_tensor.shape[-1] == self.num_agents:
                rewards_tensor = rewards_tensor.unsqueeze(-1)
            else:
                rewards_tensor = rewards_tensor.view(self.num_envs, 1, -1).repeat(1, self.num_agents, 1)
        return rewards_tensor

    def to_agent_dones(self, dones: Any) -> torch.Tensor:
        dones_tensor = self._extract_tensor(dones).to(self.device, dtype=torch.bool)
        if dones_tensor.dim() == 0:
            dones_tensor = dones_tensor.view(1, 1).repeat(self.num_envs, self.num_agents)
        elif dones_tensor.dim() == 1:
            dones_tensor = dones_tensor.view(self.num_envs, 1).repeat(1, self.num_agents)
        return dones_tensor

    def _infer_agent_action_dim(self) -> int:
        action_dim = None
        if hasattr(self.env, "action_manager"):
            action_dim = getattr(self.env.action_manager, "action_dim", None)
        if action_dim is None and hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "action_manager"):
            action_dim = getattr(self.env.unwrapped.action_manager, "action_dim", None)
        if action_dim is None:
            action_space = getattr(self.env, "action_space", None)
            if action_space is not None and getattr(action_space, "shape", None):
                action_dim = int(action_space.shape[-1])
        if action_dim is None:
            raise ValueError("Cannot infer platoon action dimension from env.")
        if int(action_dim) % self.num_agents != 0:
            raise ValueError(f"Action dim {action_dim} is not divisible by num_agents={self.num_agents}")
        return int(action_dim) // self.num_agents

    def _extract_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        if isinstance(value, dict):
            for key in ("policy", "obs", "observations", "state", "states", "joint_obs"):
                if key in value:
                    return self._extract_tensor(value[key])
            for item in value.values():
                try:
                    return self._extract_tensor(item)
                except Exception:
                    continue
        if isinstance(value, (tuple, list)):
            for item in value:
                try:
                    return self._extract_tensor(item)
                except Exception:
                    continue
        return torch.as_tensor(value, device=self.device)


    def _should_log(self, level: str) -> bool:
        if self._log_level == "off":
            return False
        if self._log_level == "debug":
            return True
        # basic
        return level == "basic"


def build_platoon_algorithm_router(env: Any, cfg: Any, runner_cfg: Any | None = None) -> PlatoonAlgorithmRouter:
    return PlatoonAlgorithmRouter(env, cfg, runner_cfg)
