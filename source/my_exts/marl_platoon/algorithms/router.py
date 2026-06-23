"""Task-local algorithm router for marl_platoon.

This module keeps the official IsaacLab/RSL-RL entrypoint intact. PPO remains a
no-op here because it is still handled by the stable RSL-RL runner. HAPPO is
built as an internal task component and can be stepped explicitly by task-side
code once the rollout loop is connected.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

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
        self.num_agents = int(getattr(cfg, "num_agents", 5))
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
        self.inference_only = False
        self.deterministic_actions = False
        self.enable_eval_attack = False
        self._metrics_csv_path: Path | None = None
        self._metrics_csv_file = None
        self._metrics_csv_writer: csv.DictWriter | None = None
        self._last_raw_joint_actions: torch.Tensor | None = None
        self._last_executed_joint_actions: torch.Tensor | None = None
        self._last_local_reward_metrics: dict[str, float] = {}
        self._metrics_fieldnames = [
            "update",
            "env_step",
            "sim_time_s",
            "num_envs",
            "num_agents",
            "command_speed_mean",
            "leader_speed_mean",
            "platoon_speed_mean",
            "leader_speed_error_mean",
            "speed_error_abs_mean",
            "gap_error_mean",
            "gap_error_abs_mean",
            "gap_error_max_abs",
            "lateral_error_abs_mean",
            "lateral_error_abs_max",
            "lateral_worst_pair",
            "lateral_pair_1_abs_mean",
            "lateral_pair_2_abs_mean",
            "lateral_pair_3_abs_mean",
            "lateral_pair_4_abs_mean",
            "lateral_pair_1_signed_mean",
            "lateral_pair_2_signed_mean",
            "lateral_pair_3_signed_mean",
            "lateral_pair_4_signed_mean",
            "centerline_error_abs_mean",
            "centerline_error_abs_max",
            "centerline_robot_1_abs_mean",
            "centerline_robot_2_abs_mean",
            "centerline_robot_3_abs_mean",
            "centerline_robot_4_abs_mean",
            "centerline_robot_5_abs_mean",
            "centerline_robot_1_signed_mean",
            "centerline_robot_2_signed_mean",
            "centerline_robot_3_signed_mean",
            "centerline_robot_4_signed_mean",
            "centerline_robot_5_signed_mean",
            "heading_error_abs_mean",
            "heading_error_abs_max",
            "pair_heading_error_abs_mean",
            "pair_heading_error_abs_max",
            "min_pair_gap_mean",
            "collision_rate",
            "reward_env_mean",
            "done_rate",
            "termination_time_out",
            "termination_reset_on_bad_ori",
            "action_abs_mean",
            "leader_action_abs_mean",
            "raw_action_turn_abs_mean",
            "executed_action_turn_abs_mean",
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
            "local_reward_shaping_mean",
            "local_reward_centerline_penalty_mean",
            "local_reward_pair_lateral_penalty_mean",
            "local_reward_heading_penalty_mean",
            "local_reward_turn_penalty_mean",
            "reward_leader_motion",
            "reward_leader_progress",
            "reward_formation",
            "reward_collision_risk",
            "reward_forward_drive",
            "reward_stall_penalty",
            "reward_lateral_correct",
            "reward_lateral_velocity",
            "reward_centerline_lateral",
            "reward_heading_align",
            "reward_action_rate",
            "reward_true_success",
            "reward_true_fail",
            "policy_loss",
            "entropy",
            "value_loss",
            "ratio",
            "actor_grad_norm",
            "critic_grad_norm",
            "attack_enabled",
            "attack_level",
            "attack_mode",
            "attack_target_mode",
            "attack_max_fdi_pos",
            "attack_max_fdi_acc",
            "attack_max_dos_rate",
            "attack_late_obj_scale",
            "attack_late_update_interval",
            "fdi_abs_mean",
            "dos_rate",
            "obs_fdi_abs_mean",
            "obs_dos_rate",
            "obs_dos_budget",
            "obs_drop_count",
            "obs_elem_n",
            "act_fdi_abs_mean",
            "act_dos_rate",
            "act_dos_budget",
            "act_drop_count",
            "act_elem_n",
            "attack_objective",
            "attack_g_loss",
            "attack_d_loss",
            "attack_realism_loss",
            "attack_seq_realism_loss",
            "attack_phy_ctx_norm",
            "attack_ref_template_id",
            "physical_cost",
            "physical_spacing_cost",
            "physical_velocity_cost",
            "physical_acceleration_cost",
            "physical_jerk_cost",
            "physical_overspeed_cost",
            "physical_lateral_cost",
            "physical_heading_cost",
            "physical_backward_cost",
            "physical_forward_deficit_cost",
            "physical_collision_cost",
            "teacher_update_active",
            "teacher_shaping_mean",
            "teacher_loss",
            "teacher_base_loss",
            "teacher_delta_j",
            "teacher_norm_delta_j",
            "teacher_outer_coef_scale",
            "teacher_outer_loss",
            "teacher_advantage_corr",
            "teacher_physical_adv_std",
            "teacher_shaping_adv_std",
            "teacher_window_samples",
            "teacher_grad_norm",
            "shield_warn_rate",
            "shield_critical_rate",
            "shield_trigger_rate",
            "shield_scale_mean",
            "shield_lateral_rate",
            "shield_lateral_critical_rate",
            "shield_lateral_turn_mean",
            "shield_centerline_rate",
            "shield_centerline_turn_mean",
            "shield_first_follower_centerline_rate",
            "shield_first_follower_centerline_turn_mean",
        ]

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
        if self.runner is not None and self.info is not None:
            if (
                self.info.obs_dim == obs_dim
                and self.info.act_dim == act_dim
                and self.info.share_obs_dim == share_obs_dim
            ):
                self.runner.buffer.set_initial_obs(agent_obs, share_obs)
                if self.inference_only:
                    self.runner.prep_rollout()
                return self.runner

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
            num_mini_batches=int(getattr(self.cfg, "happo_num_mini_batches", 16)),
            factor_eval_chunk_size=int(getattr(self.cfg, "happo_factor_chunk_size", 4096)),
            entropy_coef=float(getattr(self.cfg, "happo_entropy_coef", 0.01)),
            init_noise_std=float(getattr(self.cfg, "happo_init_noise_std", 1.0)),
            log_std_min=float(getattr(self.cfg, "happo_log_std_min", -20.0)),
            log_std_max=float(getattr(self.cfg, "happo_log_std_max", 2.0)),
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
                act_dim=act_dim,
                shaping_coef=float(getattr(self.cfg, "teacher_shaping_coef", 0.0)),
                lr=float(getattr(self.cfg, "teacher_lr", 1.0e-4)),
                update_interval=int(getattr(self.cfg, "teacher_update_interval", 5)),
                shaping_clip=float(getattr(self.cfg, "teacher_shaping_clip", 0.05)),
                action_penalty_coef=float(getattr(self.cfg, "teacher_action_penalty_coef", 0.0)),
                device=str(self.device),
            ),
            env=self.env,
        )
        attack_level = str(getattr(self.cfg, "attack_level", "off")).lower()
        attack_presets = {
            "off": {"enabled": False, "max_fdi_pos": 0.0, "max_fdi_acc": 0.0, "max_dos_rate": 0.0},
            "light": {"enabled": True, "max_fdi_pos": 1.0, "max_fdi_acc": 0.10, "max_dos_rate": 0.02},
            "easy": {"enabled": True, "max_fdi_pos": 2.0, "max_fdi_acc": 0.5, "max_dos_rate": 0.05},
            "medium": {"enabled": True, "max_fdi_pos": 5.0, "max_fdi_acc": 1.5, "max_dos_rate": 0.15},
            "hard": {"enabled": True, "max_fdi_pos": 8.0, "max_fdi_acc": 3.0, "max_dos_rate": 0.30},
        }
        if attack_level not in attack_presets:
            valid = ", ".join(sorted(attack_presets))
            raise ValueError(f"Unsupported attack_level={attack_level!r}. Valid attack levels: {valid}")
        preset = attack_presets[attack_level]

        # Override priority:
        # 1) explicit cfg values if present and not None
        # 2) level preset defaults
        cfg_dict = getattr(self.cfg, "__dict__", {}) if self.cfg is not None else {}
        max_fdi_pos = self._optional_nonnegative_cfg(cfg_dict.get("max_fdi_pos", None))
        max_fdi_acc = self._optional_nonnegative_cfg(cfg_dict.get("max_fdi_acc", None))
        max_dos_rate = self._optional_nonnegative_cfg(cfg_dict.get("max_dos_rate", None))

        attack_cfg = AttackModuleCfg(
            max_fdi_pos=float(preset["max_fdi_pos"] if max_fdi_pos is None else max_fdi_pos),
            max_fdi_acc=float(preset["max_fdi_acc"] if max_fdi_acc is None else max_fdi_acc),
            max_dos_rate=float(preset["max_dos_rate"] if max_dos_rate is None else max_dos_rate),
            enabled=bool(flags.enable_attack and preset["enabled"]),
            target_mode=str(getattr(self.cfg, "attack_target_mode", "all")),
            obs_dim=obs_dim,
            seed=int(getattr(self.cfg, "attack_seed", 3407)),
            mode=str(getattr(self.cfg, "attack_mode", "profile")),
            noise_dim=int(getattr(self.cfg, "attack_noise_dim", 16)),
            hidden_dim=int(getattr(self.cfg, "attack_hidden_dim", 128)),
            realism_coef=float(getattr(self.cfg, "attack_realism_coef", 0.10)),
            generator_lr=float(getattr(self.cfg, "attack_generator_lr", 1.0e-4)),
            discriminator_lr=float(getattr(self.cfg, "attack_discriminator_lr", 1.0e-4)),
            update_interval=int(getattr(self.cfg, "attack_update_interval", 4)),
            attack_obj_coef=float(getattr(self.cfg, "attack_obj_coef", 1.0)),
            reward_proxy_coef=float(getattr(self.cfg, "attack_reward_proxy_coef", 1.0)),
            dos_proxy_coef=float(getattr(self.cfg, "attack_dos_proxy_coef", 0.3)),
            attack_objective_clip=float(getattr(self.cfg, "attack_objective_clip", 20.0)),
            curriculum_warmup_updates=int(getattr(self.cfg, "attack_curriculum_warmup_updates", 2000)),
            curriculum_start_mode=str(getattr(self.cfg, "attack_curriculum_start_mode", "profile")),
            decay_start_updates=int(getattr(self.cfg, "attack_decay_start_updates", 5000)),
            min_attack_obj_scale=float(getattr(self.cfg, "attack_min_obj_scale", 0.25)),
            min_update_freq_scale=float(getattr(self.cfg, "attack_min_update_freq_scale", 0.25)),
            decay_half_life_updates=int(getattr(self.cfg, "attack_decay_half_life_updates", 2500)),
        )
        attacker = RandomFDIDoSAttackModule(attack_cfg) if attack_cfg.enabled else NoOpAttackModule(attack_cfg)
        self._attack_config_text = (
            f"level={attack_level}, enabled={int(attack_cfg.enabled)}, "
            f"max_fdi_pos={attack_cfg.max_fdi_pos:.3f}, "
            f"max_fdi_acc={attack_cfg.max_fdi_acc:.3f}, "
            f"max_dos_rate={attack_cfg.max_dos_rate:.3f}, "
            f"target_mode={attack_cfg.target_mode}, mode={attack_cfg.mode}, seed={attack_cfg.seed}, "
            f"curriculum=({attack_cfg.curriculum_start_mode}->{attack_cfg.mode}@{attack_cfg.curriculum_warmup_updates}), "
            f"obj_coef={attack_cfg.attack_obj_coef:.3f}, "
            f"reward_coef={attack_cfg.reward_proxy_coef:.3f}, "
            f"dos_coef={attack_cfg.dos_proxy_coef:.3f}"
        )
        safety_cfg = getattr(getattr(self.env, "cfg", None), "safety_shield", None)
        shield = BadHeadingShieldModule(
            self.env,
            BadHeadingShieldCfg(
                enabled=True,
                d_crit=float(getattr(safety_cfg, "d_crit", 0.50)),
                d_drop=float(getattr(safety_cfg, "d_drop", 1.20)),
                brake_action=float(getattr(safety_cfg, "brake_action", 0.0)),
                catchup_action=float(getattr(safety_cfg, "catchup_action", -0.35)),
                lateral_tol=float(getattr(safety_cfg, "lateral_tol", 0.08)),
                lateral_crit=float(getattr(safety_cfg, "lateral_crit", 0.35)),
                lateral_turn_gain=float(getattr(safety_cfg, "lateral_turn_gain", 0.70)),
                lateral_turn_clip=float(getattr(safety_cfg, "lateral_turn_clip", 0.18)),
                lateral_velocity_gain=float(getattr(safety_cfg, "lateral_velocity_gain", 0.25)),
                centerline_turn_gain=float(getattr(safety_cfg, "centerline_turn_gain", 0.0)),
                centerline_turn_clip=float(getattr(safety_cfg, "centerline_turn_clip", 0.0)),
                first_follower_lateral_gain_scale=float(getattr(safety_cfg, "first_follower_lateral_gain_scale", 1.0)),
                first_follower_lateral_clip_scale=float(getattr(safety_cfg, "first_follower_lateral_clip_scale", 1.0)),
                first_follower_lateral_clip_max=float(getattr(safety_cfg, "first_follower_lateral_clip_max", 0.18)),
                first_follower_centerline_gain=float(getattr(safety_cfg, "first_follower_centerline_gain", 0.0)),
                first_follower_centerline_clip=float(getattr(safety_cfg, "first_follower_centerline_clip", 0.0)),
                pair2_lateral_gain_scale=float(getattr(safety_cfg, "pair2_lateral_gain_scale", 1.0)),
                pair2_lateral_clip_scale=float(getattr(safety_cfg, "pair2_lateral_clip_scale", 1.0)),
                pair2_lateral_clip_max=float(getattr(safety_cfg, "pair2_lateral_clip_max", 0.18)),
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

    @staticmethod
    def _optional_nonnegative_cfg(value: Any) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if out >= 0.0 else None

    def set_inference_mode(
        self,
        *,
        deterministic: bool = True,
        disable_updates: bool = True,
        disable_attack: bool = True,
    ) -> None:
        """Configure task-local modules for play/evaluation."""
        self.deterministic_actions = bool(deterministic)
        self.inference_only = bool(disable_updates)
        self.enable_eval_attack = not bool(disable_attack)
        if self.pipeline is not None and disable_attack:
            self.pipeline.flags.enable_attack = False
        if self.runner is not None:
            self.runner.prep_rollout()

    def state_dict(self) -> dict:
        """Return checkpoint state for task-local HAPPO modules."""
        if self.runner is None:
            return {}
        return {
            "version": 1,
            "algorithm": self.algorithm,
            "num_agents": self.num_agents,
            "info": self.info.__dict__ if self.info is not None else None,
            "happo_update_count": self.happo_update_count,
            "happo_action_steps": self.happo_action_steps,
            "runner": self.runner.state_dict(),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """Load checkpoint state for task-local HAPPO modules."""
        if self.runner is None:
            raise RuntimeError("HAPPO runner has not been built. Call build_from_sample() first.")
        if not state or "runner" not in state:
            raise ValueError("Checkpoint does not contain task-local HAPPO runner state.")
        if int(state.get("num_agents", self.num_agents)) != self.num_agents:
            raise ValueError(
                f"HAPPO num_agents mismatch: checkpoint={state.get('num_agents')} current={self.num_agents}"
            )
        self.runner.load_state_dict(state["runner"], strict=strict)
        self.happo_update_count = int(state.get("happo_update_count", self.happo_update_count))
        self.happo_action_steps = int(state.get("happo_action_steps", self.happo_action_steps))
        if self.runner is not None:
            self.runner.prep_rollout()

    def act(self, obs: Any, deterministic: bool = False) -> torch.Tensor:
        """Produce flat IsaacLab actions from HAPPO for one rollout step."""
        if self.runner is None:
            raise RuntimeError("HAPPO runner has not been built. Call build_from_sample() first.")
        agent_obs = self.to_agent_obs(obs)
        deterministic = bool(deterministic or self.deterministic_actions)
        if self.pipeline is not None:
            student_obs = self.pipeline.preprocess_obs_for_student(agent_obs)
            joint_actions = self.pipeline.student.act(student_obs, deterministic=deterministic)
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
        if self.runner is None or self.inference_only:
            return
        if self.pipeline is None:
            required = ("_last_joint_actions", "_last_joint_log_probs", "_last_values")
            if not all(hasattr(self, name) for name in required):
                raise RuntimeError("No pending HAPPO action to insert. Call act() before observe_step().")
        elif not getattr(self.pipeline.student, "has_pending_action", False):
            raise RuntimeError("No pending HAPPO action to insert. Call act() before observe_step().")
        next_agent_obs = self.to_agent_obs(next_obs)
        self._last_next_obs = next_agent_obs
        env_rewards_tensor = self.to_agent_rewards(rewards)
        rewards_tensor = self._apply_local_agent_reward_shaping(env_rewards_tensor)
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
                            f"mode={atk_info.get('attack_mode', 'n/a')}, "
                            f"fdi_abs_mean={float(atk_info.get('fdi_abs_mean', 0.0)):.4f}, "
                            f"dos_rate={float(atk_info.get('dos_rate', 0.0)):.4f}, "
                            f"obs_fdi={float(atk_info.get('obs_fdi_abs_mean', 0.0)):.4f}, "
                            f"obs_dos={float(atk_info.get('obs_dos_rate', 0.0)):.4f}, "
                            f"act_fdi={float(atk_info.get('act_fdi_abs_mean', 0.0)):.4f}, "
                            f"act_dos={float(atk_info.get('act_dos_rate', 0.0)):.4f}, "
                            f"act_budget={float(atk_info.get('act_dos_budget', 0.0)):.4f}, "
                            f"act_drop_n={float(atk_info.get('act_drop_count', 0.0)):.0f}, "
                            f"act_elem_n={float(atk_info.get('act_elem_n', 0.0)):.0f}, "
                            f"stats_bad={int(atk_info.get('attack_stats_inconsistent', 0.0))}, "
                            f"g_loss={float(atk_info.get('attack_g_loss', 0.0)):.4f}, "
                            f"d_loss={float(atk_info.get('attack_d_loss', 0.0)):.4f}, "
                            f"realism_loss={float(atk_info.get('attack_realism_loss', 0.0)):.4f}, "
                            f"seq_realism={float(atk_info.get('attack_seq_realism_loss', 0.0)):.4f}, "
                            f"obj_att={float(atk_info.get('attack_objective', 0.0)):.4f}, "
                            f"obj_s={float(atk_info.get('obj_spacing', 0.0)):.4f}, "
                            f"obj_v={float(atk_info.get('obj_vel', 0.0)):.4f}, "
                            f"obj_a={float(atk_info.get('obj_acc', 0.0)):.4f}, "
                            f"obj_j={float(atk_info.get('obj_jerk', 0.0)):.4f}, "
                            f"obj_phy={float(atk_info.get('obj_physical_cost', 0.0)):.4f}, "
                            f"obj_col={float(atk_info.get('obj_collision', 0.0)):.4f}, "
                            f"obj_bd={float(atk_info.get('obj_bad_done', 0.0)):.4f}, "
                            f"obj_en={float(atk_info.get('obj_energy', 0.0)):.4f}, "
                            f"phy_ctx={float(atk_info.get('attack_phy_ctx_norm', 0.0)):.4f}, "
                            f"ref_tpl={atk_info.get('attack_ref_template_id', 'none')}, "
                            f"uniq_ref20={float(atk_info.get('attack_unique_ref_tpl_count', 0.0)):.0f}, "
                            f"config=({getattr(self, '_attack_config_text', 'n/a')})"
                        )
                    tea_info = logs.get("teacher_update")
                    if isinstance(tea_info, dict):
                        print(
                            "[Platoon Teacher] stats: "
                            f"update={self.happo_update_count}, "
                            f"shaping_mean={float(tea_info.get('teacher_shaping_mean', 0.0)):.4f}, "
                            f"teacher_loss={float(tea_info.get('teacher_loss', 0.0)):.4f}, "
                            f"teacher_base_loss={float(tea_info.get('teacher_base_loss', 0.0)):.4f}, "
                            f"teacher_consistency_loss={float(tea_info.get('teacher_consistency_loss', 0.0)):.4f}, "
                            f"teacher_delta_j={float(tea_info.get('teacher_delta_j', 0.0)):.4f}, "
                            f"teacher_norm_delta_j={float(tea_info.get('teacher_norm_delta_j', 0.0)):.4f}, "
                            f"teacher_outer_coef_scale={float(tea_info.get('teacher_outer_coef_scale', 0.0)):.4f}, "
                            f"teacher_outer_loss={float(tea_info.get('teacher_outer_loss', 0.0)):.4f}, "
                            f"teacher_adv_corr={float(tea_info.get('teacher_advantage_corr', 0.0)):.4f}, "
                            f"phy_adv_std={float(tea_info.get('teacher_physical_adv_std', 0.0)):.4f}, "
                            f"shape_adv_std={float(tea_info.get('teacher_shaping_adv_std', 0.0)):.4f}, "
                            f"window_samples={float(tea_info.get('teacher_window_samples', 0.0)):.0f}, "
                            f"teacher_grad_norm={float(tea_info.get('teacher_grad_norm', 0.0)):.4f}"
                        )
                    if self.pipeline is not None and hasattr(self.pipeline.shield, "get_stats"):
                        shield_stats = self.pipeline.shield.get_stats()
                        if isinstance(shield_stats, dict):
                            print(
                                "[Platoon Shield] stats: "
                                f"update={self.happo_update_count}, "
                                f"warn_rate={float(shield_stats.get('shield_warn_rate', 0.0)):.4f}, "
                                f"critical_rate={float(shield_stats.get('shield_critical_rate', 0.0)):.4f}, "
                                f"trigger_rate={float(shield_stats.get('shield_trigger_rate', 0.0)):.4f}, "
                                f"scale_mean={float(shield_stats.get('shield_scale_mean', 1.0)):.4f}"
                            )
                self._write_metrics_csv_row(update_info, logs, env_rewards_tensor, dones_tensor)
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

    def _ensure_metrics_csv(self) -> csv.DictWriter | None:
        if self._metrics_csv_writer is not None:
            return self._metrics_csv_writer
        log_dir = getattr(getattr(self.env, "cfg", None), "log_dir", None)
        if log_dir is None:
            log_dir = Path.cwd() / "logs" / "platoon_metrics"
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_csv_path = log_dir / "platoon_metrics.csv"
        self._metrics_csv_file = self._metrics_csv_path.open("w", newline="")
        self._metrics_csv_writer = csv.DictWriter(self._metrics_csv_file, fieldnames=self._metrics_fieldnames)
        self._metrics_csv_writer.writeheader()
        print(f"[Platoon Metrics] writing CSV to: {self._metrics_csv_path}")
        return self._metrics_csv_writer

    def _write_metrics_csv_row(
        self,
        update_info: dict[str, float] | None,
        logs: dict[str, Any] | None,
        rewards_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
    ) -> None:
        writer = self._ensure_metrics_csv()
        if writer is None:
            return
        logs = logs or {}
        row: dict[str, Any] = {key: "" for key in self._metrics_fieldnames}
        row.update(self._collect_platoon_metrics(rewards_tensor, dones_tensor))
        row.update(self._collect_local_reward_metrics())
        row.update(self._collect_reward_term_metrics())
        row.update(self._collect_termination_metrics())
        row.update(self._collect_update_metrics(update_info or {}))
        row.update(self._collect_attack_metrics(logs.get("attacker_update")))
        row.update(self._collect_teacher_metrics(logs.get("teacher_update")))
        row.update(self._collect_shield_metrics())
        writer.writerow({key: row.get(key, "") for key in self._metrics_fieldnames})
        if self._metrics_csv_file is not None:
            self._metrics_csv_file.flush()

    def _collect_platoon_metrics(self, rewards_tensor: torch.Tensor, dones_tensor: torch.Tensor) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "update": self.happo_update_count,
            "env_step": int(getattr(self.env, "common_step_counter", self.rollout_steps)),
            "sim_time_s": float(getattr(self.env, "common_step_counter", self.rollout_steps))
            * float(getattr(self.env, "step_dt", 0.0)),
            "num_envs": self.num_envs,
            "num_agents": self.num_agents,
            "reward_env_mean": float(rewards_tensor.mean().item()),
            "done_rate": float(dones_tensor.float().mean().item()),
        }
        robots = ["robot", "robot_2", "robot_3", "robot_4", "robot_5"][: self.num_agents]
        try:
            command = self.env.command_manager.get_command("base_velocity").to(self.device)
            command_speed = command[:, 0]
            metrics["command_speed_mean"] = float(command_speed.mean().item())
        except Exception:
            command_speed = torch.zeros(self.num_envs, device=self.device)
            metrics["command_speed_mean"] = 0.0

        try:
            speeds = torch.stack([self.env.scene[name].data.root_lin_vel_b[:, 0].to(self.device) for name in robots], dim=1)
            metrics["leader_speed_mean"] = float(speeds[:, 0].mean().item())
            metrics["platoon_speed_mean"] = float(speeds.mean().item())
            metrics["leader_speed_error_mean"] = float((command_speed - speeds[:, 0]).mean().item())
            metrics["speed_error_abs_mean"] = float((command_speed.view(-1, 1) - speeds).abs().mean().item())
        except Exception:
            speeds = None

        try:
            centerline_error = torch.stack(
                [self._env_local_y(name) for name in robots],
                dim=1,
            )
            metrics["centerline_error_abs_mean"] = float(centerline_error.abs().mean().item())
            metrics["centerline_error_abs_max"] = float(centerline_error.abs().max().item())
            centerline_abs = centerline_error.abs().mean(dim=0)
            centerline_signed = centerline_error.mean(dim=0)
            for robot_idx in range(min(len(robots), centerline_abs.numel())):
                metrics[f"centerline_robot_{robot_idx + 1}_abs_mean"] = float(centerline_abs[robot_idx].item())
                metrics[f"centerline_robot_{robot_idx + 1}_signed_mean"] = float(centerline_signed[robot_idx].item())
        except Exception:
            pass

        gap_errors = []
        lateral_errors = []
        pair_gaps = []
        pair_heading_errors = []
        try:
            target_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            heading_vecs = [
                torch.nn.functional.normalize(
                    quat_apply(self.env.scene[name].data.root_quat_w.to(self.device), target_dir)[:, :2],
                    dim=-1,
                )
                for name in robots
            ]
            heading_x = torch.stack([heading[:, 0] for heading in heading_vecs], dim=1).clamp(-1.0, 1.0)
            heading_error = torch.acos(heading_x)
            metrics["heading_error_abs_mean"] = float(heading_error.abs().mean().item())
            metrics["heading_error_abs_max"] = float(heading_error.abs().max().item())

            for idx in range(1, len(robots)):
                pred = self.env.scene[robots[idx - 1]]
                foll = self.env.scene[robots[idx]]
                diff = foll.data.root_pos_w.to(self.device) - pred.data.root_pos_w.to(self.device)
                rel = quat_apply_inverse(pred.data.root_quat_w.to(self.device), diff)
                follower_speed = torch.clamp(foll.data.root_lin_vel_b[:, 0].to(self.device), min=0.0)
                desired_gap = 1.5 + 0.6 * follower_speed
                gap_errors.append(rel[:, 0] + desired_gap)
                lateral_errors.append(rel[:, 1])
                pair_gaps.append(torch.norm(diff[:, :2], dim=1))
                pair_dot = torch.sum(heading_vecs[idx - 1] * heading_vecs[idx], dim=-1).clamp(-1.0, 1.0)
                pair_heading_errors.append(torch.acos(pair_dot))
            if gap_errors:
                gap_error = torch.stack(gap_errors, dim=1)
                lateral_error = torch.stack(lateral_errors, dim=1)
                pair_gap = torch.stack(pair_gaps, dim=1)
                pair_heading_error = torch.stack(pair_heading_errors, dim=1)
                metrics["gap_error_mean"] = float(gap_error.mean().item())
                metrics["gap_error_abs_mean"] = float(gap_error.abs().mean().item())
                metrics["gap_error_max_abs"] = float(gap_error.abs().max().item())
                metrics["lateral_error_abs_mean"] = float(lateral_error.abs().mean().item())
                metrics["lateral_error_abs_max"] = float(lateral_error.abs().max().item())
                lateral_pair_abs = lateral_error.abs().mean(dim=0)
                lateral_pair_signed = lateral_error.mean(dim=0)
                worst_pair = int(torch.argmax(lateral_pair_abs).item()) + 1
                metrics["lateral_worst_pair"] = worst_pair
                for pair_idx in range(lateral_pair_abs.numel()):
                    metrics[f"lateral_pair_{pair_idx + 1}_abs_mean"] = float(lateral_pair_abs[pair_idx].item())
                    metrics[f"lateral_pair_{pair_idx + 1}_signed_mean"] = float(lateral_pair_signed[pair_idx].item())
                metrics["pair_heading_error_abs_mean"] = float(pair_heading_error.abs().mean().item())
                metrics["pair_heading_error_abs_max"] = float(pair_heading_error.abs().max().item())
                metrics["min_pair_gap_mean"] = float(pair_gap.min(dim=1).values.mean().item())
                metrics["collision_rate"] = float((pair_gap < 0.30).float().mean().item())
        except Exception:
            pass

        try:
            actions = self.pipeline.student.pending_actions if self.pipeline is not None else None
            if actions is not None:
                metrics["action_abs_mean"] = float(actions.abs().mean().item())
                metrics["leader_action_abs_mean"] = float(actions[:, 0, :].abs().mean().item())
            metrics.update(self._collect_action_turn_metrics(self._last_raw_joint_actions, "raw"))
            metrics.update(self._collect_action_turn_metrics(self._last_executed_joint_actions, "executed"))
        except Exception:
            pass
        return metrics

    def record_action_diagnostics(
        self,
        raw_joint_actions: torch.Tensor | None,
        executed_joint_actions: torch.Tensor | None,
    ) -> None:
        """Store latest semantic action tensors for per-agent turn-bias metrics."""
        self._last_raw_joint_actions = (
            raw_joint_actions.detach().clone().to(self.device, dtype=torch.float32)
            if raw_joint_actions is not None
            else None
        )
        self._last_executed_joint_actions = (
            executed_joint_actions.detach().clone().to(self.device, dtype=torch.float32)
            if executed_joint_actions is not None
            else None
        )

    def _collect_action_turn_metrics(self, actions: torch.Tensor | None, prefix: str) -> dict[str, Any]:
        if actions is None or actions.dim() != 3 or actions.shape[-1] < 4:
            return {}
        turn = self._semantic_action_physical_turn(actions)
        metrics: dict[str, Any] = {
            f"{prefix}_action_turn_abs_mean": float(turn.abs().mean().item()),
        }
        for agent_idx in range(min(self.num_agents, turn.shape[1])):
            metrics[f"{prefix}_action_turn_agent_{agent_idx + 1}"] = float(turn[:, agent_idx].mean().item())
        return metrics

    def _semantic_action_physical_turn(self, actions: torch.Tensor) -> torch.Tensor:
        """Return semantic differential turn before the URDF wheel-axis adapter.

        HAPPO/shield actions use a semantic convention where equal left/right
        values mean straight driving.  The URDF adapter is applied only at the
        IsaacLab action boundary, so applying the adapter sign here would make
        straight forward commands look like a large turn.
        """
        left = actions[:, :, 0:2].mean(dim=-1)
        right = actions[:, :, 2:4].mean(dim=-1)
        return right - left

    def _collect_local_reward_metrics(self) -> dict[str, Any]:
        return dict(self._last_local_reward_metrics)

    def _env_local_y(self, asset_name: str) -> torch.Tensor:
        pos_y = self.env.scene[asset_name].data.root_pos_w[:, 1].to(self.device)
        env_origins = getattr(self.env.scene, "env_origins", None)
        if env_origins is not None:
            pos_y = pos_y - env_origins[:, 1].to(device=pos_y.device, dtype=pos_y.dtype)
        return pos_y

    def _collect_reward_term_metrics(self) -> dict[str, Any]:
        reward_manager = getattr(self.env, "reward_manager", None)
        term_names = getattr(reward_manager, "_term_names", [])
        step_reward = getattr(reward_manager, "_step_reward", None)
        if step_reward is None or not term_names:
            return {}
        wanted = {
            "leader_motion",
            "leader_progress",
            "formation",
            "collision_risk",
            "forward_drive",
            "stall_penalty",
            "lateral_correct",
            "lateral_velocity",
            "centerline_lateral",
            "heading_align",
            "action_rate",
            "true_success",
            "true_fail",
        }
        metrics: dict[str, Any] = {}
        for idx, name in enumerate(term_names):
            if name in wanted:
                metrics[f"reward_{name}"] = float(step_reward[:, idx].mean().item())
        return metrics

    def _collect_termination_metrics(self) -> dict[str, Any]:
        termination_manager = getattr(self.env, "termination_manager", None)
        term_names = getattr(termination_manager, "_term_names", [])
        last_episode_dones = getattr(termination_manager, "_last_episode_dones", None)
        if last_episode_dones is None or not term_names:
            return {}
        metrics: dict[str, Any] = {}
        for idx, name in enumerate(term_names):
            if name in {"time_out", "reset_on_bad_ori"}:
                metrics[f"termination_{name}"] = float(last_episode_dones[:, idx].float().mean().item())
        return metrics

    def _collect_update_metrics(self, update_info: dict[str, float]) -> dict[str, Any]:
        return {
            "policy_loss": float(update_info.get("policy_loss", 0.0)),
            "entropy": float(update_info.get("dist_entropy", update_info.get("entropy", 0.0))),
            "value_loss": float(update_info.get("value_loss", 0.0)),
            "ratio": float(update_info.get("ratio", 0.0)),
            "actor_grad_norm": float(update_info.get("actor_grad_norm", 0.0)),
            "critic_grad_norm": float(update_info.get("critic_grad_norm", 0.0)),
        }

    def _collect_attack_metrics(self, attack_info: Any) -> dict[str, Any]:
        attacker = self.pipeline.attacker if self.pipeline is not None else None
        cfg = getattr(attacker, "cfg", None)
        info = attack_info if isinstance(attack_info, dict) else {}
        return {
            "attack_enabled": float(info.get("attack_enabled", float(getattr(cfg, "enabled", 0.0)))),
            "attack_level": str(getattr(self.cfg, "attack_level", "")),
            "attack_mode": str(info.get("attack_mode", getattr(attacker, "_runtime_mode", getattr(cfg, "mode", "")))),
            "attack_target_mode": str(getattr(cfg, "target_mode", "")),
            "attack_max_fdi_pos": float(getattr(cfg, "max_fdi_pos", 0.0)),
            "attack_max_fdi_acc": float(getattr(cfg, "max_fdi_acc", 0.0)),
            "attack_max_dos_rate": float(getattr(cfg, "max_dos_rate", 0.0)),
            "attack_late_obj_scale": float(info.get("attack_late_obj_scale", 0.0)),
            "attack_late_update_interval": float(info.get("attack_late_update_interval", 0.0)),
            "fdi_abs_mean": float(info.get("fdi_abs_mean", getattr(attacker, "last_fdi_mean", 0.0))),
            "dos_rate": float(info.get("dos_rate", getattr(attacker, "last_dos_rate", 0.0))),
            "obs_fdi_abs_mean": float(info.get("obs_fdi_abs_mean", getattr(attacker, "last_obs_fdi_mean", 0.0))),
            "obs_dos_rate": float(info.get("obs_dos_rate", getattr(attacker, "last_obs_dos_rate", 0.0))),
            "obs_dos_budget": float(info.get("obs_dos_budget", getattr(attacker, "last_obs_dos_budget", 0.0))),
            "obs_drop_count": float(info.get("obs_drop_count", getattr(attacker, "last_obs_drop_count", 0.0))),
            "obs_elem_n": float(info.get("obs_elem_n", getattr(attacker, "last_obs_elem_n", 0.0))),
            "act_fdi_abs_mean": float(info.get("act_fdi_abs_mean", getattr(attacker, "last_act_fdi_mean", 0.0))),
            "act_dos_rate": float(info.get("act_dos_rate", getattr(attacker, "last_act_dos_rate", 0.0))),
            "act_dos_budget": float(info.get("act_dos_budget", getattr(attacker, "last_act_dos_budget", 0.0))),
            "act_drop_count": float(info.get("act_drop_count", getattr(attacker, "last_act_drop_count", 0.0))),
            "act_elem_n": float(info.get("act_elem_n", getattr(attacker, "last_act_elem_n", 0.0))),
            "attack_objective": float(info.get("attack_objective", getattr(attacker, "last_attack_obj", 0.0))),
            "attack_g_loss": float(info.get("attack_g_loss", getattr(attacker, "last_g_loss", 0.0))),
            "attack_d_loss": float(info.get("attack_d_loss", getattr(attacker, "last_d_loss", 0.0))),
            "attack_realism_loss": float(info.get("attack_realism_loss", getattr(attacker, "last_realism_loss", 0.0))),
            "attack_seq_realism_loss": float(info.get("attack_seq_realism_loss", getattr(attacker, "last_seq_realism_loss", 0.0))),
            "attack_phy_ctx_norm": float(info.get("attack_phy_ctx_norm", 0.0)),
            "attack_ref_template_id": str(info.get("attack_ref_template_id", getattr(attacker, "last_ref_template_id", ""))),
        }

    def _collect_teacher_metrics(self, teacher_info: Any) -> dict[str, Any]:
        teacher = self.pipeline.teacher if self.pipeline is not None else None
        info = teacher_info if isinstance(teacher_info, dict) else {}
        physical_context = teacher.get_physical_attack_context() if hasattr(teacher, "get_physical_attack_context") else {}
        metrics = {
            "teacher_update_active": 1.0 if isinstance(teacher_info, dict) else 0.0,
            "teacher_shaping_mean": float(info.get("teacher_shaping_mean", getattr(teacher, "last_shaping_mean", 0.0))),
            "teacher_loss": float(info.get("teacher_loss", getattr(teacher, "last_teacher_loss", 0.0))),
            "teacher_base_loss": float(info.get("teacher_base_loss", 0.0)),
            "teacher_delta_j": float(info.get("teacher_delta_j", 0.0)),
            "teacher_norm_delta_j": float(info.get("teacher_norm_delta_j", 0.0)),
            "teacher_outer_coef_scale": float(info.get("teacher_outer_coef_scale", 0.0)),
            "teacher_outer_loss": float(info.get("teacher_outer_loss", 0.0)),
            "teacher_advantage_corr": float(info.get("teacher_advantage_corr", 0.0)),
            "teacher_physical_adv_std": float(info.get("teacher_physical_adv_std", 0.0)),
            "teacher_shaping_adv_std": float(info.get("teacher_shaping_adv_std", 0.0)),
            "teacher_window_samples": float(info.get("teacher_window_samples", 0.0)),
            "teacher_grad_norm": float(info.get("teacher_grad_norm", 0.0)),
        }
        for key in (
            "physical_cost",
            "physical_spacing_cost",
            "physical_velocity_cost",
            "physical_acceleration_cost",
            "physical_jerk_cost",
            "physical_overspeed_cost",
            "physical_lateral_cost",
            "physical_heading_cost",
            "physical_backward_cost",
            "physical_forward_deficit_cost",
            "physical_collision_cost",
        ):
            metrics[key] = float(physical_context.get(key, 0.0))
        return metrics

    def _collect_shield_metrics(self) -> dict[str, Any]:
        if self.pipeline is None or not hasattr(self.pipeline.shield, "get_stats"):
            return {}
        stats = self.pipeline.shield.get_stats()
        if not isinstance(stats, dict):
            return {}
        return {
            "shield_warn_rate": float(stats.get("shield_warn_rate", 0.0)),
            "shield_critical_rate": float(stats.get("shield_critical_rate", 0.0)),
            "shield_trigger_rate": float(stats.get("shield_trigger_rate", 0.0)),
            "shield_scale_mean": float(stats.get("shield_scale_mean", 1.0)),
            "shield_lateral_rate": float(stats.get("shield_lateral_rate", 0.0)),
            "shield_lateral_critical_rate": float(stats.get("shield_lateral_critical_rate", 0.0)),
            "shield_lateral_turn_mean": float(stats.get("shield_lateral_turn_mean", 0.0)),
            "shield_centerline_rate": float(stats.get("shield_centerline_rate", 0.0)),
            "shield_centerline_turn_mean": float(stats.get("shield_centerline_turn_mean", 0.0)),
            "shield_first_follower_centerline_rate": float(
                stats.get("shield_first_follower_centerline_rate", 0.0)
            ),
            "shield_first_follower_centerline_turn_mean": float(
                stats.get("shield_first_follower_centerline_turn_mean", 0.0)
            ),
        }

    def train_if_ready(self, next_obs: Any) -> tuple[list[dict[str, float]], dict[str, float]] | None:
        """Run one HAPPO update when a rollout segment is full."""
        if self.inference_only or self.runner is None or not self.ready_to_train():
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

    @staticmethod
    def _soft_hard_penalty(abs_error: torch.Tensor, deadzone: float, critical: float, hard_weight: float) -> torch.Tensor:
        soft_error = torch.relu(abs_error - float(deadzone))
        hard_error = torch.relu(abs_error - float(critical))
        return soft_error + float(hard_weight) * torch.square(hard_error)

    def _apply_local_agent_reward_shaping(self, rewards_tensor: torch.Tensor) -> torch.Tensor:
        """Add per-agent physical shaping before HAPPO stores rewards."""
        self._last_local_reward_metrics = {}
        if not bool(getattr(self.cfg, "local_reward_shaping", True)):
            return rewards_tensor

        centerline_coef = float(getattr(self.cfg, "local_reward_centerline_coef", 0.0))
        pair_lateral_coef = float(getattr(self.cfg, "local_reward_pair_lateral_coef", 0.0))
        heading_coef = float(getattr(self.cfg, "local_reward_heading_coef", 0.0))
        turn_coef = float(getattr(self.cfg, "local_reward_turn_coef", 0.0))
        first_centerline_scale = float(getattr(self.cfg, "local_reward_first_follower_centerline_scale", 1.0))
        first_pair_lateral_scale = float(getattr(self.cfg, "local_reward_first_follower_pair_lateral_scale", 1.0))
        first_turn_scale = float(getattr(self.cfg, "local_reward_first_follower_turn_scale", 1.0))
        last_centerline_scale = float(getattr(self.cfg, "local_reward_last_follower_centerline_scale", 1.0))
        last_pair_lateral_scale = float(getattr(self.cfg, "local_reward_last_follower_pair_lateral_scale", 1.0))
        last_turn_scale = float(getattr(self.cfg, "local_reward_last_follower_turn_scale", 1.0))
        if max(centerline_coef, pair_lateral_coef, heading_coef, turn_coef) <= 0.0:
            return rewards_tensor

        try:
            robots = ["robot", "robot_2", "robot_3", "robot_4", "robot_5"][: self.num_agents]
            num_envs = rewards_tensor.shape[0]
            num_agents = min(self.num_agents, rewards_tensor.shape[1], len(robots))
            centerline_penalty = torch.zeros((num_envs, num_agents), device=self.device)
            pair_lateral_penalty = torch.zeros_like(centerline_penalty)
            heading_penalty = torch.zeros_like(centerline_penalty)
            turn_penalty = torch.zeros_like(centerline_penalty)

            target_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
            heading_vecs = []
            for agent_idx, name in enumerate(robots[:num_agents]):
                centerline_y = self._env_local_y(name)
                centerline_penalty[:, agent_idx] = self._soft_hard_penalty(
                    centerline_y.abs(),
                    deadzone=0.05,
                    critical=0.35,
                    hard_weight=3.0,
                )
                heading = quat_apply(self.env.scene[name].data.root_quat_w.to(self.device), target_dir)[:, :2]
                heading = torch.nn.functional.normalize(heading, dim=-1)
                heading_vecs.append(heading)
                heading_penalty[:, agent_idx] = heading[:, 1].abs()

            for agent_idx in range(1, num_agents):
                pred = self.env.scene[robots[agent_idx - 1]]
                foll = self.env.scene[robots[agent_idx]]
                rel = quat_apply_inverse(
                    pred.data.root_quat_w.to(self.device),
                    foll.data.root_pos_w.to(self.device) - pred.data.root_pos_w.to(self.device),
                )
                pair_lateral_penalty[:, agent_idx] = self._soft_hard_penalty(
                    rel[:, 1].abs(),
                    deadzone=0.05,
                    critical=0.25,
                    hard_weight=4.0,
                )
                pair_alignment = torch.sum(heading_vecs[agent_idx - 1] * heading_vecs[agent_idx], dim=-1).clamp(-1.0, 1.0)
                heading_penalty[:, agent_idx] += 0.5 * (1.0 - pair_alignment)

            actions = None
            if self.pipeline is not None and getattr(self.pipeline.student, "has_pending_action", False):
                actions = self.pipeline.student.pending_actions
            elif self._last_raw_joint_actions is not None:
                actions = self._last_raw_joint_actions
            if actions is not None and actions.dim() == 3 and actions.shape[-1] >= 4:
                usable_agents = min(num_agents, actions.shape[1])
                turn = self._semantic_action_physical_turn(actions[:, :usable_agents, :])
                turn_penalty[:, :usable_agents] = turn.abs()

            if num_agents > 1:
                centerline_penalty[:, 1] *= first_centerline_scale
                pair_lateral_penalty[:, 1] *= first_pair_lateral_scale
                turn_penalty[:, 1] *= first_turn_scale
                last_idx = num_agents - 1
                centerline_penalty[:, last_idx] *= last_centerline_scale
                pair_lateral_penalty[:, last_idx] *= last_pair_lateral_scale
                turn_penalty[:, last_idx] *= last_turn_scale

            shaping = -(
                centerline_coef * centerline_penalty
                + pair_lateral_coef * pair_lateral_penalty
                + heading_coef * heading_penalty
                + turn_coef * turn_penalty
            )
            self._last_local_reward_metrics = {
                "local_reward_shaping_mean": float(shaping.mean().item()),
                "local_reward_centerline_penalty_mean": float(centerline_penalty.mean().item()),
                "local_reward_pair_lateral_penalty_mean": float(pair_lateral_penalty.mean().item()),
                "local_reward_heading_penalty_mean": float(heading_penalty.mean().item()),
                "local_reward_turn_penalty_mean": float(turn_penalty.mean().item()),
            }
            return rewards_tensor + shaping.unsqueeze(-1)
        except Exception:
            return rewards_tensor

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
