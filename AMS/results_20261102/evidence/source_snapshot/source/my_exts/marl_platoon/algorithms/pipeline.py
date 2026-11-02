"""Modular training architecture for task-internal student/teacher/attacker/shield.

This module defines a lightweight scheduler and component interfaces so the
platoon task can evolve toward the full paper pipeline while keeping the
official IsaacLab entrypoint unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PipelineScheduleCfg:
    """Update schedule for paper-style multi-module training."""

    student_every_steps: int = 32
    attacker_every_student_updates: int = 4
    teacher_every_student_updates: int = 8
    shield_every_steps: int = 1
    curriculum_enabled: bool = False
    acquisition_updates: int = 0
    expert_decay_updates: int = 0
    attack_adaptation_updates: int = 0
    expert_start_ratio: float = 1.0
    expert_end_ratio: float = 0.0
    clean_fraction_start: float = 1.0
    clean_fraction_end: float = 0.0


@dataclass
class PipelineFlags:
    enable_teacher: bool = False
    enable_attack: bool = False
    enable_shield: bool = False


@dataclass
class StepBatch:
    """Shared transition container passed across pipeline modules."""

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_obs: torch.Tensor
    extras: Any | None = None


class StudentModule:
    """Student learner interface (HAPPO/PPO/etc)."""

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def observe(self, batch: StepBatch) -> None:
        raise NotImplementedError

    def maybe_update(self) -> dict[str, float] | None:
        raise NotImplementedError


class TeacherModule:
    """Teacher interface for reward shaping and meta updates."""

    def shape_reward(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        return rewards

    def get_physical_attack_context(self) -> dict[str, float]:
        return {}

    def maybe_update(self, context: dict[str, Any]) -> dict[str, float] | None:
        return None


class AttackerModule:
    """Attacker interface for observation/action disturbance."""

    def perturb_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def perturb_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def maybe_update(self, context: dict[str, Any]) -> dict[str, float] | None:
        return None


class ShieldModule:
    """Shield interface for safety projection/filter."""

    def project_action(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return actions


class NoOpTeacher(TeacherModule):
    pass


class NoOpAttacker(AttackerModule):
    pass


class NoOpShield(ShieldModule):
    pass


class PlatoonTrainingPipeline:
    """Task-internal scheduler for student/teacher/attacker/shield modules.

    The pipeline is intentionally lightweight and non-invasive. It can be wired
    under the existing IsaacLab task wrappers to orchestrate module calls and
    update cadence without introducing a new training entrypoint.
    """

    def __init__(
        self,
        *,
        student: StudentModule,
        teacher: TeacherModule | None = None,
        attacker: AttackerModule | None = None,
        shield: ShieldModule | None = None,
        schedule: PipelineScheduleCfg | None = None,
        flags: PipelineFlags | None = None,
    ):
        self.student = student
        self.teacher = teacher or NoOpTeacher()
        self.attacker = attacker or NoOpAttacker()
        self.shield = shield or NoOpShield()
        self.schedule = schedule or PipelineScheduleCfg()
        self.flags = flags or PipelineFlags()

        self.total_steps = 0
        self.student_updates = 0
        self.attacker_updates = 0
        self.teacher_updates = 0
        # Step-1 teacher outer-objective cache (window-level, first-order only).
        self._teacher_outer_j_pre: float | None = None
        self._teacher_outer_j_latest = 0.0
        self._teacher_outer_delta_j = 0.0
        self._last_action_trace: dict[str, torch.Tensor] | None = None
        self.curriculum_stage = "disabled"
        self.curriculum_expert_ratio = 0.0
        self.curriculum_clean_fraction = 0.0
        self._teacher_score_sum = 0.0
        self._teacher_score_count = 0
        self._teacher_previous_window_mean: float | None = None
        self._clean_gate_approved = False
        self._attack_start_update: int | None = None
        self._apply_curriculum()

    def state_dict(self) -> dict[str, Any]:
        state = {"total_steps": self.total_steps, "student_updates": self.student_updates,
                 "attacker_updates": self.attacker_updates, "teacher_updates": self.teacher_updates}
        if hasattr(self.teacher, "model"):
            state["teacher_model"] = self.teacher.model.state_dict()
            state["teacher_optimizer"] = self.teacher.optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("total_steps", "student_updates", "attacker_updates", "teacher_updates"):
            setattr(self, key, int(state.get(key, getattr(self, key))))
        if "teacher_model" in state and hasattr(self.teacher, "model"):
            self.teacher.model.load_state_dict(state["teacher_model"])
            self.teacher.optimizer.load_state_dict(state["teacher_optimizer"])
        # Simulator state and partial physical windows are not restored. A
        # fresh clean validation must approve attacks after a process restart.
        self._clean_gate_approved = False
        self._attack_start_update = None
        self._teacher_previous_window_mean = None
        self._teacher_score_sum = 0.0
        self._teacher_score_count = 0
        self._apply_curriculum()

    def approve_clean_gate(self, evidence: dict[str, Any]) -> None:
        """Called only with a verified clean gate by the external stage driver.

        Training rewards/update counts never approve a promotion. The driver
        must verify checkpoint-bound raw evaluation artifacts before this call.
        """
        required = ("completion", "instant", "dwell")
        safety = ("collision", "road_any", "road_termination", "attack_enabled")
        if not evidence.get("checkpoint_sha256") or not all(evidence.get(k) == 1 for k in required):
            raise ValueError("Clean gate requires checkpoint-bound full completion evidence")
        if not all(evidence.get(k) == 0 for k in safety):
            raise ValueError("Clean gate requires zero attack/collision/road violations")
        if self.student_updates < self.schedule.acquisition_updates + self.schedule.expert_decay_updates:
            raise ValueError("Withdraw expert control before approving clean gate")
        self._clean_gate_approved = True
        self._attack_start_update = self.student_updates
        self._apply_curriculum()

    def preprocess_obs_for_student(self, obs: torch.Tensor) -> torch.Tensor:
        if self.flags.enable_attack:
            return self.attacker.perturb_obs(obs)
        return obs

    def postprocess_action_for_env(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Keep all semantic action stages.  The policy action remains the one
        # used for the on-policy likelihood, while physical objectives must use
        # what survived the attack and safety projection.
        policy = actions.detach().clone()
        result = policy
        if self.flags.enable_attack:
            result = self.attacker.perturb_action(result)
        attacked = result.detach().clone()
        if self.flags.enable_shield and self.total_steps % max(self.schedule.shield_every_steps, 1) == 0:
            result = self.shield.project_action(obs, result)
        shield = result.detach().clone()
        self._last_action_trace = {
            "policy_actions": policy,
            "attacked_actions": attacked,
            "shield_actions": shield,
            "executed_semantic_actions": shield.detach().clone(),
        }
        return result

    def consume_action_trace(self) -> dict[str, torch.Tensor] | None:
        trace = self._last_action_trace
        self._last_action_trace = None
        if trace is None:
            return None
        return {key: value.detach().clone() for key, value in trace.items()}

    def _apply_curriculum(self) -> dict[str, float | str]:
        """Coordinate clean acquisition, expert withdrawal and attack ramping."""
        cfg = self.schedule
        if not cfg.curriculum_enabled:
            self.curriculum_stage = "disabled"
            return {"curriculum_stage": self.curriculum_stage, "curriculum_expert_ratio": 0.0,
                    "curriculum_clean_fraction": 0.0}
        update = int(self.student_updates)
        acquisition = max(int(cfg.acquisition_updates), 0)
        decay = max(int(cfg.expert_decay_updates), 0)
        adaptation = max(int(cfg.attack_adaptation_updates), 0)
        if update < acquisition:
            stage = "clean_acquisition"
            progress = 0.0
        elif update < acquisition + decay:
            stage = "expert_decay"
            progress = (update - acquisition) / max(decay, 1)
        elif not self._clean_gate_approved:
            stage = "awaiting_clean_validation"
            progress = 1.0
        elif update - int(self._attack_start_update if self._attack_start_update is not None else update) < adaptation:
            stage = "attack_adaptation"
            progress = 1.0
        else:
            stage = "awaiting_hard_validation"
            progress = 1.0
        expert_ratio = float(cfg.expert_start_ratio) + progress * (
            float(cfg.expert_end_ratio) - float(cfg.expert_start_ratio)
        )
        # Stay entirely clean through acquisition AND expert withdrawal. Attack
        # mixing starts only after a strict external clean gate, on its own clock.
        attack_progress = 0.0 if not self._clean_gate_approved else min(
            max((update - int(self._attack_start_update or 0)) / max(adaptation, 1), 0.0), 1.0
        )
        clean_fraction = 1.0 + attack_progress * (
            float(cfg.clean_fraction_end) - 1.0
        )
        self.curriculum_stage = stage
        self.curriculum_expert_ratio = min(max(expert_ratio, 0.0), 1.0)
        self.curriculum_clean_fraction = min(max(clean_fraction, 0.0), 1.0)
        if hasattr(self.shield, "cfg"):
            self.shield.cfg.expert_path_tracking_enabled = self.curriculum_expert_ratio > 0.0
            self.shield.cfg.expert_blend_ratio = self.curriculum_expert_ratio
            self.shield.cfg.expert_blend_end_ratio = self.curriculum_expert_ratio
            self.shield.cfg.expert_blend_decay_steps = 0
        if hasattr(self.attacker, "cfg"):
            self.attacker.cfg.clean_env_fraction = self.curriculum_clean_fraction
        return {
            "curriculum_stage": self.curriculum_stage,
            "curriculum_expert_ratio": self.curriculum_expert_ratio,
            "curriculum_clean_fraction": self.curriculum_clean_fraction,
        }

    def _compute_teacher_outer_score_j(self, batch: StepBatch) -> float:
        """Compute paper-aligned outer objective score J.

        This is the task-side bridge to the paper's global control objective.
        It uses the shaped reward plus termination cost only; the detailed
        physical-cost decomposition is supplied by the teacher context.
        """
        reward_mean = float(batch.rewards.mean().item())
        bad_done_rate = float(batch.dones.float().mean().item())
        return reward_mean - bad_done_rate

    def _compute_teacher_physical_outer_score(self, batch: StepBatch, physical_context: dict[str, float]) -> float:
        physical_cost = physical_context.get("physical_cost")
        if physical_cost is None:
            return self._compute_teacher_outer_score_j(batch)
        bad_done_rate = float(batch.dones.float().mean().item())
        collision_cost = float(physical_context.get("physical_collision_cost", 0.0))
        # Paper global cost: J = E[mean_i c_i(t) + C_col I_col(t)].
        # The teacher context already provides mean_i c_i(t)-compatible
        # physical_cost, so do not double count its sub-terms here.
        return -float(physical_cost) - collision_cost - bad_done_rate

    def process_transition(self, batch: StepBatch) -> dict[str, Any]:
        self.total_steps += 1

        physical_attack_context: dict[str, float] = {}
        action_trace = batch.extras if isinstance(batch.extras, dict) else {}
        teacher_actions = action_trace.get("executed_semantic_actions", batch.actions)
        if self.flags.enable_teacher:
            batch.rewards = self.teacher.shape_reward(batch.obs, teacher_actions, batch.rewards)
            physical_attack_context = self.teacher.get_physical_attack_context()

        teacher_outer_j = self._compute_teacher_physical_outer_score(batch, physical_attack_context)
        self._teacher_score_sum += teacher_outer_j
        self._teacher_score_count += 1
        self._teacher_outer_j_latest = teacher_outer_j

        # Generate the next policy observation exactly once.  The router caches
        # this tensor for the next action and the buffer uses the same tensor for
        # bootstrap/update, preserving the old-log-prob observation identity.
        batch.next_obs = self.preprocess_obs_for_student(batch.next_obs)
        self.student.observe(batch)
        update_info = self.student.maybe_update()

        logs: dict[str, Any] = {"student_update": update_info}

        if update_info is not None:
            self.student_updates += 1
            update_info.update(self._apply_curriculum())
            set_student_updates = getattr(self.attacker, "set_student_updates", None)
            if callable(set_student_updates):
                set_student_updates(self.student_updates)
            if self.flags.enable_attack and self.student_updates % max(self.schedule.attacker_every_student_updates, 1) == 0:
                rewards = batch.rewards
                dones = batch.dones
                attack_context = {
                    "student_updates": self.student_updates,
                    "reward_mean": float(rewards.mean().item()),
                    "bad_done_rate": float(dones.float().mean().item()),
                }
                attack_context.update(physical_attack_context)
                atk_info = self.attacker.maybe_update(attack_context)
                self.attacker_updates += 1
                logs["attacker_update"] = atk_info
            if self.flags.enable_teacher and self.student_updates % max(self.schedule.teacher_every_student_updates, 1) == 0:
                # Compare complete non-overlapping rollout windows. The latest
                # window follows the preceding student updates; it is NOT a
                # measurement of the update that has just run above.
                j_post = self._teacher_score_sum / max(self._teacher_score_count, 1)
                j_pre = self._teacher_previous_window_mean
                if j_pre is None:
                    j_pre = j_post  # first complete window has no pre baseline
                self._teacher_outer_delta_j = j_post - j_pre
                tea_info = self.teacher.maybe_update(
                    {
                        "student_updates": self.student_updates,
                        "teacher_j_pre": j_pre,
                        "teacher_j_post": j_post,
                        "teacher_delta_j": self._teacher_outer_delta_j,
                        "teacher_window_steps": self._teacher_score_count,
                        "teacher_post_policy_update": max(self.student_updates - 1, 0),
                        "teacher_outer_mode": "first_order_window_covariance_proxy",
                    }
                )
                self._teacher_previous_window_mean = j_post
                self._teacher_score_sum = 0.0
                self._teacher_score_count = 0
                self.teacher_updates += 1
                logs["teacher_update"] = tea_info

        return logs
