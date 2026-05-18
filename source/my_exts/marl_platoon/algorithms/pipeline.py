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

    def preprocess_obs_for_student(self, obs: torch.Tensor) -> torch.Tensor:
        if self.flags.enable_attack:
            return self.attacker.perturb_obs(obs)
        return obs

    def postprocess_action_for_env(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        result = actions
        if self.flags.enable_attack:
            result = self.attacker.perturb_action(result)
        if self.flags.enable_shield and self.total_steps % max(self.schedule.shield_every_steps, 1) == 0:
            result = self.shield.project_action(obs, result)
        return result

    def _compute_teacher_outer_score_j(self, batch: StepBatch) -> float:
        """Compute Step-1 outer objective score J for teacher updates.

        This is a first-order meta proxy (no second-order gradient):
            J = reward_mean + forward_drive - no_backward_penalty - bad_done_rate

        where:
        - reward_mean: current shaped reward mean over env/agent dims
        - forward_drive: positive forward command magnitude proxy
        - no_backward_penalty: negative forward command magnitude proxy
        - bad_done_rate: termination proxy (includes reset_on_bad_ori when present)

        The teacher outer signal is then defined over windows:
            delta_j = J_post - J_pre
        """
        reward_mean = float(batch.rewards.mean().item())
        forward_component = batch.actions[..., 0]
        forward_drive = float(torch.relu(forward_component).mean().item())
        no_backward_penalty = float(torch.relu(-forward_component).mean().item())
        bad_done_rate = float(batch.dones.float().mean().item())
        return reward_mean + forward_drive - no_backward_penalty - bad_done_rate

    def process_transition(self, batch: StepBatch) -> dict[str, Any]:
        self.total_steps += 1
        teacher_outer_j = self._compute_teacher_outer_score_j(batch)
        if self._teacher_outer_j_pre is None:
            self._teacher_outer_j_pre = teacher_outer_j
        self._teacher_outer_j_latest = teacher_outer_j

        if self.flags.enable_teacher:
            batch.rewards = self.teacher.shape_reward(batch.obs, batch.actions, batch.rewards)

        self.student.observe(batch)
        update_info = self.student.maybe_update()

        logs: dict[str, Any] = {"student_update": update_info}

        if update_info is not None:
            self.student_updates += 1
            if self.flags.enable_attack and self.student_updates % max(self.schedule.attacker_every_student_updates, 1) == 0:
                forward_component = batch.actions[..., 0]
                rewards = batch.rewards
                dones = batch.dones
                attack_context = {
                    "student_updates": self.student_updates,
                    "reward_mean": float(rewards.mean().item()),
                    "forward_drive": float(torch.relu(forward_component).mean().item()),
                    "no_backward_penalty": float(torch.relu(-forward_component).mean().item()),
                    "bad_done_rate": float(dones.float().mean().item()),
                    # lightweight physical proxies for CA-GAN conditioning
                    "spacing_err_proxy": float(rewards.abs().mean().item()),
                    "vel_err_proxy": float(torch.relu(forward_component.abs() - 0.5).mean().item()),
                    "acc_proxy": float(torch.relu(batch.actions[..., 0].abs() - 0.2).mean().item()),
                    "jerk_proxy": float(torch.relu(batch.actions.diff(dim=0).abs().mean() if batch.actions.shape[0] > 1 else torch.tensor(0.0, device=batch.actions.device)).item()),
                }
                atk_info = self.attacker.maybe_update(attack_context)
                self.attacker_updates += 1
                logs["attacker_update"] = atk_info
            if self.flags.enable_teacher and self.student_updates % max(self.schedule.teacher_every_student_updates, 1) == 0:
                j_pre = self._teacher_outer_j_pre if self._teacher_outer_j_pre is not None else self._teacher_outer_j_latest
                j_post = self._teacher_outer_j_latest
                self._teacher_outer_delta_j = j_post - j_pre
                tea_info = self.teacher.maybe_update(
                    {
                        "student_updates": self.student_updates,
                        "teacher_j_pre": j_pre,
                        "teacher_j_post": j_post,
                        "teacher_delta_j": self._teacher_outer_delta_j,
                    }
                )
                self._teacher_outer_j_pre = self._teacher_outer_j_latest
                self.teacher_updates += 1
                logs["teacher_update"] = tea_info

        return logs
