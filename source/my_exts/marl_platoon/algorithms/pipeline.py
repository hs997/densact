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

    def process_transition(self, batch: StepBatch) -> dict[str, Any]:
        self.total_steps += 1

        if self.flags.enable_teacher:
            batch.rewards = self.teacher.shape_reward(batch.obs, batch.actions, batch.rewards)

        self.student.observe(batch)
        update_info = self.student.maybe_update()

        logs: dict[str, Any] = {"student_update": update_info}

        if update_info is not None:
            self.student_updates += 1
            if self.flags.enable_attack and self.student_updates % max(self.schedule.attacker_every_student_updates, 1) == 0:
                atk_info = self.attacker.maybe_update({"student_updates": self.student_updates})
                self.attacker_updates += 1
                logs["attacker_update"] = atk_info
            if self.flags.enable_teacher and self.student_updates % max(self.schedule.teacher_every_student_updates, 1) == 0:
                tea_info = self.teacher.maybe_update({"student_updates": self.student_updates})
                self.teacher_updates += 1
                logs["teacher_update"] = tea_info

        return logs
