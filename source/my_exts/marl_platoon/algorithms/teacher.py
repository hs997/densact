"""Teacher modules for the platoon training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from marl_platoon.algorithms.pipeline import TeacherModule
from marl_platoon.tasks.platoon.teacher import RewardTeacher


@dataclass
class RewardTeacherCfg:
    obs_dim: int
    act_dim: int
    shaping_coef: float = 0.0
    device: str = "cpu"
    lr: float = 1.0e-4
    update_interval: int = 5
    shaping_clip: float = 0.05
    action_penalty_coef: float = 0.0


class RewardTeacherModule(TeacherModule):
    """Minimal trainable teacher adapter.

    This is not the full paper meta-gradient yet; it adds a stable teacher
    optimization loop so the module can learn rather than stay random.
    """

    def __init__(self, cfg: RewardTeacherCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = RewardTeacher(obs_dim=cfg.obs_dim, act_dim=cfg.act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.last_shaping_mean = 0.0
        self.last_teacher_loss = 0.0
        self._step = 0
        self._cached_obs: torch.Tensor | None = None
        self._cached_actions: torch.Tensor | None = None
        self._cached_rewards: torch.Tensor | None = None

    def shape_reward(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        teacher_obs = obs[:, 0, :] if obs.dim() == 3 else obs
        flat_actions = actions.reshape(actions.shape[0], -1)
        with torch.no_grad():
            shaping = self.model(teacher_obs.to(self.device), flat_actions.to(self.device)).view(rewards.shape[0], 1, 1)
            if self.cfg.shaping_clip > 0.0:
                shaping = torch.clamp(shaping, -self.cfg.shaping_clip, self.cfg.shaping_clip)
        self.last_shaping_mean = float(shaping.mean().item())
        self._cached_obs = teacher_obs.detach().clone()
        self._cached_actions = flat_actions.detach().clone()
        self._cached_rewards = rewards.detach().clone()

        if self.cfg.shaping_coef <= 0.0:
            return rewards
        delta = self.cfg.shaping_coef * shaping
        if self.cfg.action_penalty_coef > 0.0:
            action_penalty = flat_actions.pow(2).mean(dim=-1).view(rewards.shape[0], 1, 1)
            delta = delta - self.cfg.action_penalty_coef * action_penalty
        return rewards + delta

    def maybe_update(self, context: dict[str, Any]) -> dict[str, float] | None:
        self._step += 1
        if self._step % max(self.cfg.update_interval, 1) != 0:
            return None
        if self._cached_obs is None or self._cached_actions is None or self._cached_rewards is None:
            return None

        self.model.train()
        pred = self.model(self._cached_obs.to(self.device), self._cached_actions.to(self.device)).view(-1, 1, 1)
        target = self._cached_rewards.to(self.device)
        # Minimal stable objective: fit teacher output to normalized reward signal.
        target = (target - target.mean()) / (target.std().clamp_min(1.0e-5))
        loss = torch.nn.functional.mse_loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()

        self.last_teacher_loss = float(loss.item())
        return {
            "teacher_shaping_mean": self.last_shaping_mean,
            "teacher_loss": self.last_teacher_loss,
            "teacher_grad_norm": float(grad_norm),
        }
