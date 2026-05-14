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
    reward_ema_tau: float = 0.97
    consistency_coef: float = 0.05
    no_backward_coef: float = 0.10
    forward_drive_coef: float = 0.10
    forward_drive_target: float = 0.10
    outer_delta_coef: float = 0.10
    outer_delta_warmup_updates: int = 20
    outer_delta_ramp_updates: int = 50


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
        self._cached_prev_obs: torch.Tensor | None = None
        self._cached_prev_actions: torch.Tensor | None = None
        self._cached_physical_score: torch.Tensor | None = None
        # Step-3 paper-aligned first-order meta signal buffer.
        # With num_envs=1, a single transition has no non-zero centered advantage;
        # therefore we accumulate a teacher-update window and estimate
        # E[A_phy * A_phi] over that window, matching the paper's
        # advantage-correlation estimator more closely than single-batch stats.
        self._window_obs: list[torch.Tensor] = []
        self._window_actions: list[torch.Tensor] = []
        self._window_physical_scores: list[torch.Tensor] = []
        self._reward_ema: torch.Tensor | None = None
        self._delta_j_ema = 0.0
        self._delta_j_var_ema = 1.0

    def shape_reward(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        teacher_obs = obs[:, 0, :] if obs.dim() == 3 else obs
        flat_actions = actions.reshape(actions.shape[0], -1)
        with torch.no_grad():
            shaping = self.model(teacher_obs.to(self.device), flat_actions.to(self.device)).view(rewards.shape[0], 1, 1)
            if self.cfg.shaping_clip > 0.0:
                shaping = torch.clamp(shaping, -self.cfg.shaping_clip, self.cfg.shaping_clip)
        self.last_shaping_mean = float(shaping.mean().item())
        if self._cached_obs is not None:
            self._cached_prev_obs = self._cached_obs.detach().clone()
        if self._cached_actions is not None:
            self._cached_prev_actions = self._cached_actions.detach().clone()
        self._cached_obs = teacher_obs.detach().clone()
        self._cached_actions = flat_actions.detach().clone()
        self._cached_rewards = rewards.detach().clone()
        # Build a lightweight physical-score proxy from available training-time signals.
        # This is the implementation proxy of A_phy in the paper's
        # first-order advantage-correlation estimator. It is intentionally
        # computed before applying teacher shaping so it reflects the current
        # transition's physical tendency rather than the teacher-modified reward.
        forward_component = flat_actions[..., 0].view(rewards.shape[0], 1, 1)
        no_backward_penalty = torch.relu(-forward_component)
        forward_drive = torch.relu(forward_component)
        action_energy_penalty = flat_actions.pow(2).mean(dim=-1, keepdim=True).view(rewards.shape[0], 1, 1)
        physical_score = rewards.mean(dim=(1, 2), keepdim=True) + forward_drive - no_backward_penalty - 0.1 * action_energy_penalty
        self._cached_physical_score = physical_score.detach().clone()

        # Accumulate the teacher-update window for Step-3 meta attribution.
        # The actual correlation is computed in maybe_update() and then the
        # window is cleared, so each teacher update receives delayed credit
        # from the shaping behavior applied over the preceding window.
        self._window_obs.append(teacher_obs.detach().clone())
        self._window_actions.append(flat_actions.detach().clone())
        self._window_physical_scores.append(physical_score.detach().clone())

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

        rewards = self._cached_rewards.to(self.device)
        if self._reward_ema is None:
            self._reward_ema = rewards.mean(dim=0, keepdim=True).detach()
        self._reward_ema = self.cfg.reward_ema_tau * self._reward_ema + (1.0 - self.cfg.reward_ema_tau) * rewards.mean(dim=0, keepdim=True).detach()

        # Stage-2 objective (more policy-aware than plain reward fitting):
        # 1) advantage-like target against EMA baseline
        # 2) temporal consistency regularization to avoid shaping oscillation
        target_adv = rewards - self._reward_ema
        target_adv = target_adv / target_adv.std(unbiased=False).clamp_min(1.0e-5)
        base_loss = torch.nn.functional.mse_loss(pred, target_adv)

        consistency_loss = torch.tensor(0.0, device=self.device)
        if self._cached_prev_obs is not None and self._cached_prev_actions is not None:
            with torch.no_grad():
                prev_pred = self.model(
                    self._cached_prev_obs.to(self.device),
                    self._cached_prev_actions.to(self.device),
                ).view(-1, 1, 1)
            consistency_loss = torch.nn.functional.mse_loss(pred, prev_pred)

        # Behavior-aligned proxies (scheme A):
        # - no_backward_proxy: penalize negative forward command components
        # - forward_drive_proxy: penalize insufficient positive forward command
        actions = self._cached_actions.to(self.device)
        forward_component = actions[..., 0]
        no_backward_loss = torch.relu(-forward_component).mean()
        forward_drive_loss = torch.relu(self.cfg.forward_drive_target - forward_component).mean()

        raw_delta_j = float(context.get("teacher_delta_j", 0.0))
        self._delta_j_ema = 0.97 * self._delta_j_ema + 0.03 * raw_delta_j
        centered = raw_delta_j - self._delta_j_ema
        self._delta_j_var_ema = 0.97 * self._delta_j_var_ema + 0.03 * (centered * centered)
        delta_std = (self._delta_j_var_ema + 1.0e-6) ** 0.5
        norm_delta_j = centered / delta_std

        student_updates = int(context.get("student_updates", 0))
        if student_updates <= self.cfg.outer_delta_warmup_updates:
            outer_coef_scale = 0.0
        else:
            progress = (student_updates - self.cfg.outer_delta_warmup_updates) / max(self.cfg.outer_delta_ramp_updates, 1)
            outer_coef_scale = float(min(max(progress, 0.0), 1.0))

        outer_delta_j = torch.tensor(norm_delta_j, device=self.device)

        # Step-3 paper-aligned outer objective:
        #   L_outer = - norm_delta_j * E_window[A_phy * A_phi]
        # where A_phy is the centered physical-score proxy and A_phi is the
        # centered teacher shaping output. The window estimate avoids the
        # num_envs=1 degeneracy where a single centered sample would always
        # produce zero advantage and zero correlation.
        if self._window_obs and self._window_actions and self._window_physical_scores:
            window_obs = torch.cat(self._window_obs, dim=0).to(self.device)
            window_actions = torch.cat(self._window_actions, dim=0).to(self.device)
            window_physical_score = torch.cat(self._window_physical_scores, dim=0).to(self.device).detach().view(-1, 1, 1)
            window_pred = self.model(window_obs, window_actions).view(-1, 1, 1)
        else:
            window_physical_score = self._cached_physical_score.to(self.device).detach().view_as(pred)
            window_pred = pred
        physical_adv = window_physical_score - window_physical_score.mean()
        teacher_adv = window_pred - window_pred.mean()
        advantage_corr = (physical_adv * teacher_adv).mean()
        outer_loss = -outer_delta_j * advantage_corr

        loss = (
            base_loss
            + self.cfg.consistency_coef * consistency_loss
            + self.cfg.no_backward_coef * no_backward_loss
            + self.cfg.forward_drive_coef * forward_drive_loss
            + (self.cfg.outer_delta_coef * outer_coef_scale) * outer_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()

        window_sample_count = len(self._window_physical_scores)
        self._window_obs.clear()
        self._window_actions.clear()
        self._window_physical_scores.clear()

        self.last_teacher_loss = float(loss.item())
        return {
            "teacher_shaping_mean": self.last_shaping_mean,
            "teacher_loss": self.last_teacher_loss,
            "teacher_grad_norm": float(grad_norm),
            "teacher_base_loss": float(base_loss.item()),
            "teacher_consistency_loss": float(consistency_loss.item()),
            "teacher_no_backward_loss": float(no_backward_loss.item()),
            "teacher_forward_drive_loss": float(forward_drive_loss.item()),
            "teacher_raw_delta_j": float(raw_delta_j),
            "teacher_norm_delta_j": float(outer_delta_j.item()),
            "teacher_outer_coef_scale": float(outer_coef_scale),
            "teacher_outer_loss": float(outer_loss.item()),
            "teacher_advantage_corr": float(advantage_corr.item()),
            "teacher_physical_adv_std": float(physical_adv.std(unbiased=False).item()),
            "teacher_shaping_adv_std": float(teacher_adv.std(unbiased=False).item()),
            "teacher_window_samples": float(window_sample_count),
            "teacher_delta_j": float(raw_delta_j),
        }
