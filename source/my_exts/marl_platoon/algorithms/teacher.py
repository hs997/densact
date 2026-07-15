"""Teacher modules for the platoon training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from isaaclab.utils.math import quat_apply_inverse

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
    physical_cost_clip: float = 50.0
    target_adv_clip: float = 5.0
    outer_loss_clip: float = 10.0
    no_backward_coef: float = 0.0
    forward_drive_coef: float = 0.0
    forward_drive_target: float = 0.10
    outer_delta_coef: float = 0.02
    outer_delta_warmup_updates: int = 20
    outer_delta_ramp_updates: int = 50
    target_gap: float = 1.50
    headway: float = 0.60
    collision_gap: float = 0.30
    lambda_spacing: float = 1.0
    lambda_velocity: float = 0.5
    lambda_acceleration: float = 0.25
    lambda_jerk: float = 0.10
    lambda_overspeed: float = 0.25
    lambda_centerline: float = 1.0
    lambda_lateral: float = 0.5
    lambda_heading: float = 2.0
    lambda_backward: float = 1.0
    lambda_forward_deficit: float = 0.5
    lambda_collision: float = 10.0
    lambda_action_energy: float = 0.02


class RewardTeacherModule(TeacherModule):
    """Minimal trainable teacher adapter.

    This is not the full paper meta-gradient yet; it adds a stable teacher
    optimization loop so the module can learn rather than stay random.
    """

    def __init__(self, cfg: RewardTeacherCfg, env: Any | None = None):
        self.cfg = cfg
        self.env = env
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
        self.robots = ["robot", "robot_2", "robot_3", "robot_4", "robot_5"]
        self._prev_clean_vel: torch.Tensor | None = None
        self._prev_clean_acc: torch.Tensor | None = None
        self._prev_actions_for_score: torch.Tensor | None = None
        self._last_physical_context: dict[str, float] = {}

    def _compute_clean_physical_score(self, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Compute a clean-state platoon physical score during centralized training."""
        if actions.dim() == 2:
            if rewards.dim() == 3:
                num_agents = rewards.shape[1]
            else:
                num_agents = len(self.robots)
            actions_3d = actions.view(actions.shape[0], num_agents, -1)
        else:
            actions_3d = actions

        if self.env is None or not hasattr(self.env, "scene"):
            reward_term = rewards if rewards.dim() == 3 else rewards.view(actions_3d.shape[0], 1, 1)
            if reward_term.shape[1] == 1 and actions_3d.shape[1] > 1:
                reward_term = reward_term.repeat(1, actions_3d.shape[1], 1)
            proxy_cost = (-reward_term).mean().clamp_min(0.0)
            self._last_physical_context = {
                "physical_cost": float(proxy_cost.item()),
                "physical_spacing_cost": float(proxy_cost.item()),
                "physical_velocity_cost": 0.0,
                "physical_acceleration_cost": 0.0,
                "physical_jerk_cost": 0.0,
                "physical_overspeed_cost": 0.0,
                "physical_collision_cost": 0.0,
            }
            return (-proxy_cost).view(actions_3d.shape[0], actions_3d.shape[1], 1)

        num_envs = actions_3d.shape[0]
        num_agents = actions_3d.shape[1]
        physical_cost = torch.zeros(num_envs, num_agents, device=self.device)
        spacing_metric = torch.zeros(num_envs, device=self.device)
        velocity_metric = torch.zeros_like(spacing_metric)
        acceleration_metric = torch.zeros_like(spacing_metric)
        jerk_metric = torch.zeros_like(spacing_metric)
        overspeed_metric = torch.zeros_like(spacing_metric)
        lateral_metric = torch.zeros_like(spacing_metric)
        centerline_metric = torch.zeros_like(spacing_metric)
        heading_metric = torch.zeros_like(spacing_metric)
        backward_metric = torch.zeros_like(spacing_metric)
        forward_deficit_metric = torch.zeros_like(spacing_metric)
        collision_metric = torch.zeros_like(spacing_metric)

        clean_vel = []
        for name in self.robots[:num_agents]:
            clean_vel.append(self.env.scene[name].data.root_lin_vel_b[:, 0].to(self.device))
        clean_vel_t = torch.stack(clean_vel, dim=1)
        if self._prev_clean_vel is None or self._prev_clean_vel.shape != clean_vel_t.shape:
            clean_acc_t = torch.zeros_like(clean_vel_t)
        else:
            clean_acc_t = clean_vel_t - self._prev_clean_vel
        if self._prev_clean_acc is None or self._prev_clean_acc.shape != clean_acc_t.shape:
            clean_jerk_t = torch.zeros_like(clean_acc_t)
        else:
            clean_jerk_t = clean_acc_t - self._prev_clean_acc

        try:
            command = self.env.command_manager.get_command("base_velocity").to(self.device)
            target_speed = command[:, 0].view(num_envs, 1)
        except Exception:
            target_speed = torch.zeros(num_envs, 1, device=self.device)

        target_heading = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        for idx, name in enumerate(self.robots[:num_agents]):
            robot = self.env.scene[name]
            speed_x = robot.data.root_lin_vel_b[:, 0].to(self.device)
            root_pos = robot.data.root_pos_w.to(self.device)
            if hasattr(self.env.scene, "env_origins"):
                centerline_y = root_pos[:, 1] - self.env.scene.env_origins.to(self.device)[:, 1]
            else:
                centerline_y = root_pos[:, 1]
            centerline_i = centerline_y.pow(2)
            heading_vec = torch.nn.functional.normalize(
                torch.nan_to_num(robot.data.root_quat_w.to(self.device), nan=0.0), dim=-1
            )
            from isaaclab.utils.math import quat_apply
            world_heading = quat_apply(heading_vec, target_heading)
            heading_i = torch.relu(-world_heading[:, 0]).pow(2)
            backward_i = torch.relu(-speed_x).pow(2)
            forward_deficit_i = torch.relu(target_speed.squeeze(1) - speed_x).pow(2)
            physical_cost[:, idx] = physical_cost[:, idx] + (
                self.cfg.lambda_centerline * centerline_i
                + self.cfg.lambda_heading * heading_i
                + self.cfg.lambda_backward * backward_i
                + self.cfg.lambda_forward_deficit * forward_deficit_i
            )
            centerline_metric = centerline_metric + centerline_i
            heading_metric = heading_metric + heading_i
            backward_metric = backward_metric + backward_i
            forward_deficit_metric = forward_deficit_metric + forward_deficit_i

        for idx in range(1, min(len(self.robots), num_agents)):
            pred = self.env.scene[self.robots[idx - 1]]
            foll = self.env.scene[self.robots[idx]]
            rel_world = pred.data.root_pos_w.to(self.device) - foll.data.root_pos_w.to(self.device)
            rel_local = quat_apply_inverse(pred.data.root_quat_w.to(self.device), rel_world)
            gap = rel_local[:, 0].abs()
            lateral = rel_local[:, 1]
            vel_err = pred.data.root_lin_vel_b[:, :2].to(self.device) - foll.data.root_lin_vel_b[:, :2].to(self.device)
            acc_err = clean_acc_t[:, idx - 1] - clean_acc_t[:, idx]
            follower_speed = torch.clamp(foll.data.root_lin_vel_b[:, 0].to(self.device), min=0.0)
            desired_gap = self.cfg.target_gap + self.cfg.headway * follower_speed
            overspeed = torch.relu(follower_speed - target_speed.squeeze(1))
            spacing_i = (gap - desired_gap).pow(2)
            velocity_i = vel_err.pow(2).sum(dim=-1)
            acceleration_i = acc_err.pow(2)
            jerk_i = clean_jerk_t[:, idx].pow(2)
            overspeed_i = overspeed.pow(2)
            lateral_i = lateral.pow(2)
            collision_i = (gap < self.cfg.collision_gap).float()

            physical_cost[:, idx] = (
                self.cfg.lambda_spacing * spacing_i
                + self.cfg.lambda_velocity * velocity_i
                + self.cfg.lambda_acceleration * acceleration_i
                + self.cfg.lambda_jerk * jerk_i
                + self.cfg.lambda_overspeed * overspeed_i
                + self.cfg.lambda_lateral * lateral_i
                + self.cfg.lambda_collision * collision_i
            )
            spacing_metric = spacing_metric + spacing_i
            velocity_metric = velocity_metric + velocity_i
            acceleration_metric = acceleration_metric + acceleration_i
            jerk_metric = jerk_metric + jerk_i
            overspeed_metric = overspeed_metric + overspeed_i
            lateral_metric = lateral_metric + lateral_i
            collision_metric = collision_metric + collision_i

        action_energy = actions_3d.pow(2).mean(dim=-1)
        if self._prev_actions_for_score is None or self._prev_actions_for_score.shape != actions_3d.shape:
            action_jerk = torch.zeros_like(action_energy)
        else:
            action_jerk = (actions_3d - self._prev_actions_for_score).pow(2).mean(dim=-1)
        physical_cost = (
            physical_cost
            + self.cfg.lambda_action_energy * action_energy
            + self.cfg.lambda_jerk * action_jerk
        )
        if num_agents > 1:
            physical_cost[:, 0] = physical_cost[:, 1:].mean(dim=1)
        physical_cost = torch.clamp(physical_cost, min=0.0, max=self.cfg.physical_cost_clip)

        self._prev_clean_vel = clean_vel_t.detach().clone()
        self._prev_clean_acc = clean_acc_t.detach().clone()
        self._prev_actions_for_score = actions_3d.detach().clone()
        pair_count = float(max(min(len(self.robots), num_agents) - 1, 1))
        self._last_physical_context = {
            "physical_cost": float(physical_cost.mean().item()),
            "physical_spacing_cost": float((spacing_metric / pair_count).mean().item()),
            "physical_velocity_cost": float((velocity_metric / pair_count).mean().item()),
            "physical_acceleration_cost": float((acceleration_metric / pair_count).mean().item()),
            "physical_jerk_cost": float(((jerk_metric / pair_count) + action_jerk.mean(dim=1)).mean().item()),
            "physical_overspeed_cost": float((overspeed_metric / pair_count).mean().item()),
            "physical_centerline_cost": float((centerline_metric / float(max(num_agents, 1))).mean().item()),
            "physical_lateral_cost": float((lateral_metric / pair_count).mean().item()),
            "physical_heading_cost": float((heading_metric / float(max(num_agents, 1))).mean().item()),
            "physical_backward_cost": float((backward_metric / float(max(num_agents, 1))).mean().item()),
            "physical_forward_deficit_cost": float((forward_deficit_metric / float(max(num_agents, 1))).mean().item()),
            "physical_collision_cost": float((collision_metric / pair_count).mean().item()),
        }
        return (-physical_cost).view(num_envs, num_agents, 1)

    def get_physical_attack_context(self) -> dict[str, float]:
        return dict(self._last_physical_context)

    def shape_reward(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            num_envs, num_agents, obs_dim = obs.shape
            act_dim = actions.shape[-1]
            teacher_obs = obs.reshape(num_envs * num_agents, obs_dim)
            teacher_actions = actions.reshape(num_envs * num_agents, act_dim)
        else:
            num_envs, obs_dim = obs.shape
            num_agents = 1
            teacher_obs = obs
            teacher_actions = actions.reshape(num_envs, -1)
        with torch.no_grad():
            shaping = self.model(teacher_obs.to(self.device), teacher_actions.to(self.device)).view(num_envs, num_agents, 1)
            if self.cfg.shaping_clip > 0.0:
                shaping = torch.clamp(shaping, -self.cfg.shaping_clip, self.cfg.shaping_clip)
        self.last_shaping_mean = float(shaping.mean().item())
        if self._cached_obs is not None:
            self._cached_prev_obs = self._cached_obs.detach().clone()
        if self._cached_actions is not None:
            self._cached_prev_actions = self._cached_actions.detach().clone()
        self._cached_obs = teacher_obs.detach().clone()
        self._cached_actions = teacher_actions.detach().clone()
        self._cached_rewards = rewards.reshape(num_envs * num_agents, 1, 1).detach().clone()
        # Clean physical score for the upper-level Teacher. This uses the
        # uncompromised IsaacLab scene state available during centralized
        # training, so it is not tied to the corrupted student observation.
        physical_score = self._compute_clean_physical_score(actions.to(self.device), rewards.to(self.device))
        physical_score_flat = physical_score.reshape(num_envs * num_agents, 1, 1)
        self._cached_physical_score = physical_score_flat.detach().clone()

        # Accumulate the teacher-update window for Step-3 meta attribution.
        # The actual correlation is computed in maybe_update() and then the
        # window is cleared, so each teacher update receives delayed credit
        # from the shaping behavior applied over the preceding window.
        self._window_obs.append(teacher_obs.detach().clone())
        self._window_actions.append(teacher_actions.detach().clone())
        self._window_physical_scores.append(physical_score_flat.detach().clone())

        if self.cfg.shaping_coef <= 0.0:
            return rewards
        delta = self.cfg.shaping_coef * shaping
        if self.cfg.action_penalty_coef > 0.0:
            action_penalty = actions.pow(2).mean(dim=-1, keepdim=True)
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

        physical_score = self._cached_physical_score.to(self.device)
        if self._reward_ema is None:
            self._reward_ema = physical_score.mean(dim=0, keepdim=True).detach()
        self._reward_ema = (
            self.cfg.reward_ema_tau * self._reward_ema
            + (1.0 - self.cfg.reward_ema_tau) * physical_score.mean(dim=0, keepdim=True).detach()
        )

        # Physics-aware objective: the Teacher learns F_phi so that the lower-level
        # shaped-reward advantage is aligned with the clean physical objective.
        target_adv = physical_score - self._reward_ema.mean(dim=1, keepdim=True)
        target_adv = target_adv / target_adv.std(unbiased=False).clamp_min(1.0e-5)
        target_adv = torch.clamp(target_adv, -self.cfg.target_adv_clip, self.cfg.target_adv_clip)
        base_loss = torch.nn.functional.mse_loss(pred, target_adv)

        # Paper-aligned teacher update uses the physical regret signal and the
        # shaped reward correlation. We keep only the minimum training bridge
        # terms needed for stable optimization.
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
        outer_loss = torch.clamp(outer_loss, -self.cfg.outer_loss_clip, self.cfg.outer_loss_clip)

        loss = base_loss + (self.cfg.outer_delta_coef * outer_coef_scale) * outer_loss

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
