"""Shield modules for the platoon training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.utils.math import quat_apply_inverse

from marl_platoon.algorithms.pipeline import ShieldModule


@dataclass
class BadHeadingShieldCfg:
    enabled: bool = False
    d_crit: float = 0.50
    d_drop: float = 1.45
    brake_action: float = 0.0
    catchup_action: float = -0.35
    catchup_lateral_limit: float = -1.0
    catchup_centerline_limit: float = -1.0
    delta_v: float = 0.05
    lateral_tol: float = 0.035
    lateral_crit: float = 0.55
    lateral_turn_gain: float = 0.30
    lateral_turn_clip: float = 0.08
    lateral_velocity_gain: float = 0.10
    centerline_turn_gain: float = 0.25
    centerline_turn_clip: float = 0.08
    first_follower_lateral_gain_scale: float = 1.0
    first_follower_lateral_clip_scale: float = 1.0
    first_follower_lateral_clip_max: float = 0.18
    first_follower_centerline_gain: float = 0.0
    first_follower_centerline_clip: float = 0.0
    pair2_lateral_gain_scale: float = 1.0
    pair2_lateral_clip_scale: float = 1.0
    pair2_lateral_clip_max: float = 0.18
    pair3_lateral_gain_scale: float = 1.0
    pair3_lateral_clip_scale: float = 1.0
    pair3_lateral_clip_max: float = 0.18
    pair4_lateral_gain_scale: float = 1.0
    pair4_lateral_clip_scale: float = 1.0
    pair4_lateral_clip_max: float = 0.18
    forward_bias_gain: float = 0.0
    forward_bias_clip: float = 0.0
    forward_bias_speed_margin: float = 0.02
    forward_bias_min_command: float = 0.0
    forward_bias_min_gap: float = 0.75


class BadHeadingShieldModule(ShieldModule):
    """Training-time distance and lane-center action projection.

    The class name is kept for config compatibility, but the implementation is
    now the paper-style safety shield:

    - if a follower is too close to its predecessor, project to a braking action;
    - if it is dropping out and not already faster than its predecessor, project
      to a mild catch-up action;
    - if it drifts laterally from the predecessor centerline, add a bounded
      differential-drive steering correction before the URDF wheel-axis adapter.
    """

    def __init__(self, env, cfg: BadHeadingShieldCfg | None = None):
        self.env = env
        self.cfg = cfg or BadHeadingShieldCfg()
        self.robots = ["robot", "robot_2", "robot_3", "robot_4", "robot_5"]

        # Runtime stats (aggregated over calls) for logging.
        self.calls = 0
        self.warn_count = 0
        self.critical_count = 0
        self.trigger_count = 0
        self.scale_sum = 0.0
        self.scale_samples = 0
        self.lateral_count = 0
        self.lateral_critical_count = 0
        self.lateral_turn_abs_sum = 0.0
        self.lateral_turn_samples = 0
        self.centerline_count = 0
        self.centerline_turn_abs_sum = 0.0
        self.centerline_turn_samples = 0
        self.first_follower_centerline_count = 0
        self.first_follower_centerline_turn_abs_sum = 0.0
        self.first_follower_centerline_turn_samples = 0
        self.forward_bias_count = 0
        self.forward_bias_abs_sum = 0.0
        self.forward_bias_samples = 0

    def _env_local_y(self, asset_name: str) -> torch.Tensor:
        pos_y = self.env.scene[asset_name].data.root_pos_w[:, 1]
        env_origins = getattr(self.env.scene, "env_origins", None)
        if env_origins is not None:
            pos_y = pos_y - env_origins[:, 1].to(device=pos_y.device, dtype=pos_y.dtype)
        return pos_y

    def _brake_mask_for_agent(self, agent_id: int) -> torch.Tensor:
        mask = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        if agent_id <= 0 or agent_id >= len(self.robots):
            return mask
        pred = self.env.scene[self.robots[agent_id - 1]]
        foll = self.env.scene[self.robots[agent_id]]
        rel_world = pred.data.root_pos_w - foll.data.root_pos_w
        rel_local = quat_apply_inverse(pred.data.root_quat_w, rel_world)
        return rel_local[:, 0].abs() <= self.cfg.d_crit

    def project_action(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if not self.cfg.enabled:
            return actions
        if actions.dim() != 3:
            return actions

        guarded = actions.clone()
        num_envs = guarded.shape[0]
        warn_any = torch.zeros(num_envs, device=self.env.device, dtype=torch.bool)
        critical_any = torch.zeros_like(warn_any)
        lateral_any = torch.zeros_like(warn_any)
        lateral_critical_any = torch.zeros_like(warn_any)
        centerline_any = torch.zeros_like(warn_any)

        for agent_id in range(1, min(len(self.robots), guarded.shape[1])):
            pred_name = self.robots[agent_id - 1]
            foll_name = self.robots[agent_id]
            pred = self.env.scene[pred_name]
            foll = self.env.scene[foll_name]

            rel_world = pred.data.root_pos_w - foll.data.root_pos_w
            rel_local = quat_apply_inverse(pred.data.root_quat_w, rel_world)
            gap = rel_local[:, 0].abs()
            pred_v = pred.data.root_lin_vel_b[:, 0]
            foll_v = foll.data.root_lin_vel_b[:, 0]

            brake_mask = gap <= self.cfg.d_crit
            catchup_mask = (gap >= self.cfg.d_drop) & (foll_v <= pred_v + self.cfg.delta_v) & (~brake_mask)
            if self.cfg.catchup_lateral_limit > 0.0:
                catchup_mask &= torch.abs(rel_local[:, 1]) <= float(self.cfg.catchup_lateral_limit)
            if self.cfg.catchup_centerline_limit > 0.0:
                catchup_mask &= torch.abs(self._env_local_y(foll_name)) <= float(self.cfg.catchup_centerline_limit)

            if brake_mask.any():
                guarded[brake_mask, agent_id, :] = float(self.cfg.brake_action)
            if catchup_mask.any():
                guarded[catchup_mask, agent_id, :] = float(self.cfg.catchup_action)

            critical_any |= brake_mask
            warn_any |= catchup_mask

            if guarded.shape[-1] == 4 and self.cfg.lateral_turn_clip > 0.0 and self.cfg.lateral_turn_gain > 0.0:
                follower_from_pred = quat_apply_inverse(
                    pred.data.root_quat_w,
                    foll.data.root_pos_w - pred.data.root_pos_w,
                )
                lateral_offset = follower_from_pred[:, 1]
                lateral_rate = torch.zeros_like(lateral_offset)
                pred_vel_w = getattr(pred.data, "root_lin_vel_w", None)
                foll_vel_w = getattr(foll.data, "root_lin_vel_w", None)
                if pred_vel_w is not None and foll_vel_w is not None:
                    rel_vel_local = quat_apply_inverse(pred.data.root_quat_w, foll_vel_w - pred_vel_w)
                    lateral_rate = rel_vel_local[:, 1]

                lateral_signal = lateral_offset + float(self.cfg.lateral_velocity_gain) * lateral_rate
                lateral_mask = (torch.abs(lateral_offset) > self.cfg.lateral_tol) & (~brake_mask)
                lateral_critical_mask = torch.abs(lateral_offset) > self.cfg.lateral_crit
                turn_gain = float(self.cfg.lateral_turn_gain)
                turn_clip = float(self.cfg.lateral_turn_clip)
                if agent_id == 1:
                    turn_gain *= float(self.cfg.first_follower_lateral_gain_scale)
                    turn_clip = min(
                        float(self.cfg.first_follower_lateral_clip_max),
                        turn_clip * float(self.cfg.first_follower_lateral_clip_scale),
                    )
                if agent_id == 2:  # pair_2 follower: robot_3 relative to robot_2
                    turn_gain *= float(self.cfg.pair2_lateral_gain_scale)
                    turn_clip = min(float(self.cfg.pair2_lateral_clip_max), turn_clip * float(self.cfg.pair2_lateral_clip_scale))
                if agent_id == 3:  # pair_3 follower: robot_4 relative to robot_3
                    turn_gain *= float(self.cfg.pair3_lateral_gain_scale)
                    turn_clip = min(float(self.cfg.pair3_lateral_clip_max), turn_clip * float(self.cfg.pair3_lateral_clip_scale))
                if agent_id == 4:  # pair_4 follower: robot_5 relative to robot_4
                    turn_gain *= float(self.cfg.pair4_lateral_gain_scale)
                    turn_clip = min(float(self.cfg.pair4_lateral_clip_max), turn_clip * float(self.cfg.pair4_lateral_clip_scale))
                turn_signal = turn_gain * lateral_signal
                turn = torch.clamp(turn_signal, min=-turn_clip, max=turn_clip)
                turn = torch.where(lateral_mask, turn, torch.zeros_like(turn))

                if lateral_mask.any():
                    # Pre-adapter convention: more-negative same-sign wheel
                    # commands drive forward. Positive lateral offset means the
                    # follower is left of its predecessor, so left wheels get
                    # more forward command and right wheels less forward command.
                    guarded[:, agent_id, 0:2] -= turn.unsqueeze(-1)
                    guarded[:, agent_id, 2:4] += turn.unsqueeze(-1)
                    self.lateral_turn_abs_sum += float(turn[lateral_mask].abs().sum().item())
                    self.lateral_turn_samples += int(lateral_mask.sum().item())

                lateral_any |= lateral_mask
                lateral_critical_any |= lateral_critical_mask
                warn_any |= lateral_mask
                critical_any |= lateral_critical_mask

        if guarded.shape[-1] == 4 and self.cfg.centerline_turn_gain > 0.0 and self.cfg.centerline_turn_clip > 0.0:
            for agent_id in range(min(len(self.robots), guarded.shape[1])):
                gain = float(self.cfg.centerline_turn_gain)
                clip = float(self.cfg.centerline_turn_clip)
                if agent_id == 1:
                    gain = float(self.cfg.first_follower_centerline_gain) if self.cfg.first_follower_centerline_gain > 0.0 else gain
                    clip = float(self.cfg.first_follower_centerline_clip) if self.cfg.first_follower_centerline_clip > 0.0 else clip
                if gain <= 0.0 or clip <= 0.0:
                    continue

                centerline_y = self._env_local_y(self.robots[agent_id])
                brake_mask = self._brake_mask_for_agent(agent_id)
                centerline_mask = (torch.abs(centerline_y) > self.cfg.lateral_tol) & (~brake_mask)
                centerline_turn = torch.clamp(gain * centerline_y, min=-clip, max=clip)
                centerline_turn = torch.where(centerline_mask, centerline_turn, torch.zeros_like(centerline_turn))

                if centerline_mask.any():
                    guarded[:, agent_id, 0:2] -= centerline_turn.unsqueeze(-1)
                    guarded[:, agent_id, 2:4] += centerline_turn.unsqueeze(-1)
                    sample_count = int(centerline_mask.sum().item())
                    turn_abs_sum = float(centerline_turn[centerline_mask].abs().sum().item())
                    self.centerline_count += sample_count
                    self.centerline_turn_abs_sum += turn_abs_sum
                    self.centerline_turn_samples += sample_count
                    if agent_id == 1:
                        self.first_follower_centerline_count += sample_count
                        self.first_follower_centerline_turn_abs_sum += turn_abs_sum
                        self.first_follower_centerline_turn_samples += sample_count

                centerline_any |= centerline_mask
            warn_any |= centerline_any

        if guarded.shape[-1] == 4 and self.cfg.forward_bias_gain > 0.0 and self.cfg.forward_bias_clip > 0.0:
            try:
                command_x = self.env.command_manager.get_command("base_velocity")[:, 0].to(
                    device=self.env.device,
                    dtype=guarded.dtype,
                )
            except Exception:
                command_x = torch.zeros(num_envs, device=self.env.device, dtype=guarded.dtype)
            command_mask = command_x >= float(self.cfg.forward_bias_min_command)
            for agent_id in range(min(len(self.robots), guarded.shape[1])):
                robot_name = self.robots[agent_id]
                speed = self.env.scene[robot_name].data.root_lin_vel_b[:, 0].to(
                    device=self.env.device,
                    dtype=guarded.dtype,
                )
                speed_deficit = command_x - speed - float(self.cfg.forward_bias_speed_margin)
                bias = torch.clamp(
                    float(self.cfg.forward_bias_gain) * speed_deficit,
                    min=0.0,
                    max=float(self.cfg.forward_bias_clip),
                )
                bias_mask = command_mask & (bias > 0.0)
                if agent_id > 0 and self.cfg.forward_bias_min_gap > 0.0:
                    pred = self.env.scene[self.robots[agent_id - 1]]
                    foll = self.env.scene[robot_name]
                    rel_world = pred.data.root_pos_w - foll.data.root_pos_w
                    rel_local = quat_apply_inverse(pred.data.root_quat_w, rel_world)
                    bias_mask &= rel_local[:, 0].abs() >= float(self.cfg.forward_bias_min_gap)
                if bias_mask.any():
                    # Pre-adapter convention: equal negative semantic wheel commands
                    # drive straight forward. Subtracting the same bias from all
                    # wheels raises longitudinal authority without changing turn.
                    guarded[bias_mask, agent_id, :] -= bias[bias_mask].unsqueeze(-1)
                    sample_count = int(bias_mask.sum().item())
                    self.forward_bias_count += sample_count
                    self.forward_bias_abs_sum += float(bias[bias_mask].abs().sum().item())
                    self.forward_bias_samples += sample_count

        warn_n = int(warn_any.sum().item())
        critical_n = int(critical_any.sum().item())
        total_n = int(num_envs)

        self.warn_count += warn_n
        self.critical_count += critical_n
        self.trigger_count += int((warn_n + critical_n) > 0)
        self.scale_samples += total_n
        self.scale_sum += float(guarded.abs().mean().item()) * total_n
        self.lateral_count += int(lateral_any.sum().item())
        self.lateral_critical_count += int(lateral_critical_any.sum().item())
        return guarded

    def get_stats(self) -> dict[str, float]:
        calls = max(self.calls, 1)
        samples = max(self.scale_samples, 1)
        lateral_turn_samples = max(self.lateral_turn_samples, 1)
        all_centerline_turn_samples = max(self.centerline_turn_samples, 1)
        first_centerline_turn_samples = max(self.first_follower_centerline_turn_samples, 1)
        forward_bias_samples = max(self.forward_bias_samples, 1)
        return {
            "shield_warn_rate": float(self.warn_count / samples),
            "shield_critical_rate": float(self.critical_count / samples),
            "shield_trigger_rate": float(self.trigger_count / calls),
            "shield_scale_mean": float(self.scale_sum / samples),
            "shield_lateral_rate": float(self.lateral_count / samples),
            "shield_lateral_critical_rate": float(self.lateral_critical_count / samples),
            "shield_lateral_turn_mean": float(self.lateral_turn_abs_sum / lateral_turn_samples),
            "shield_centerline_rate": float(self.centerline_count / samples),
            "shield_centerline_turn_mean": float(self.centerline_turn_abs_sum / all_centerline_turn_samples),
            "shield_first_follower_centerline_rate": float(self.first_follower_centerline_count / samples),
            "shield_first_follower_centerline_turn_mean": float(
                self.first_follower_centerline_turn_abs_sum / first_centerline_turn_samples
            ),
            "shield_forward_bias_rate": float(self.forward_bias_count / samples),
            "shield_forward_bias_mean": float(self.forward_bias_abs_sum / forward_bias_samples),
        }
