"""Shield modules for the platoon training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.utils.math import quat_apply_inverse

from marl_platoon.trajectory import get_platoon_lane_change_reference, road_boundary_enforced, road_boundary_state

from marl_platoon.algorithms.pipeline import ShieldModule


@dataclass
class BadHeadingShieldCfg:
    enabled: bool = False
    d_crit: float = 0.50
    d_drop: float = 1.45
    brake_action: float = 0.0
    catchup_action: float = 0.35
    catchup_lateral_limit: float = -1.0
    catchup_centerline_limit: float = -1.0
    # Follow the same constant-time-headway spacing model used by reward and
    # completion checks.  A fixed 1.5 m trigger otherwise classifies the
    # normal moving gap (roughly 1.7 m) as a dropout.
    dynamic_catchup_gap: bool = True
    catchup_time_headway_s: float = 0.60
    catchup_gap_tolerance_m: float = 0.10
    delta_v: float = 0.05
    lateral_tol: float = 0.035
    lateral_crit: float = 0.55
    lateral_turn_gain: float = 0.30
    lateral_turn_clip: float = 0.08
    lateral_velocity_gain: float = 0.10
    centerline_turn_gain: float = 0.25
    centerline_turn_clip: float = 0.08
    centerline_turn_sign: float = 1.0
    boundary_turn_gain: float = 0.0
    boundary_turn_clip: float = 0.0
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
    forward_bias_leader_gain_scale: float = 1.0
    forward_bias_leader_clip_scale: float = 1.0
    # Deterministic clean-acquisition controller.  This deliberately lives in
    # the action projection so a non-converged policy cannot prevent the five
    # physical Ackermann cars from first experiencing the complete path.
    expert_path_tracking_enabled: bool = False
    expert_blend_ratio: float = 1.0
    expert_blend_end_ratio: float = 0.0
    expert_blend_hold_steps: int = 0
    expert_blend_decay_steps: int = 0
    expert_speed_action: float = 0.32
    expert_heading_gain: float = 1.40
    expert_cross_track_gain: float = 0.80
    expert_steering_clip: float = 0.40
    expert_wheelbase_m: float = 0.14325
    expert_max_steering_rad: float = 0.32


class BadHeadingShieldModule(ShieldModule):
    """Training-time distance and lane-center action projection.

    The class name is kept for config compatibility, but the implementation is
    now the paper-style safety shield:

    - if a follower is too close to its predecessor, project to a braking action;
    - if it is dropping out and not already faster than its predecessor, project
      to a mild catch-up action;
    - if it drifts laterally from the predecessor centerline, add a bounded
      steering correction. Two-dimensional actions are interpreted as
      ``[forward_speed, front_steering]``; legacy four-wheel actions retain their
      differential-drive interpretation.
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
        self.batch_trigger_count = 0
        self.catchup_count = 0
        self.brake_count = 0
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
        self.boundary_count = 0
        self.boundary_critical_count = 0
        self.boundary_turn_abs_sum = 0.0
        self.boundary_turn_samples = 0
        self.forward_bias_count = 0
        self.forward_bias_abs_sum = 0.0
        self.forward_bias_samples = 0
        self.last_expert_actions: torch.Tensor | None = None

    def _catchup_threshold(self, follower_speed: torch.Tensor) -> torch.Tensor:
        threshold = torch.full_like(follower_speed, float(self.cfg.d_drop))
        if self.cfg.dynamic_catchup_gap:
            threshold = (
                threshold
                + float(self.cfg.catchup_time_headway_s) * torch.clamp(follower_speed, min=0.0)
                + float(self.cfg.catchup_gap_tolerance_m)
            )
        return threshold

    def _env_local_y(self, asset_name: str) -> torch.Tensor:
        pos_y = self.env.scene[asset_name].data.root_pos_w[:, 1]
        env_origins = getattr(self.env.scene, "env_origins", None)
        if env_origins is not None:
            pos_y = pos_y - env_origins[:, 1].to(device=pos_y.device, dtype=pos_y.dtype)
        try:
            _, reference_y, _, _, _ = get_platoon_lane_change_reference(self.env, (asset_name,))
            pos_y = pos_y - reference_y[:, 0].to(device=pos_y.device, dtype=pos_y.dtype)
        except Exception:
            pass
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

    def _apply_expert_path_tracking(self, guarded: torch.Tensor) -> torch.Tensor:
        """Override 2-D actions with a shared-path Ackermann tracking law.

        Each car evaluates the same geometric path at its own world-space x.
        The action remains ``[rear-wheel speed, front-knuckle steering]``;
        therefore this controller never commands rear-wheel steering.
        """
        if not self.cfg.expert_path_tracking_enabled or guarded.shape[-1] != 2:
            return guarded

        active_robots = self.robots[: guarded.shape[1]]
        _, reference_y, reference_heading, reference_velocity_w, reference_yaw_rate = (
            get_platoon_lane_change_reference(self.env, active_robots)
        )
        actual_y = torch.stack(
            [self.env.scene[name].data.root_pos_w[:, 1] for name in active_robots], dim=1
        )
        env_origins = getattr(self.env.scene, "env_origins", None)
        if env_origins is not None:
            actual_y = actual_y - env_origins[:, 1].to(device=actual_y.device, dtype=actual_y.dtype).unsqueeze(1)

        quaternions = torch.stack(
            [self.env.scene[name].data.root_quat_w for name in active_robots], dim=1
        )
        qw, qx, qy, qz = quaternions.unbind(dim=-1)
        actual_heading = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        heading_error = torch.atan2(
            torch.sin(reference_heading - actual_heading),
            torch.cos(reference_heading - actual_heading),
        )
        cross_track_error = actual_y - reference_y

        reference_speed = torch.linalg.vector_norm(reference_velocity_w[..., :2], dim=-1).clamp_min(0.05)
        curvature = reference_yaw_rate / reference_speed
        feedforward_steering = torch.atan(float(self.cfg.expert_wheelbase_m) * curvature)
        desired_steering = (
            feedforward_steering
            + float(self.cfg.expert_heading_gain) * heading_error
            - float(self.cfg.expert_cross_track_gain) * cross_track_error
        )
        normalized_steering = desired_steering / max(float(self.cfg.expert_max_steering_rad), 1.0e-6)
        normalized_steering = torch.clamp(
            normalized_steering,
            min=-float(self.cfg.expert_steering_clip),
            max=float(self.cfg.expert_steering_clip),
        )

        expert_actions = guarded.clone()
        expert_actions[:, : len(active_robots), 0] = float(self.cfg.expert_speed_action)
        expert_actions[:, : len(active_robots), 1] = normalized_steering.to(dtype=guarded.dtype)
        self.last_expert_actions = expert_actions.detach().clone()
        blend = min(max(float(self.cfg.expert_blend_ratio), 0.0), 1.0)
        decay_steps = max(int(self.cfg.expert_blend_decay_steps), 0)
        if decay_steps > 0:
            elapsed = max(self.calls - max(int(self.cfg.expert_blend_hold_steps), 0), 0)
            progress = min(float(elapsed) / float(decay_steps), 1.0)
            end = min(max(float(self.cfg.expert_blend_end_ratio), 0.0), 1.0)
            blend = blend + progress * (end - blend)
        self._last_expert_blend_ratio = blend
        return torch.lerp(guarded, expert_actions, blend)

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
            catchup_threshold = self._catchup_threshold(foll_v)
            catchup_mask = (gap >= catchup_threshold) & (foll_v <= pred_v + self.cfg.delta_v) & (~brake_mask)
            if self.cfg.catchup_lateral_limit > 0.0:
                catchup_mask &= torch.abs(rel_local[:, 1]) <= float(self.cfg.catchup_lateral_limit)
            if self.cfg.catchup_centerline_limit > 0.0:
                catchup_mask &= torch.abs(self._env_local_y(foll_name)) <= float(self.cfg.catchup_centerline_limit)

            if brake_mask.any():
                if guarded.shape[-1] == 2:
                    guarded[brake_mask, agent_id, 0] = 0.0
                else:
                    guarded[brake_mask, agent_id, :] = float(self.cfg.brake_action)
            if catchup_mask.any():
                if guarded.shape[-1] == 2:
                    guarded[catchup_mask, agent_id, 0] = abs(float(self.cfg.catchup_action))
                else:
                    guarded[catchup_mask, agent_id, :] = float(self.cfg.catchup_action)

            critical_any |= brake_mask
            warn_any |= catchup_mask
            self.brake_count += int(brake_mask.sum().item())
            self.catchup_count += int(catchup_mask.sum().item())

            if guarded.shape[-1] in (2, 4) and self.cfg.lateral_turn_clip > 0.0 and self.cfg.lateral_turn_gain > 0.0:
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
                    if guarded.shape[-1] == 2:
                        # Positive y is left of the predecessor, so command a right turn.
                        guarded[:, agent_id, 1] -= turn
                    else:
                        guarded[:, agent_id, 0:2] -= turn.unsqueeze(-1)
                        guarded[:, agent_id, 2:4] += turn.unsqueeze(-1)
                    self.lateral_turn_abs_sum += float(turn[lateral_mask].abs().sum().item())
                    self.lateral_turn_samples += int(lateral_mask.sum().item())

                lateral_any |= lateral_mask
                lateral_critical_any |= lateral_critical_mask
                warn_any |= lateral_mask
                critical_any |= lateral_critical_mask

        if guarded.shape[-1] in (2, 4) and self.cfg.centerline_turn_gain > 0.0 and self.cfg.centerline_turn_clip > 0.0:
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
                centerline_turn = torch.clamp(
                    float(self.cfg.centerline_turn_sign) * gain * centerline_y,
                    min=-clip,
                    max=clip,
                )
                centerline_turn = torch.where(centerline_mask, centerline_turn, torch.zeros_like(centerline_turn))

                if centerline_mask.any():
                    if guarded.shape[-1] == 2:
                        guarded[:, agent_id, 1] -= centerline_turn
                    else:
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

        if (
            guarded.shape[-1] in (2, 4)
            and road_boundary_enforced(self.env)
            and self.cfg.boundary_turn_gain > 0.0
            and self.cfg.boundary_turn_clip > 0.0
        ):
            active_robots = self.robots[: guarded.shape[1]]
            exceedance, _, lower_y, upper_y = road_boundary_state(self.env, active_robots)
            command_term = self.env.command_manager._terms.get("base_velocity")
            command_cfg = command_term.cfg
            line_half = 0.5 * float(command_cfg.road_edge_line_width_m)
            near_edge = float(command_cfg.road_near_edge_y) - line_half
            far_edge = float(command_cfg.road_far_edge_y) + line_half
            margin = float(command_cfg.road_boundary_margin_m)
            near_pressure = torch.relu(upper_y - (near_edge - margin))
            far_pressure = torch.relu((far_edge + margin) - lower_y)
            boundary_signal = near_pressure - far_pressure
            boundary_mask = (near_pressure > 0.0) | (far_pressure > 0.0)
            boundary_critical = exceedance > 0.0
            boundary_turn = torch.clamp(
                float(self.cfg.boundary_turn_gain) * boundary_signal,
                min=-float(self.cfg.boundary_turn_clip),
                max=float(self.cfg.boundary_turn_clip),
            )
            boundary_turn = torch.where(boundary_mask, boundary_turn, torch.zeros_like(boundary_turn))
            if guarded.shape[-1] == 2:
                guarded[:, : len(active_robots), 1] -= boundary_turn
            else:
                guarded[:, : len(active_robots), 0:2] -= boundary_turn.unsqueeze(-1)
                guarded[:, : len(active_robots), 2:4] += boundary_turn.unsqueeze(-1)
            self.boundary_count += int(boundary_mask.sum().item())
            self.boundary_critical_count += int(boundary_critical.sum().item())
            self.boundary_turn_abs_sum += float(boundary_turn[boundary_mask].abs().sum().item())
            self.boundary_turn_samples += int(boundary_mask.sum().item())
            warn_any |= boundary_mask.any(dim=1)
            critical_any |= boundary_critical.any(dim=1)

        if guarded.shape[-1] in (2, 4) and self.cfg.forward_bias_gain > 0.0 and self.cfg.forward_bias_clip > 0.0:
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
                gain = float(self.cfg.forward_bias_gain)
                clip = float(self.cfg.forward_bias_clip)
                if agent_id == 0:
                    gain *= float(self.cfg.forward_bias_leader_gain_scale)
                    clip *= float(self.cfg.forward_bias_leader_clip_scale)
                speed_deficit = command_x - speed - float(self.cfg.forward_bias_speed_margin)
                bias = torch.clamp(
                    gain * speed_deficit,
                    min=0.0,
                    max=clip,
                )
                bias_mask = command_mask & (bias > 0.0)
                if agent_id > 0 and self.cfg.forward_bias_min_gap > 0.0:
                    pred = self.env.scene[self.robots[agent_id - 1]]
                    foll = self.env.scene[robot_name]
                    rel_world = pred.data.root_pos_w - foll.data.root_pos_w
                    rel_local = quat_apply_inverse(pred.data.root_quat_w, rel_world)
                    bias_mask &= rel_local[:, 0].abs() >= float(self.cfg.forward_bias_min_gap)
                if bias_mask.any():
                    if guarded.shape[-1] == 2:
                        guarded[bias_mask, agent_id, 0] += bias[bias_mask]
                    else:
                        # Legacy pre-adapter convention: more negative is forward.
                        guarded[bias_mask, agent_id, :] -= bias[bias_mask].unsqueeze(-1)
                    sample_count = int(bias_mask.sum().item())
                    self.forward_bias_count += sample_count
                    self.forward_bias_abs_sum += float(bias[bias_mask].abs().sum().item())
                    self.forward_bias_samples += sample_count

        # The expert is intentionally applied last: during clean acquisition it
        # is the authoritative controller, while the legacy learned-policy
        # shield remains unchanged when the flag is disabled.
        guarded = self._apply_expert_path_tracking(guarded)

        warn_n = int(warn_any.sum().item())
        critical_n = int(critical_any.sum().item())
        total_n = int(num_envs)

        self.warn_count += warn_n
        self.critical_count += critical_n
        trigger_n = int((warn_any | critical_any).sum().item())
        self.trigger_count += trigger_n
        self.batch_trigger_count += int(trigger_n > 0)
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
        boundary_turn_samples = max(self.boundary_turn_samples, 1)
        vehicle_samples = max(self.scale_samples * len(self.robots), 1)
        follower_samples = max(self.scale_samples * max(len(self.robots) - 1, 1), 1)
        forward_bias_samples = max(self.forward_bias_samples, 1)
        return {
            "shield_warn_rate": float(self.warn_count / samples),
            "shield_critical_rate": float(self.critical_count / samples),
            "shield_trigger_rate": float(self.trigger_count / samples),
            "shield_batch_trigger_rate": float(self.batch_trigger_count / calls),
            "shield_catchup_rate": float(self.catchup_count / follower_samples),
            "shield_brake_rate": float(self.brake_count / follower_samples),
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
            "shield_boundary_rate": float(self.boundary_count / vehicle_samples),
            "shield_boundary_critical_rate": float(self.boundary_critical_count / vehicle_samples),
            "shield_boundary_turn_mean": float(self.boundary_turn_abs_sum / boundary_turn_samples),
            "shield_forward_bias_rate": float(self.forward_bias_count / samples),
            "shield_forward_bias_mean": float(self.forward_bias_abs_sum / forward_bias_samples),
            "shield_expert_blend_ratio": float(getattr(self, "_last_expert_blend_ratio", 0.0)),
        }
