"""Temporal and geometric lane-change references shared by the platoon stack."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse


PLATOON_ASSET_NAMES = ("robot", "robot_2", "robot_3", "robot_4", "robot_5")


def quintic_lane_change_profile(
    elapsed_s: torch.Tensor,
    duration_s: float,
    start_y: float,
    end_y: float,
    speed_x_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute progress, lateral position/velocity, heading, and yaw rate."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    u = torch.clamp(elapsed_s / duration_s, min=0.0, max=1.0)
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    smooth = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    smooth_d1 = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
    smooth_d2 = 60.0 * u - 180.0 * u2 + 120.0 * u3

    delta_y = float(end_y - start_y)
    speed_x_w = torch.clamp(speed_x_w, min=1.0e-4)
    speed_y_w = delta_y * smooth_d1 / duration_s
    accel_y_w = delta_y * smooth_d2 / (duration_s * duration_s)
    reference_y = float(start_y) + delta_y * smooth
    reference_heading = torch.atan2(speed_y_w, speed_x_w)
    reference_yaw_rate = speed_x_w * accel_y_w / (speed_x_w.square() + speed_y_w.square())
    return u, reference_y, speed_y_w, reference_heading, reference_yaw_rate


def quintic_lane_change_spatial_profile(
    local_x: torch.Tensor,
    start_x: float,
    length_m: float,
    start_y: float,
    end_y: float,
    speed_x_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one geometric quintic lane-change path at arbitrary vehicle x positions."""

    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    u = torch.clamp((local_x - float(start_x)) / float(length_m), min=0.0, max=1.0)
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    smooth = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    smooth_d1 = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
    smooth_d2 = 60.0 * u - 180.0 * u2 + 120.0 * u3

    delta_y = float(end_y - start_y)
    dy_dx = delta_y * smooth_d1 / float(length_m)
    d2y_dx2 = delta_y * smooth_d2 / (float(length_m) * float(length_m))
    reference_y = float(start_y) + delta_y * smooth
    reference_heading = torch.atan(dy_dx)
    speed_x_w = torch.clamp(speed_x_w, min=1.0e-4)
    speed_y_w = speed_x_w * dy_dx
    reference_yaw_rate = speed_x_w * d2y_dx2 / (1.0 + dy_dx.square())
    return u, reference_y, speed_y_w, reference_heading, reference_yaw_rate


def _env_local_position(env, asset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    pos = env.scene[asset_name].data.root_pos_w
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        origins = env_origins.to(device=pos.device, dtype=pos.dtype)
        pos_x = pos_x - origins[:, 0]
        pos_y = pos_y - origins[:, 1]
    return pos_x, pos_y


def _lane_change_term(env):
    return getattr(getattr(env, "command_manager", None), "_terms", {}).get("base_velocity")


class LaneChangeVelocityCommand(UniformVelocityCommand):
    """Uniform longitudinal speed plus a smooth world-frame lane-change tangent."""

    cfg: "LaneChangeVelocityCommandCfg"

    def __init__(self, cfg: "LaneChangeVelocityCommandCfg", env):
        if cfg.lane_change_duration_s <= 0.0:
            raise ValueError("lane_change_duration_s must be positive")
        if cfg.lane_change_length_m <= 0.0:
            raise ValueError("lane_change_length_m must be positive")
        super().__init__(cfg, env)
        self.nominal_speed_w = torch.zeros(self.num_envs, device=self.device)
        self.reference_progress = torch.zeros(self.num_envs, device=self.device)
        self.reference_y = torch.full((self.num_envs,), float(cfg.lane_start_y), device=self.device)
        self.reference_y_velocity_w = torch.zeros(self.num_envs, device=self.device)
        self.reference_heading = torch.zeros(self.num_envs, device=self.device)
        self.reference_yaw_rate = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.nominal_speed_w[env_ids] = self.vel_command_b[env_ids, 0]

    def _update_command(self):
        if not self.cfg.lane_change_enabled:
            super()._update_command()
            self.reference_progress.zero_()
            self.reference_y.fill_(float(self.cfg.lane_start_y))
            self.reference_y_velocity_w.zero_()
            self.reference_heading.zero_()
            self.reference_yaw_rate.zero_()
            return

        speed_x_w = torch.clamp(self.nominal_speed_w, min=1.0e-4)
        if self.cfg.lane_change_spatial:
            leader_x, _ = _env_local_position(self._env, self.cfg.asset_name)
            u, reference_y, speed_y_w, reference_heading, reference_yaw_rate = (
                quintic_lane_change_spatial_profile(
                    leader_x,
                    float(self.cfg.lane_change_start_x),
                    float(self.cfg.lane_change_length_m),
                    float(self.cfg.lane_start_y),
                    float(self.cfg.lane_end_y),
                    speed_x_w,
                )
            )
        else:
            duration = float(self.cfg.lane_change_duration_s)
            elapsed = self._env.episode_length_buf.to(dtype=torch.float32) * float(self._env.step_dt)
            u, reference_y, speed_y_w, reference_heading, reference_yaw_rate = quintic_lane_change_profile(
                elapsed,
                duration,
                float(self.cfg.lane_start_y),
                float(self.cfg.lane_end_y),
                speed_x_w,
            )

        self.reference_progress.copy_(u)
        self.reference_y.copy_(reference_y)
        self.reference_y_velocity_w.copy_(speed_y_w)
        self.reference_heading.copy_(reference_heading)
        self.reference_yaw_rate.copy_(reference_yaw_rate)

        # The Ackermann platform is non-holonomic. Give it a feasible tangent-speed/yaw-rate
        # command; path-relative y/heading errors close the spatial loop instead of requesting
        # an unattainable body-frame lateral velocity.
        tangent_speed = torch.sqrt(speed_x_w.square() + speed_y_w.square())
        self.vel_command_b[:, 0].copy_(tangent_speed)
        self.vel_command_b[:, 1].zero_()
        self.vel_command_b[:, 2].copy_(self.reference_yaw_rate)


@configclass
class LaneChangeVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for a default-neutral smooth lane-change command."""

    class_type: type = LaneChangeVelocityCommand
    lane_change_enabled: bool = False
    lane_change_spatial: bool = False
    lane_start_y: float = 0.0
    lane_end_y: float = -1.20
    lane_change_duration_s: float = 20.0
    lane_change_start_x: float = -8.75
    lane_change_length_m: float = 7.50
    enforce_road_boundaries: bool = False
    road_near_edge_y: float = 0.47
    road_far_edge_y: float = -1.72
    road_edge_line_width_m: float = 0.026
    vehicle_length_m: float = 0.240
    vehicle_width_m: float = 0.200
    road_boundary_margin_m: float = 0.05
    road_boundary_termination_exceedance_m: float = 0.10
    completion_progress: float = 0.99
    completion_lateral_tolerance_m: float = 0.12
    completion_heading_tolerance_rad: float = 0.12
    completion_gap_tolerance_m: float = 0.35
    completion_pair_lateral_tolerance_m: float = 0.20
    completion_dwell_s: float = 2.0


def get_lane_change_reference(env):
    """Return `(progress, y, heading, velocity_w, yaw_rate)` for every environment."""

    term = _lane_change_term(env)
    if term is not None and bool(getattr(getattr(term, "cfg", None), "lane_change_enabled", False)):
        speed_x_w = term.nominal_speed_w
        velocity_w = torch.stack(
            (speed_x_w, term.reference_y_velocity_w, torch.zeros_like(speed_x_w)), dim=-1
        )
        return (
            term.reference_progress,
            term.reference_y,
            term.reference_heading,
            velocity_w,
            term.reference_yaw_rate,
        )

    zeros = torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command("base_velocity")
    velocity_w = torch.stack((command[:, 0], zeros, zeros), dim=-1)
    return zeros, zeros, zeros, velocity_w, zeros


def get_platoon_lane_change_reference(env, asset_names=PLATOON_ASSET_NAMES):
    """Return per-vehicle spatial references with shape ``[env, vehicle, ...]``."""

    names = tuple(asset_names)
    term = _lane_change_term(env)
    enabled = term is not None and bool(getattr(getattr(term, "cfg", None), "lane_change_enabled", False))
    if enabled and bool(getattr(term.cfg, "lane_change_spatial", False)):
        local_x = torch.stack([_env_local_position(env, name)[0] for name in names], dim=1)
        speed_x_w = term.nominal_speed_w.view(-1, 1).expand_as(local_x)
        progress, reference_y, speed_y_w, heading, yaw_rate = quintic_lane_change_spatial_profile(
            local_x,
            float(term.cfg.lane_change_start_x),
            float(term.cfg.lane_change_length_m),
            float(term.cfg.lane_start_y),
            float(term.cfg.lane_end_y),
            speed_x_w,
        )
        velocity_w = torch.stack((speed_x_w, speed_y_w, torch.zeros_like(speed_x_w)), dim=-1)
        return progress, reference_y, heading, velocity_w, yaw_rate

    progress, reference_y, heading, velocity_w, yaw_rate = get_lane_change_reference(env)
    count = len(names)
    return (
        progress.view(-1, 1).expand(-1, count),
        reference_y.view(-1, 1).expand(-1, count),
        heading.view(-1, 1).expand(-1, count),
        velocity_w.view(env.num_envs, 1, 3).expand(-1, count, -1),
        yaw_rate.view(-1, 1).expand(-1, count),
    )


def reference_heading_vectors(env, asset_names=None) -> torch.Tensor:
    """World-frame unit tangent of the active scalar or per-vehicle reference."""

    if asset_names is None:
        _, _, heading, _, _ = get_lane_change_reference(env)
    else:
        _, _, heading, _, _ = get_platoon_lane_change_reference(env, asset_names)
    return torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)


def road_boundary_state(env, asset_names=PLATOON_ASSET_NAMES):
    """Return yaw-aware footprint exceedance, clearance, lower-y, and upper-y."""

    names = tuple(asset_names)
    term = _lane_change_term(env)
    cfg = getattr(term, "cfg", None)
    near_edge = float(getattr(cfg, "road_near_edge_y", 0.47)) - 0.5 * float(
        getattr(cfg, "road_edge_line_width_m", 0.026)
    )
    far_edge = float(getattr(cfg, "road_far_edge_y", -1.72)) + 0.5 * float(
        getattr(cfg, "road_edge_line_width_m", 0.026)
    )
    half_length = 0.5 * float(getattr(cfg, "vehicle_length_m", 0.240))
    half_width = 0.5 * float(getattr(cfg, "vehicle_width_m", 0.200))

    centers_y = []
    lateral_halves = []
    for name in names:
        _, pos_y = _env_local_position(env, name)
        quat = env.scene[name].data.root_quat_w
        w, x, y, z = quat.unbind(dim=-1)
        heading_sin = 2.0 * (w * z + x * y)
        heading_cos = 1.0 - 2.0 * (y.square() + z.square())
        lateral_half = heading_sin.abs() * half_length + heading_cos.abs() * half_width
        centers_y.append(pos_y)
        lateral_halves.append(lateral_half)

    center_y = torch.stack(centers_y, dim=1)
    lateral_half = torch.stack(lateral_halves, dim=1)
    upper_y = center_y + lateral_half
    lower_y = center_y - lateral_half
    clearance = torch.minimum(float(near_edge) - upper_y, lower_y - float(far_edge))
    exceedance = torch.relu(-clearance)
    return exceedance, clearance, lower_y, upper_y


def road_boundary_enforced(env) -> bool:
    term = _lane_change_term(env)
    return bool(getattr(getattr(term, "cfg", None), "enforce_road_boundaries", False))


def platoon_lane_change_completion_state(env, asset_names=PLATOON_ASSET_NAMES) -> dict[str, torch.Tensor]:
    """Return strict per-environment whole-platoon lane-change completion diagnostics."""

    names = tuple(asset_names)
    term = _lane_change_term(env)
    cfg = getattr(term, "cfg", None)
    progress, reference_y, reference_heading, _, _ = get_platoon_lane_change_reference(env, names)
    center_y = torch.stack([_env_local_position(env, name)[1] for name in names], dim=1)
    center_error = (center_y - reference_y).abs()

    target_x = torch.ones(env.num_envs, 3, device=env.device)
    target_x[:, 1:] = 0.0
    headings = []
    for name in names:
        heading = quat_apply(env.scene[name].data.root_quat_w, target_x)[:, :2]
        headings.append(torch.nn.functional.normalize(heading, dim=-1))
    heading_vectors = torch.stack(headings, dim=1)
    reference_vectors = torch.stack((torch.cos(reference_heading), torch.sin(reference_heading)), dim=-1)
    heading_error = torch.acos(torch.sum(heading_vectors * reference_vectors, dim=-1).clamp(-1.0, 1.0))

    gap_errors = []
    pair_lateral = []
    pair_distances = []
    for idx in range(1, len(names)):
        pred = env.scene[names[idx - 1]]
        foll = env.scene[names[idx]]
        diff = foll.data.root_pos_w - pred.data.root_pos_w
        rel = quat_apply_inverse(pred.data.root_quat_w, diff)
        follower_speed = torch.clamp(foll.data.root_lin_vel_b[:, 0], min=0.0)
        desired_gap = 1.5 + 0.6 * follower_speed
        gap_errors.append((rel[:, 0] + desired_gap).abs())
        pair_lateral.append(rel[:, 1].abs())
        pair_distances.append(torch.norm(diff[:, :2], dim=-1))
    if gap_errors:
        max_gap_error = torch.stack(gap_errors, dim=1).max(dim=1).values
        max_pair_lateral = torch.stack(pair_lateral, dim=1).max(dim=1).values
        min_pair_distance = torch.stack(pair_distances, dim=1).min(dim=1).values
    else:
        zeros = torch.zeros(env.num_envs, device=env.device)
        max_gap_error = zeros
        max_pair_lateral = zeros
        min_pair_distance = torch.full_like(zeros, float("inf"))

    exceedance, clearance, _, _ = road_boundary_state(env, names)
    min_progress = progress.min(dim=1).values
    max_center_error = center_error.max(dim=1).values
    max_heading_error = heading_error.max(dim=1).values
    max_road_exceedance = exceedance.max(dim=1).values
    complete = (
        (min_progress >= float(getattr(cfg, "completion_progress", 0.99)))
        & (max_center_error <= float(getattr(cfg, "completion_lateral_tolerance_m", 0.12)))
        & (max_heading_error <= float(getattr(cfg, "completion_heading_tolerance_rad", 0.12)))
        & (max_gap_error <= float(getattr(cfg, "completion_gap_tolerance_m", 0.35)))
        & (max_pair_lateral <= float(getattr(cfg, "completion_pair_lateral_tolerance_m", 0.20)))
        & (min_pair_distance >= 0.30)
        & (max_road_exceedance <= 0.0)
    )
    return {
        "complete": complete,
        "min_progress": min_progress,
        "max_center_error": max_center_error,
        "max_heading_error": max_heading_error,
        "max_gap_error": max_gap_error,
        "max_pair_lateral": max_pair_lateral,
        "min_pair_distance": min_pair_distance,
        "min_road_clearance": clearance.min(dim=1).values,
        "max_road_exceedance": max_road_exceedance,
    }
