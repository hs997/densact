"""Two-dimensional Ackermann action for the Wheeltec mini AKM vehicle."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class AckermannAction(ActionTerm):
    """Map ``[normalized_speed, normalized_steering]`` to rear drive and front steering joints.

    The rear axle receives wheel angular-velocity targets only.  The two front
    knuckles receive Ackermann-compatible position targets and their wheel-spin
    joints remain passive, so the rear wheels can never steer.
    """

    cfg: "AckermannActionCfg"

    def __init__(self, cfg: "AckermannActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._rear_ids, self._rear_names = self._asset.find_joints(
            list(cfg.rear_wheel_joint_names), preserve_order=True
        )
        self._steer_ids, self._steer_names = self._asset.find_joints(
            list(cfg.front_steer_joint_names), preserve_order=True
        )
        if len(self._rear_ids) != 2 or len(self._steer_ids) != 2:
            raise ValueError(
                "AckermannAction requires exactly two rear wheel and two front steering joints; "
                f"resolved rear={self._rear_names}, steer={self._steer_names}"
            )
        if cfg.wheelbase_m <= 0.0 or cfg.track_width_m <= 0.0 or cfg.wheel_radius_m <= 0.0:
            raise ValueError("Ackermann wheelbase, track width, and wheel radius must be positive")
        self._raw_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._rear_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._steer_targets = torch.zeros((self.num_envs, 2), device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Physical ``[center_speed_mps, center_steering_rad]`` commands."""
        return self._processed_actions

    @property
    def rear_wheel_velocity_targets(self) -> torch.Tensor:
        return self._rear_targets

    @property
    def front_steering_targets(self) -> torch.Tensor:
        return self._steer_targets

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(torch.clamp(actions, -1.0, 1.0))
        speed = self._raw_actions[:, 0] * float(self.cfg.max_speed_mps)
        center_steer = self._raw_actions[:, 1] * float(self.cfg.max_steering_rad)
        self._processed_actions[:, 0].copy_(speed)
        self._processed_actions[:, 1].copy_(center_steer)

        wheelbase = float(self.cfg.wheelbase_m)
        half_track = 0.5 * float(self.cfg.track_width_m)
        tan_center = torch.tan(center_steer)
        yaw_rate = speed * tan_center / wheelbase
        left_speed = speed - yaw_rate * half_track
        right_speed = speed + yaw_rate * half_track
        radius = float(self.cfg.wheel_radius_m)
        self._rear_targets[:, 0].copy_(left_speed / radius * float(self.cfg.rear_left_axis_sign))
        self._rear_targets[:, 1].copy_(right_speed / radius * float(self.cfg.rear_right_axis_sign))

        left_denom = wheelbase - half_track * tan_center
        right_denom = wheelbase + half_track * tan_center
        eps = 1.0e-4
        left_fallback = torch.where(left_denom >= 0.0, eps, -eps)
        right_fallback = torch.where(right_denom >= 0.0, eps, -eps)
        left_denom = torch.where(left_denom.abs() < eps, left_fallback, left_denom)
        right_denom = torch.where(right_denom.abs() < eps, right_fallback, right_denom)
        left_angle = torch.atan2(wheelbase * tan_center, left_denom)
        right_angle = torch.atan2(wheelbase * tan_center, right_denom)
        limit = float(self.cfg.front_joint_limit_rad)
        self._steer_targets[:, 0].copy_(
            torch.clamp(left_angle * float(self.cfg.front_left_axis_sign), -limit, limit)
        )
        self._steer_targets[:, 1].copy_(
            torch.clamp(right_angle * float(self.cfg.front_right_axis_sign), -limit, limit)
        )
        # Snapshot issued commands before auto-reset can clear ActionTerm data.
        # These are actuator TARGETS, not measured tire speed/steering motion.
        if not hasattr(self._env, "_platoon_ackermann_action_trace"):
            self._env._platoon_ackermann_action_trace = {}
        self._env._platoon_ackermann_action_trace[self.cfg.asset_name] = {
            "executed_semantic_actions": self._raw_actions.detach().clone(),
            "physical_command_actions": self._processed_actions.detach().clone(),
            "rear_velocity_targets": self._rear_targets.detach().clone(),
            "front_steering_targets": self._steer_targets.detach().clone(),
        }

    def apply_actions(self) -> None:
        self._asset.set_joint_velocity_target(self._rear_targets, joint_ids=self._rear_ids)
        self._asset.set_joint_position_target(self._steer_targets, joint_ids=self._steer_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._rear_targets[env_ids] = 0.0
        self._steer_targets[env_ids] = 0.0


@configclass
class AckermannActionCfg(ActionTermCfg):
    """Configuration for :class:`AckermannAction`."""

    class_type: type = AckermannAction
    rear_wheel_joint_names: tuple[str, str] = ("lb_joint", "rb_joint")
    front_steer_joint_names: tuple[str, str] = ("left_joint", "right_joint")
    wheelbase_m: float = 0.143250
    track_width_m: float = 0.104000
    wheel_radius_m: float = 0.0320
    # HAPPO deployment clips normalized actions to +/-0.4; 1.2 m/s keeps the
    # reachable forward range at 0.48 m/s for the 0.30--0.45 m/s command band.
    max_speed_mps: float = 1.20
    max_steering_rad: float = 0.32
    front_joint_limit_rad: float = 0.39
    # This URDF's rear-wheel axes are opposite to body-forward rotation.
    rear_left_axis_sign: float = -1.0
    rear_right_axis_sign: float = -1.0
    front_left_axis_sign: float = -1.0
    front_right_axis_sign: float = 1.0
