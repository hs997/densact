"""Shield modules for the platoon training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.utils.math import quat_apply

from marl_platoon.algorithms.pipeline import ShieldModule


@dataclass
class BadHeadingShieldCfg:
    enabled: bool = False
    action_scale: float = 0.2
    heading_warn_threshold: float = 0.35
    heading_critical_threshold: float = 0.20
    warn_scale: float = 0.5


class BadHeadingShieldModule(ShieldModule):
    """Lightweight safety guard used as the current shield placeholder."""

    def __init__(self, env, cfg: BadHeadingShieldCfg | None = None):
        self.env = env
        self.cfg = cfg or BadHeadingShieldCfg()
        self.robots = ["robot", "robot_2", "robot_3", "robot_4"]

    def project_action(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return actions
        target_dir = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).repeat(self.env.num_envs, 1)
        min_heading_x = torch.ones(self.env.num_envs, device=self.env.device)
        for name in self.robots:
            quat = self.env.scene[name].data.root_quat_w
            heading = quat_apply(quat, target_dir)
            min_heading_x = torch.minimum(min_heading_x, heading[:, 0])

        critical_mask = min_heading_x < self.cfg.heading_critical_threshold
        warn_mask = (min_heading_x < self.cfg.heading_warn_threshold) & (~critical_mask)

        if not (critical_mask.any() or warn_mask.any()):
            return actions

        guarded = actions.clone()
        if warn_mask.any() and self.cfg.warn_scale > 0.0:
            guarded[warn_mask] = guarded[warn_mask] * self.cfg.warn_scale
        if critical_mask.any() and self.cfg.action_scale > 0.0:
            guarded[critical_mask] = guarded[critical_mask] * self.cfg.action_scale
        return guarded
