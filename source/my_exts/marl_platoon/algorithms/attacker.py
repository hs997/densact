"""Attacker modules for the platoon training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from marl_platoon.algorithms.pipeline import AttackerModule


@dataclass
class AttackModuleCfg:
    max_fdi_pos: float = 8.0
    max_fdi_acc: float = 3.0
    max_dos_rate: float = 0.30
    enabled: bool = False
    target_mode: str = "all"  # all|rel_pos|vel
    obs_dim: int = 18
    seed: int = 3407


class NoOpAttackModule(AttackerModule):
    """Placeholder attacker that preserves interfaces without perturbing data."""

    def __init__(self, cfg: AttackModuleCfg | None = None):
        self.cfg = cfg or AttackModuleCfg(enabled=False)

    def perturb_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def perturb_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def maybe_update(self, context: dict[str, Any]) -> dict[str, float] | None:
        return {"attack_enabled": float(self.cfg.enabled)}


class RandomFDIDoSAttackModule(AttackerModule):
    """Minimal attacker: bounded random FDI on obs + random DoS-style zero-out."""

    def __init__(self, cfg: AttackModuleCfg | None = None):
        self.cfg = cfg or AttackModuleCfg(enabled=False)
        self.last_fdi_mean = 0.0
        self.last_dos_rate = 0.0
        self._generator_cpu = torch.Generator(device="cpu")
        self._generator_cpu.manual_seed(int(self.cfg.seed))

    def _build_target_mask(self, obs: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.target_mode).lower()
        mask = torch.ones_like(obs, dtype=torch.bool)
        if mode == "all":
            return mask
        # platoon policy obs layout: rel_pos_chain(12) + last_vel(3) + last_ang_vel(3)
        rel_pos_end = min(12, obs.shape[-1])
        vel_end = min(18, obs.shape[-1])
        if mode == "rel_pos":
            mask[:] = False
            mask[..., :rel_pos_end] = True
        elif mode == "vel":
            mask[:] = False
            mask[..., rel_pos_end:vel_end] = True
        return mask

    def perturb_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return obs
        out = obs.clone()
        target_mask = self._build_target_mask(out)

        # FDI-like bounded additive perturbation.
        if out.is_cuda:
            eps = torch.empty_like(out).uniform_(-self.cfg.max_fdi_pos, self.cfg.max_fdi_pos)
        else:
            eps = torch.empty_like(out).uniform_(
                -self.cfg.max_fdi_pos,
                self.cfg.max_fdi_pos,
                generator=self._generator_cpu,
            )
        fdi_noise = 0.02 * eps
        fdi_noise = torch.where(target_mask, fdi_noise, torch.zeros_like(fdi_noise))
        out = out + fdi_noise
        self.last_fdi_mean = float(fdi_noise.abs().mean().item())

        # DoS-like random channel drop with configured probability.
        if self.cfg.max_dos_rate > 0.0:
            if out.is_cuda:
                rand_mask = torch.rand_like(out)
            else:
                rand_mask = torch.rand_like(out, generator=self._generator_cpu)
            drop_mask = (rand_mask < self.cfg.max_dos_rate) & target_mask
            out = torch.where(drop_mask, torch.zeros_like(out), out)
            self.last_dos_rate = float(drop_mask.float().mean().item())
        else:
            self.last_dos_rate = 0.0

        return out

    def perturb_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def maybe_update(self, context: dict[str, Any]) -> dict[str, float] | None:
        return {
            "attack_enabled": float(self.cfg.enabled),
            "fdi_abs_mean": self.last_fdi_mean,
            "dos_rate": self.last_dos_rate,
        }
