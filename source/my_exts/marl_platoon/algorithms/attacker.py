"""Attacker modules for the platoon training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
from pathlib import Path

import torch
from torch import nn

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
    mode: str = "profile"  # profile|generator|cagan
    noise_dim: int = 16
    hidden_dim: int = 128
    realism_coef: float = 0.10
    generator_lr: float = 1.0e-4
    discriminator_lr: float = 1.0e-4
    update_interval: int = 4
    attack_obj_coef: float = 1.0
    reward_proxy_coef: float = 1.0  # fallback only when no clean physical context is available
    dos_proxy_coef: float = 0.3
    attack_objective_clip: float = 20.0
    curriculum_warmup_updates: int = 2000
    curriculum_start_mode: str = "profile"
    decay_start_updates: int = 5000
    min_attack_obj_scale: float = 0.25
    min_update_freq_scale: float = 0.25
    decay_half_life_updates: int = 2500


class AttackGenerator(nn.Module):
    """CA-GAN generator G(z, s_t) -> (f_p, beta_p, f_a, beta_a)."""

    def __init__(self, obs_dim: int, noise_dim: int, hidden_dim: int, action_dim: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + noise_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.f_p_head = nn.Linear(hidden_dim, obs_dim)
        self.beta_p_head = nn.Linear(hidden_dim, obs_dim)
        self.f_a_head = nn.Linear(hidden_dim, action_dim)
        self.beta_a_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, obs: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, noise], dim=-1)
        feat = self.net(x)
        f_p_raw = torch.tanh(self.f_p_head(feat))
        beta_p_logits = self.beta_p_head(feat)
        f_a_raw = torch.tanh(self.f_a_head(feat))
        beta_a_logits = self.beta_a_head(feat)
        return f_p_raw, beta_p_logits, f_a_raw, beta_a_logits


class AttackDiscriminator(nn.Module):
    """Conditional discriminator D(xi | s_t), xi=(f_p,beta_p,f_a,beta_a)."""

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 3 + action_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        f_p: torch.Tensor,
        beta_p: torch.Tensor,
        f_a: torch.Tensor,
        beta_a: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([obs, f_p, beta_p, f_a, beta_a], dim=-1))


class AttackSequenceDiscriminator(nn.Module):
    """Temporal D for xi_t=(f_p,beta_p,f_a,beta_a) sequences."""

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int = 4):
        super().__init__()
        in_dim = obs_dim * 3 + action_dim * 2
        self.encoder = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs_seq: torch.Tensor,
        f_p_seq: torch.Tensor,
        beta_p_seq: torch.Tensor,
        f_a_seq: torch.Tensor,
        beta_a_seq: torch.Tensor,
    ) -> torch.Tensor:
        seq = torch.cat([obs_seq, f_p_seq, beta_p_seq, f_a_seq, beta_a_seq], dim=-1)
        _, h = self.encoder(seq)
        return self.head(h[-1])


class ReferenceAttackLibrary:
    """Template sampler backed by literature JSON library."""

    def __init__(self, cfg: AttackModuleCfg):
        self.cfg = cfg
        lib_path = Path(__file__).with_name("attack_reference_library.json")
        self.templates: list[dict[str, Any]] = []
        if lib_path.exists():
            try:
                data = json.loads(lib_path.read_text())
                for paper in data.get("papers", []):
                    self.templates.extend(paper.get("attack_family", []))
            except Exception:
                self.templates = []

    def _sample_template(self, recent_ids: list[str] | None = None) -> dict[str, Any] | None:
        if not self.templates:
            return None
        recent_set = set(recent_ids or [])
        candidates = [t for t in self.templates if str(t.get("template_id", "")) not in recent_set]
        if not candidates:
            candidates = self.templates
        weights = torch.tensor([float(t.get("sample_weight", 1.0)) for t in candidates], dtype=torch.float32)
        probs = weights / weights.sum().clamp_min(1.0e-6)
        idx = int(torch.multinomial(probs, 1).item())
        return candidates[idx]

    def sample(
        self,
        obs: torch.Tensor,
        action_dim: int = 4,
        context: dict[str, Any] | None = None,
        recent_ids: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        f_p = torch.zeros_like(obs)
        beta_p = torch.ones_like(obs)
        f_a = torch.zeros(obs.shape[0], action_dim, device=obs.device, dtype=obs.dtype)
        beta_a = torch.ones_like(f_a)
        if obs.numel() == 0:
            return f_p, beta_p, f_a, beta_a, "none"
        tpl = self._sample_template(recent_ids=recent_ids)
        if tpl is None:
            drop_p = (torch.rand_like(obs) < min(max(self.cfg.max_dos_rate, 0.0), 1.0)).float()
            drop_a = (torch.rand_like(f_a) < min(max(self.cfg.max_dos_rate, 0.0), 1.0)).float()
            f_p = 0.5 * self.cfg.max_fdi_pos * torch.sign(torch.randn_like(obs)) * (1.0 - drop_p)
            f_a = 0.5 * self.cfg.max_fdi_acc * torch.sign(torch.randn_like(f_a)) * (1.0 - drop_a)
            return f_p, 1.0 - drop_p, f_a, 1.0 - drop_a, "fallback_random"

        timing = tpl.get("timing", {})
        mode = str(timing.get("mode", "continuous")).lower()
        step = int((context or {}).get("student_updates", 0))
        active = True
        start_step = int(timing.get("start_step", 0))
        if step < start_step:
            active = False
        period = int(timing.get("period", 0))
        burst_len = int(timing.get("burst_len", 0))
        duty = float(timing.get("duty_cycle", 0.0))
        if mode == "periodic" and period > 0:
            active = (step % period) < max(burst_len, int(period * duty))
        elif mode == "burst" and period > 0:
            active = (step % period) < max(burst_len, 1)
        elif mode == "event_triggered":
            no_backward = float((context or {}).get("no_backward_penalty", 0.0))
            bad_done = float((context or {}).get("bad_done_rate", 0.0))
            active = active and (no_backward > 0.05 or bad_done > 0.0)

        template_id = str(tpl.get("template_id", "unknown"))
        if not active:
            return f_p, beta_p, f_a, beta_a, template_id

        fdi_cfg = tpl.get("fdi", {})
        amp = fdi_cfg.get("amplitude", {})
        pos_cfg = amp.get("pos", {})
        acc_cfg = amp.get("acc", {})
        pos_max = min(float(pos_cfg.get("max", 0.0)), self.cfg.max_fdi_pos)
        acc_max = min(float(acc_cfg.get("max", 0.0)), self.cfg.max_fdi_acc)
        sign_p = torch.sign(torch.randn_like(obs))
        sign_a = torch.sign(torch.randn_like(f_a))
        f_p = pos_max * sign_p
        f_a = acc_max * sign_a
        noise_std = float(fdi_cfg.get("noise_std", 0.0))
        if noise_std > 0.0:
            f_p = f_p + noise_std * torch.randn_like(obs)
            f_a = f_a + noise_std * torch.randn_like(f_a)

        dos_cfg = tpl.get("dos", {})
        if bool(dos_cfg.get("enabled", False)):
            drop_rate = min(float(dos_cfg.get("drop_rate", 0.0)), self.cfg.max_dos_rate)
            beta_p = 1.0 - (torch.rand_like(obs) < drop_rate).float()
            beta_a = 1.0 - (torch.rand_like(f_a) < drop_rate).float()

        # FDI is effective only when the channel is active, matching Eq. (attack_model).
        f_p = f_p * beta_p
        f_a = f_a * beta_a
        return f_p, beta_p, f_a, beta_a, template_id


class NoOpAttackModule(AttackerModule):
    """Disabled attacker that preserves the training-pipeline interface."""

    def __init__(self, cfg: AttackModuleCfg | None = None):
        self.cfg = cfg or AttackModuleCfg(enabled=False)

    def perturb_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def perturb_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def maybe_update(self, context: dict[str, Any]) -> dict[str, float] | None:
        return {"attack_enabled": float(self.cfg.enabled)}


class RandomFDIDoSAttackModule(AttackerModule):
    """Profile/generator/CA-GAN attacker with bounded FDI and DoS outputs.

    Modes:
    - profile: previous bounded random FDI/DoS baseline.
    - generator: state-conditioned G(z, s) attack generation without GAN updates.
    - cagan: state-conditioned CA-GAN update with realism regularization.
    """

    def __init__(self, cfg: AttackModuleCfg | None = None):
        self.cfg = cfg or AttackModuleCfg(enabled=False)
        self.last_fdi_mean = 0.0
        self.last_dos_rate = 0.0
        self.last_obs_fdi_mean = 0.0   # f_p magnitude
        self.last_obs_dos_rate = 0.0   # 1 - beta_p
        self.last_act_fdi_mean = 0.0   # f_a magnitude
        self.last_act_dos_rate = 0.0   # 1 - beta_a
        self.last_act_dos_budget = float(self.cfg.max_dos_rate)
        self.last_beta_p = 1.0 - self.last_obs_dos_rate
        self.last_beta_a = 1.0 - self.last_act_dos_rate
        self.last_act_drop_count = 0.0
        self.last_g_loss = 0.0
        self.last_d_loss = 0.0
        self.last_realism_loss = 0.0
        self.last_attack_obj = 0.0
        self.last_seq_realism_loss = 0.0
        self.last_obj_spacing = 0.0
        self.last_obj_vel = 0.0
        self.last_obj_acc = 0.0
        self.last_obj_jerk = 0.0
        self.last_obj_bad_done = 0.0
        self.last_obj_energy = 0.0
        self.last_act_elem_n = 0.0
        self.last_stats_inconsistent = 0.0
        self._cached_obs: torch.Tensor | None = None
        self._cached_fdi: torch.Tensor | None = None
        self._cached_dos: torch.Tensor | None = None
        self._cached_f_p: torch.Tensor | None = None
        self._cached_beta_p: torch.Tensor | None = None
        self._cached_f_a: torch.Tensor | None = None
        self._cached_beta_a: torch.Tensor | None = None
        self._cached_fallback_disruption_score: float = 0.0
        self._cached_attack_objective: float = 0.0
        self._pending_f_a: torch.Tensor | None = None
        self._pending_beta_a: torch.Tensor | None = None
        self._last_valid_obs: torch.Tensor | None = None
        self._last_valid_action: torch.Tensor | None = None
        self._updates = 0
        self._generator_cpu = torch.Generator(device="cpu")
        self._generator_cpu.manual_seed(int(self.cfg.seed))
        self.generator: AttackGenerator | None = None
        self.discriminator: AttackDiscriminator | None = None
        self.sequence_discriminator: AttackSequenceDiscriminator | None = None
        self.reference_library = ReferenceAttackLibrary(self.cfg)
        self.last_ref_template_id = "none"
        self._recent_ref_templates: list[str] = []
        self.last_unique_ref_tpl_count = 0.0
        self._seq_obs: list[torch.Tensor] = []
        self._seq_f_p: list[torch.Tensor] = []
        self._seq_beta_p: list[torch.Tensor] = []
        self._seq_f_a: list[torch.Tensor] = []
        self._seq_beta_a: list[torch.Tensor] = []
        self.seq_window = 8
        self.generator_optimizer: torch.optim.Optimizer | None = None
        self.discriminator_optimizer: torch.optim.Optimizer | None = None
        self.sequence_discriminator_optimizer: torch.optim.Optimizer | None = None
        self._runtime_mode = str(self.cfg.mode).lower()

    def _get_runtime_mode(self, student_updates: int | None = None) -> str:
        progress_updates = self._updates if student_updates is None else student_updates
        if self.cfg.curriculum_warmup_updates > 0 and progress_updates < self.cfg.curriculum_warmup_updates:
            return str(self.cfg.curriculum_start_mode).lower()
        return str(self.cfg.mode).lower()

    def _get_late_stage_attack_scale(self, student_updates: int) -> float:
        if self.cfg.decay_start_updates <= 0 or student_updates <= self.cfg.decay_start_updates:
            return 1.0
        half_life = max(float(self.cfg.decay_half_life_updates), 1.0)
        decay_steps = float(student_updates - self.cfg.decay_start_updates)
        scale = 0.5 ** (decay_steps / half_life)
        return float(max(self.cfg.min_attack_obj_scale, min(1.0, scale)))

    def _get_late_stage_update_interval(self, student_updates: int) -> int:
        if self.cfg.decay_start_updates <= 0 or student_updates <= self.cfg.decay_start_updates:
            return max(self.cfg.update_interval, 1)
        half_life = max(float(self.cfg.decay_half_life_updates), 1.0)
        decay_steps = float(student_updates - self.cfg.decay_start_updates)
        freq_scale = 0.5 ** (decay_steps / half_life)
        freq_scale = max(self.cfg.min_update_freq_scale, min(1.0, freq_scale))
        return max(int(round(max(self.cfg.update_interval, 1) / freq_scale)), 1)

    def _build_target_mask(self, obs: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.target_mode).lower()
        mask = torch.ones_like(obs, dtype=torch.bool)
        if mode == "all":
            return mask
        # Local platoon obs layout: [rel_x, rel_y, rel_vx, rel_vy, own_vx, own_vy, yaw_rate, role_id].
        # Older global-chain layouts are still handled by the fallback split.
        if obs.shape[-1] <= 10:
            rel_pos_end = min(4, obs.shape[-1])
            vel_end = min(6, obs.shape[-1])
        else:
            rel_pos_end = max(obs.shape[-1] - 6, 0)
            vel_end = max(obs.shape[-1] - 3, rel_pos_end)
        if mode == "rel_pos":
            mask[:] = False
            mask[..., :rel_pos_end] = True
        elif mode == "vel":
            mask[:] = False
            mask[..., rel_pos_end:vel_end] = True
        return mask

    def _module_has_inference_params(self, module: nn.Module | None) -> bool:
        if module is None:
            return False
        return any(p.is_inference() for p in module.parameters())

    def _ensure_neural_modules(self, device: torch.device) -> None:
        # Important: modules may be first touched during env stepping where
        # inference mode can be active. Build them with inference disabled so
        # autograd is valid later in cagan updates.
        need_new_g = self.generator is None or self._module_has_inference_params(self.generator)
        need_new_d = self.discriminator is None or self._module_has_inference_params(self.discriminator)
        need_new_sd = self.sequence_discriminator is None or self._module_has_inference_params(self.sequence_discriminator)
        if need_new_g or need_new_d or need_new_sd:
            with torch.inference_mode(False):
                if need_new_g:
                    self.generator = AttackGenerator(self.cfg.obs_dim, self.cfg.noise_dim, self.cfg.hidden_dim).to(device)
                    self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.generator_lr)
                if need_new_d:
                    self.discriminator = AttackDiscriminator(self.cfg.obs_dim, self.cfg.hidden_dim).to(device)
                    self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.discriminator_lr)
                if need_new_sd:
                    self.sequence_discriminator = AttackSequenceDiscriminator(self.cfg.obs_dim, self.cfg.hidden_dim).to(device)
                    self.sequence_discriminator_optimizer = torch.optim.Adam(self.sequence_discriminator.parameters(), lr=self.cfg.discriminator_lr)

    def _sample_generator_attack(
        self, obs: torch.Tensor, target_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_neural_modules(obs.device)
        flat_obs = obs.reshape(-1, obs.shape[-1])
        noise = torch.randn(flat_obs.shape[0], self.cfg.noise_dim, device=obs.device)
        assert self.generator is not None
        f_p_raw, beta_p_logits, f_a_raw, beta_a_logits = self.generator(flat_obs, noise)
        f_p = (self.cfg.max_fdi_pos * f_p_raw).view_as(obs)
        p_drop = torch.sigmoid(beta_p_logits).view_as(obs) * min(max(self.cfg.max_dos_rate, 0.0), 1.0)
        beta_p = 1.0 - (torch.rand_like(obs) < p_drop).float()
        f_p = torch.where(target_mask, f_p * beta_p, torch.zeros_like(f_p))
        beta_p = torch.where(target_mask, beta_p, torch.ones_like(beta_p))
        self._pending_f_a = self.cfg.max_fdi_acc * f_a_raw.detach()
        p_drop_a = torch.sigmoid(beta_a_logits).detach() * min(max(self.cfg.max_dos_rate, 0.0), 1.0)
        self._pending_beta_a = 1.0 - (torch.rand_like(p_drop_a) < p_drop_a).float()
        self._pending_f_a = self._pending_f_a * self._pending_beta_a
        return f_p, beta_p, self._pending_f_a, self._pending_beta_a

    def perturb_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return obs
        out = obs.clone()
        target_mask = self._build_target_mask(out)
        mode = self._get_runtime_mode()
        self._runtime_mode = mode

        if mode in {"generator", "cagan"}:
            f_p, beta_p, pending_f_a, pending_beta_a = self._sample_generator_attack(out, target_mask)
            fdi_noise = f_p
            drop_mask = (beta_p <= 0.5)
            # enforce observation-channel DoS budget for beta_p consistency
            if self.cfg.max_dos_rate > 0.0:
                flat_n = out.numel()
                k = int(self.cfg.max_dos_rate * flat_n)
                if k > 0:
                    rand_flat = torch.rand(flat_n, device=out.device)
                    topk_idx = torch.topk(rand_flat, k=k, largest=False).indices
                    budget_mask = torch.zeros(flat_n, device=out.device, dtype=torch.bool)
                    budget_mask[topk_idx] = True
                    budget_mask = budget_mask.view_as(out)
                    drop_mask = drop_mask & budget_mask
                    beta_p = (~drop_mask).float()
                    fdi_noise = fdi_noise * beta_p
                else:
                    drop_mask = torch.zeros_like(out, dtype=torch.bool)
        else:
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

            # DoS-like random channel drop with configured probability.
            if self.cfg.max_dos_rate > 0.0:
                rand_mask = torch.rand(out.shape, device=out.device)
                drop_mask = (rand_mask < self.cfg.max_dos_rate) & target_mask
            else:
                drop_mask = torch.zeros_like(out, dtype=torch.bool)

        out = out + fdi_noise
        # DoS hold strategy: use ZOH-style last valid observation by default.
        if self._last_valid_obs is None or self._last_valid_obs.shape != out.shape:
            self._last_valid_obs = obs.detach().clone()
        out = torch.where(drop_mask, self._last_valid_obs.to(out.device), out)
        self.last_obs_fdi_mean = float(fdi_noise.abs().mean().item())
        self.last_obs_dos_rate = float(drop_mask.float().mean().item())
        self.last_beta_p = 1.0 - self.last_obs_dos_rate
        self.last_obs_drop_count = float(drop_mask.sum().item())
        self.last_obs_elem_n = float(drop_mask.numel())
        self.last_obs_dos_budget = float(self.cfg.max_dos_rate)
        self.last_fdi_mean = 0.5 * (self.last_obs_fdi_mean + self.last_act_fdi_mean)
        self.last_dos_rate = 0.5 * (self.last_obs_dos_rate + self.last_act_dos_rate)
        # update last valid obs from non-dropped entries
        self._last_valid_obs = torch.where(drop_mask, self._last_valid_obs.to(out.device), out).detach().clone()
        self._cached_obs = obs.detach().clone()
        self._cached_fdi = fdi_noise.detach().clone()
        self._cached_dos = drop_mask.float().detach().clone()
        self._cached_f_p = fdi_noise.detach().clone()
        self._cached_beta_p = (~drop_mask).float().detach().clone()
        action_dim = int(self.generator.action_dim) if self.generator is not None else 4
        action_shape = (*obs.shape[:-1], action_dim)
        if self._pending_f_a is not None and self._pending_f_a.numel() == int(torch.tensor(action_shape).prod().item()):
            f_a = self._pending_f_a.to(obs.device, dtype=obs.dtype).view(action_shape)
            beta_a = self._pending_beta_a.to(obs.device, dtype=obs.dtype).view(action_shape)
        else:
            f_a = torch.zeros(action_shape, device=obs.device, dtype=obs.dtype)
            beta_a = torch.ones_like(f_a)
        self._cached_f_a = f_a.detach().clone()
        self._cached_beta_a = beta_a.detach().clone()
        self._seq_obs.append(self._cached_obs)
        self._seq_f_p.append(self._cached_f_p)
        self._seq_beta_p.append(self._cached_beta_p)
        self._seq_f_a.append(self._cached_f_a)
        self._seq_beta_a.append(self._cached_beta_a)
        if len(self._seq_obs) > self.seq_window:
            self._seq_obs = self._seq_obs[-self.seq_window:]
            self._seq_f_p = self._seq_f_p[-self.seq_window:]
            self._seq_beta_p = self._seq_beta_p[-self.seq_window:]
            self._seq_f_a = self._seq_f_a[-self.seq_window:]
            self._seq_beta_a = self._seq_beta_a[-self.seq_window:]
        return out

    def perturb_action(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return actions
        out = actions.clone()
        mode = self._get_runtime_mode()
        self._runtime_mode = mode
        if mode in {"generator", "cagan"} and self._pending_f_a is not None and self._pending_beta_a is not None:
            fdi_u = self._pending_f_a.to(out.device, dtype=out.dtype).view_as(out)
            beta_a = self._pending_beta_a.to(out.device, dtype=out.dtype).view_as(out)
            drop_mask = beta_a <= 0.5
        else:
            if out.is_cuda:
                fdi_u = torch.empty_like(out).uniform_(-self.cfg.max_fdi_acc, self.cfg.max_fdi_acc)
            else:
                fdi_u = torch.empty_like(out).uniform_(
                    -self.cfg.max_fdi_acc,
                    self.cfg.max_fdi_acc,
                    generator=self._generator_cpu,
                )
            if self.cfg.max_dos_rate > 0.0:
                rand_mask = torch.rand(out.shape, device=out.device)
                drop_mask = rand_mask < self.cfg.max_dos_rate
            else:
                drop_mask = torch.zeros_like(out, dtype=torch.bool)

        if self.cfg.max_dos_rate > 0.0:
            flat_n = out.numel()
            k = int(self.cfg.max_dos_rate * flat_n)
            if k <= 0:
                drop_mask = torch.zeros_like(out, dtype=torch.bool)
            else:
                flat_drop = drop_mask.reshape(-1)
                if int(flat_drop.sum().item()) > k:
                    priorities = torch.rand(flat_n, device=out.device)
                    priorities = torch.where(flat_drop, priorities, torch.ones_like(priorities) + 1.0)
                    topk_idx = torch.topk(priorities, k=k, largest=False).indices
                    limited = torch.zeros(flat_n, device=out.device, dtype=torch.bool)
                    limited[topk_idx] = True
                    drop_mask = limited.view_as(out)
        else:
            drop_mask = torch.zeros_like(out, dtype=torch.bool)

        fdi_u = torch.clamp(fdi_u, -self.cfg.max_fdi_acc, self.cfg.max_fdi_acc)
        fdi_u = fdi_u * (~drop_mask).to(dtype=out.dtype)
        out = out + fdi_u

        if self._last_valid_action is None or self._last_valid_action.shape != out.shape:
            self._last_valid_action = actions.detach().clone()
        out = torch.where(drop_mask, self._last_valid_action.to(out.device), out)
        self._last_valid_action = torch.where(drop_mask, self._last_valid_action.to(out.device), out).detach().clone()

        self.last_act_fdi_mean = float(fdi_u.abs().mean().item())
        self.last_act_dos_rate = float(drop_mask.float().mean().item())
        self.last_beta_a = 1.0 - self.last_act_dos_rate
        self.last_act_drop_count = float(drop_mask.sum().item())
        self.last_act_elem_n = float(drop_mask.numel())
        self.last_act_dos_budget = float(self.cfg.max_dos_rate)

        self.last_fdi_mean = 0.5 * (self.last_obs_fdi_mean + self.last_act_fdi_mean)
        self.last_dos_rate = 0.5 * (self.last_obs_dos_rate + self.last_act_dos_rate)
        return out

    def maybe_update(self, context: dict[str, Any]) -> dict[str, float] | None:
        self._updates += 1
        # Paper objective direction: maximize clean physical degradation while
        # penalizing attack energy and DoS use for realism/stealthiness.
        bad_done_rate = float(context.get("bad_done_rate", 0.0))
        physical_cost = float(context.get("physical_cost", 0.0))
        spacing_cost = float(context.get("physical_spacing_cost", 0.0))
        velocity_cost = float(context.get("physical_velocity_cost", 0.0))
        acceleration_cost = float(context.get("physical_acceleration_cost", 0.0))
        jerk_cost = float(context.get("physical_jerk_cost", 0.0))
        overspeed_cost = float(context.get("physical_overspeed_cost", 0.0))
        collision_cost = float(context.get("physical_collision_cost", 0.0))
        l_adv_fdi = abs(self.last_obs_fdi_mean) + abs(self.last_act_fdi_mean)
        l_adv_dos = abs(self.last_obs_dos_rate) + abs(self.last_act_dos_rate)
        self.last_obj_spacing = spacing_cost
        self.last_obj_vel = velocity_cost
        self.last_obj_acc = acceleration_cost
        self.last_obj_jerk = jerk_cost
        self.last_obj_bad_done = bad_done_rate
        self.last_obj_energy = l_adv_fdi + l_adv_dos
        raw_attack_objective = physical_cost + collision_cost + bad_done_rate - l_adv_fdi - l_adv_dos
        self._cached_attack_objective = float(
            max(min(raw_attack_objective, self.cfg.attack_objective_clip), -self.cfg.attack_objective_clip)
        )

        student_updates = int(context.get("student_updates", self._updates))
        mode = self._get_runtime_mode(student_updates)
        self._runtime_mode = mode
        attack_obj_scale = self._get_late_stage_attack_scale(student_updates)
        update_interval = self._get_late_stage_update_interval(student_updates)
        if self.cfg.enabled and mode == "cagan" and self._updates % update_interval == 0:
            self._update_cagan(context, attack_obj_scale=attack_obj_scale)

        expected_act_dos = (self.last_act_drop_count / self.last_act_elem_n) if self.last_act_elem_n > 0 else 0.0
        self.last_stats_inconsistent = float(abs(self.last_act_dos_rate - expected_act_dos) > 1.0e-6)
        phy_ctx_norm = float(
            (abs(self.last_obj_spacing)
             + abs(self.last_obj_vel)
             + abs(self.last_obj_acc)
             + abs(self.last_obj_jerk)
             + abs(overspeed_cost)
             + abs(collision_cost))
            / 6.0
        )
        return {
            "attack_enabled": float(self.cfg.enabled),
            "attack_mode": mode,
            "attack_target_mode_cfg": str(self.cfg.mode).lower(),
            "attack_curriculum_warmup_updates": float(self.cfg.curriculum_warmup_updates),
            "attack_late_obj_scale": attack_obj_scale,
            "attack_late_update_interval": float(update_interval),
            "fdi_abs_mean": self.last_fdi_mean,
            "dos_rate": self.last_dos_rate,
            "obs_fdi_abs_mean": self.last_obs_fdi_mean,
            "obs_dos_rate": self.last_obs_dos_rate,
            "obs_dos_budget": self.last_obs_dos_budget,
            "obs_drop_count": self.last_obs_drop_count,
            "obs_elem_n": self.last_obs_elem_n,
            "act_fdi_abs_mean": self.last_act_fdi_mean,
            "act_dos_rate": self.last_act_dos_rate,
            "act_dos_budget": self.last_act_dos_budget,
            "act_drop_count": self.last_act_drop_count,
            "act_elem_n": self.last_act_elem_n,
            "beta_p_mean": self.last_beta_p,
            "beta_a_mean": self.last_beta_a,
            "attack_stats_inconsistent": self.last_stats_inconsistent,
            "attack_g_loss": self.last_g_loss,
            "attack_d_loss": self.last_d_loss,
            "attack_realism_loss": self.last_realism_loss,
            "attack_seq_realism_loss": self.last_seq_realism_loss,
            "attack_objective": self.last_attack_obj,
            "obj_spacing": self.last_obj_spacing,
            "obj_vel": self.last_obj_vel,
            "obj_acc": self.last_obj_acc,
            "obj_jerk": self.last_obj_jerk,
            "obj_bad_done": self.last_obj_bad_done,
            "obj_energy": self.last_obj_energy,
            "obj_physical_cost": physical_cost,
            "obj_overspeed": overspeed_cost,
            "obj_collision": collision_cost,
            "attack_phy_ctx_norm": phy_ctx_norm,
            "attack_cond_dim": 4.0,
            "attack_ref_template_id": self.last_ref_template_id,
            "attack_unique_ref_tpl_count": self.last_unique_ref_tpl_count,
        }

    def _update_cagan(self, context: dict[str, Any], attack_obj_scale: float = 1.0) -> None:
        if (
            self._cached_obs is None
            or self._cached_f_p is None
            or self._cached_beta_p is None
            or self._cached_f_a is None
            or self._cached_beta_a is None
        ):
            return
        self._ensure_neural_modules(self._cached_obs.device)
        assert self.discriminator is not None
        assert self.discriminator_optimizer is not None
        assert self.sequence_discriminator is not None
        assert self.sequence_discriminator_optimizer is not None
        # Convert possible inference tensors (from env step path) into fresh
        # regular tensors before autograd-enabled discriminator updates.
        obs = self._cached_obs.detach().clone().reshape(-1, self._cached_obs.shape[-1])
        fake_f_p = self._cached_f_p.detach().clone().reshape(-1, self._cached_f_p.shape[-1])
        fake_beta_p = self._cached_beta_p.detach().clone().reshape(-1, self._cached_beta_p.shape[-1])
        fake_f_a = self._cached_f_a.detach().clone().reshape(-1, self._cached_f_a.shape[-1])
        fake_beta_a = self._cached_beta_a.detach().clone().reshape(-1, self._cached_beta_a.shape[-1])
        real_f_p, real_beta_p, real_f_a, real_beta_a, ref_template_id = self.reference_library.sample(
            obs,
            action_dim=fake_f_a.shape[-1],
            context=context,
            recent_ids=self._recent_ref_templates[-3:],
        )
        self.last_ref_template_id = ref_template_id
        self._recent_ref_templates.append(ref_template_id)
        if len(self._recent_ref_templates) > 64:
            self._recent_ref_templates = self._recent_ref_templates[-64:]
        self.last_unique_ref_tpl_count = float(len(set(self._recent_ref_templates[-20:])))

        real_logits = self.discriminator(obs, real_f_p, real_beta_p, real_f_a, real_beta_a)
        fake_logits = self.discriminator(
            obs, fake_f_p.detach(), fake_beta_p.detach(), fake_f_a.detach(), fake_beta_a.detach()
        )
        d_loss = torch.nn.functional.softplus(-real_logits).mean() + torch.nn.functional.softplus(fake_logits).mean()
        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()
        self.last_d_loss = float(d_loss.item())

        # Generator objective:
        # Re-sample differentiable attacks from G so the clean physical attack
        # objective modulates gradients on the generator output. The sampled DoS
        # is non-differentiable during environment rollout, so generator training
        # uses its dropout probability.
        assert self.generator is not None
        assert self.generator_optimizer is not None
        noise = torch.randn(obs.shape[0], self.cfg.noise_dim, device=obs.device)
        f_p_raw_g, beta_p_logits_g, f_a_raw_g, beta_a_logits_g = self.generator(obs, noise)
        f_p_g = self.cfg.max_fdi_pos * f_p_raw_g
        f_a_g = self.cfg.max_fdi_acc * f_a_raw_g
        p_drop_g = torch.sigmoid(beta_p_logits_g) * min(max(self.cfg.max_dos_rate, 0.0), 1.0)
        p_drop_a_g = torch.sigmoid(beta_a_logits_g) * min(max(self.cfg.max_dos_rate, 0.0), 1.0)
        beta_p_g = 1.0 - p_drop_g
        beta_a_g = 1.0 - p_drop_a_g
        fake_logits_for_g = self.discriminator(obs, f_p_g, beta_p_g, f_a_g, beta_a_g)
        realism_loss = torch.nn.functional.softplus(-fake_logits_for_g).mean()
        attack_obj = torch.tensor(self._cached_attack_objective, device=obs.device)
        # Paper-aligned generator objective:
        # maximize J(theta_C, theta_G) - lambda_real * L_real.
        # Since the simulator is not differentiable through the rollout, the
        # clean physical attack objective is used as the rollout-level scalar
        # weight for the generator's differentiable attack distribution.
        expected_attack_cost = (
            f_p_g.abs().mean()
            + f_a_g.abs().mean()
            + p_drop_g.mean()
            + p_drop_a_g.mean()
        )
        effective_attack_obj_coef = self.cfg.attack_obj_coef * float(attack_obj_scale)
        g_loss = -effective_attack_obj_coef * attack_obj.detach() * expected_attack_cost + self.cfg.realism_coef * realism_loss
        self.generator_optimizer.zero_grad()
        g_loss.backward()
        self.generator_optimizer.step()

        seq_realism_loss = torch.tensor(0.0, device=obs.device)
        if len(self._seq_obs) >= self.seq_window:
            obs_seq = torch.cat(self._seq_obs[-self.seq_window:], dim=0)
            f_p_seq = torch.cat(self._seq_f_p[-self.seq_window:], dim=0)
            beta_p_seq = torch.cat(self._seq_beta_p[-self.seq_window:], dim=0)
            f_a_seq = torch.cat(self._seq_f_a[-self.seq_window:], dim=0)
            beta_a_seq = torch.cat(self._seq_beta_a[-self.seq_window:], dim=0)
            if obs_seq.dim() == 2:
                obs_seq = obs_seq.unsqueeze(0)
                f_p_seq = f_p_seq.unsqueeze(0)
                beta_p_seq = beta_p_seq.unsqueeze(0)
                f_a_seq = f_a_seq.unsqueeze(0)
                beta_a_seq = beta_a_seq.unsqueeze(0)
            seq_fake_logits = self.sequence_discriminator(obs_seq, f_p_seq, beta_p_seq, f_a_seq, beta_a_seq)
            seq_realism_loss = torch.nn.functional.softplus(-seq_fake_logits).mean()
            self.sequence_discriminator_optimizer.zero_grad()
            seq_realism_loss.backward()
            self.sequence_discriminator_optimizer.step()

        self.last_realism_loss = float(realism_loss.item())
        self.last_seq_realism_loss = float(seq_realism_loss.item())
        self.last_attack_obj = float(attack_obj.item())
        self.last_g_loss = float(g_loss.item())
