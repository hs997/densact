"""HAPPO actor update logic for the marl_platoon task.

This is a task-local port of the actor-side objective from
`HARL-main/harl/algorithms/actors/happo.py`. It intentionally avoids importing
HARL runners or environment factories so the IsaacLab official entrypoint stays
in control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


@dataclass
class HAPPOActorCfg:
    obs_dim: int
    act_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: type[nn.Module] = nn.ELU
    init_noise_std: float = 1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    clip_param: float = 0.2
    ppo_epoch: int = 5
    actor_num_mini_batch: int = 4
    entropy_coef: float = 0.01
    lr: float = 3.0e-4
    max_grad_norm: float = 1.0
    action_aggregation: str = "prod"
    use_policy_active_masks: bool = True
    update_mode: str = "ppo"
    trpo_kl_threshold: float = 0.01
    trpo_cg_iters: int = 10
    trpo_damping: float = 0.1
    trpo_line_search_steps: int = 10
    trpo_accept_ratio: float = 0.5
    trpo_backtrack_coeff: float = 0.8


class GaussianActorNet(nn.Module):
    """Simple continuous Gaussian actor compatible with IsaacLab tensor actions."""

    def __init__(self, cfg: HAPPOActorCfg):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        last_dim = cfg.obs_dim
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(cfg.activation())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, cfg.act_dim))
        self.mean_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.ones(cfg.act_dim) * torch.log(torch.tensor(cfg.init_noise_std)))

    def distribution(self, obs: torch.Tensor) -> Independent:
        mean = self.mean_net(obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        safe_log_std = torch.nan_to_num(
            self.log_std,
            nan=0.0,
            posinf=float(self.cfg.log_std_max),
            neginf=float(self.cfg.log_std_min),
        )
        safe_log_std = torch.clamp(
            safe_log_std,
            min=float(self.cfg.log_std_min),
            max=float(self.cfg.log_std_max),
        )
        std = torch.exp(safe_log_std).clamp_min(1.0e-6).expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        actions = dist.mean if deterministic else dist.rsample()
        action_log_probs = dist.log_prob(actions).unsqueeze(-1)
        return actions, action_log_probs

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action_log_probs = dist.log_prob(actions).unsqueeze(-1)
        entropy = dist.entropy().mean()
        return action_log_probs, entropy


class HAPPOActor:
    """Actor-side HAPPO objective with HARL-style factor weighting."""

    def __init__(self, cfg: HAPPOActorCfg, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = GaussianActorNet(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs = obs.to(self.device, dtype=torch.float32)
            return self.actor(obs, deterministic=deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode(False), torch.enable_grad():
            obs = obs.detach().clone().to(self.device, dtype=torch.float32)
            actions = actions.detach().clone().to(self.device, dtype=torch.float32)
            return self.actor.evaluate_actions(obs, actions)

    def update(self, sample: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if str(self.cfg.update_mode).lower() == "trpo":
            return self._trpo_update(sample)

        old_action_log_probs = sample["old_action_log_probs"].detach().clone().to(self.device, dtype=torch.float32)
        adv_targ = sample["advantages"].detach().clone().to(self.device, dtype=torch.float32)
        active_masks = sample["active_masks"].detach().clone().to(self.device, dtype=torch.float32)
        factor = sample["factor"].detach().clone().to(self.device, dtype=torch.float32)

        # RSL-RL may call env.step() from an inference/no-grad rollout context.
        # HAPPO updates happen from that path, so explicitly re-enable autograd
        # and disable inference mode before building the actor loss graph.
        with torch.inference_mode(False), torch.enable_grad():
            action_log_probs, dist_entropy = self.evaluate_actions(sample["obs"], sample["actions"])
            if not torch.isfinite(action_log_probs).all() or not torch.isfinite(dist_entropy):
                zero = torch.tensor(0.0, device=self.device)
                one = torch.ones_like(old_action_log_probs)
                return zero, zero, zero, one
            imp_weights = getattr(torch, self.cfg.action_aggregation)(
                torch.exp(torch.clamp(action_log_probs - old_action_log_probs, min=-20.0, max=20.0)),
                dim=-1,
                keepdim=True,
            )
            if str(self.cfg.update_mode).lower() == "a2c":
                surrogate = imp_weights * adv_targ
            else:
                surr1 = imp_weights * adv_targ
                surr2 = torch.clamp(imp_weights, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * adv_targ
                surrogate = torch.min(surr1, surr2)

            policy_terms = -torch.sum(factor * surrogate, dim=-1, keepdim=True)
            if self.cfg.use_policy_active_masks:
                policy_loss = (policy_terms * active_masks).sum() / active_masks.sum().clamp_min(1.0)
            else:
                policy_loss = policy_terms.mean()

            self.optimizer.zero_grad()
            (policy_loss - dist_entropy * self.cfg.entropy_coef).backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                self.actor.log_std.clamp_(min=float(self.cfg.log_std_min), max=float(self.cfg.log_std_max))
        return policy_loss, dist_entropy, grad_norm, imp_weights.detach()

    def _trpo_update(self, sample: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = sample["obs"].detach().clone().to(self.device, dtype=torch.float32)
        actions = sample["actions"].detach().clone().to(self.device, dtype=torch.float32)
        old_action_log_probs = sample["old_action_log_probs"].detach().clone().to(self.device, dtype=torch.float32)
        adv_targ = sample["advantages"].detach().clone().to(self.device, dtype=torch.float32)
        active_masks = sample["active_masks"].detach().clone().to(self.device, dtype=torch.float32)
        factor = sample["factor"].detach().clone().to(self.device, dtype=torch.float32)

        with torch.inference_mode(False), torch.enable_grad():
            with torch.no_grad():
                old_dist = self.actor.distribution(obs)
                old_mean = old_dist.base_dist.loc.detach().clone()
                old_std = old_dist.base_dist.scale.detach().clone()
                old_params = self._flat_params().detach().clone()

            objective, dist_entropy, ratio = self._trpo_objective(
                obs,
                actions,
                old_action_log_probs,
                adv_targ,
                active_masks,
                factor,
            )
            grads = torch.autograd.grad(objective, self.actor.parameters(), allow_unused=True)
            loss_grad = self._flat_grad(grads).detach()
            grad_norm = torch.norm(loss_grad)
            if not torch.isfinite(grad_norm) or float(grad_norm.item()) <= 1.0e-12:
                zero = torch.tensor(0.0, device=self.device)
                return -objective.detach(), dist_entropy.detach(), zero, ratio.detach()

            step_dir = self._conjugate_gradient(
                lambda vector: self._fisher_vector_product(obs, old_mean, old_std, active_masks, vector),
                loss_grad,
                int(self.cfg.trpo_cg_iters),
            )
            fvp_step = self._fisher_vector_product(obs, old_mean, old_std, active_masks, step_dir)
            shs = 0.5 * torch.dot(step_dir, fvp_step)
            if not torch.isfinite(shs) or float(shs.item()) <= 1.0e-12:
                self._set_flat_params(old_params)
                zero = torch.tensor(0.0, device=self.device)
                return -objective.detach(), dist_entropy.detach(), zero, ratio.detach()

            full_step = torch.sqrt(torch.tensor(float(self.cfg.trpo_kl_threshold), device=self.device) / shs) * step_dir
            expected_improve = torch.dot(loss_grad, full_step).detach()
            old_objective = objective.detach()
            accepted_objective = old_objective
            accepted_entropy = dist_entropy.detach()
            accepted_ratio = ratio.detach()
            accepted = False

            for step_idx in range(int(self.cfg.trpo_line_search_steps)):
                fraction = float(self.cfg.trpo_backtrack_coeff) ** step_idx
                self._set_flat_params(old_params + fraction * full_step)
                new_objective, new_entropy, new_ratio = self._trpo_objective(
                    obs,
                    actions,
                    old_action_log_probs,
                    adv_targ,
                    active_masks,
                    factor,
                )
                kl = self._normal_kl(old_mean, old_std, obs, active_masks).mean()
                improve = new_objective.detach() - old_objective
                expected = expected_improve * fraction
                improve_ratio = improve / expected.clamp_min(1.0e-12)
                if (
                    torch.isfinite(kl)
                    and torch.isfinite(improve)
                    and float(kl.item()) <= float(self.cfg.trpo_kl_threshold)
                    and float(improve.item()) > 0.0
                    and float(improve_ratio.item()) >= float(self.cfg.trpo_accept_ratio)
                ):
                    accepted_objective = new_objective.detach()
                    accepted_entropy = new_entropy.detach()
                    accepted_ratio = new_ratio.detach()
                    accepted = True
                    break

            if not accepted:
                self._set_flat_params(old_params)

            with torch.no_grad():
                self.actor.log_std.clamp_(min=float(self.cfg.log_std_min), max=float(self.cfg.log_std_max))
            return -accepted_objective, accepted_entropy, grad_norm, accepted_ratio

    def _trpo_objective(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_action_log_probs: torch.Tensor,
        adv_targ: torch.Tensor,
        active_masks: torch.Tensor,
        factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, actions)
        ratio = torch.exp(torch.clamp(action_log_probs - old_action_log_probs, min=-20.0, max=20.0))
        surrogate = torch.sum(ratio * factor * adv_targ, dim=-1, keepdim=True)
        if self.cfg.use_policy_active_masks:
            objective = (surrogate * active_masks).sum() / active_masks.sum().clamp_min(1.0)
        else:
            objective = surrogate.mean()
        return objective, dist_entropy, ratio

    def _normal_kl(
        self,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        obs: torch.Tensor,
        active_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        new_dist = self.actor.distribution(obs)
        new_mean = new_dist.base_dist.loc
        new_std = new_dist.base_dist.scale
        var_ratio = (old_std / new_std).pow(2)
        mean_term = ((new_mean - old_mean) / new_std).pow(2)
        kl = 0.5 * (var_ratio + mean_term - 1.0 - torch.log(var_ratio.clamp_min(1.0e-12)))
        kl = kl.sum(dim=-1, keepdim=True)
        if active_masks is not None and self.cfg.use_policy_active_masks:
            return kl * active_masks
        return kl

    def _fisher_vector_product(
        self,
        obs: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        active_masks: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        kl = self._normal_kl(old_mean, old_std, obs, active_masks)
        if self.cfg.use_policy_active_masks:
            kl_mean = kl.sum() / active_masks.sum().clamp_min(1.0)
        else:
            kl_mean = kl.mean()
        kl_grad = torch.autograd.grad(kl_mean, self.actor.parameters(), create_graph=True, allow_unused=True)
        flat_kl_grad = self._flat_grad(kl_grad)
        kl_grad_vector = torch.dot(flat_kl_grad, vector)
        kl_hessian_vector = torch.autograd.grad(kl_grad_vector, self.actor.parameters(), allow_unused=True)
        return self._flat_grad(kl_hessian_vector).detach() + float(self.cfg.trpo_damping) * vector

    def _conjugate_gradient(self, avp_fn, b: torch.Tensor, nsteps: int, residual_tol: float = 1.0e-10) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(max(nsteps, 1)):
            avp = avp_fn(p)
            denom = torch.dot(p, avp).clamp_min(1.0e-12)
            alpha = rdotr / denom
            x = x + alpha * p
            r = r - alpha * avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr.clamp_min(1.0e-12)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _flat_params(self) -> torch.Tensor:
        return torch.cat([param.data.view(-1) for param in self.actor.parameters()])

    def _set_flat_params(self, flat_params: torch.Tensor) -> None:
        offset = 0
        for param in self.actor.parameters():
            numel = param.numel()
            param.data.copy_(flat_params[offset : offset + numel].view_as(param))
            offset += numel

    def _flat_grad(self, grads) -> torch.Tensor:
        flat = []
        for param, grad in zip(self.actor.parameters(), grads):
            if grad is None:
                flat.append(torch.zeros_like(param).view(-1))
            else:
                flat.append(grad.contiguous().view(-1))
        return torch.cat(flat)

    def train_on_buffer(self, buffer, advantages: torch.Tensor, agent_id: int, factor: torch.Tensor) -> dict[str, float]:
        train_info = {"policy_loss": 0.0, "dist_entropy": 0.0, "actor_grad_norm": 0.0, "ratio": 0.0}
        num_updates = 0
        if str(self.cfg.update_mode).lower() == "trpo":
            sample = buffer.actor_full_batch(agent_id, advantages, factor)
            policy_loss, dist_entropy, grad_norm, imp_weights = self.update(sample)
            return {
                "policy_loss": float(policy_loss.item()),
                "dist_entropy": float(dist_entropy.item()),
                "actor_grad_norm": float(grad_norm),
                "ratio": float(imp_weights.mean().item()),
            }
        for _ in range(self.cfg.ppo_epoch):
            for sample in buffer.actor_minibatches(agent_id, advantages, factor, self.cfg.actor_num_mini_batch):
                policy_loss, dist_entropy, grad_norm, imp_weights = self.update(sample)
                train_info["policy_loss"] += float(policy_loss.item())
                train_info["dist_entropy"] += float(dist_entropy.item())
                train_info["actor_grad_norm"] += float(grad_norm)
                train_info["ratio"] += float(imp_weights.mean().item())
                num_updates += 1
        if num_updates > 0:
            for key in train_info:
                train_info[key] /= num_updates
        return train_info

    def train_shared_on_buffer(self, buffer, advantages: torch.Tensor, factor: torch.Tensor) -> dict[str, float]:
        train_info = {"policy_loss": 0.0, "dist_entropy": 0.0, "actor_grad_norm": 0.0, "ratio": 0.0}
        num_updates = 0
        for _ in range(self.cfg.ppo_epoch):
            for sample in buffer.shared_actor_minibatches(advantages, factor, self.cfg.actor_num_mini_batch):
                policy_loss, dist_entropy, grad_norm, imp_weights = self.update(sample)
                train_info["policy_loss"] += float(policy_loss.item())
                train_info["dist_entropy"] += float(dist_entropy.item())
                train_info["actor_grad_norm"] += float(grad_norm)
                train_info["ratio"] += float(imp_weights.mean().item())
                num_updates += 1
        if num_updates > 0:
            for key in train_info:
                train_info[key] /= num_updates
        return train_info

    def prep_training(self) -> None:
        self.actor.train()

    def prep_rollout(self) -> None:
        self.actor.eval()
