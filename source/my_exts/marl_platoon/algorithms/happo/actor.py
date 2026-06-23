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
            surr1 = imp_weights * adv_targ
            surr2 = torch.clamp(imp_weights, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * adv_targ

            policy_terms = -torch.sum(factor * torch.min(surr1, surr2), dim=-1, keepdim=True)
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

    def train_on_buffer(self, buffer, advantages: torch.Tensor, agent_id: int, factor: torch.Tensor) -> dict[str, float]:
        train_info = {"policy_loss": 0.0, "dist_entropy": 0.0, "actor_grad_norm": 0.0, "ratio": 0.0}
        num_updates = 0
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

    def prep_training(self) -> None:
        self.actor.train()

    def prep_rollout(self) -> None:
        self.actor.eval()
