"""Centralized critic for the task-local HAPPO implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HAPPOCriticCfg:
    share_obs_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: type[nn.Module] = nn.ELU
    clip_param: float = 0.2
    critic_epoch: int = 5
    critic_num_mini_batch: int = 4
    value_loss_coef: float = 1.0
    lr: float = 3.0e-4
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True


class ValueNet(nn.Module):
    def __init__(self, cfg: HAPPOCriticCfg):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = cfg.share_obs_dim
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(cfg.activation())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, share_obs: torch.Tensor) -> torch.Tensor:
        return self.net(share_obs)


class HAPPOCritic:
    """Centralized V critic mirroring HARL's `VCritic` at minimal scope."""

    def __init__(self, cfg: HAPPOCriticCfg, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.critic = ValueNet(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def get_values(self, share_obs: torch.Tensor) -> torch.Tensor:
        share_obs = share_obs.to(self.device, dtype=torch.float32)
        return self.critic(share_obs)

    def value_loss(self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_clipped_value_loss:
            value_pred_clipped = old_values + (values - old_values).clamp(-self.cfg.clip_param, self.cfg.clip_param)
            value_loss_original = (returns - values).pow(2)
            value_loss_clipped = (returns - value_pred_clipped).pow(2)
            return torch.max(value_loss_original, value_loss_clipped).mean()
        return (returns - values).pow(2).mean()

    def update(self, sample: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        share_obs = sample["share_obs"].detach().clone().to(self.device, dtype=torch.float32)
        old_values = sample["value_preds"].detach().clone().to(self.device, dtype=torch.float32)
        returns = sample["returns"].detach().clone().to(self.device, dtype=torch.float32)
        # RSL-RL may call env.step() from an inference/no-grad rollout context.
        # HAPPO updates happen from that path, so explicitly re-enable autograd
        # and disable inference mode before building the critic loss graph.
        with torch.inference_mode(False), torch.enable_grad():
            values = self.critic(share_obs)
            value_loss = self.value_loss(values, old_values, returns)

            self.optimizer.zero_grad()
            (value_loss * self.cfg.value_loss_coef).backward()
            grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
        return value_loss, grad_norm

    def train_on_buffer(self, buffer) -> dict[str, float]:
        train_info = {"value_loss": 0.0, "critic_grad_norm": 0.0}
        num_updates = 0
        for _ in range(self.cfg.critic_epoch):
            for sample in buffer.critic_minibatches(self.cfg.critic_num_mini_batch):
                value_loss, grad_norm = self.update(sample)
                train_info["value_loss"] += float(value_loss.item())
                train_info["critic_grad_norm"] += float(grad_norm)
                num_updates += 1
        if num_updates > 0:
            for key in train_info:
                train_info[key] /= num_updates
        return train_info

    def prep_training(self) -> None:
        self.critic.train()

    def prep_rollout(self) -> None:
        self.critic.eval()
