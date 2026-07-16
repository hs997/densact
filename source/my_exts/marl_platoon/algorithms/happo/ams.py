"""Action Manifold Smoothing adapted to task-local on-policy HAPPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HAPPOAMSCfg:
    share_obs_dim: int
    joint_action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: type[nn.Module] = nn.ELU
    gamma: float = 0.99
    lr: float = 1.0e-4
    tau: float = 0.01
    num_neighbors: int = 8
    neighborhood_radius: float = 0.25
    action_limit: float = 0.4
    refresh_every: int = 1000
    huber_beta: float = 0.3
    q_epochs: int = 1
    num_mini_batches: int = 16
    max_grad_norm: float = 0.5
    target_clip: float = 100.0
    eval_chunk_size: int = 8192


class ActionValueNet(nn.Module):
    def __init__(self, cfg: HAPPOAMSCfg):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = cfg.share_obs_dim + cfg.joint_action_dim
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(cfg.activation())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, share_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((share_obs, joint_actions), dim=-1)
        return self.net(inputs)


class OrthogonalActionSampler:
    """QR-based action directions matching the geometric core of AMS."""

    def __init__(self, action_dim: int, num_neighbors: int, device: torch.device, refresh_every: int):
        self.action_dim = int(action_dim)
        self.num_neighbors = min(max(int(num_neighbors), 1), self.action_dim)
        self.device = device
        self.refresh_every = max(int(refresh_every), 1)
        self.calls = 0
        self.basis = torch.empty((self.action_dim, self.num_neighbors), device=device)
        self._refresh()

    def _refresh(self) -> None:
        random_matrix = torch.randn(self.action_dim, self.action_dim, device=self.device)
        basis, _ = torch.linalg.qr(random_matrix)
        self.basis.copy_(basis[:, : self.num_neighbors])

    def sample(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        self.calls += 1
        if self.calls % self.refresh_every == 0:
            self._refresh()
        directions = self.basis.T.to(dtype=dtype).unsqueeze(0)
        return directions.expand(int(batch_size), -1, -1)


class HAPPOActionManifoldSmoother:
    """Twin auxiliary Q critic with symmetric action-neighborhood targets."""

    def __init__(self, cfg: HAPPOAMSCfg, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        if cfg.share_obs_dim <= 0 or cfg.joint_action_dim <= 0:
            raise ValueError("AMS observation and joint-action dimensions must be positive.")
        if cfg.action_limit <= 0.0:
            raise ValueError("AMS action_limit must be positive.")
        if cfg.neighborhood_radius < 0.0:
            raise ValueError("AMS neighborhood_radius must be non-negative.")
        if not 0.0 <= cfg.gamma <= 1.0:
            raise ValueError("AMS gamma must be in [0, 1].")
        if not 0.0 < cfg.tau <= 1.0:
            raise ValueError("AMS tau must be in (0, 1].")
        if cfg.lr <= 0.0 or cfg.target_clip <= 0.0:
            raise ValueError("AMS lr and target_clip must be positive.")
        self.q1 = ActionValueNet(cfg).to(self.device)
        self.q2 = ActionValueNet(cfg).to(self.device)
        self.target_q1 = ActionValueNet(cfg).to(self.device)
        self.target_q2 = ActionValueNet(cfg).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.optimizer = torch.optim.Adam((*self.q1.parameters(), *self.q2.parameters()), lr=cfg.lr)
        self.sampler = OrthogonalActionSampler(
            cfg.joint_action_dim,
            cfg.num_neighbors,
            self.device,
            cfg.refresh_every,
        )

    def _min_q(
        self,
        share_obs: torch.Tensor,
        joint_actions: torch.Tensor,
        *,
        target: bool,
    ) -> torch.Tensor:
        q1 = self.target_q1 if target else self.q1
        q2 = self.target_q2 if target else self.q2
        outputs = []
        chunk_size = max(int(self.cfg.eval_chunk_size), 1)
        for start in range(0, share_obs.shape[0], chunk_size):
            end = min(start + chunk_size, share_obs.shape[0])
            first = q1(share_obs[start:end], joint_actions[start:end])
            second = q2(share_obs[start:end], joint_actions[start:end])
            outputs.append(torch.minimum(first, second))
        return torch.cat(outputs, dim=0)

    def smoothed_q(
        self,
        share_obs: torch.Tensor,
        center_actions: torch.Tensor,
        *,
        target: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(share_obs.shape[0])
        directions = self.sampler.sample(batch_size, center_actions.dtype)
        epsilon = float(self.cfg.neighborhood_radius) * float(self.cfg.action_limit)
        offsets = epsilon * directions
        limit = abs(float(self.cfg.action_limit))
        plus = torch.clamp(center_actions.unsqueeze(1) + offsets, -limit, limit)
        minus = torch.clamp(center_actions.unsqueeze(1) - offsets, -limit, limit)
        neighbors = torch.cat((plus, minus), dim=1)
        num_samples = int(neighbors.shape[1])
        obs_flat = share_obs.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, share_obs.shape[-1])
        action_flat = neighbors.reshape(-1, center_actions.shape[-1])
        values = self._min_q(obs_flat, action_flat, target=target).reshape(batch_size, num_samples)
        return values.mean(dim=1, keepdim=True), values.var(dim=1, unbiased=False, keepdim=True)

    def train_on_buffer(self, buffer, next_policy_actions: torch.Tensor) -> dict[str, float]:
        batch_size = buffer.cfg.episode_length * buffer.cfg.num_envs
        share_obs = buffer.share_obs[:-1].reshape(batch_size, buffer.cfg.share_obs_dim)
        next_share_obs = buffer.share_obs[1:].reshape(batch_size, buffer.cfg.share_obs_dim)
        limit = abs(float(self.cfg.action_limit))
        joint_actions = buffer.actions.reshape(batch_size, self.cfg.joint_action_dim).clamp(-limit, limit)
        next_policy_actions = next_policy_actions.reshape(batch_size, self.cfg.joint_action_dim)
        rewards = buffer.rewards.mean(dim=2).reshape(batch_size, 1)
        masks = (~torch.all(buffer.dones, dim=2)).to(dtype=torch.float32).reshape(batch_size, 1)

        totals = {
            "ams_q_loss": 0.0,
            "ams_q_grad_norm": 0.0,
            "ams_anchor_q_mean": 0.0,
            "ams_target_q_mean": 0.0,
            "ams_neighborhood_var": 0.0,
        }
        num_updates = 0
        mini_batch_size = max(batch_size // max(int(self.cfg.num_mini_batches), 1), 1)
        with torch.inference_mode(False), torch.enable_grad():
            for _ in range(max(int(self.cfg.q_epochs), 1)):
                indices = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, mini_batch_size):
                    mb = indices[start : start + mini_batch_size]
                    with torch.no_grad():
                        next_q, neighbor_var = self.smoothed_q(
                            next_share_obs[mb],
                            next_policy_actions[mb],
                            target=True,
                        )
                        target_q = rewards[mb] + float(self.cfg.gamma) * masks[mb] * next_q
                        target_clip = abs(float(self.cfg.target_clip))
                        target_q = torch.nan_to_num(target_q, nan=0.0, posinf=target_clip, neginf=-target_clip)
                        target_q = torch.clamp(target_q, -target_clip, target_clip)

                    q1_values = self.q1(share_obs[mb], joint_actions[mb])
                    q2_values = self.q2(share_obs[mb], joint_actions[mb])
                    q_loss = F.smooth_l1_loss(q1_values, target_q, beta=float(self.cfg.huber_beta))
                    q_loss = q_loss + F.smooth_l1_loss(q2_values, target_q, beta=float(self.cfg.huber_beta))
                    if not torch.isfinite(q_loss):
                        continue

                    self.optimizer.zero_grad()
                    q_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        (*self.q1.parameters(), *self.q2.parameters()),
                        float(self.cfg.max_grad_norm),
                    )
                    self.optimizer.step()

                    totals["ams_q_loss"] += float(q_loss.item())
                    totals["ams_q_grad_norm"] += float(grad_norm)
                    totals["ams_anchor_q_mean"] += float(torch.minimum(q1_values, q2_values).mean().item())
                    totals["ams_target_q_mean"] += float(target_q.mean().item())
                    totals["ams_neighborhood_var"] += float(neighbor_var.mean().item())
                    num_updates += 1

        self._polyak_update()
        if num_updates > 0:
            for key in totals:
                totals[key] /= num_updates
        totals["ams_num_neighbors"] = float(self.sampler.num_neighbors)
        totals["ams_radius"] = float(self.cfg.neighborhood_radius)
        return totals

    @torch.no_grad()
    def local_advantage(
        self,
        share_obs: torch.Tensor,
        executed_actions: torch.Tensor,
        policy_center_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        anchor_q = self._min_q(share_obs, executed_actions, target=False)
        neighborhood_q, neighborhood_var = self.smoothed_q(share_obs, policy_center_actions, target=False)
        advantage = torch.nan_to_num(anchor_q - neighborhood_q, nan=0.0, posinf=0.0, neginf=0.0)
        info = {
            "ams_local_adv_mean": float(advantage.mean().item()),
            "ams_local_adv_std": float(advantage.std(unbiased=False).item()),
            "ams_neighborhood_q_mean": float(neighborhood_q.mean().item()),
            "ams_neighborhood_var": float(neighborhood_var.mean().item()),
        }
        return advantage, info

    @torch.no_grad()
    def _polyak_update(self) -> None:
        tau = float(self.cfg.tau)
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)

    def state_dict(self) -> dict:
        return {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sampler_basis": self.sampler.basis.detach().clone(),
            "sampler_calls": self.sampler.calls,
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.q1.load_state_dict(state["q1"], strict=strict)
        self.q2.load_state_dict(state["q2"], strict=strict)
        self.target_q1.load_state_dict(state.get("target_q1", state["q1"]), strict=strict)
        self.target_q2.load_state_dict(state.get("target_q2", state["q2"]), strict=strict)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        basis = state.get("sampler_basis")
        if basis is not None and basis.shape == self.sampler.basis.shape:
            self.sampler.basis.copy_(basis.to(self.device))
        self.sampler.calls = int(state.get("sampler_calls", self.sampler.calls))

    def prep_training(self) -> None:
        self.q1.train()
        self.q2.train()
        self.target_q1.eval()
        self.target_q2.eval()

    def prep_rollout(self) -> None:
        self.q1.eval()
        self.q2.eval()
        self.target_q1.eval()
        self.target_q2.eval()
