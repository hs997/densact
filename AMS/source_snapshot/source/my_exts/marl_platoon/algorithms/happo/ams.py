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
    geometry: str = "ackermann"
    wheelbase_m: float = 0.14325
    max_speed_mps: float = 1.20
    max_steering_rad: float = 0.32
    curvature_limit: float = 3.0
    risk_coef: float = 0.0
    use_executed_action: bool = True


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
        if cfg.geometry not in {"ackermann", "euclidean"}:
            raise ValueError("AMS geometry must be ackermann or euclidean")
        if cfg.geometry == "ackermann" and (cfg.joint_action_dim % 2 or cfg.wheelbase_m <= 0 or cfg.max_steering_rad <= 0):
            raise ValueError("Ackermann AMS needs speed/steering pairs and positive geometry")
        if cfg.risk_coef < 0:
            raise ValueError("AMS risk_coef must be nonnegative")
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

    def project_feasible(self, actions: torch.Tensor) -> torch.Tensor:
        """Project normalized joint actions onto the Ackermann action set."""
        limit = abs(float(self.cfg.action_limit))
        result = torch.clamp(actions, -limit, limit)
        if str(self.cfg.geometry).lower() != "ackermann" or result.shape[-1] < 2:
            return result
        steer_bound = abs(float(self.cfg.max_steering_rad))
        if float(self.cfg.curvature_limit) > 0.0:
            curvature_steer = torch.atan(
                torch.as_tensor(
                    float(self.cfg.wheelbase_m) * float(self.cfg.curvature_limit),
                    device=result.device,
                    dtype=result.dtype,
                )
            )
            steer_bound = min(steer_bound, abs(float(curvature_steer.item())))
        normalized_bound = min(1.0, steer_bound / max(abs(float(self.cfg.max_steering_rad)), 1.0e-6))
        for steer_index in range(1, result.shape[-1], 2):
            result[..., steer_index] = torch.clamp(result[..., steer_index], -normalized_bound, normalized_bound)
        return result

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
        plus = self.project_feasible(center_actions.unsqueeze(1) + offsets)
        minus = self.project_feasible(center_actions.unsqueeze(1) - offsets)
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
        source_actions = buffer.executed_actions if self.cfg.use_executed_action else buffer.actions
        # Do not relabel an observed transition by projecting its action to a
        # different command. Feasibility projection is only for counterfactuals.
        joint_actions = source_actions.reshape(batch_size, self.cfg.joint_action_dim)
        next_policy_actions = self.project_feasible(next_policy_actions.reshape(batch_size, self.cfg.joint_action_dim))
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
        policy_center_actions = self.project_feasible(policy_center_actions)
        anchor_q = self._min_q(share_obs, executed_actions, target=False)
        neighborhood_q, neighborhood_var = self.smoothed_q(share_obs, policy_center_actions, target=False)
        advantage = torch.nan_to_num(
            anchor_q - neighborhood_q - float(self.cfg.risk_coef) * torch.sqrt(neighborhood_var.clamp_min(0.0)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        info = {
            "ams_local_adv_mean": float(advantage.mean().item()),
            "ams_local_adv_std": float(advantage.std(unbiased=False).item()),
            "ams_neighborhood_q_mean": float(neighborhood_q.mean().item()),
            "ams_neighborhood_var": float(neighborhood_var.mean().item()),
        }
        return advantage, info

    @torch.no_grad()
    def local_agent_advantage(
        self,
        share_obs: torch.Tensor,
        executed_actions: torch.Tensor,
        policy_center_actions: torch.Tensor,
        num_agents: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Smooth one vehicle's action block at a time for local credit."""
        policy_center_actions = self.project_feasible(policy_center_actions)
        batch_size = int(share_obs.shape[0])
        if num_agents <= 0 or policy_center_actions.shape[-1] % num_agents:
            raise ValueError("Joint action dimension must be divisible by num_agents")
        act_dim = policy_center_actions.shape[-1] // num_agents
        anchor_q = self._min_q(share_obs, executed_actions, target=False)
        directions = self.sampler.sample(batch_size, policy_center_actions.dtype)
        offsets = float(self.cfg.neighborhood_radius) * float(self.cfg.action_limit) * directions
        local_values = []
        local_vars = []
        for agent_id in range(int(num_agents)):
            start, end = agent_id * act_dim, (agent_id + 1) * act_dim
            # Hold all OTHER cars at their executed action. Replace only this
            # car by its policy center/neighborhood (conditional local credit).
            center = executed_actions.clone()
            center[:, start:end] = policy_center_actions[:, start:end]
            local_offsets = offsets.clone()
            local_offsets[..., :start] = 0.0
            local_offsets[..., end:] = 0.0
            neighbors = torch.cat(
                (
                    self.project_feasible(center.unsqueeze(1) + local_offsets),
                    self.project_feasible(center.unsqueeze(1) - local_offsets),
                ),
                dim=1,
            )
            neighbors[..., :start] = executed_actions[:, None, :start]
            neighbors[..., end:] = executed_actions[:, None, end:]
            sample_count = int(neighbors.shape[1])
            obs_flat = share_obs.unsqueeze(1).expand(-1, sample_count, -1).reshape(-1, share_obs.shape[-1])
            action_flat = neighbors.reshape(-1, policy_center_actions.shape[-1])
            values = self._min_q(obs_flat, action_flat, target=False).reshape(batch_size, sample_count)
            local_values.append(values.mean(dim=1, keepdim=True))
            local_vars.append(values.var(dim=1, unbiased=False, keepdim=True))
        neighborhood_q = torch.cat(local_values, dim=1)
        neighborhood_var = torch.cat(local_vars, dim=1)
        advantage = anchor_q - neighborhood_q - float(self.cfg.risk_coef) * torch.sqrt(neighborhood_var.clamp_min(0.0))
        info = {
            "ams_local_adv_mean": float(advantage.mean().item()),
            "ams_local_adv_std": float(advantage.std(unbiased=False).item()),
            "ams_neighborhood_q_mean": float(neighborhood_q.mean().item()),
            "ams_neighborhood_var": float(neighborhood_var.mean().item()),
        }
        info["ams_risk_coef"] = float(self.cfg.risk_coef)
        return torch.nan_to_num(advantage, nan=0.0, posinf=0.0, neginf=0.0), info

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
