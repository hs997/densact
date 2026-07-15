"""Rollout buffer for task-local HAPPO.

Shapes follow HARL's on-policy runner convention:
- obs: `[T + 1, N, A, obs_dim]`
- share_obs: `[T + 1, N, share_obs_dim]`
- actions/log_probs/rewards/dones: rollout-major tensors
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HAPPORolloutBufferCfg:
    episode_length: int
    num_envs: int
    num_agents: int
    obs_dim: int
    share_obs_dim: int
    act_dim: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"


class HAPPORolloutBuffer:
    """Minimal feed-forward rollout buffer for HAPPO in IsaacLab tasks."""

    def __init__(self, cfg: HAPPORolloutBufferCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        shape_prefix = (cfg.episode_length + 1, cfg.num_envs)
        agent_prefix = (cfg.episode_length + 1, cfg.num_envs, cfg.num_agents)
        step_agent_prefix = (cfg.episode_length, cfg.num_envs, cfg.num_agents)

        self.obs = torch.zeros((*agent_prefix, cfg.obs_dim), device=self.device)
        self.share_obs = torch.zeros((*shape_prefix, cfg.share_obs_dim), device=self.device)
        self.actions = torch.zeros((*step_agent_prefix, cfg.act_dim), device=self.device)
        self.action_log_probs = torch.zeros((*step_agent_prefix, 1), device=self.device)
        self.rewards = torch.zeros((*step_agent_prefix, 1), device=self.device)
        self.dones = torch.zeros((*step_agent_prefix, 1), dtype=torch.bool, device=self.device)
        self.active_masks = torch.ones((*agent_prefix, 1), device=self.device)
        self.value_preds = torch.zeros((cfg.episode_length + 1, cfg.num_envs, 1), device=self.device)
        self.returns = torch.zeros_like(self.value_preds)
        self.step = 0

    def insert(
        self,
        obs: torch.Tensor,
        share_obs: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        idx = self.step
        self.obs[idx + 1].copy_(obs.detach().clone().to(self.device, dtype=torch.float32))
        self.share_obs[idx + 1].copy_(share_obs.detach().clone().to(self.device, dtype=torch.float32))
        self.actions[idx].copy_(actions.detach().clone().to(self.device, dtype=torch.float32))
        self.action_log_probs[idx].copy_(action_log_probs.detach().clone().to(self.device, dtype=torch.float32))
        self.rewards[idx].copy_(rewards.detach().clone().to(self.device, dtype=torch.float32))
        dones_tensor = dones.detach().clone().to(self.device, dtype=torch.bool).unsqueeze(-1) if dones.dim() == 2 else dones.detach().clone().to(self.device, dtype=torch.bool)
        self.dones[idx].copy_(dones_tensor)
        self.value_preds[idx].copy_(values.detach().clone().to(self.device, dtype=torch.float32))
        self.active_masks[idx + 1].copy_((~dones_tensor).to(dtype=torch.float32))
        self.step = (self.step + 1) % self.cfg.episode_length

    def set_initial_obs(self, obs: torch.Tensor, share_obs: torch.Tensor) -> None:
        self.obs[0].copy_(obs.to(self.device, dtype=torch.float32))
        self.share_obs[0].copy_(share_obs.to(self.device, dtype=torch.float32))

    def compute_returns(self, next_values: torch.Tensor) -> torch.Tensor:
        self.value_preds[-1].copy_(next_values.to(self.device, dtype=torch.float32))
        gae = torch.zeros((self.cfg.num_envs, 1), device=self.device)
        for step in reversed(range(self.cfg.episode_length)):
            mask = (~torch.all(self.dones[step], dim=1)).to(dtype=torch.float32)
            reward = self.rewards[step].mean(dim=1)
            delta = reward + self.cfg.gamma * self.value_preds[step + 1] * mask - self.value_preds[step]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            self.returns[step] = gae + self.value_preds[step]
        advantages = self.returns[:-1] - self.value_preds[:-1]
        return advantages.unsqueeze(2).repeat(1, 1, self.cfg.num_agents, 1)

    def actor_minibatches(
        self,
        agent_id: int,
        advantages: torch.Tensor,
        factor: torch.Tensor,
        num_mini_batches: int,
    ):
        batch_size = self.cfg.episode_length * self.cfg.num_envs
        mini_batch_size = max(batch_size // num_mini_batches, 1)
        indices = torch.randperm(batch_size, device=self.device)

        obs = self.obs[:-1, :, agent_id].reshape(batch_size, self.cfg.obs_dim)
        actions = self.actions[:, :, agent_id].reshape(batch_size, self.cfg.act_dim)
        old_log_probs = self.action_log_probs[:, :, agent_id].reshape(batch_size, 1)
        adv = advantages[:, :, agent_id].reshape(batch_size, 1)
        masks = self.active_masks[:-1, :, agent_id].reshape(batch_size, 1)
        factor_flat = factor.reshape(batch_size, 1)

        for start in range(0, batch_size, mini_batch_size):
            mb_inds = indices[start : start + mini_batch_size]
            yield {
                "obs": obs[mb_inds].clone(),
                "actions": actions[mb_inds].clone(),
                "old_action_log_probs": old_log_probs[mb_inds].clone(),
                "advantages": adv[mb_inds].clone(),
                "active_masks": masks[mb_inds].clone(),
                "factor": factor_flat[mb_inds].clone(),
            }

    def actor_full_batch(self, agent_id: int, advantages: torch.Tensor, factor: torch.Tensor):
        batch_size = self.cfg.episode_length * self.cfg.num_envs
        return {
            "obs": self.obs[:-1, :, agent_id].reshape(batch_size, self.cfg.obs_dim).clone(),
            "actions": self.actions[:, :, agent_id].reshape(batch_size, self.cfg.act_dim).clone(),
            "old_action_log_probs": self.action_log_probs[:, :, agent_id].reshape(batch_size, 1).clone(),
            "advantages": advantages[:, :, agent_id].reshape(batch_size, 1).clone(),
            "active_masks": self.active_masks[:-1, :, agent_id].reshape(batch_size, 1).clone(),
            "factor": factor.reshape(batch_size, 1).clone(),
        }

    def shared_actor_minibatches(
        self,
        advantages: torch.Tensor,
        factor: torch.Tensor,
        num_mini_batches: int,
    ):
        batch_size = self.cfg.episode_length * self.cfg.num_envs * self.cfg.num_agents
        mini_batch_size = max(batch_size // num_mini_batches, 1)
        indices = torch.randperm(batch_size, device=self.device)

        obs = self.obs[:-1].reshape(batch_size, self.cfg.obs_dim)
        actions = self.actions.reshape(batch_size, self.cfg.act_dim)
        old_log_probs = self.action_log_probs.reshape(batch_size, 1)
        adv = advantages.reshape(batch_size, 1)
        masks = self.active_masks[:-1].reshape(batch_size, 1)
        shared_factor = factor.unsqueeze(2).expand(-1, -1, self.cfg.num_agents, -1).reshape(batch_size, 1)

        for start in range(0, batch_size, mini_batch_size):
            mb_inds = indices[start : start + mini_batch_size]
            yield {
                "obs": obs[mb_inds].clone(),
                "actions": actions[mb_inds].clone(),
                "old_action_log_probs": old_log_probs[mb_inds].clone(),
                "advantages": adv[mb_inds].clone(),
                "active_masks": masks[mb_inds].clone(),
                "factor": shared_factor[mb_inds].clone(),
            }

    def critic_minibatches(self, num_mini_batches: int):
        batch_size = self.cfg.episode_length * self.cfg.num_envs
        mini_batch_size = max(batch_size // num_mini_batches, 1)
        indices = torch.randperm(batch_size, device=self.device)

        share_obs = self.share_obs[:-1].reshape(batch_size, self.cfg.share_obs_dim)
        value_preds = self.value_preds[:-1].reshape(batch_size, 1)
        returns = self.returns[:-1].reshape(batch_size, 1)

        for start in range(0, batch_size, mini_batch_size):
            mb_inds = indices[start : start + mini_batch_size]
            yield {
                "share_obs": share_obs[mb_inds].clone(),
                "value_preds": value_preds[mb_inds].clone(),
                "returns": returns[mb_inds].clone(),
            }

    def after_update(self) -> None:
        self.obs[0].copy_(self.obs[-1])
        self.share_obs[0].copy_(self.share_obs[-1])
        self.active_masks[0].copy_(self.active_masks[-1])
        self.step = 0
