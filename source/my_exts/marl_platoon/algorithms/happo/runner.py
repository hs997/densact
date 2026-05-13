"""Task-local HAPPO runner skeleton for marl_platoon.

The official IsaacLab launch path remains responsible for creating the app and
environment. This runner is a reusable internal component that can later be
called from the platoon task algorithm switch when `algorithm == "happo"`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .actor import HAPPOActor, HAPPOActorCfg
from .buffer import HAPPORolloutBuffer, HAPPORolloutBufferCfg
from .critic import HAPPOCritic, HAPPOCriticCfg


@dataclass
class PlatoonHAPPOCfg:
    num_agents: int
    obs_dim: int
    act_dim: int
    share_obs_dim: int
    num_envs: int
    episode_length: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    hidden_dims: tuple[int, ...] = (256, 256)
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-4
    clip_param: float = 0.1
    ppo_epoch: int = 3
    num_mini_batches: int = 4
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cpu"
    fixed_order: bool = True


class PlatoonHAPPORunner:
    """Minimal HAPPO coordinator ported from HARL's HA runner structure."""

    def __init__(self, cfg: PlatoonHAPPOCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        actor_cfg = HAPPOActorCfg(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dims=cfg.hidden_dims,
            clip_param=cfg.clip_param,
            ppo_epoch=cfg.ppo_epoch,
            actor_num_mini_batch=cfg.num_mini_batches,
            entropy_coef=cfg.entropy_coef,
            lr=cfg.actor_lr,
            max_grad_norm=cfg.max_grad_norm,
        )
        critic_cfg = HAPPOCriticCfg(
            share_obs_dim=cfg.share_obs_dim,
            hidden_dims=cfg.hidden_dims,
            clip_param=cfg.clip_param,
            critic_epoch=cfg.ppo_epoch,
            critic_num_mini_batch=cfg.num_mini_batches,
            lr=cfg.critic_lr,
            max_grad_norm=cfg.max_grad_norm,
        )
        buffer_cfg = HAPPORolloutBufferCfg(
            episode_length=cfg.episode_length,
            num_envs=cfg.num_envs,
            num_agents=cfg.num_agents,
            obs_dim=cfg.obs_dim,
            share_obs_dim=cfg.share_obs_dim,
            act_dim=cfg.act_dim,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=cfg.device,
        )
        self.actors = [HAPPOActor(actor_cfg, self.device) for _ in range(cfg.num_agents)]
        self.critic = HAPPOCritic(critic_cfg, self.device)
        self.buffer = HAPPORolloutBuffer(buffer_cfg)

    def act(self, obs: torch.Tensor, share_obs: torch.Tensor, deterministic: bool = False):
        """Return joint actions, log-probs, and centralized values.

        Args:
            obs: Tensor shaped `[num_envs, num_agents, obs_dim]`.
            share_obs: Tensor shaped `[num_envs, share_obs_dim]`.
        """
        actions = []
        action_log_probs = []
        for agent_id, actor in enumerate(self.actors):
            agent_actions, agent_log_probs = actor.act(obs[:, agent_id], deterministic=deterministic)
            actions.append(agent_actions)
            action_log_probs.append(agent_log_probs)
        values = self.critic.get_values(share_obs)
        return torch.stack(actions, dim=1), torch.stack(action_log_probs, dim=1), values

    def train(self, next_share_obs: torch.Tensor) -> tuple[list[dict[str, float]], dict[str, float]]:
        """Run one HAPPO update from the collected rollout buffer."""
        with torch.no_grad():
            next_values = self.critic.get_values(next_share_obs)
        advantages = self.buffer.compute_returns(next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1.0e-5))

        factor = torch.ones((self.cfg.episode_length, self.cfg.num_envs, 1), device=self.device)
        agent_order = list(range(self.cfg.num_agents))
        if not self.cfg.fixed_order:
            agent_order = torch.randperm(self.cfg.num_agents).tolist()

        actor_train_infos: list[dict[str, float]] = []
        for agent_id in agent_order:
            with torch.no_grad():
                obs = self.buffer.obs[:-1, :, agent_id].reshape(-1, self.cfg.obs_dim)
                actions = self.buffer.actions[:, :, agent_id].reshape(-1, self.cfg.act_dim)
                old_log_probs, _ = self.actors[agent_id].evaluate_actions(obs, actions)
            actor_info = self.actors[agent_id].train_on_buffer(self.buffer, advantages, agent_id, factor)
            with torch.no_grad():
                new_log_probs, _ = self.actors[agent_id].evaluate_actions(obs, actions)
                ratio = torch.exp(new_log_probs - old_log_probs).reshape(self.cfg.episode_length, self.cfg.num_envs, 1)
                factor = factor * ratio
            actor_train_infos.append(actor_info)

        critic_train_info = self.critic.train_on_buffer(self.buffer)
        self.buffer.after_update()
        return actor_train_infos, critic_train_info

    def prep_rollout(self) -> None:
        for actor in self.actors:
            actor.prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self) -> None:
        for actor in self.actors:
            actor.prep_training()
        self.critic.prep_training()
