"""Task-local HAPPO runner for marl_platoon.

The official IsaacLab launch path remains responsible for creating the app and
environment. This runner is the lower-level HAPPO-style trust-region optimizer
used by the task-local MGRS pipeline when `algorithm == "happo"`.
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
    factor_eval_chunk_size: int = 8192
    entropy_coef: float = 0.01
    init_noise_std: float = 1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
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
            init_noise_std=cfg.init_noise_std,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
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
                old_log_probs = self.buffer.action_log_probs[:, :, agent_id].reshape(-1, 1)
            actor_info = self.actors[agent_id].train_on_buffer(self.buffer, advantages, agent_id, factor)
            with torch.no_grad():
                new_log_probs = self._evaluate_actor_log_probs_chunked(agent_id)
                ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs, min=-20.0, max=20.0)).reshape(
                    self.cfg.episode_length, self.cfg.num_envs, 1
                )
                factor = factor * ratio
            actor_train_infos.append(actor_info)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        critic_train_info = self.critic.train_on_buffer(self.buffer)
        self.buffer.after_update()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return actor_train_infos, critic_train_info

    def _evaluate_actor_log_probs_chunked(self, agent_id: int) -> torch.Tensor:
        batch_size = self.cfg.episode_length * self.cfg.num_envs
        chunk_size = max(int(self.cfg.factor_eval_chunk_size), 1)
        obs = self.buffer.obs[:-1, :, agent_id].reshape(batch_size, self.cfg.obs_dim)
        actions = self.buffer.actions[:, :, agent_id].reshape(batch_size, self.cfg.act_dim)
        log_prob_chunks: list[torch.Tensor] = []
        actor_net = self.actors[agent_id].actor
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            action_log_probs, _ = actor_net.evaluate_actions(obs[start:end], actions[start:end])
            log_prob_chunks.append(action_log_probs.detach())
        return torch.cat(log_prob_chunks, dim=0)

    def state_dict(self) -> dict:
        """Return trainable HAPPO state for checkpointing."""
        return {
            "cfg": dict(self.cfg.__dict__),
            "actors": [actor.actor.state_dict() for actor in self.actors],
            "actor_optimizers": [actor.optimizer.state_dict() for actor in self.actors],
            "critic": self.critic.critic.state_dict(),
            "critic_optimizer": self.critic.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """Load trainable HAPPO state from a checkpoint."""
        actors = state.get("actors", [])
        if len(actors) != len(self.actors):
            raise ValueError(f"HAPPO actor count mismatch: got {len(actors)}, expected {len(self.actors)}")
        for actor, actor_state in zip(self.actors, actors):
            actor.actor.load_state_dict(actor_state, strict=strict)

        actor_optimizers = state.get("actor_optimizers", [])
        if len(actor_optimizers) == len(self.actors):
            for actor, optimizer_state in zip(self.actors, actor_optimizers):
                actor.optimizer.load_state_dict(optimizer_state)

        if "critic" not in state:
            raise ValueError("HAPPO checkpoint is missing critic state.")
        self.critic.critic.load_state_dict(state["critic"], strict=strict)
        if "critic_optimizer" in state:
            self.critic.optimizer.load_state_dict(state["critic_optimizer"])

    def prep_rollout(self) -> None:
        for actor in self.actors:
            actor.prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self) -> None:
        for actor in self.actors:
            actor.prep_training()
        self.critic.prep_training()
