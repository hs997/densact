"""Task-local HAPPO runner for marl_platoon.

The official IsaacLab launch path remains responsible for creating the app and
environment. This runner is the lower-level HAPPO-style trust-region optimizer
used by the task-local MGRS pipeline when `algorithm == "happo"`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .actor import HAPPOActor, HAPPOActorCfg
from .ams import HAPPOAMSCfg, HAPPOActionManifoldSmoother
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
    use_happo_factor: bool = True
    share_actor: bool = False
    actor_update_mode: str = "ppo"
    trpo_kl_threshold: float = 0.01
    trpo_cg_iters: int = 10
    trpo_damping: float = 0.1
    trpo_line_search_steps: int = 10
    trpo_accept_ratio: float = 0.5
    trpo_backtrack_coeff: float = 0.8
    ams_enabled: bool = False
    ams_lr: float = 1.0e-4
    ams_tau: float = 0.01
    ams_num_neighbors: int = 8
    ams_neighborhood_radius: float = 0.25
    ams_action_limit: float = 0.4
    ams_refresh_every: int = 1000
    ams_huber_beta: float = 0.3
    ams_q_epochs: int = 1
    ams_num_mini_batches: int = 16
    ams_max_grad_norm: float = 0.5
    ams_target_clip: float = 100.0
    ams_eval_chunk_size: int = 8192
    ams_advantage_weight: float = 0.10
    ams_warmup_updates: int = 50
    ams_ramp_updates: int = 100


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
            update_mode=cfg.actor_update_mode,
            trpo_kl_threshold=cfg.trpo_kl_threshold,
            trpo_cg_iters=cfg.trpo_cg_iters,
            trpo_damping=cfg.trpo_damping,
            trpo_line_search_steps=cfg.trpo_line_search_steps,
            trpo_accept_ratio=cfg.trpo_accept_ratio,
            trpo_backtrack_coeff=cfg.trpo_backtrack_coeff,
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
        if cfg.share_actor:
            shared_actor = HAPPOActor(actor_cfg, self.device)
            self.actors = [shared_actor for _ in range(cfg.num_agents)]
        else:
            self.actors = [HAPPOActor(actor_cfg, self.device) for _ in range(cfg.num_agents)]
        self.critic = HAPPOCritic(critic_cfg, self.device)
        self.buffer = HAPPORolloutBuffer(buffer_cfg)
        self.update_count = 0
        self.ams: HAPPOActionManifoldSmoother | None = None
        if cfg.ams_enabled:
            ams_cfg = HAPPOAMSCfg(
                share_obs_dim=cfg.share_obs_dim,
                joint_action_dim=cfg.num_agents * cfg.act_dim,
                hidden_dims=cfg.hidden_dims,
                gamma=cfg.gamma,
                lr=cfg.ams_lr,
                tau=cfg.ams_tau,
                num_neighbors=cfg.ams_num_neighbors,
                neighborhood_radius=cfg.ams_neighborhood_radius,
                action_limit=cfg.ams_action_limit,
                refresh_every=cfg.ams_refresh_every,
                huber_beta=cfg.ams_huber_beta,
                q_epochs=cfg.ams_q_epochs,
                num_mini_batches=cfg.ams_num_mini_batches,
                max_grad_norm=cfg.ams_max_grad_norm,
                target_clip=cfg.ams_target_clip,
                eval_chunk_size=cfg.ams_eval_chunk_size,
            )
            self.ams = HAPPOActionManifoldSmoother(ams_cfg, self.device)

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
        advantages = self._normalize_advantage(self.buffer.compute_returns(next_values))
        self.update_count += 1

        ams_info: dict[str, float] = {"ams_enabled": float(self.ams is not None)}
        if self.ams is not None:
            policy_actions = self._policy_joint_actions(self.buffer.obs)
            ams_info.update(self.ams.train_on_buffer(self.buffer, policy_actions[1:]))
            batch_size = self.cfg.episode_length * self.cfg.num_envs
            action_limit = abs(float(self.cfg.ams_action_limit))
            executed_actions = self.buffer.actions.reshape(batch_size, -1).clamp(-action_limit, action_limit)
            local_advantage, local_info = self.ams.local_advantage(
                self.buffer.share_obs[:-1].reshape(batch_size, self.cfg.share_obs_dim),
                executed_actions,
                policy_actions[:-1].reshape(batch_size, -1),
            )
            ams_info.update(local_info)
            effective_weight = self._ams_advantage_weight()
            ams_info["ams_advantage_weight"] = effective_weight
            if effective_weight > 0.0:
                local_advantage = local_advantage.reshape(
                    self.cfg.episode_length, self.cfg.num_envs, 1, 1
                ).repeat(1, 1, self.cfg.num_agents, 1)
                local_advantage = self._normalize_advantage(local_advantage)
                advantages = self._normalize_advantage(advantages + effective_weight * local_advantage)

        factor = torch.ones((self.cfg.episode_length, self.cfg.num_envs, 1), device=self.device)
        if self.cfg.share_actor and not self.cfg.use_happo_factor:
            actor_info = self.actors[0].train_shared_on_buffer(self.buffer, advantages, factor)
            actor_train_infos = [dict(actor_info) for _ in range(self.cfg.num_agents)]
            critic_train_info = self.critic.train_on_buffer(self.buffer)
            critic_train_info.update(ams_info)
            self.buffer.after_update()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return actor_train_infos, critic_train_info

        agent_order = list(range(self.cfg.num_agents))
        if not self.cfg.fixed_order:
            agent_order = torch.randperm(self.cfg.num_agents).tolist()

        actor_train_infos: list[dict[str, float]] = []
        for agent_id in agent_order:
            with torch.no_grad():
                old_log_probs = self.buffer.action_log_probs[:, :, agent_id].reshape(-1, 1)
            actor_info = self.actors[agent_id].train_on_buffer(self.buffer, advantages, agent_id, factor)
            if self.cfg.use_happo_factor:
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
        critic_train_info.update(ams_info)
        self.buffer.after_update()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return actor_train_infos, critic_train_info

    @staticmethod
    def _normalize_advantage(advantages: torch.Tensor) -> torch.Tensor:
        finite = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        return (finite - finite.mean()) / finite.std(unbiased=False).clamp_min(1.0e-5)

    @torch.no_grad()
    def _policy_joint_actions(self, obs: torch.Tensor) -> torch.Tensor:
        leading_shape = obs.shape[:-2]
        limit = abs(float(self.cfg.ams_action_limit))
        actions = []
        for agent_id, actor in enumerate(self.actors):
            agent_obs = obs[..., agent_id, :].reshape(-1, self.cfg.obs_dim)
            mean = actor.actor.distribution(agent_obs).mean
            actions.append(mean.reshape(*leading_shape, self.cfg.act_dim).clamp(-limit, limit))
        return torch.stack(actions, dim=-2)

    def _ams_advantage_weight(self) -> float:
        if self.ams is None or self.update_count <= int(self.cfg.ams_warmup_updates):
            return 0.0
        ramp_updates = max(int(self.cfg.ams_ramp_updates), 1)
        ramp = min((self.update_count - int(self.cfg.ams_warmup_updates)) / ramp_updates, 1.0)
        return float(self.cfg.ams_advantage_weight) * ramp

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
        state = {
            "cfg": dict(self.cfg.__dict__),
            "update_count": self.update_count,
            "actors": [actor.actor.state_dict() for actor in self.actors],
            "actor_optimizers": [actor.optimizer.state_dict() for actor in self.actors],
            "critic": self.critic.critic.state_dict(),
            "critic_optimizer": self.critic.optimizer.state_dict(),
        }
        if self.ams is not None:
            state["ams"] = self.ams.state_dict()
        return state

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
        self.update_count = int(state.get("update_count", self.update_count))
        if self.ams is not None:
            if "ams" not in state:
                if strict:
                    raise ValueError("AMS is enabled but the HAPPO checkpoint is missing AMS state.")
            else:
                self.ams.load_state_dict(state["ams"], strict=strict)

    def prep_rollout(self) -> None:
        for actor in self.actors:
            actor.prep_rollout()
        self.critic.prep_rollout()
        if self.ams is not None:
            self.ams.prep_rollout()

    def prep_training(self) -> None:
        for actor in self.actors:
            actor.prep_training()
        self.critic.prep_training()
        if self.ams is not None:
            self.ams.prep_training()
