"""Adapters for reusing HARL/HAPPO as the lower-level student learner.

This module does not replace HARL internals. Instead, it provides the minimal
conversion utilities needed to bridge Isaac Lab's manager-based env outputs
with HARL's expected multi-agent tensor layouts.

The intended usage is:

1. Collect Isaac Lab observations/actions from the platoon env.
2. Reshape them into HARL's (n_envs, n_agents, dim) layout.
3. Feed them into HARL's actor/buffer/update code.
4. Apply the resulting joint actions back to Isaac Lab.

The actual rollout/training loop is intentionally kept outside this helper so
we can first verify data layout compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


@dataclass
class HAPPOLayout:
    """Simple container describing the multi-agent layout."""

    num_agents: int
    obs_dim: int
    act_dim: int


def flatten_joint_action(actions: torch.Tensor) -> torch.Tensor:
    """Convert (n_envs, n_agents, act_dim) -> (n_envs, n_agents * act_dim)."""
    if actions.dim() != 3:
        raise ValueError(f"Expected 3D actions, got shape {tuple(actions.shape)}")
    return actions.reshape(actions.shape[0], -1)


def unflatten_joint_action(actions: torch.Tensor, num_agents: int, act_dim: int) -> torch.Tensor:
    """Convert (n_envs, n_agents * act_dim) -> (n_envs, n_agents, act_dim)."""
    if actions.dim() != 2:
        raise ValueError(f"Expected 2D actions, got shape {tuple(actions.shape)}")
    expected = num_agents * act_dim
    if actions.shape[-1] != expected:
        raise ValueError(f"Action dim mismatch: got {actions.shape[-1]}, expected {expected}")
    return actions.reshape(actions.shape[0], num_agents, act_dim)


def split_agent_obs(obs: torch.Tensor, num_agents: int) -> torch.Tensor:
    """Convert a flat observation into per-agent chunks.

    Supported input shapes:
    - (n_envs, num_agents * obs_dim)
    - (n_envs, obs_dim * num_agents)  # same as above
    - (n_envs, num_agents, obs_dim)   # returned as-is
    """
    if obs.dim() == 3:
        if obs.shape[1] != num_agents:
            raise ValueError(f"Expected {num_agents} agents, got {obs.shape[1]}")
        return obs

    if obs.dim() != 2:
        raise ValueError(f"Expected 2D or 3D obs, got shape {tuple(obs.shape)}")

    if obs.shape[-1] % num_agents != 0:
        raise ValueError(
            f"Observation dim {obs.shape[-1]} is not divisible by num_agents={num_agents}"
        )

    obs_dim = obs.shape[-1] // num_agents
    return obs.reshape(obs.shape[0], num_agents, obs_dim)


def merge_agent_obs(obs: torch.Tensor) -> torch.Tensor:
    """Convert (n_envs, n_agents, obs_dim) -> (n_envs, n_agents * obs_dim)."""
    if obs.dim() != 3:
        raise ValueError(f"Expected 3D obs, got shape {tuple(obs.shape)}")
    return obs.reshape(obs.shape[0], -1)


def make_agent_masks(num_envs: int, num_agents: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create default RNN/active masks for feed-forward HAPPO."""
    masks = torch.ones((num_envs, num_agents, 1), device=device, dtype=torch.float32)
    active_masks = torch.ones_like(masks)
    return masks, active_masks


def infer_flat_dims(obs: torch.Tensor, actions: torch.Tensor) -> HAPPOLayout:
    """Infer a minimal layout from a rollout sample."""
    if obs.dim() == 3:
        num_agents = obs.shape[1]
        obs_dim = obs.shape[2]
    elif obs.dim() == 2:
        num_agents = 1
        obs_dim = obs.shape[1]
    else:
        raise ValueError(f"Unsupported obs shape {tuple(obs.shape)}")

    if actions.dim() == 3:
        act_dim = actions.shape[2]
        num_agents = actions.shape[1]
    elif actions.dim() == 2:
        act_dim = actions.shape[1]
    else:
        raise ValueError(f"Unsupported action shape {tuple(actions.shape)}")

    return HAPPOLayout(num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim)


class HAPPOReplayView:
    """A thin storage helper for rollout data used by the adapter.

    This is intentionally lightweight. It stores one rollout segment and can be
    extended later into a full HARL actor buffer bridge.
    """

    def __init__(self):
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []

    def append(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        log_probs: torch.Tensor | None = None,
    ) -> None:
        self.obs.append(obs.detach().cpu())
        self.actions.append(actions.detach().cpu())
        self.rewards.append(rewards.detach().cpu())
        self.dones.append(dones.detach().cpu())
        if log_probs is not None:
            self.log_probs.append(log_probs.detach().cpu())

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()

    def is_empty(self) -> bool:
        return len(self.obs) == 0


class IsaacLabHAPPOAdapter:
    """Adapter that keeps IsaacLab outputs aligned with HARL on-policy buffers.

    The adapter intentionally stays thin: it only converts environment outputs to
    the multi-agent shapes expected by HARL's on-policy runner and exposes a tiny
    rollout view that we can later wire into HAPPO's actor/critic update path.
    """

    def __init__(
        self,
        env: Any,
        layout: HAPPOLayout | None = None,
        *,
        num_agents: int | None = None,
        obs_dim: int | None = None,
        act_dim: int | None = None,
    ):
        self.env = env
        if layout is None:
            if num_agents is None or obs_dim is None or act_dim is None:
                raise ValueError("Provide either layout or num_agents/obs_dim/act_dim")
            layout = HAPPOLayout(num_agents=int(num_agents), obs_dim=int(obs_dim), act_dim=int(act_dim))
        self.layout = layout
        self.replay = HAPPOReplayView()
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device = torch.device(getattr(env, "device", "cpu"))

    def reset(self):
        obs = self.env.reset()
        obs_tensor = self._to_tensor(obs)
        agent_obs = self._ensure_agent_obs(obs_tensor)
        share_obs = self._make_share_obs(agent_obs)
        available_actions = None
        return agent_obs, share_obs, available_actions

    def step(self, actions: torch.Tensor):
        action_tensor = self._to_tensor(actions)
        joint_actions = self._ensure_joint_actions(action_tensor)
        flat_actions = flatten_joint_action(joint_actions)
        obs, rewards, dones, infos = self.env.step(flat_actions)
        obs_tensor = self._to_tensor(obs)
        agent_obs = self._ensure_agent_obs(obs_tensor)
        share_obs = self._make_share_obs(agent_obs)
        rewards_tensor = self._ensure_reward_tensor(rewards, agent_obs.device)
        dones_tensor = self._ensure_done_tensor(dones, agent_obs.device)
        self.replay.append(agent_obs, joint_actions, rewards_tensor, dones_tensor)
        available_actions = None
        return agent_obs, share_obs, rewards_tensor, dones_tensor, infos, available_actions

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        if hasattr(value, "to_dict"):
            try:
                value = value.to_dict()
            except Exception:
                pass
        if isinstance(value, (tuple, list)):
            if len(value) == 0:
                raise TypeError("Cannot convert empty sequence to tensor")
            for item in value:
                if torch.is_tensor(item):
                    return item
                if hasattr(item, "to_dict"):
                    try:
                        item = item.to_dict()
                    except Exception:
                        pass
                if isinstance(item, dict):
                    tensor = self._dict_to_tensor(item)
                    if tensor is not None:
                        return tensor
                if isinstance(item, (tuple, list)):
                    try:
                        return self._to_tensor(item)
                    except Exception:
                        continue
            return self._to_tensor(value[0])
        if isinstance(value, dict):
            tensor = self._dict_to_tensor(value)
            if tensor is None:
                raise TypeError(f"Cannot infer tensor from dict keys={list(value.keys())}")
            return tensor
        return torch.as_tensor(value)

    def _dict_to_tensor(self, data: dict[str, Any]) -> torch.Tensor | None:
        preferred_keys = ("policy", "obs", "observations", "state", "states", "joint_obs")
        for key in preferred_keys:
            if key in data:
                tensor = self._to_tensor(data[key])
                if tensor.numel() > 0:
                    return tensor
        for _, item in data.items():
            if torch.is_tensor(item) and item.numel() > 0:
                return item
        for _, item in data.items():
            if isinstance(item, (dict, tuple, list)):
                try:
                    tensor = self._to_tensor(item)
                    if tensor.numel() > 0:
                        return tensor
                except Exception:
                    continue
        return None

    def _ensure_agent_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            if obs.shape[-1] % self.layout.num_agents == 0:
                return split_agent_obs(obs, self.layout.num_agents)
            return obs.unsqueeze(1).repeat(1, self.layout.num_agents, 1)
        if obs.dim() == 3:
            if obs.shape[1] != self.layout.num_agents:
                raise ValueError(f"Expected {self.layout.num_agents} agents, got {obs.shape[1]}")
            return obs
        raise ValueError(f"Unsupported observation shape {tuple(obs.shape)}")

    def _make_share_obs(self, agent_obs: torch.Tensor) -> torch.Tensor:
        if agent_obs.dim() != 3:
            raise ValueError(f"Expected agent obs to be 3D, got {tuple(agent_obs.shape)}")
        return merge_agent_obs(agent_obs)

    def _ensure_joint_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() == 3:
            if actions.shape[1] != self.layout.num_agents:
                raise ValueError(f"Expected {self.layout.num_agents} agents, got {actions.shape[1]}")
            return actions
        if actions.dim() == 2:
            if actions.shape[-1] % self.layout.num_agents != 0:
                raise ValueError(
                    f"Flat action dim {actions.shape[-1]} is not divisible by num_agents={self.layout.num_agents}"
                )
            act_dim = actions.shape[-1] // self.layout.num_agents
            return unflatten_joint_action(actions, self.layout.num_agents, act_dim)
        raise ValueError(f"Unsupported action shape {tuple(actions.shape)}")

    def _ensure_reward_tensor(self, rewards: Any, device: torch.device) -> torch.Tensor:
        rewards_tensor = rewards if torch.is_tensor(rewards) else torch.as_tensor(rewards)
        rewards_tensor = rewards_tensor.to(device=device, dtype=torch.float32)
        if rewards_tensor.dim() == 0:
            rewards_tensor = rewards_tensor.view(1, 1, 1).repeat(self.num_envs, self.layout.num_agents, 1)
        elif rewards_tensor.dim() == 1:
            rewards_tensor = rewards_tensor.view(self.num_envs, 1, 1).repeat(1, self.layout.num_agents, 1)
        elif rewards_tensor.dim() == 2:
            rewards_tensor = rewards_tensor.unsqueeze(-1)
        return rewards_tensor

    def _ensure_done_tensor(self, dones: Any, device: torch.device) -> torch.Tensor:
        dones_tensor = dones if torch.is_tensor(dones) else torch.as_tensor(dones)
        dones_tensor = dones_tensor.to(device=device, dtype=torch.bool)
        if dones_tensor.dim() == 0:
            dones_tensor = dones_tensor.view(1, 1).repeat(self.num_envs, self.layout.num_agents)
        elif dones_tensor.dim() == 1:
            dones_tensor = dones_tensor.view(self.num_envs, 1).repeat(1, self.layout.num_agents)
        return dones_tensor
