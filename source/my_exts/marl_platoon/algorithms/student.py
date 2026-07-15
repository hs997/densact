"""Student module adapters for the platoon training pipeline."""

from __future__ import annotations

import torch

from marl_platoon.algorithms.happo import PlatoonHAPPORunner
from marl_platoon.algorithms.pipeline import StepBatch, StudentModule
from marl_platoon.harl_adapter import merge_agent_obs


class HAPPOStudentModule(StudentModule):
    """Lower-level Student optimizer for the paper's MGRS pipeline.

    The Student does not design or alter rewards. It receives the shaped
    lower-level reward R_phi = r_env + F_phi from the pipeline/Teacher and uses
    HAPPO as the practical HATRL-style sequential policy optimizer.
    """

    def __init__(self, runner: PlatoonHAPPORunner):
        self.runner = runner
        self._pending_actions: torch.Tensor | None = None
        self._pending_log_probs: torch.Tensor | None = None
        self._pending_values: torch.Tensor | None = None
        self._last_next_obs: torch.Tensor | None = None
        self._last_update_rollout_steps = -1
        self.rollout_steps = 0

    @property
    def has_pending_action(self) -> bool:
        return self._pending_actions is not None and self._pending_log_probs is not None and self._pending_values is not None

    @property
    def pending_actions(self) -> torch.Tensor:
        if self._pending_actions is None:
            raise RuntimeError("HAPPO student has no pending actions.")
        return self._pending_actions

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        share_obs = merge_agent_obs(obs)
        joint_actions, joint_log_probs, values = self.runner.act(obs, share_obs, deterministic=deterministic)
        self._pending_actions = joint_actions
        self._pending_log_probs = joint_log_probs
        self._pending_values = values
        return joint_actions

    def observe(self, batch: StepBatch) -> None:
        if not self.has_pending_action:
            raise RuntimeError("HAPPO student has no pending action. Call act() before observe().")
        # batch.rewards is already the paper lower-level reward
        # R_phi(tilde_y_i, u_i) = r_env + F_phi(tilde_y_i, u_i), produced by the
        # pipeline/Teacher before Student observation. The Student only stores it
        # and performs policy optimization.
        self._last_next_obs = batch.next_obs
        self.runner.buffer.insert(
            batch.next_obs,
            merge_agent_obs(batch.next_obs),
            self._pending_actions,
            self._pending_log_probs,
            batch.rewards,
            batch.dones,
            self._pending_values,
        )
        self.rollout_steps += 1

    def maybe_update(self) -> dict[str, float] | None:
        if self.runner.buffer.step != 0:
            return None
        if self._last_update_rollout_steps == self.rollout_steps:
            return None
        if self._last_next_obs is None:
            return None
        next_share_obs = merge_agent_obs(self._last_next_obs)
        actor_infos, critic_info = self.runner.train(next_share_obs)
        self._last_update_rollout_steps = self.rollout_steps
        return self._summarize_update(actor_infos, critic_info)

    @staticmethod
    def _summarize_update(actor_infos: list[dict[str, float]], critic_info: dict[str, float]) -> dict[str, float]:
        policy_loss = 0.0
        actor_entropy = 0.0
        ratio_mean = 0.0
        actor_grad_norm_mean = 0.0
        if actor_infos:
            policy_loss = sum(info.get("policy_loss", 0.0) for info in actor_infos) / len(actor_infos)
            actor_entropy = sum(info.get("dist_entropy", 0.0) for info in actor_infos) / len(actor_infos)
            ratio_mean = sum(info.get("ratio", 0.0) for info in actor_infos) / len(actor_infos)
            actor_grad_norm_mean = sum(info.get("actor_grad_norm", 0.0) for info in actor_infos) / len(actor_infos)
        return {
            "policy_loss": policy_loss,
            "dist_entropy": actor_entropy,
            "ratio": ratio_mean,
            "actor_grad_norm": actor_grad_norm_mean,
            "value_loss": float(critic_info.get("value_loss", 0.0)),
            "critic_grad_norm": float(critic_info.get("critic_grad_norm", 0.0)),
        }
