import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from marl_platoon.algorithms import PlatoonAlgorithmRouter


def make_happo_routed_platoon_env(*args, **kwargs):
    """Create the platoon env with internal HAPPO routing for experiment tasks.

    The observation/action spaces are kept flat so the official RSL-RL runner can
    still launch the task unchanged. Only env reset/step are tapped to route data
    through the task-local HAPPO runner.
    """
    env = ManagerBasedRLEnv(*args, **kwargs)
    return IsaacHAPPOInternalWrapper(env, num_agents=4)


class IsaacHAPPOInternalWrapper(gym.Wrapper):
    """Flat-space wrapper that injects task-local HAPPO routing.

    Unlike `IsaacMARLWrapper`, this wrapper does not expose multi-agent spaces to
    RSL-RL. It preserves the flat IsaacLab contract and only uses structured
    multi-agent tensors internally for HAPPO rollout/action override.
    """

    def __init__(self, env: ManagerBasedRLEnv, num_agents: int):
        super().__init__(env)
        self.num_agents = num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        if "policy" in env.observation_manager.group_obs_dim:
            total_obs_dim = env.observation_manager.group_obs_dim["policy"][0]
        else:
            total_obs_dim = env.unwrapped.observation_space.shape[0]
        self.obs_dim = total_obs_dim // self.num_agents if total_obs_dim % self.num_agents == 0 else total_obs_dim
        self._broadcast_obs_to_agents = total_obs_dim % self.num_agents != 0
        self.act_dim = self._infer_total_action_dim() // self.num_agents

        env_cfg = getattr(env.unwrapped, "cfg", None)
        algorithm_cfg = getattr(env_cfg, "algorithm", None)
        self.algorithm_router = (
            PlatoonAlgorithmRouter(env.unwrapped, algorithm_cfg)
            if algorithm_cfg is not None
            else None
        )
        self.use_happo_actions = bool(getattr(algorithm_cfg, "use_happo_actions", False))
        self.freeze_outer_ppo = bool(getattr(algorithm_cfg, "freeze_outer_ppo", False))
        self.happo_action_clip = float(getattr(algorithm_cfg, "happo_action_clip", 1.0))
        self.happo_action_warmup_updates = int(getattr(algorithm_cfg, "happo_action_warmup_updates", 0))
        self._last_agent_obs = None
        self._happo_log_level = str(getattr(algorithm_cfg, "happo_log_level", "basic")).lower() if algorithm_cfg is not None else "basic"
        if algorithm_cfg is not None and self._happo_log_level != "off":
            print(
                "[Platoon Algorithm] configured: "
                f"algorithm={getattr(algorithm_cfg, 'algorithm', 'ppo')}, "
                f"use_happo_actions={self.use_happo_actions}, "
                f"teacher={getattr(algorithm_cfg, 'enable_teacher', False)}, "
                f"attack={getattr(algorithm_cfg, 'enable_attack', False)}, "
                f"shield={getattr(algorithm_cfg, 'enable_shield', False)}, "
                f"log_level={self._happo_log_level}"
            )

    def reset(self, seed=None, options=None):
        obs_dict, info = self.env.reset(seed=seed, options=options)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) and "policy" in obs_dict else obs_dict
        agent_obs = self._reshape_obs(obs)
        self._last_agent_obs = agent_obs
        if self.algorithm_router is not None:
            self.algorithm_router.build_from_sample(agent_obs)
        return obs_dict, info

    def step(self, actions):
        used_happo_actions = False
        if self._should_use_happo_actions():
            flat_actions = self.algorithm_router.act(self._last_agent_obs)
            if self.happo_action_clip > 0.0:
                flat_actions = torch.clamp(flat_actions, -self.happo_action_clip, self.happo_action_clip)
            if self.algorithm_router.pipeline is not None:
                joint_actions = flat_actions.view(self.env.num_envs, self.num_agents, self.act_dim)
                flat_actions = self.algorithm_router.pipeline.postprocess_action_for_env(
                    self._last_agent_obs, joint_actions
                ).reshape(self.env.num_envs, -1)
            used_happo_actions = True
        else:
            # Warmup stage: still run HAPPO actor forward so rollout/update can proceed,
            # but keep executing the external action to avoid early destabilization.
            if self.algorithm_router is not None and self.algorithm_router.runner is not None and self._last_agent_obs is not None:
                _ = self.algorithm_router.act(self._last_agent_obs)
            if isinstance(actions, torch.Tensor):
                flat_actions = actions.reshape(self.env.num_envs, -1)
            else:
                flat_actions = torch.tensor(actions, device=self.env.device).reshape(self.env.num_envs, -1)

        obs_dict, rew, terminated, truncated, extras = self.env.step(flat_actions)
        if self.freeze_outer_ppo:
            rew = torch.zeros_like(rew)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) and "policy" in obs_dict else obs_dict
        agent_obs = self._reshape_obs(obs)

        if self.algorithm_router is not None and self.algorithm_router.runner is not None:
            dones = torch.logical_or(terminated, truncated)
            self.algorithm_router.observe_step(agent_obs, rew, dones)
            if self.algorithm_router.pipeline is None:
                self.algorithm_router.train_if_ready(agent_obs)
        self._last_agent_obs = agent_obs
        return obs_dict, rew, terminated, truncated, extras

    def _should_use_happo_actions(self) -> bool:
        warmup_ok = True
        if self.algorithm_router is not None:
            warmup_ok = self.algorithm_router.happo_update_count >= self.happo_action_warmup_updates
        return (
            self.use_happo_actions
            and warmup_ok
            and self.algorithm_router is not None
            and self.algorithm_router.runner is not None
            and self._last_agent_obs is not None
        )

    def _infer_total_action_dim(self) -> int:
        action_manager = getattr(self.env, "action_manager", None)
        for attr_name in ("action_dim", "total_action_dim", "num_actions"):
            value = getattr(action_manager, attr_name, None)
            if value is not None:
                return int(value)
        action_space = getattr(self.env, "single_action_space", None) or getattr(self.env, "action_space", None)
        if action_space is not None and getattr(action_space, "shape", None):
            return int(action_space.shape[-1])
        if action_manager is not None and hasattr(action_manager, "_terms"):
            return sum(int(getattr(term, "action_dim", 0)) for term in action_manager._terms.values())
        raise AttributeError("Cannot infer total action dimension from IsaacLab env/action manager.")

    def _reshape_obs(self, obs_tensor):
        if not isinstance(obs_tensor, torch.Tensor):
            obs_tensor = torch.tensor(obs_tensor, device=self.env.device)
        if self._broadcast_obs_to_agents:
            return obs_tensor.view(self.env.num_envs, 1, self.obs_dim).repeat(1, self.num_agents, 1)
        return obs_tensor.view(self.env.num_envs, self.num_agents, self.obs_dim)


class IsaacMARLWrapper(gym.Wrapper):
    """Legacy wrapper kept for backward compatibility.

    NOTE:
    - The framework-mode path used by `Isaac-Marl-Platoon-HAPPO-v0` now goes
      through `IsaacHAPPOInternalWrapper` + `PlatoonTrainingPipeline`.
    - This class is retained to avoid breaking older scripts, but new task-side
      algorithm routing should use the pipeline-backed wrapper.
    """

    def __init__(self, env: ManagerBasedRLEnv, num_agents: int):
        super().__init__(env)
        self.num_agents = num_agents

        # 1. 获取原始观测空间的维度
        # Isaac Lab 通常输出一个巨大的扁平向量，我们需要知道它原本属于几个智能体
        # 假设所有智能体的观测维度是一样的
        if "policy" in env.observation_manager.group_obs_dim:
            total_obs_dim = env.observation_manager.group_obs_dim["policy"][0]
        else:
            # 如果没有 'policy' 组，尝试直接获取
            total_obs_dim = env.unwrapped.observation_space.shape[0]

        self.obs_dim = total_obs_dim // self.num_agents
        self.act_dim = env.action_manager.action_dim // self.num_agents

        print(f"[MARL Wrapper] Detected {self.num_agents} agents.")
        print(f"[MARL Wrapper] Single Agent Obs Dim: {self.obs_dim}, Act Dim: {self.act_dim}")

        env_cfg = getattr(env.unwrapped, "cfg", None)
        algorithm_cfg = getattr(env_cfg, "algorithm", None)
        self.algorithm_router = (
            PlatoonAlgorithmRouter(env.unwrapped, algorithm_cfg)
            if algorithm_cfg is not None
            else None
        )
        self.use_happo_actions = bool(getattr(algorithm_cfg, "use_happo_actions", False))
        self.freeze_outer_ppo = bool(getattr(algorithm_cfg, "freeze_outer_ppo", False))
        self._last_agent_obs = None
        if algorithm_cfg is not None:
            print(
                "[Platoon Algorithm] configured: "
                f"algorithm={getattr(algorithm_cfg, 'algorithm', 'ppo')}, "
                f"use_happo_actions={self.use_happo_actions}, "
                f"freeze_outer_ppo={self.freeze_outer_ppo}, "
                f"teacher={getattr(algorithm_cfg, 'enable_teacher', False)}, "
                f"attack={getattr(algorithm_cfg, 'enable_attack', False)}, "
                f"shield={getattr(algorithm_cfg, 'enable_shield', False)}"
            )

        # 2. 重新定义 Observation Space (告诉你的算法：输入形状变了)
        # 现在的形状是：(智能体数量, 每个智能体的观测维度)
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"),
            shape=(self.num_agents, self.obs_dim),
            dtype="float32"
        )

        # 3. 重新定义 Action Space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_agents, self.act_dim),
            dtype="float32"
        )

    def reset(self, seed=None, options=None):
        """重置环境，并处理观测数据"""
        obs_dict, info = self.env.reset(seed=seed, options=options)

        # 提取 'policy' 组的观测数据 (这是主要的训练输入)
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            obs = obs_dict["policy"]
        else:
            obs = obs_dict

        agent_obs = self._reshape_obs(obs)
        self._last_agent_obs = agent_obs
        if self.algorithm_router is not None:
            self.algorithm_router.build_from_sample(agent_obs)
        return agent_obs, info

    def step(self, actions):
        """执行动作"""
        # 你的算法输出通常是 (num_envs, num_agents, act_dim)
        # Isaac Lab 的物理引擎需要扁平的 (num_envs, num_agents * act_dim)
        if self._should_use_happo_actions():
            flat_actions = self.happo_act(self._last_agent_obs)
        elif isinstance(actions, torch.Tensor):
            flat_actions = actions.reshape(self.env.num_envs, -1)
        else:
            # 如果是 numpy 数组转 tensor
            flat_actions = torch.tensor(actions, device=self.env.device).reshape(self.env.num_envs, -1)

        # 放入仿真器跑一步
        obs_dict, rew, terminated, truncated, extras = self.env.step(flat_actions)

        # 处理返回的观测数据
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            obs = obs_dict["policy"]
        else:
            obs = obs_dict

        agent_obs = self._reshape_obs(obs)
        if self.algorithm_router is not None and self.algorithm_router.runner is not None:
            dones = torch.logical_or(terminated, truncated)
            self.algorithm_router.observe_step(agent_obs, rew, dones)
            self.algorithm_router.train_if_ready(agent_obs)
        self._last_agent_obs = agent_obs

        # 处理奖励 (Reward)
        # 如果 Isaac Lab 返回的是 (num_envs, 1) 的共享奖励，你可能需要把它广播给所有智能体
        # 这里为了通用，我们先假设直接透传，后续根据你的算法具体调整

        return agent_obs, rew, terminated, truncated, extras

    def happo_act(self, obs, deterministic: bool = False):
        """使用任务内部 HAPPO runner 产生 IsaacLab 扁平动作。"""
        if self.algorithm_router is None:
            raise RuntimeError("Platoon algorithm router is not configured.")
        return self.algorithm_router.act(obs, deterministic=deterministic)

    def _should_use_happo_actions(self) -> bool:
        return (
            self.use_happo_actions
            and self.algorithm_router is not None
            and self.algorithm_router.runner is not None
            and self._last_agent_obs is not None
        )

    def _reshape_obs(self, obs_tensor):
        """核心转换逻辑：(Batch, N*Dim) -> (Batch, N, Dim)"""
        # 确保数据是 Tensor
        if not isinstance(obs_tensor, torch.Tensor):
            obs_tensor = torch.tensor(obs_tensor, device=self.env.device)

        return obs_tensor.view(self.env.num_envs, self.num_agents, self.obs_dim)