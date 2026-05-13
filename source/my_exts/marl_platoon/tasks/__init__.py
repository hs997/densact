import gymnasium as gym

from .platoon.config import PlatoonEnvCfg
from .platoon.agents import PlatoonPPORunnerCfg, PlatoonHAPPORunnerCfg
from marl_platoon.wrappers import make_happo_routed_platoon_env


class PlatoonHAPPOEnvCfg(PlatoonEnvCfg):
    """Framework-mode env config for task-internal algorithm routing.

    This config is the source of truth for student/teacher/attacker/shield
    switches used by the pipeline-backed wrapper.
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.algorithm = "happo"
        self.algorithm.enable_teacher = True
        self.algorithm.enable_attack = True
        self.algorithm.attack_level = "hard"
        self.algorithm.enable_shield = True
        self.algorithm.use_happo_layout = True
        self.algorithm.use_happo_actions = True
        self.algorithm.freeze_outer_ppo = True

        # Keep env-side algorithm knobs in sync with runner-side defaults.
        self.algorithm.happo_actor_lr = 5.0e-5
        self.algorithm.happo_critic_lr = 5.0e-5
        self.algorithm.happo_clip_param = 0.1
        self.algorithm.happo_ppo_epoch = 3
        self.algorithm.happo_max_grad_norm = 0.3
        self.algorithm.happo_action_clip = 0.3
        self.algorithm.happo_action_warmup_updates = 30
        self.algorithm.happo_log_interval = 256
        self.algorithm.happo_log_level = "basic"
        self.algorithm.teacher_shaping_coef = 0.002
        self.algorithm.teacher_shaping_clip = 0.03
        self.algorithm.teacher_action_penalty_coef = 0.01

# Stable PPO baseline entry.
gym.register(
    id="Isaac-Marl-Platoon-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PlatoonEnvCfg,
        "rsl_rl_cfg_entry_point": PlatoonPPORunnerCfg,
    },
)

# HAPPO experiment entry. The official train.py still launches the task, while
# this entry wraps the env internally to route rollout/action data to HAPPO.
gym.register(
    id="Isaac-Marl-Platoon-HAPPO-v0",
    entry_point=make_happo_routed_platoon_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PlatoonHAPPOEnvCfg,
        "rsl_rl_cfg_entry_point": PlatoonHAPPORunnerCfg,
    },
)
