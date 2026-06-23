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

        # ------------------------------------------------------------------
        # framework_teacher_attack_v1 baseline (single source of truth)
        # NOTE: for experiments, only modify values in this block.
        # ------------------------------------------------------------------

        # Core routing switches
        self.algorithm.algorithm = "happo"
        self.algorithm.num_agents = 5
        self.algorithm.use_happo_layout = True
        self.algorithm.use_happo_actions = True
        self.algorithm.freeze_outer_ppo = True

        # Module toggles
        self.algorithm.enable_teacher = True
        self.algorithm.enable_attack = False
        self.algorithm.enable_shield = True

        # Stabilized default profile. Stronger paper ablations can override
        # these values explicitly after the Student baseline is numerically stable.
        self.algorithm.attack_level = "off"  # off|light|easy|medium|hard
        self.algorithm.attack_target_mode = "all"  # all|rel_pos|vel
        self.algorithm.attack_seed = 3407
        self.algorithm.attack_log_interval_updates = 20
        self.algorithm.attack_mode = "profile"  # profile|generator|cagan
        self.algorithm.attack_noise_dim = 16
        self.algorithm.attack_hidden_dim = 128
        self.algorithm.attack_realism_coef = 0.10
        self.algorithm.attack_generator_lr = 1.0e-4
        self.algorithm.attack_discriminator_lr = 1.0e-4
        self.algorithm.attack_update_interval = 32
        self.algorithm.attack_curriculum_warmup_updates = 2000
        self.algorithm.attack_curriculum_start_mode = "profile"
        self.algorithm.attack_decay_start_updates = 5000
        self.algorithm.attack_min_obj_scale = 0.25
        self.algorithm.attack_min_update_freq_scale = 0.25
        self.algorithm.attack_decay_half_life_updates = 2500
        self.algorithm.attack_obj_coef = 1.2
        self.algorithm.attack_reward_proxy_coef = 1.0
        self.algorithm.attack_dos_proxy_coef = 0.3

        # HAPPO baseline knobs
        self.algorithm.happo_actor_lr = 1.0e-5
        self.algorithm.happo_critic_lr = 1.0e-5
        self.algorithm.happo_clip_param = 0.05
        self.algorithm.happo_ppo_epoch = 1
        self.algorithm.happo_num_mini_batches = 16
        self.algorithm.happo_factor_chunk_size = 4096
        self.algorithm.happo_max_grad_norm = 0.05
        self.algorithm.happo_entropy_coef = 0.0
        self.algorithm.happo_init_noise_std = 0.25
        self.algorithm.happo_log_std_min = -2.995732273553991
        self.algorithm.happo_log_std_max = -1.0498221244986778
        self.algorithm.happo_action_clip = 0.4
        self.algorithm.happo_action_warmup_updates = 0
        self.algorithm.happo_urdf_wheel_sign_adapter = True
        self.algorithm.local_reward_shaping = True
        self.algorithm.local_reward_centerline_coef = 0.35
        self.algorithm.local_reward_pair_lateral_coef = 0.55
        self.algorithm.local_reward_heading_coef = 0.15
        self.algorithm.local_reward_turn_coef = 0.005
        self.algorithm.local_reward_first_follower_centerline_scale = 1.0
        self.algorithm.local_reward_first_follower_pair_lateral_scale = 1.0
        self.algorithm.local_reward_first_follower_turn_scale = 1.0
        self.algorithm.local_reward_last_follower_centerline_scale = 1.0
        self.algorithm.local_reward_last_follower_pair_lateral_scale = 1.0
        self.algorithm.local_reward_last_follower_turn_scale = 1.0
        self.algorithm.happo_log_interval = 256
        self.algorithm.happo_log_level = "basic"

        # Teacher baseline knobs
        self.algorithm.teacher_shaping_coef = 0.001
        self.algorithm.teacher_lr = 1.0e-4
        self.algorithm.teacher_update_interval = 5
        self.algorithm.teacher_shaping_clip = 0.03
        self.algorithm.teacher_action_penalty_coef = 0.0

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
