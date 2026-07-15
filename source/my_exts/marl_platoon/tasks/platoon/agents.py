from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PlatoonAlgorithmCfg:
    """Task-local algorithm switch used by the official IsaacLab entrypoint.

    Launch path stays unchanged (`rsl_rl/train.py`). These flags only control
    task-internal behavior:

    - algorithm:
      - "ppo": keep current stable PPO path.
      - "happo": enable task-internal HAPPO routing.
    - use_happo_layout:
      Use multi-agent layout conversion for HAPPO internals.
    - use_happo_actions:
      If True, wrapper overrides external PPO actions with internal HAPPO
      actions before `env.step()`.
    - enable_teacher / enable_attack / enable_shield:
      Enable the task-local MGRS Teacher, CA-GAN attacker, and safety shield.
    - happo_log_interval:
      Rollout log interval in environment steps.
    - happo_log_level:
      Logging verbosity for task-internal HAPPO logs: "off" | "basic" | "debug".
    """

    algorithm: str = "ppo"
    num_agents: int = 5
    enable_teacher: bool = False
    enable_attack: bool = False
    enable_shield: bool = False
    use_happo_layout: bool = False
    use_happo_actions: bool = False
    happo_log_interval: int = 32
    happo_log_level: str = "basic"
    freeze_outer_ppo: bool = False

    # HAPPO/MAPPO stability knobs (task-internal)
    # MAPPO uses the same centralized critic and per-agent actors but disables
    # HAPPO's sequential importance factor during actor updates.
    happo_use_factor: bool = True
    happo_share_actor: bool = False
    happo_actor_update_mode: str = "ppo"
    happo_trpo_kl_threshold: float = 0.01
    happo_trpo_cg_iters: int = 10
    happo_trpo_damping: float = 0.1
    happo_trpo_line_search_steps: int = 10
    happo_trpo_accept_ratio: float = 0.5
    happo_trpo_backtrack_coeff: float = 0.8
    happo_actor_lr: float = 1.0e-5
    happo_critic_lr: float = 1.0e-5
    happo_clip_param: float = 0.05
    happo_ppo_epoch: int = 1
    happo_num_mini_batches: int = 16
    happo_factor_chunk_size: int = 4096
    happo_max_grad_norm: float = 0.05
    happo_entropy_coef: float = 0.0
    happo_init_noise_std: float = 0.25
    happo_log_std_min: float = -2.995732273553991  # log(0.05)
    happo_log_std_max: float = -1.0498221244986778  # log(0.35)
    happo_action_clip: float = 0.4
    happo_action_warmup_updates: int = 0
    happo_bad_heading_action_scale: float = 0.2
    local_reward_shaping: bool = True
    local_reward_centerline_coef: float = 0.35
    local_reward_pair_lateral_coef: float = 0.55
    local_reward_heading_coef: float = 0.15
    local_reward_turn_coef: float = 0.005
    local_reward_first_follower_centerline_scale: float = 1.0
    local_reward_first_follower_pair_lateral_scale: float = 1.0
    local_reward_first_follower_turn_scale: float = 1.0
    local_reward_last_follower_centerline_scale: float = 1.0
    local_reward_last_follower_pair_lateral_scale: float = 1.0
    local_reward_last_follower_turn_scale: float = 1.0
    teacher_shaping_coef: float = 0.02
    teacher_lr: float = 3.0e-4
    teacher_update_interval: int = 1
    teacher_every_student_updates: int = 2
    teacher_shaping_clip: float = 0.20
    teacher_action_penalty_coef: float = 0.001
    teacher_reward_ema_tau: float = 0.95
    teacher_consistency_coef: float = 0.03
    teacher_outer_delta_coef: float = 0.08
    teacher_outer_delta_warmup_updates: int = 5
    teacher_outer_delta_ramp_updates: int = 20
    teacher_lambda_spacing: float = 1.0
    teacher_lambda_velocity: float = 0.5
    teacher_lambda_acceleration: float = 0.25
    teacher_lambda_jerk: float = 0.10
    teacher_lambda_overspeed: float = 0.25
    teacher_lambda_centerline: float = 1.0
    teacher_lambda_lateral: float = 0.5
    teacher_lambda_heading: float = 2.0
    teacher_lambda_backward: float = 1.0
    teacher_lambda_forward_deficit: float = 0.5
    teacher_lambda_collision: float = 10.0
    teacher_lambda_action_energy: float = 0.02

    # Attack profile (task-internal attacker preset): off|light|easy|medium|hard
    attack_level: str = "off"
    attack_target_mode: str = "all"  # all|rel_pos|vel
    attack_seed: int = 3407
    attack_log_interval_updates: int = 20
    attack_mode: str = "profile"  # profile|generator|cagan
    attack_noise_dim: int = 16
    attack_hidden_dim: int = 128
    attack_realism_coef: float = 0.10
    attack_generator_lr: float = 1.0e-4
    attack_discriminator_lr: float = 1.0e-4
    attack_update_interval: int = 32
    attack_curriculum_warmup_updates: int = 2000
    attack_curriculum_start_mode: str = "profile"
    attack_decay_start_updates: int = 5000
    attack_min_obj_scale: float = 0.25
    attack_min_update_freq_scale: float = 0.25
    attack_decay_half_life_updates: int = 2500
    attack_obj_coef: float = 1.2
    attack_reward_proxy_coef: float = 1.0  # fallback score coefficient when physical context is unavailable
    attack_dos_proxy_coef: float = 0.3
    max_fdi_pos: float = -1.0
    max_fdi_acc: float = -1.0
    max_dos_rate: float = -1.0


@configclass
class PlatoonPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    empirical_normalization = True

    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 50
    experiment_name = "platoon_test"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    platoon = PlatoonAlgorithmCfg()


@configclass
class PlatoonHAPPORunnerCfg(PlatoonPPORunnerCfg):
    """Task-side HAPPO hook.

    This keeps the launch surface identical while reserving a dedicated config
    branch for the eventual HARL-backed student logic.
    """

    experiment_name = "platoon_happo"
    platoon = PlatoonAlgorithmCfg(
        algorithm="happo",
        enable_teacher=False,
        enable_attack=False,
        enable_shield=False,
        use_happo_layout=True,
        use_happo_actions=True,
        freeze_outer_ppo=True,
    )
