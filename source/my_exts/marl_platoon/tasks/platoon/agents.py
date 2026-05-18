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
      Reserved feature toggles (kept off in the current HAPPO baseline).
    - happo_log_interval:
      Rollout log interval in environment steps.
    - happo_log_level:
      Logging verbosity for task-internal HAPPO logs: "off" | "basic" | "debug".
    """

    algorithm: str = "ppo"
    enable_teacher: bool = False
    enable_attack: bool = False
    enable_shield: bool = False
    use_happo_layout: bool = False
    use_happo_actions: bool = False
    happo_log_interval: int = 32
    happo_log_level: str = "basic"
    freeze_outer_ppo: bool = False

    # HAPPO stability knobs (task-internal)
    happo_actor_lr: float = 5.0e-5
    happo_critic_lr: float = 5.0e-5
    happo_clip_param: float = 0.1
    happo_ppo_epoch: int = 3
    happo_max_grad_norm: float = 0.3
    happo_action_clip: float = 0.3
    happo_action_warmup_updates: int = 30
    happo_bad_heading_action_scale: float = 0.2
    teacher_shaping_coef: float = 0.002
    teacher_lr: float = 1.0e-4
    teacher_update_interval: int = 5
    teacher_shaping_clip: float = 0.03
    teacher_action_penalty_coef: float = 0.01

    # Attack profile (task-internal attacker preset): off|easy|medium|hard
    attack_level: str = "medium"
    attack_target_mode: str = "all"  # all|rel_pos|vel
    attack_seed: int = 3407
    attack_log_interval_updates: int = 20
    attack_mode: str = "profile"  # profile|generator|cagan
    attack_noise_dim: int = 16
    attack_hidden_dim: int = 128
    attack_realism_coef: float = 0.10
    attack_generator_lr: float = 1.0e-4
    attack_discriminator_lr: float = 1.0e-4
    attack_update_interval: int = 4
    attack_obj_coef: float = 1.2
    attack_reward_proxy_coef: float = 1.0
    attack_dos_proxy_coef: float = 0.3


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
