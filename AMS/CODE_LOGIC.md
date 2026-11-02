# AMS code logic and implementation map

This document is a source-oriented guide for a writer or engineer who needs to understand the implementation without reading the whole Isaac Lab repository.

## End-to-end data flow

At each simulator step:

1. The five Ackermann vehicles observe the centralized/local task state. During attacked training, the attack wrapper may drop observations (DoS) or perturb position/acceleration and longitudinal command channels (FDI).
2. The HAPPO actor emits a two-dimensional normalized command for each vehicle: speed and center steering.
3. `AckermannAction` maps that command to wheel targets. Rear-wheel joints receive velocity targets only. Front-wheel joints receive coupled left/right steering angles from the Ackermann geometry.
4. The environment applies the safety shield and any attack-time execution effects. The executed command, not only the actor proposal, is retained for critic and teacher calculations.
5. The lane-change command computes the shared quintic spatial reference and the platoon-level completion state.
6. The reward teacher computes a clean physical score from privileged state and executed input, updates its shaping model, and contributes the attacked training reward.
7. HAPPO computes its normal factorized advantage. If AMS is enabled and past warm-up, the AMS critic evaluates a feasible local action neighborhood and returns a local agent advantage. The runner adds a ramped, normalized AMS correction before the actor update.
8. Validation exports per-environment CSV trajectories, reward/safety plots, top-down trajectories, and optional ROS replay/video artifacts.

## Ackermann actuator implementation

Source: `source/my_exts/marl_platoon/ackermann_action.py`.

`AckermannAction` is an `ActionTerm` with two commands per vehicle:

    a_i = [v_i / v_max, delta_center / delta_max]

The current configuration uses wheelbase 0.14325 m, track width 0.104 m, wheel radius 0.032 m, maximum speed 1.20 m/s, and a center-steering limit of 0.32 rad. The front steering angles are coupled by the instantaneous-turning-center relation. For a positive center angle, the inside and outside front wheel angles differ; they are not independent policy outputs. The rear joints `lb_joint` and `rb_joint` are given wheel velocity targets only, so the code enforces front-wheel steering and rear-wheel non-steering.

Do not describe the policy as a generic four-wheel-steering controller. That would be false for this implementation.

## Shared spatial path and whole-platoon completion

Source: `source/my_exts/marl_platoon/trajectory.py` and `source/my_exts/marl_platoon/tasks/platoon/config.py`.

The lane-change reference is a quintic spatial profile:

    u = clip((x - x_start) / L, 0, 1)
    h(u) = 10 u^3 - 15 u^4 + 6 u^5
    y_ref(x) = y_start + (y_end - y_start) h(u)

The verified protocol uses x_start = -7.75 m, L = 6.0 m, y_start = -0.065 m, and y_end = -1.160 m. Every vehicle queries the same spatial path as a function of its own longitudinal position; therefore the intended behavior is a sequential propagation of one path through the platoon, not five independently selected lane changes.

`platoon_lane_change_completion_state` aggregates the minimum progress, maximum centerline error, maximum heading error, maximum adjacent gap error, maximum pair lateral error, minimum pair distance, and road-boundary exceedance. The command configuration currently requires progress >= 0.99, lateral error <= 0.12 m, heading error <= 0.12 rad, gap error <= 0.35 m, pair lateral error <= 0.20 m, pair distance >= 0.30 m, road exceedance <= 0, and a 2 s completion dwell. The task reward keeps a streak and sets `complete_ever` only after the dwell requirement.

This joint gate is the source of the phrase whole-platoon completion in the paper.

## Reward teacher

Source: `source/my_exts/marl_platoon/algorithms/teacher.py` and task-side teacher helpers.

The teacher receives privileged clean physical information and the executed action. Its score combines spacing/velocity/acceleration errors, jerk, overspeed, centerline/lateral/heading/backward/forward deficits, collision, and action-energy terms. The score is clipped and normalized. A moving average with decay near 0.97 gives a stable reference. Teacher output is trained with a base consistency/regression objective and an outer alignment term that increases agreement between teacher shaping changes and the physical-score advantage over a window. Warm-up, ramp, clipping, and update intervals reduce variance.

Important wording: the code comments explicitly say that this is not the full paper meta-gradient yet. In the manuscript call it a first-order, proxy-aligned, physics-aware reward-teacher update. The original meta-gradient derivation can be presented as the ideal objective, while the implementation is its stable truncated approximation.

## AMS action-manifold smoother

Source: `source/my_exts/marl_platoon/algorithms/happo/ams.py`.

`HAPPOAMSCfg` currently includes hidden dimensions (256, 256), gamma 0.99, learning rate 1e-4, Polyak tau 0.01, eight local directions, neighborhood radius 0.25, normalized action limit 0.4, refresh interval 1000, Huber beta 0.3, one Q epoch, 16 mini-batches, gradient norm 0.5, target clip 100, and evaluation chunks of 8192. Geometry is `ackermann`; wheelbase 0.14325, maximum speed 1.2, maximum center steer 0.32, and curvature limit 3.0.

The central objects are:

- `ActionValueNet`: centralized observation plus joint action to a scalar Q value.
- `OrthogonalActionSampler`: QR-generated orthogonal directions for a low-variance local neighborhood.
- `HAPPOActionManifoldSmoother.project_feasible`: clips normalized actions and rejects steering values that violate the Ackermann curvature bound.
- `smoothed_q`: evaluates symmetric actions around the center, uses the conservative minimum of twin critics, and returns neighborhood mean and variance.
- `train_on_buffer`: anchors the target to the executed action, bootstraps from next-state smoothed Q, fits twin critics with Huber loss, clips gradients, and applies Polyak target updates.
- `local_advantage`: compares Q at the executed action with the smoothed neighborhood value and subtracts an optional risk term proportional to neighborhood standard deviation.
- `local_agent_advantage`: reduces the joint/local signal to per-agent credit for HAPPO.

The executed-action anchor matters under FDI: the critic is trained on what the plant actually received, not only what the actor intended.

## HAPPO integration and schedules

Source: `source/my_exts/marl_platoon/algorithms/happo/runner.py`.

The runner first computes standard HAPPO advantages. If AMS is enabled it trains the smoother, computes local agent advantages, normalizes the correction, and adds it to the standard advantage. The code defaults are correction weight 0.10, 50 warm-up updates, and a 100-update ramp. The verified matrix launcher overrides these to 0.02, 5, and 15 respectively; use the manifest rather than the class default when reproducing a result. This is a training-only auxiliary signal; deployment does not require privileged state, the teacher, or the neighborhood critic.

The retention dataset and behavior-cloning warm start are loaded from the real trajectory path. Retention is controlled by `happo_retention_enabled`, `happo_retention_trajectory_path`, BC coefficient, KL coefficient, batch size, and optional curriculum schedule.

## Configuration and routing

`source/my_exts/marl_platoon/tasks/platoon/config.py` stores algorithm flags and numerical values. The important switches are `enable_teacher`, `happo_ams_enabled`, `happo_share_actor`, `happo_use_factor`, retention settings, attack settings, and the Ackermann lane-change command settings.

`source/my_exts/marl_platoon/algorithms/router.py` translates the configuration into the selected algorithm components. The method flags are teacher, AMS, retention, and actor sharing. Verify these flags in every experiment manifest: the old AMS-only pilot used a different actor-sharing setting and is not directly comparable.

## Training and validation orchestration

- `scripts/tools/run_ackermann_verified_followup.py` maps seed index 0, 1, 2 to 42, 3407, 777; performs clean gating before attack levels; reuses the last passing checkpoint; writes manifests/status and matrix CSV files.
- `scripts/tools/run_ackermann_orthogonal_multiseed.sh` runs the four cells `happo_base_shared`, `happo_meta`, `happo_ams_only`, and `happo_meta_ams` sequentially for three training seeds.
- `scripts/tools/run_medium_algorithm_comparison.sh` contains the six-algorithm comparison. The corrected AMS-only branch sets `happo_share_actor=true` for both train and evaluation.
- `scripts/tools/ackermann_validation.py` wraps evaluation, exports trajectory CSVs, and can generate ROS replay/video artifacts.

The intended attack curriculum is clean/off, then light, easy-0/easy-a/easy-b, med-a/med-b, hard-a, and hard-b. A failed gate must stop escalation for that method and preserve the last passing model. Never replace a passing checkpoint with a failed candidate merely because it is newer. The current matrix uses the bounded `profile` attack mode; `generator` and `cagan` are separate code paths and must not be implied by profile results.

## Minimal source inspection commands

From the Isaac Lab project:

    rg -n "class AckermannAction|rear|front|steer" source/my_exts/marl_platoon/ackermann_action.py
    rg -n "quintic|completion|dwell|pair_lateral|road" source/my_exts/marl_platoon/trajectory.py source/my_exts/marl_platoon/tasks/platoon/config.py
    rg -n "class HAPPOAction|smoothed_q|local_advantage|executed" source/my_exts/marl_platoon/algorithms/happo/ams.py
    rg -n "ams_advantage|warmup|ramp|local_agent" source/my_exts/marl_platoon/algorithms/happo/runner.py
    python3 scripts/tools/test_ackermann_followup.py

