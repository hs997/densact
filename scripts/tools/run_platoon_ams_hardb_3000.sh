#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}"
TAG="${TAG:-hardb_ams_$(date +%Y%m%d_%H%M%S)}"

export ROOT TAG
export PIPELINE_LOG="${PIPELINE_LOG:-${ROOT}/train_${TAG}.log}"
export RESULT_ROOT="${RESULT_ROOT:-${ROOT}/logs/rsl_rl/platoon_happo/${TAG}_package}"
export RESULT_TAR="${RESULT_TAR:-${RESULT_ROOT}.tar.gz}"
export PROGRESS_NOTES_PATH="${PROGRESS_NOTES_PATH:-/home/cnc/SSD_1T/xzw/ams_debug_notes.md}"

export NUM_ENVS="${NUM_ENVS:-64}"
export EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-32}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
export EVAL_EVERY="${EVAL_EVERY:-300}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export FRESH_ENV_PER_CHECKPOINT="${FRESH_ENV_PER_CHECKPOINT:-0}"
export SEED="${SEED:-42}"
export COMPARE_ALGOS="happo_meta_ams"
export PARALLEL_TRAIN_JOBS=1

export ATTACK_PROFILE_LABEL=hard_b
export ATTACK_LEVEL=hard
export ATTACK_MODE=profile
export ATTACK_MAX_FDI_POS=4.0
export ATTACK_MAX_FDI_ACC=1.30
export ATTACK_MAX_DOS_RATE=0.18
export HAPPO_ACTION_CLIP=0.4
export COMMAND_SPEED_RANGE='[0.30,0.45]'
export SAFETY_D_DROP=1.45
export SAFETY_CATCHUP_ACTION=-0.355
export WHEEL_ACTION_SCALE=12.5

export HAPPO_META_SHARE_ACTOR=true
export HAPPO_META_TEACHER_SHAPING_COEF=0.016
export HAPPO_META_TEACHER_LR=2.0e-4
export HAPPO_META_TEACHER_SHAPING_CLIP=0.16
export HAPPO_META_TEACHER_CONSISTENCY_COEF=0.04
export HAPPO_META_TEACHER_OUTER_DELTA_COEF=0.045
export HAPPO_META_TEACHER_OUTER_DELTA_WARMUP_UPDATES=5
export HAPPO_META_TEACHER_OUTER_DELTA_RAMP_UPDATES=30
export HAPPO_META_TEACHER_LAMBDA_SPACING=1.45
export HAPPO_META_TEACHER_LAMBDA_VELOCITY=0.55
export HAPPO_META_TEACHER_LAMBDA_CENTERLINE=2.25
export HAPPO_META_TEACHER_LAMBDA_LATERAL=1.85
export HAPPO_META_TEACHER_LAMBDA_HEADING=1.10
export HAPPO_META_TEACHER_LAMBDA_FORWARD_DEFICIT=0.35
export HAPPO_META_TEACHER_LAMBDA_ACTION_ENERGY=0.008
export HAPPO_META_EXTRA_OVERRIDES="env.algorithm.local_reward_gap_coef=0.18 env.algorithm.local_reward_centerline_coef=0.42 env.algorithm.local_reward_pair_lateral_coef=0.66 env.algorithm.local_reward_heading_coef=0.16 env.algorithm.local_reward_turn_coef=0.004 env.safety_shield.d_crit=1.505 env.safety_shield.d_drop=1.42 env.safety_shield.catchup_action=-0.365 env.safety_shield.lateral_tol=0.005 env.safety_shield.lateral_turn_gain=0.40 env.safety_shield.centerline_turn_gain=0.90 env.safety_shield.centerline_turn_clip=0.22 env.safety_shield.first_follower_centerline_gain=1.05 env.safety_shield.first_follower_centerline_clip=0.22 env.safety_shield.forward_bias_gain=0.18 env.safety_shield.forward_bias_clip=0.018 env.safety_shield.forward_bias_min_command=0.30 env.safety_shield.forward_bias_speed_margin=-0.02 env.safety_shield.forward_bias_min_gap=1.45 env.safety_shield.forward_bias_leader_gain_scale=2.5 env.safety_shield.forward_bias_leader_clip_scale=2.0"
export HAPPO_META_TRAIN_EXTRA_OVERRIDES="env.rewards.formation.weight=2.35 env.rewards.leader_progress.weight=6.25 env.rewards.forward_drive.weight=6.40 env.rewards.lateral_correct.weight=-3.20 env.rewards.centerline_lateral.weight=-0.40"
export HAPPO_META_EVAL_EXTRA_OVERRIDES="env.safety_shield.lateral_turn_gain=0.44 env.safety_shield.lateral_turn_clip=0.09 env.safety_shield.lateral_velocity_gain=0.12 env.safety_shield.first_follower_lateral_gain_scale=0.80 env.safety_shield.pair2_lateral_gain_scale=1.25 env.safety_shield.centerline_turn_gain=1.15 env.safety_shield.centerline_turn_clip=0.24"

export HAPPO_AMS_LR="${HAPPO_AMS_LR:-1.0e-4}"
export HAPPO_AMS_TAU="${HAPPO_AMS_TAU:-0.01}"
export HAPPO_AMS_NUM_NEIGHBORS="${HAPPO_AMS_NUM_NEIGHBORS:-8}"
export HAPPO_AMS_NEIGHBORHOOD_RADIUS="${HAPPO_AMS_NEIGHBORHOOD_RADIUS:-0.25}"
export HAPPO_AMS_REFRESH_EVERY="${HAPPO_AMS_REFRESH_EVERY:-1000}"
export HAPPO_AMS_HUBER_BETA="${HAPPO_AMS_HUBER_BETA:-0.3}"
export HAPPO_AMS_Q_EPOCHS="${HAPPO_AMS_Q_EPOCHS:-1}"
export HAPPO_AMS_NUM_MINI_BATCHES="${HAPPO_AMS_NUM_MINI_BATCHES:-16}"
export HAPPO_AMS_MAX_GRAD_NORM="${HAPPO_AMS_MAX_GRAD_NORM:-0.5}"
export HAPPO_AMS_TARGET_CLIP="${HAPPO_AMS_TARGET_CLIP:-100.0}"
export HAPPO_AMS_EVAL_CHUNK_SIZE="${HAPPO_AMS_EVAL_CHUNK_SIZE:-8192}"
export HAPPO_AMS_ADVANTAGE_WEIGHT="${HAPPO_AMS_ADVANTAGE_WEIGHT:-0.10}"
export HAPPO_AMS_WARMUP_UPDATES="${HAPPO_AMS_WARMUP_UPDATES:-50}"
export HAPPO_AMS_RAMP_UPDATES="${HAPPO_AMS_RAMP_UPDATES:-100}"

exec bash "${ROOT}/scripts/tools/run_medium_algorithm_comparison.sh"
