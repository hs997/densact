#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/cnc/SSD_1T/xzw/isaac-sim}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
NUM_ENVS="${NUM_ENVS:-32}"
SEED="${SEED:-42}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
CHECKPOINTS="${CHECKPOINTS:?set CHECKPOINTS to a comma-separated checkpoint list}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
KIT_ARGS="${KIT_ARGS:---/rtx/verifyDriverVersion/enabled=false --/app/viewport/grid/enabled=false --/app/viewport/defaults/guide/grid/visible=false}"
FRESH_ENV_PER_CHECKPOINT="${FRESH_ENV_PER_CHECKPOINT:-0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

cd "${ROOT}" || exit 1

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
unset _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
unset VIRTUAL_ENV PYTHONHOME

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/source/isaaclab:${ROOT}/source/isaaclab_rl:${ROOT}/source/isaaclab_tasks:${ROOT}/source/my_exts:${ISAAC_SIM_ROOT}/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"

COMMON_OVERRIDES=(
  "env.algorithm.enable_attack=true"
  "env.algorithm.attack_level=medium"
  "env.algorithm.attack_mode=profile"
  "env.algorithm.attack_curriculum_warmup_updates=0"
  "env.algorithm.max_fdi_pos=2.0"
  "env.algorithm.max_fdi_acc=0.50"
  "env.algorithm.max_dos_rate=0.10"
  "env.algorithm.enable_shield=true"
  "env.algorithm.happo_action_clip=0.4"
  "env.commands.base_velocity.ranges.lin_vel_x=[0.30,0.45]"
  "env.safety_shield.d_drop=1.45"
  "env.safety_shield.catchup_action=-0.355"
  "env.safety_shield.lateral_turn_gain=0.32"
  "env.safety_shield.centerline_turn_gain=0.28"
  "env.safety_shield.pair3_lateral_gain_scale=0.85"
  "env.safety_shield.pair4_lateral_gain_scale=0.90"
  "env.actions.joint_vel_1.scale=12.5"
  "env.actions.joint_vel_2.scale=12.5"
  "env.actions.joint_vel_3.scale=12.5"
  "env.actions.joint_vel_4.scale=12.5"
  "env.actions.joint_vel_5.scale=12.5"
)

fresh_args=()
if [[ "${FRESH_ENV_PER_CHECKPOINT}" == "1" ]]; then
  fresh_args+=(--fresh_env_per_checkpoint)
fi
read -r -a extra_override_items <<< "${EXTRA_OVERRIDES}"

"${ISAAC_PYTHON}" scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py \
  --task Isaac-Marl-Platoon-HAPPO-v0 \
  --num_envs "${NUM_ENVS}" \
  --seed "${SEED}" \
  --headless \
  --eval_checkpoints "${CHECKPOINTS}" \
  --eval_steps "${EVAL_STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --output_dir "${OUT_DIR}" \
  --enable_attack_eval \
  "${fresh_args[@]}" \
  --kit_args="${KIT_ARGS}" \
  "${COMMON_OVERRIDES[@]}" \
  "${extra_override_items[@]}"
