#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/source/isaaclab:${ROOT_DIR}/source/isaaclab_rl:${ROOT_DIR}/source/isaaclab_tasks:${ROOT_DIR}/source/my_exts:/home/cnc/SSD_1T/xzw/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-/home/cnc/SSD_1T/xzw/isaac-sim/python.sh}"
TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"
TASK="${TASK:-Isaac-Marl-Platoon-HAPPO-v0}"
NUM_ENVS="${NUM_ENVS:-64}"
LOG_ROOT="${ROOT_DIR}/logs/rsl_rl/platoon_happo"

STAGE1_ITERS="${STAGE1_ITERS:-3000}"
STAGE2_ITERS="${STAGE2_ITERS:-2000}"
STAGE3_ITERS="${STAGE3_ITERS:-2000}"
STAGE4_ITERS="${STAGE4_ITERS:-2000}"
STAGE5_ITERS="${STAGE5_ITERS:-1000}"
CHECK_WINDOW="${CHECK_WINDOW:-50}"
AUTO_PROMOTE="${AUTO_PROMOTE:-1}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-}"

if [[ "${NUM_ENVS}" =~ ^[0-9]+$ && "${NUM_ENVS}" -gt 128 ]]; then
  echo "[curriculum] WARNING: NUM_ENVS=${NUM_ENVS} can OOM on RTX 3080 10GB. Use NUM_ENVS=64 first."
fi

# Optional start point. Examples:
#   START_RUN=2026-06-17_17-34-19 START_CHECKPOINT=model_final.pt scripts/tools/run_platoon_curriculum.sh
#   START_STAGE=stage4_medium_profile START_RUN=2026-06-18_16-16-14_stage3_easy_attack scripts/tools/run_platoon_curriculum.sh
START_RUN="${START_RUN:-}"
START_CHECKPOINT="${START_CHECKPOINT:-model_final.pt}"
START_STAGE="${START_STAGE:-stage1_stable}"
LAST_RUN_DIR=""

stage_rank() {
  case "$1" in
    stage1_stable) echo 1 ;;
    stage2_speedup) echo 2 ;;
    stage3_easy_attack) echo 3 ;;
    stage4_medium_profile) echo 4 ;;
    stage5_cagan) echo 5 ;;
    *)
      echo "[curriculum] ERROR: unknown START_STAGE=$1" >&2
      echo "[curriculum] valid stages: stage1_stable stage2_speedup stage3_easy_attack stage4_medium_profile stage5_cagan" >&2
      exit 2
      ;;
  esac
}

START_RANK="$(stage_rank "${START_STAGE}")"
if [[ "${START_RANK}" != "1" && -z "${START_RUN}" ]]; then
  echo "[curriculum] ERROR: START_STAGE=${START_STAGE} requires START_RUN=<previous_stage_run_dir>" >&2
  exit 2
fi

latest_run_for_stage() {
  local stage="$1"
  find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${stage}" -printf '%T@ %f\n' \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

run_stage() {
  local stage="$1"
  local iterations="$2"
  local resume_run="$3"
  local checkpoint="$4"
  shift 4

  local resume_args=()
  if [[ -n "${resume_run}" ]]; then
    resume_args=(--resume --load_run "${resume_run}" --checkpoint "${checkpoint}")
    echo "[curriculum] ${stage}: resuming from ${resume_run}/${checkpoint}"
  else
    echo "[curriculum] ${stage}: starting from scratch"
  fi

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --headless \
    --max_iterations "${iterations}" \
    --run_name "${stage}" \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    "${resume_args[@]}" \
    "$@"

  local run_dir
  run_dir="$(latest_run_for_stage "${stage}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[curriculum] ERROR: could not locate run directory for ${stage}" >&2
    exit 2
  fi
  echo "[curriculum] ${stage} completed: ${run_dir}"
  LAST_RUN_DIR="${run_dir}"
}

check_stage() {
  local stage="$1"
  local run_dir="$2"
  local csv_path="${LOG_ROOT}/${run_dir}/platoon_metrics.csv"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[curriculum] ERROR: missing metrics CSV: ${csv_path}" >&2
    exit 3
  fi
  python3 scripts/tools/check_platoon_stage.py "${csv_path}" --stage "${stage}" --window "${CHECK_WINDOW}"
}

maybe_check_or_stop() {
  local stage="$1"
  local run_dir="$2"
  if [[ "${AUTO_PROMOTE}" == "0" ]]; then
    echo "[curriculum] AUTO_PROMOTE=0, not checking ${stage}; next resume point is ${run_dir}/model_final.pt"
    return 0
  fi
  if ! check_stage "${stage}" "${run_dir}"; then
    echo "[curriculum] ${stage} did not pass promotion criteria."
    echo "[curriculum] Inspect: ${LOG_ROOT}/${run_dir}/platoon_metrics.csv"
    echo "[curriculum] Plot: python3 scripts/tools/plot_platoon_metrics.py ${LOG_ROOT}/${run_dir}/platoon_metrics.csv --smooth 10"
    exit 10
  fi
}

maybe_stop_after_stage() {
  local stage="$1"
  local run_dir="$2"
  if [[ -n "${STOP_AFTER_STAGE}" && "${STOP_AFTER_STAGE}" == "${stage}" ]]; then
    echo "[curriculum] STOP_AFTER_STAGE=${stage}; stopping after ${run_dir}/model_final.pt"
    exit 0
  fi
}

if [[ "${START_RANK}" -le 1 ]]; then
  run_stage stage1_stable "${STAGE1_ITERS}" "${START_RUN}" "${START_CHECKPOINT}" \
    "env.commands.base_velocity.ranges.lin_vel_x=[0.6,0.8]" \
    "env.algorithm.happo_action_clip=0.4" \
    "env.algorithm.happo_action_warmup_updates=10" \
    "env.algorithm.enable_attack=false" \
    "env.actions.joint_vel_1.scale=12.0" \
    "env.actions.joint_vel_2.scale=12.0" \
    "env.actions.joint_vel_3.scale=12.0" \
    "env.actions.joint_vel_4.scale=12.0" \
    "env.actions.joint_vel_5.scale=12.0"
  stage1_run="${LAST_RUN_DIR}"
  maybe_check_or_stop stage1_stable "${stage1_run}"
  maybe_stop_after_stage stage1_stable "${stage1_run}"
else
  stage1_run="${START_RUN}"
fi

stage2_checkpoint="model_final.pt"
if [[ "${START_RANK}" -eq 2 ]]; then
  stage2_checkpoint="${START_CHECKPOINT}"
fi
if [[ "${START_RANK}" -le 2 ]]; then
  run_stage stage2_speedup "${STAGE2_ITERS}" "${stage1_run}" "${stage2_checkpoint}" \
    "env.commands.base_velocity.ranges.lin_vel_x=[0.7,0.9]" \
    "env.algorithm.happo_action_clip=0.5" \
    "env.algorithm.happo_action_warmup_updates=5" \
    "env.algorithm.enable_attack=false" \
    "env.actions.joint_vel_1.scale=14.0" \
    "env.actions.joint_vel_2.scale=14.0" \
    "env.actions.joint_vel_3.scale=14.0" \
    "env.actions.joint_vel_4.scale=14.0" \
    "env.actions.joint_vel_5.scale=14.0"
  stage2_run="${LAST_RUN_DIR}"
  maybe_check_or_stop stage2_speedup "${stage2_run}"
  maybe_stop_after_stage stage2_speedup "${stage2_run}"
else
  stage2_run="${START_RUN}"
fi

stage3_checkpoint="model_final.pt"
if [[ "${START_RANK}" -eq 3 ]]; then
  stage3_checkpoint="${START_CHECKPOINT}"
fi
if [[ "${START_RANK}" -le 3 ]]; then
  run_stage stage3_easy_attack "${STAGE3_ITERS}" "${stage2_run}" "${stage3_checkpoint}" \
    "env.commands.base_velocity.ranges.lin_vel_x=[0.7,0.9]" \
    "env.algorithm.happo_action_clip=0.5" \
    "env.algorithm.happo_action_warmup_updates=0" \
    "env.algorithm.enable_attack=true" \
    "env.algorithm.attack_level=easy" \
    "env.algorithm.attack_mode=profile" \
    "env.algorithm.max_fdi_acc=0.20" \
    "env.algorithm.max_dos_rate=0.03" \
    "env.actions.joint_vel_1.scale=14.0" \
    "env.actions.joint_vel_2.scale=14.0" \
    "env.actions.joint_vel_3.scale=14.0" \
    "env.actions.joint_vel_4.scale=14.0" \
    "env.actions.joint_vel_5.scale=14.0"
  stage3_run="${LAST_RUN_DIR}"
  maybe_check_or_stop stage3_easy_attack "${stage3_run}"
  maybe_stop_after_stage stage3_easy_attack "${stage3_run}"
else
  stage3_run="${START_RUN}"
fi

stage4_checkpoint="model_final.pt"
if [[ "${START_RANK}" -eq 4 ]]; then
  stage4_checkpoint="${START_CHECKPOINT}"
fi
if [[ "${START_RANK}" -le 4 ]]; then
  run_stage stage4_medium_profile "${STAGE4_ITERS}" "${stage3_run}" "${stage4_checkpoint}" \
    "env.commands.base_velocity.ranges.lin_vel_x=[0.75,0.9]" \
    "env.algorithm.happo_action_clip=0.55" \
    "env.algorithm.happo_action_warmup_updates=0" \
    "env.algorithm.enable_attack=true" \
    "env.algorithm.attack_level=medium" \
    "env.algorithm.attack_mode=profile" \
    "env.algorithm.max_fdi_acc=0.25" \
    "env.algorithm.max_dos_rate=0.05" \
    "env.algorithm.attack_curriculum_warmup_updates=0" \
    "env.actions.joint_vel_1.scale=14.0" \
    "env.actions.joint_vel_2.scale=14.0" \
    "env.actions.joint_vel_3.scale=14.0" \
    "env.actions.joint_vel_4.scale=14.0" \
    "env.actions.joint_vel_5.scale=14.0"
  stage4_run="${LAST_RUN_DIR}"
  maybe_check_or_stop stage4_medium_profile "${stage4_run}"
  maybe_stop_after_stage stage4_medium_profile "${stage4_run}"
else
  stage4_run="${START_RUN}"
fi

stage5_checkpoint="model_final.pt"
if [[ "${START_RANK}" -eq 5 ]]; then
  stage5_checkpoint="${START_CHECKPOINT}"
fi
run_stage stage5_cagan "${STAGE5_ITERS}" "${stage4_run}" "${stage5_checkpoint}" \
    "env.commands.base_velocity.ranges.lin_vel_x=[0.7,0.85]" \
    "env.algorithm.happo_action_clip=0.45" \
    "env.algorithm.happo_action_warmup_updates=0" \
    "env.algorithm.enable_attack=true" \
    "env.algorithm.attack_level=light" \
    "env.algorithm.attack_mode=cagan" \
    "env.algorithm.max_fdi_pos=1.0" \
    "env.algorithm.max_fdi_acc=0.10" \
    "env.algorithm.max_dos_rate=0.02" \
    "env.algorithm.attack_curriculum_warmup_updates=500" \
    "env.actions.joint_vel_1.scale=12.0" \
    "env.actions.joint_vel_2.scale=12.0" \
    "env.actions.joint_vel_3.scale=12.0" \
    "env.actions.joint_vel_4.scale=12.0" \
    "env.actions.joint_vel_5.scale=12.0"
stage5_run="${LAST_RUN_DIR}"
maybe_check_or_stop stage5_cagan "${stage5_run}"

echo "[curriculum] all stages completed."
echo "[curriculum] final run: ${stage5_run}"
echo "[curriculum] final checkpoint: ${LOG_ROOT}/${stage5_run}/model_final.pt"
