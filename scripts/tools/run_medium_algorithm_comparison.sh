#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/cnc/SSD_1T/xzw/isaac-sim}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
LOG_ROOT="${ROOT}/logs/rsl_rl/platoon_happo"

TAG="${TAG:-medium_compare_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_LOG="${PIPELINE_LOG:-${ROOT}/train_${TAG}.log}"
RESULT_ROOT="${RESULT_ROOT:-${LOG_ROOT}/${TAG}_package}"
RESULT_TAR="${RESULT_TAR:-${RESULT_ROOT}.tar.gz}"

NUM_ENVS="${NUM_ENVS:-64}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-32}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1200}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
FRESH_ENV_PER_CHECKPOINT="${FRESH_ENV_PER_CHECKPOINT:-0}"
SEED="${SEED:-42}"
SKIP_EVAL="${SKIP_EVAL:-0}"
COMPARE_ALGOS="${COMPARE_ALGOS:-mappo happo_no_meta happo_meta harl_mappo_shared harl_haa2c harl_hatrpo}"
PARALLEL_TRAIN_JOBS="${PARALLEL_TRAIN_JOBS:-1}"
PARALLEL_TRAIN_STAGGER_SEC="${PARALLEL_TRAIN_STAGGER_SEC:-0}"
ATTACK_PROFILE_LABEL="${ATTACK_PROFILE_LABEL:-medium}"
ATTACK_LEVEL="${ATTACK_LEVEL:-medium}"
ATTACK_MODE="${ATTACK_MODE:-profile}"
ATTACK_MAX_FDI_POS="${ATTACK_MAX_FDI_POS:-2.0}"
ATTACK_MAX_FDI_ACC="${ATTACK_MAX_FDI_ACC:-0.50}"
ATTACK_MAX_DOS_RATE="${ATTACK_MAX_DOS_RATE:-0.10}"
HAPPO_ACTION_CLIP="${HAPPO_ACTION_CLIP:-0.4}"
COMMAND_SPEED_RANGE="${COMMAND_SPEED_RANGE:-[0.30,0.45]}"
SAFETY_D_DROP="${SAFETY_D_DROP:-1.45}"
SAFETY_CATCHUP_ACTION="${SAFETY_CATCHUP_ACTION:--0.355}"
SAFETY_LATERAL_TURN_GAIN="${SAFETY_LATERAL_TURN_GAIN:-0.32}"
SAFETY_CENTERLINE_TURN_GAIN="${SAFETY_CENTERLINE_TURN_GAIN:-0.28}"
SAFETY_PAIR3_LATERAL_GAIN_SCALE="${SAFETY_PAIR3_LATERAL_GAIN_SCALE:-0.85}"
SAFETY_PAIR4_LATERAL_GAIN_SCALE="${SAFETY_PAIR4_LATERAL_GAIN_SCALE:-0.90}"
WHEEL_ACTION_SCALE="${WHEEL_ACTION_SCALE:-12.5}"
KIT_ARGS="${KIT_ARGS:---/rtx/verifyDriverVersion/enabled=false --/app/viewport/grid/enabled=false --/app/viewport/defaults/guide/grid/visible=false}"

if [[ "${MEDIUM_COMPARE_LOG_ACTIVE:-0}" != "1" ]]; then
  export MEDIUM_COMPARE_LOG_ACTIVE=1
  mkdir -p "$(dirname "${PIPELINE_LOG}")"
  exec >> "${PIPELINE_LOG}" 2>&1
fi

cd "${ROOT}" || exit 1
trap 'status=$?; echo "[MEDIUM_COMPARE] exit status=${status} at line=${LINENO}"' EXIT

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
unset _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
unset VIRTUAL_ENV PYTHONHOME

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/source/isaaclab:${ROOT}/source/isaaclab_rl:${ROOT}/source/isaaclab_tasks:${ROOT}/source/my_exts:${ISAAC_SIM_ROOT}/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"

COMMON_OVERRIDES=(
  "env.algorithm.enable_attack=true"
  "env.algorithm.attack_level=${ATTACK_LEVEL}"
  "env.algorithm.attack_mode=${ATTACK_MODE}"
  "env.algorithm.attack_curriculum_warmup_updates=0"
  "env.algorithm.max_fdi_pos=${ATTACK_MAX_FDI_POS}"
  "env.algorithm.max_fdi_acc=${ATTACK_MAX_FDI_ACC}"
  "env.algorithm.max_dos_rate=${ATTACK_MAX_DOS_RATE}"
  "env.algorithm.enable_shield=true"
  "env.algorithm.happo_action_clip=${HAPPO_ACTION_CLIP}"
  "env.commands.base_velocity.ranges.lin_vel_x=${COMMAND_SPEED_RANGE}"
  "env.safety_shield.d_drop=${SAFETY_D_DROP}"
  "env.safety_shield.catchup_action=${SAFETY_CATCHUP_ACTION}"
  "env.safety_shield.lateral_turn_gain=${SAFETY_LATERAL_TURN_GAIN}"
  "env.safety_shield.centerline_turn_gain=${SAFETY_CENTERLINE_TURN_GAIN}"
  "env.safety_shield.pair3_lateral_gain_scale=${SAFETY_PAIR3_LATERAL_GAIN_SCALE}"
  "env.safety_shield.pair4_lateral_gain_scale=${SAFETY_PAIR4_LATERAL_GAIN_SCALE}"
  "env.actions.joint_vel_1.scale=${WHEEL_ACTION_SCALE}"
  "env.actions.joint_vel_2.scale=${WHEEL_ACTION_SCALE}"
  "env.actions.joint_vel_3.scale=${WHEEL_ACTION_SCALE}"
  "env.actions.joint_vel_4.scale=${WHEEL_ACTION_SCALE}"
  "env.actions.joint_vel_5.scale=${WHEEL_ACTION_SCALE}"
)

HAPPO_META_SHARE_ACTOR="${HAPPO_META_SHARE_ACTOR:-true}"
HAPPO_META_TEACHER_SHAPING_COEF="${HAPPO_META_TEACHER_SHAPING_COEF:-0.015}"
HAPPO_META_TEACHER_LR="${HAPPO_META_TEACHER_LR:-2.0e-4}"
HAPPO_META_TEACHER_SHAPING_CLIP="${HAPPO_META_TEACHER_SHAPING_CLIP:-0.15}"
HAPPO_META_TEACHER_CONSISTENCY_COEF="${HAPPO_META_TEACHER_CONSISTENCY_COEF:-0.04}"
HAPPO_META_TEACHER_OUTER_DELTA_COEF="${HAPPO_META_TEACHER_OUTER_DELTA_COEF:-0.04}"
HAPPO_META_TEACHER_OUTER_DELTA_WARMUP_UPDATES="${HAPPO_META_TEACHER_OUTER_DELTA_WARMUP_UPDATES:-5}"
HAPPO_META_TEACHER_OUTER_DELTA_RAMP_UPDATES="${HAPPO_META_TEACHER_OUTER_DELTA_RAMP_UPDATES:-30}"
HAPPO_META_TEACHER_LAMBDA_SPACING="${HAPPO_META_TEACHER_LAMBDA_SPACING:-1.20}"
HAPPO_META_TEACHER_LAMBDA_VELOCITY="${HAPPO_META_TEACHER_LAMBDA_VELOCITY:-0.40}"
HAPPO_META_TEACHER_LAMBDA_CENTERLINE="${HAPPO_META_TEACHER_LAMBDA_CENTERLINE:-2.50}"
HAPPO_META_TEACHER_LAMBDA_LATERAL="${HAPPO_META_TEACHER_LAMBDA_LATERAL:-2.00}"
HAPPO_META_TEACHER_LAMBDA_HEADING="${HAPPO_META_TEACHER_LAMBDA_HEADING:-1.20}"
HAPPO_META_TEACHER_LAMBDA_FORWARD_DEFICIT="${HAPPO_META_TEACHER_LAMBDA_FORWARD_DEFICIT:-0.25}"
HAPPO_META_TEACHER_LAMBDA_ACTION_ENERGY="${HAPPO_META_TEACHER_LAMBDA_ACTION_ENERGY:-0.01}"
HAPPO_META_EXTRA_OVERRIDES="${HAPPO_META_EXTRA_OVERRIDES:-}"
HAPPO_META_TRAIN_EXTRA_OVERRIDES="${HAPPO_META_TRAIN_EXTRA_OVERRIDES:-}"
HAPPO_META_EVAL_EXTRA_OVERRIDES="${HAPPO_META_EVAL_EXTRA_OVERRIDES:-}"

HAPPO_META_OVERRIDES=(
  "env.algorithm.algorithm=happo"
  "env.algorithm.enable_teacher=true"
  "env.algorithm.happo_use_factor=true"
  "env.algorithm.happo_share_actor=${HAPPO_META_SHARE_ACTOR}"
  "env.algorithm.happo_actor_update_mode=ppo"
  "env.algorithm.teacher_shaping_coef=${HAPPO_META_TEACHER_SHAPING_COEF}"
  "env.algorithm.teacher_lr=${HAPPO_META_TEACHER_LR}"
  "env.algorithm.teacher_update_interval=1"
  "env.algorithm.teacher_every_student_updates=2"
  "env.algorithm.teacher_shaping_clip=${HAPPO_META_TEACHER_SHAPING_CLIP}"
  "env.algorithm.teacher_action_penalty_coef=0.001"
  "env.algorithm.teacher_reward_ema_tau=0.95"
  "env.algorithm.teacher_consistency_coef=${HAPPO_META_TEACHER_CONSISTENCY_COEF}"
  "env.algorithm.teacher_outer_delta_coef=${HAPPO_META_TEACHER_OUTER_DELTA_COEF}"
  "env.algorithm.teacher_outer_delta_warmup_updates=${HAPPO_META_TEACHER_OUTER_DELTA_WARMUP_UPDATES}"
  "env.algorithm.teacher_outer_delta_ramp_updates=${HAPPO_META_TEACHER_OUTER_DELTA_RAMP_UPDATES}"
  "env.algorithm.teacher_lambda_spacing=${HAPPO_META_TEACHER_LAMBDA_SPACING}"
  "env.algorithm.teacher_lambda_velocity=${HAPPO_META_TEACHER_LAMBDA_VELOCITY}"
  "env.algorithm.teacher_lambda_centerline=${HAPPO_META_TEACHER_LAMBDA_CENTERLINE}"
  "env.algorithm.teacher_lambda_lateral=${HAPPO_META_TEACHER_LAMBDA_LATERAL}"
  "env.algorithm.teacher_lambda_heading=${HAPPO_META_TEACHER_LAMBDA_HEADING}"
  "env.algorithm.teacher_lambda_forward_deficit=${HAPPO_META_TEACHER_LAMBDA_FORWARD_DEFICIT}"
  "env.algorithm.teacher_lambda_action_energy=${HAPPO_META_TEACHER_LAMBDA_ACTION_ENERGY}"
)
read -r -a HAPPO_META_EXTRA_OVERRIDE_ITEMS <<< "${HAPPO_META_EXTRA_OVERRIDES}"
read -r -a HAPPO_META_TRAIN_EXTRA_OVERRIDE_ITEMS <<< "${HAPPO_META_TRAIN_EXTRA_OVERRIDES}"
read -r -a HAPPO_META_EVAL_EXTRA_OVERRIDE_ITEMS <<< "${HAPPO_META_EVAL_EXTRA_OVERRIDES}"

latest_run_for_name() {
  local run_name="$1"
  find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

select_checkpoints() {
  local run_dir="$1"
  python3 - "${run_dir}" "${EVAL_EVERY}" "${MAX_ITERATIONS}" <<'PY'
from pathlib import Path
import re
import sys

run = Path(sys.argv[1])
every = int(sys.argv[2])
max_iterations = int(sys.argv[3])
items = []
for path in run.glob("model_*.pt"):
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match:
        iteration = int(match.group(1))
        if iteration % every == 0:
            items.append((iteration, path))
items.sort(key=lambda item: item[0])
final = run / "model_final.pt"
if final.exists():
    items.append((max_iterations, final))
print(",".join(str(path.resolve()) for _, path in items))
PY
}

run_train() {
  local label="$1"
  shift
  local run_name="${TAG}_${label}"
  echo
  echo "[MEDIUM_COMPARE] train label=${label} run_name=${run_name} max_iterations=${MAX_ITERATIONS}"
  "${ISAAC_PYTHON}" scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs "${NUM_ENVS}" \
    --headless \
    --seed "${SEED}" \
    --run_name "${run_name}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --kit_args="${KIT_ARGS}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
  local run_dir
  run_dir="$(latest_run_for_name "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] could not locate run dir for ${run_name}" >&2
    exit 2
  fi
  mkdir -p "${RESULT_ROOT}/training"
  {
    flock -x 200
    echo "${label}=${run_dir}" >> "${RESULT_ROOT}/training/runs.txt"
  } 200>"${RESULT_ROOT}/training/runs.lock"
  echo "[MEDIUM_COMPARE] completed ${label}: ${run_dir}"
}

run_train_label() {
  local label="$1"
  case "${label}" in
    mappo)
      run_train "${label}" \
        "env.algorithm.algorithm=mappo" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=false" \
        "env.algorithm.happo_share_actor=false" \
        "env.algorithm.happo_actor_update_mode=ppo"
      ;;
    happo_no_meta)
      run_train "${label}" \
        "env.algorithm.algorithm=happo" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=true" \
        "env.algorithm.happo_share_actor=false" \
        "env.algorithm.happo_actor_update_mode=ppo"
      ;;
    happo_meta)
      run_train "${label}" \
        "${HAPPO_META_OVERRIDES[@]}" \
        "${HAPPO_META_EXTRA_OVERRIDE_ITEMS[@]}" \
        "${HAPPO_META_TRAIN_EXTRA_OVERRIDE_ITEMS[@]}"
      ;;
    harl_mappo_shared)
      run_train "${label}" \
        "env.algorithm.algorithm=mappo" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=false" \
        "env.algorithm.happo_share_actor=true" \
        "env.algorithm.happo_actor_update_mode=ppo"
      ;;
    harl_haa2c)
      run_train "${label}" \
        "env.algorithm.algorithm=haa2c" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=true" \
        "env.algorithm.happo_share_actor=true" \
        "env.algorithm.happo_actor_update_mode=a2c"
      ;;
    harl_hatrpo)
      run_train "${label}" \
        "env.algorithm.algorithm=hatrpo" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=true" \
        "env.algorithm.happo_share_actor=true" \
        "env.algorithm.happo_actor_update_mode=trpo" \
        "env.algorithm.happo_trpo_kl_threshold=0.01" \
        "env.algorithm.happo_trpo_cg_iters=10" \
        "env.algorithm.happo_trpo_damping=0.1" \
        "env.algorithm.happo_trpo_line_search_steps=10"
      ;;
    *)
      echo "[ERROR] unsupported COMPARE_ALGOS label: ${label}" >&2
      exit 4
      ;;
  esac
}

run_eval_label() {
  local label="$1"
  local run_dir="$2"
  case "${label}" in
    mappo)
      run_eval "${label}" "${run_dir}" \
        "env.algorithm.algorithm=mappo" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=false" \
        "env.algorithm.happo_share_actor=false" \
        "env.algorithm.happo_actor_update_mode=ppo"
      ;;
    happo_no_meta)
      run_eval "${label}" "${run_dir}" \
        "env.algorithm.algorithm=happo" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=true" \
        "env.algorithm.happo_share_actor=false" \
        "env.algorithm.happo_actor_update_mode=ppo"
      ;;
    happo_meta)
      run_eval "${label}" "${run_dir}" \
        "${HAPPO_META_OVERRIDES[@]}" \
        "${HAPPO_META_EXTRA_OVERRIDE_ITEMS[@]}" \
        "${HAPPO_META_EVAL_EXTRA_OVERRIDE_ITEMS[@]}"
      ;;
    harl_mappo_shared)
      run_eval "${label}" "${run_dir}" \
        "env.algorithm.algorithm=mappo" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=false" \
        "env.algorithm.happo_share_actor=true" \
        "env.algorithm.happo_actor_update_mode=ppo"
      ;;
    harl_haa2c)
      run_eval "${label}" "${run_dir}" \
        "env.algorithm.algorithm=haa2c" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=true" \
        "env.algorithm.happo_share_actor=true" \
        "env.algorithm.happo_actor_update_mode=a2c"
      ;;
    harl_hatrpo)
      run_eval "${label}" "${run_dir}" \
        "env.algorithm.algorithm=hatrpo" \
        "env.algorithm.enable_teacher=false" \
        "env.algorithm.happo_use_factor=true" \
        "env.algorithm.happo_share_actor=true" \
        "env.algorithm.happo_actor_update_mode=trpo" \
        "env.algorithm.happo_trpo_kl_threshold=0.01" \
        "env.algorithm.happo_trpo_cg_iters=10" \
        "env.algorithm.happo_trpo_damping=0.1" \
        "env.algorithm.happo_trpo_line_search_steps=10"
      ;;
    *)
      echo "[ERROR] unsupported eval label: ${label}" >&2
      exit 5
      ;;
  esac
}

run_eval() {
  local label="$1"
  local run_dir="$2"
  shift 2
  local checkpoints
  checkpoints="$(select_checkpoints "${run_dir}")"
  if [[ -z "${checkpoints}" ]]; then
    echo "[ERROR] no checkpoints selected for ${label}: ${run_dir}" >&2
    exit 3
  fi
  local output_dir="${RESULT_ROOT}/evaluation/${label}"
  local fresh_env_args=()
  if [[ "${FRESH_ENV_PER_CHECKPOINT}" == "1" ]]; then
    fresh_env_args+=(--fresh_env_per_checkpoint)
  fi
  echo
  echo "[MEDIUM_COMPARE] eval label=${label} checkpoints=${checkpoints}"
  "${ISAAC_PYTHON}" scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs "${EVAL_NUM_ENVS}" \
    --seed "${SEED}" \
    --headless \
    --eval_checkpoints "${checkpoints}" \
    --eval_steps "${EVAL_STEPS}" \
    --warmup_steps 0 \
    --output_dir "${output_dir}" \
    --enable_attack_eval \
    "${fresh_env_args[@]}" \
    --kit_args="${KIT_ARGS}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

copy_artifacts() {
  local label="$1"
  local run_dir="$2"
  local dst="${RESULT_ROOT}/training/${label}"
  mkdir -p "${dst}"
  cp -a "${run_dir}/platoon_metrics.csv" "${dst}/" 2>/dev/null || true
  cp -a "${run_dir}/params" "${dst}/" 2>/dev/null || true
  cp -a "${run_dir}/model_final.pt" "${dst}/" 2>/dev/null || true
  cp -a "${run_dir}/model_best.pt" "${dst}/" 2>/dev/null || true
}

echo "[MEDIUM_COMPARE] tag=${TAG}"
echo "[MEDIUM_COMPARE] log=${PIPELINE_LOG}"
echo "[MEDIUM_COMPARE] result_root=${RESULT_ROOT}"
echo "[MEDIUM_COMPARE] iterations=${MAX_ITERATIONS} eval_every=${EVAL_EVERY} eval_steps=${EVAL_STEPS}"
echo "[MEDIUM_COMPARE] compare_algos=${COMPARE_ALGOS}"
echo "[MEDIUM_COMPARE] parallel_train_jobs=${PARALLEL_TRAIN_JOBS}"
echo "[MEDIUM_COMPARE] parallel_train_stagger_sec=${PARALLEL_TRAIN_STAGGER_SEC}"
echo "[MEDIUM_COMPARE] attack_profile=${ATTACK_PROFILE_LABEL} level=${ATTACK_LEVEL} mode=${ATTACK_MODE} max_fdi_pos=${ATTACK_MAX_FDI_POS} max_fdi_acc=${ATTACK_MAX_FDI_ACC} max_dos_rate=${ATTACK_MAX_DOS_RATE}"

rm -rf "${RESULT_ROOT}"
mkdir -p "${RESULT_ROOT}/training" "${RESULT_ROOT}/evaluation"
rm -f "${RESULT_ROOT}/training/runs.txt" "${RESULT_ROOT}/training/runs.lock"
{
  echo "tag=${TAG}"
  echo "attack_profile_label=${ATTACK_PROFILE_LABEL}"
  echo "attack_level=${ATTACK_LEVEL}"
  echo "attack_mode=${ATTACK_MODE}"
  echo "attack_max_fdi_pos=${ATTACK_MAX_FDI_POS}"
  echo "attack_max_fdi_acc=${ATTACK_MAX_FDI_ACC}"
  echo "attack_max_dos_rate=${ATTACK_MAX_DOS_RATE}"
  echo "happo_action_clip=${HAPPO_ACTION_CLIP}"
  echo "command_speed_range=${COMMAND_SPEED_RANGE}"
  echo "safety_d_drop=${SAFETY_D_DROP}"
  echo "safety_catchup_action=${SAFETY_CATCHUP_ACTION}"
  echo "wheel_action_scale=${WHEEL_ACTION_SCALE}"
  echo "max_iterations=${MAX_ITERATIONS}"
  echo "eval_every=${EVAL_EVERY}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "fresh_env_per_checkpoint=${FRESH_ENV_PER_CHECKPOINT}"
  echo "seed=${SEED}"
  echo "compare_algos=${COMPARE_ALGOS}"
  echo "parallel_train_jobs=${PARALLEL_TRAIN_JOBS}"
  echo "parallel_train_stagger_sec=${PARALLEL_TRAIN_STAGGER_SEC}"
  echo "happo_meta_share_actor=${HAPPO_META_SHARE_ACTOR}"
  echo "happo_meta_teacher_shaping_coef=${HAPPO_META_TEACHER_SHAPING_COEF}"
  echo "happo_meta_teacher_lr=${HAPPO_META_TEACHER_LR}"
  echo "happo_meta_teacher_shaping_clip=${HAPPO_META_TEACHER_SHAPING_CLIP}"
  echo "happo_meta_teacher_consistency_coef=${HAPPO_META_TEACHER_CONSISTENCY_COEF}"
  echo "happo_meta_teacher_outer_delta_coef=${HAPPO_META_TEACHER_OUTER_DELTA_COEF}"
  echo "happo_meta_teacher_outer_delta_warmup_updates=${HAPPO_META_TEACHER_OUTER_DELTA_WARMUP_UPDATES}"
  echo "happo_meta_teacher_outer_delta_ramp_updates=${HAPPO_META_TEACHER_OUTER_DELTA_RAMP_UPDATES}"
  echo "happo_meta_teacher_lambda_spacing=${HAPPO_META_TEACHER_LAMBDA_SPACING}"
  echo "happo_meta_teacher_lambda_velocity=${HAPPO_META_TEACHER_LAMBDA_VELOCITY}"
  echo "happo_meta_teacher_lambda_centerline=${HAPPO_META_TEACHER_LAMBDA_CENTERLINE}"
  echo "happo_meta_teacher_lambda_lateral=${HAPPO_META_TEACHER_LAMBDA_LATERAL}"
  echo "happo_meta_teacher_lambda_heading=${HAPPO_META_TEACHER_LAMBDA_HEADING}"
  echo "happo_meta_teacher_lambda_forward_deficit=${HAPPO_META_TEACHER_LAMBDA_FORWARD_DEFICIT}"
  echo "happo_meta_teacher_lambda_action_energy=${HAPPO_META_TEACHER_LAMBDA_ACTION_ENERGY}"
  echo "happo_meta_extra_overrides=${HAPPO_META_EXTRA_OVERRIDES}"
  echo "happo_meta_train_extra_overrides=${HAPPO_META_TRAIN_EXTRA_OVERRIDES}"
  echo "happo_meta_eval_extra_overrides=${HAPPO_META_EVAL_EXTRA_OVERRIDES}"
  echo "local_backup=/home/cnc/SSD_1T/xzw/IsaacLab-main/backups/pre_medium_compare_code_assets_20260701_022215.tar"
  echo "git_backup_commit=79b8290"
  echo "git_current_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} > "${RESULT_ROOT}/manifest.txt"

read -r -a compare_algo_items <<< "${COMPARE_ALGOS}"

if (( PARALLEL_TRAIN_JOBS > 1 )); then
  batch_pids=()
  batch_labels=()
  launched_labels=0
  total_labels="${#compare_algo_items[@]}"
  for label in "${compare_algo_items[@]}"; do
    label_log="${RESULT_ROOT}/training/${label}_train.log"
    echo "[MEDIUM_COMPARE] parallel train start label=${label} log=${label_log}"
    (
      run_train_label "${label}"
    ) > "${label_log}" 2>&1 &
    batch_pids+=("$!")
    batch_labels+=("${label}")
    launched_labels=$((launched_labels + 1))
    if (( PARALLEL_TRAIN_STAGGER_SEC > 0 && ${#batch_pids[@]} < PARALLEL_TRAIN_JOBS && launched_labels < total_labels )); then
      echo "[MEDIUM_COMPARE] stagger sleep ${PARALLEL_TRAIN_STAGGER_SEC}s before next parallel launch"
      sleep "${PARALLEL_TRAIN_STAGGER_SEC}"
    fi
    if (( ${#batch_pids[@]} >= PARALLEL_TRAIN_JOBS )); then
      for idx in "${!batch_pids[@]}"; do
        pid="${batch_pids[$idx]}"
        wait_label="${batch_labels[$idx]}"
        echo "[MEDIUM_COMPARE] waiting label=${wait_label} pid=${pid}"
        if ! wait "${pid}"; then
          echo "[ERROR] train failed for ${wait_label}; see ${RESULT_ROOT}/training/${wait_label}_train.log" >&2
          exit 6
        fi
        echo "[MEDIUM_COMPARE] train finished label=${wait_label}"
      done
      batch_pids=()
      batch_labels=()
    fi
  done
  if (( ${#batch_pids[@]} > 0 )); then
    for idx in "${!batch_pids[@]}"; do
      pid="${batch_pids[$idx]}"
      wait_label="${batch_labels[$idx]}"
      echo "[MEDIUM_COMPARE] waiting label=${wait_label} pid=${pid}"
      if ! wait "${pid}"; then
        echo "[ERROR] train failed for ${wait_label}; see ${RESULT_ROOT}/training/${wait_label}_train.log" >&2
        exit 6
      fi
      echo "[MEDIUM_COMPARE] train finished label=${wait_label}"
    done
  fi
else
  for label in "${compare_algo_items[@]}"; do
    run_train_label "${label}"
  done
fi

while IFS='=' read -r label run_dir; do
  copy_artifacts "${label}" "${run_dir}"
done < "${RESULT_ROOT}/training/runs.txt"

if [[ "${SKIP_EVAL}" != "1" ]]; then
  while IFS='=' read -r label run_dir; do
    run_eval_label "${label}" "${run_dir}"
  done < "${RESULT_ROOT}/training/runs.txt"

  python3 scripts/tools/plot_medium_algorithm_comparison.py --result-root "${RESULT_ROOT}"
fi

cp -a "${PIPELINE_LOG}" "${RESULT_ROOT}/pipeline.log" 2>/dev/null || true
cp -a debug_notes.md "${RESULT_ROOT}/debug_notes.md" 2>/dev/null || true
git diff > "${RESULT_ROOT}/git_diff_after_medium_compare.diff" || true
tar -czf "${RESULT_TAR}" -C "$(dirname "${RESULT_ROOT}")" "$(basename "${RESULT_ROOT}")"
echo "[MEDIUM_COMPARE] result root: ${RESULT_ROOT}"
echo "[MEDIUM_COMPARE] result tar:  ${RESULT_TAR}"
