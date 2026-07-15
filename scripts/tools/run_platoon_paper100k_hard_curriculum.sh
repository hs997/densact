#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/cnc/SSD_1T/xzw/isaac-sim}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
LOG_ROOT="${ROOT}/logs/rsl_rl/platoon_happo"

RUN_TAG="${RUN_TAG:-paper100k_hard_curriculum_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_LOG="${PIPELINE_LOG:-${ROOT}/train_${RUN_TAG}.log}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${LOG_ROOT}/${RUN_TAG}_package}"
PACKAGE_TAR="${PACKAGE_TAR:-${PACKAGE_ROOT}.tar.gz}"

START_RUN="${START_RUN:-2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000}"
START_CHECKPOINT="${START_CHECKPOINT:-model_best.pt}"
NUM_ENVS="${NUM_ENVS:-64}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
EVAL_WARMUP_STEPS="${EVAL_WARMUP_STEPS:-100}"
PLOT_SMOOTH="${PLOT_SMOOTH:-100}"
DRY_RUN="${DRY_RUN:-0}"
STAGE_LIMIT="${STAGE_LIMIT:-5}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"

KIT_ARGS="${KIT_ARGS:---/rtx/verifyDriverVersion/enabled=false --/app/viewport/grid/enabled=false --/app/viewport/defaults/guide/grid/visible=false}"

if [[ "${PLATOON_PIPELINE_LOG_ACTIVE:-0}" != "1" ]]; then
  export PLATOON_PIPELINE_LOG_ACTIVE=1
  mkdir -p "$(dirname "${PIPELINE_LOG}")"
  exec >> "${PIPELINE_LOG}" 2>&1
fi

cd "${ROOT}" || exit 1

# Isaac Sim's bundled python.sh expects to run outside an active conda env.
# Clear only interpreter-manager variables; keep project paths and CUDA/ROS libs intact.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
unset _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
unset VIRTUAL_ENV PYTHONHOME

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/source/isaaclab:${ROOT}/source/isaaclab_rl:${ROOT}/source/isaaclab_tasks:${ROOT}/source/my_exts:${ISAAC_SIM_ROOT}/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"

STAGE_LABELS=(off light easy medium hard)
STAGE_ITERS=(10000 15000 20000 25000 30000)

if [[ -n "${STAGE_ITERS_OVERRIDE:-}" ]]; then
  read -r -a STAGE_ITERS <<< "${STAGE_ITERS_OVERRIDE}"
fi
if [[ "${#STAGE_ITERS[@]}" -ne "${#STAGE_LABELS[@]}" ]]; then
  echo "[ERROR] STAGE_ITERS must contain ${#STAGE_LABELS[@]} values, got ${#STAGE_ITERS[@]}" >&2
  exit 2
fi

COMMON_OVERRIDES=(
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

RUN_DIRS=()

latest_run_for_name() {
  local run_name="$1"
  find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

stage_attack_overrides() {
  local level="$1"
  if [[ "${level}" == "off" ]]; then
    printf '%s\n' \
      "env.algorithm.enable_attack=false" \
      "env.algorithm.attack_level=off" \
      "env.algorithm.attack_mode=profile"
  else
    printf '%s\n' \
      "env.algorithm.enable_attack=true" \
      "env.algorithm.attack_level=${level}" \
      "env.algorithm.attack_mode=profile" \
      "env.algorithm.attack_curriculum_warmup_updates=0"
  fi
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
  fi
}

copy_run_artifacts() {
  local label="$1"
  local run_dir="$2"
  local dst="${PACKAGE_ROOT}/training_runs/${label}"
  mkdir -p "${dst}/checkpoints"
  copy_if_exists "${run_dir}/platoon_metrics.csv" "${dst}/platoon_metrics.csv"
  copy_if_exists "${run_dir}/params" "${dst}/params"
  copy_if_exists "${run_dir}/figures" "${dst}/figures"
  copy_if_exists "${run_dir}/model_best.pt" "${dst}/checkpoints/model_best.pt"
  copy_if_exists "${run_dir}/model_final.pt" "${dst}/checkpoints/model_final.pt"
  copy_if_exists "${run_dir}/git" "${dst}/git"
}

run_stage() {
  local label="$1"
  local iterations="$2"
  local load_run="$3"
  local checkpoint="$4"
  local run_name="${RUN_TAG}_s_${label}"
  mapfile -t attack_overrides < <(stage_attack_overrides "${label}")

  echo
  echo "[PIPELINE] stage=${label} iterations=${iterations} load_run=${load_run} checkpoint=${checkpoint}"
  echo "[PIPELINE] using ISAAC_PYTHON=${ISAAC_PYTHON}"
  set +e
  "${ISAAC_PYTHON}" scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs "${NUM_ENVS}" \
    --headless \
    --resume \
    --load_run "${load_run}" \
    --checkpoint "${checkpoint}" \
    --run_name "${run_name}" \
    --max_iterations "${iterations}" \
    --kit_args="${KIT_ARGS}" \
    "${COMMON_OVERRIDES[@]}" \
    "${attack_overrides[@]}"
  local train_status=$?
  set -e
  echo "[PIPELINE] train exit status for ${label}: ${train_status}"
  if [[ "${train_status}" -ne 0 ]]; then
    exit "${train_status}"
  fi

  local run_dir
  run_dir="$(latest_run_for_name "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] could not locate completed run for ${run_name}" >&2
    exit 2
  fi
  RUN_DIRS+=("${run_dir}")
  echo "[PIPELINE] completed ${label}: ${run_dir}"

  if [[ -f "${run_dir}/platoon_metrics.csv" ]]; then
    python3 scripts/tools/plot_platoon_metrics.py "${run_dir}/platoon_metrics.csv" --smooth 100 || true
    python3 scripts/reinforcement_learning/rsl_rl/plot_platoon_metrics.py "${run_dir}" --rolling 100 || true
  fi
}

run_eval() {
  local level="$1"
  local checkpoint="$2"
  local output_dir="${PACKAGE_ROOT}/evaluation/${level}"
  mapfile -t attack_overrides < <(stage_attack_overrides "${level}")

  echo
  echo "[PIPELINE] eval level=${level} checkpoint=${checkpoint}"
  "${ISAAC_PYTHON}" scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs "${NUM_ENVS}" \
    --headless \
    --eval_checkpoints "${checkpoint}" \
    --eval_steps "${EVAL_STEPS}" \
    --warmup_steps "${EVAL_WARMUP_STEPS}" \
    --output_dir "${output_dir}" \
    --enable_attack_eval \
    --kit_args="${KIT_ARGS}" \
    "${COMMON_OVERRIDES[@]}" \
    "${attack_overrides[@]}"
}

make_package() {
  echo
  echo "[PIPELINE] packaging outputs under ${PACKAGE_ROOT}"
  rm -rf "${PACKAGE_ROOT}"
  mkdir -p "${PACKAGE_ROOT}/training_runs" "${PACKAGE_ROOT}/paper_figures"

  {
    echo "run_tag=${RUN_TAG}"
    echo "started_from=${START_RUN}/${START_CHECKPOINT}"
    echo "num_envs=${NUM_ENVS}"
    echo "stage_labels=${STAGE_LABELS[*]}"
    echo "stage_iters=${STAGE_ITERS[*]}"
    echo "eval_steps=${EVAL_STEPS}"
    echo "pipeline_log=${PIPELINE_LOG}"
    echo "package_root=${PACKAGE_ROOT}"
    echo "package_tar=${PACKAGE_TAR}"
    echo
    echo "attack presets from router.py at launch:"
    sed -n '304,310p' source/my_exts/marl_platoon/algorithms/router.py
    echo
    echo "completed runs:"
  } > "${PACKAGE_ROOT}/manifest.txt"

  for idx in "${!RUN_DIRS[@]}"; do
    local label="${STAGE_LABELS[$idx]}"
    local run_dir="${RUN_DIRS[$idx]}"
    echo "${label}=${run_dir}" >> "${PACKAGE_ROOT}/manifest.txt"
    copy_run_artifacts "${label}" "${run_dir}"
  done

  copy_if_exists "${PIPELINE_LOG}" "${PACKAGE_ROOT}/pipeline.log"
  copy_if_exists "${ROOT}/debug_notes.md" "${PACKAGE_ROOT}/debug_notes.md"
  git diff > "${PACKAGE_ROOT}/git_diff_at_packaging.diff" || true

  local final_run="${RUN_DIRS[$(( ${#RUN_DIRS[@]} - 1 ))]}"
  local final_checkpoint="${final_run}/model_final.pt"
  if [[ ! -f "${final_checkpoint}" ]]; then
    final_checkpoint="${final_run}/model_best.pt"
  fi

  for level in "${STAGE_LABELS[@]}"; do
    run_eval "${level}" "${final_checkpoint}"
  done

  local figure_args=()
  for idx in "${!RUN_DIRS[@]}"; do
    figure_args+=(--run "${STAGE_LABELS[$idx]}=${RUN_DIRS[$idx]}")
  done
  for level in "${STAGE_LABELS[@]}"; do
    figure_args+=(--eval "${level}=${PACKAGE_ROOT}/evaluation/${level}")
  done
  python3 scripts/tools/plot_platoon_paper_figures.py \
    --output-dir "${PACKAGE_ROOT}/paper_figures" \
    --smooth "${PLOT_SMOOTH}" \
    "${figure_args[@]}"

  tar -czf "${PACKAGE_TAR}" -C "$(dirname "${PACKAGE_ROOT}")" "$(basename "${PACKAGE_ROOT}")"
  echo "[PIPELINE] package root: ${PACKAGE_ROOT}"
  echo "[PIPELINE] package tar:  ${PACKAGE_TAR}"
}

echo "[PIPELINE] run_tag=${RUN_TAG}"
echo "[PIPELINE] log=${PIPELINE_LOG}"
echo "[PIPELINE] package=${PACKAGE_TAR}"
echo "[PIPELINE] start=${START_RUN}/${START_CHECKPOINT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[PIPELINE] dry run only; no training will be launched."
  for idx in "${!STAGE_LABELS[@]}"; do
    echo "[PIPELINE] dry stage ${STAGE_LABELS[$idx]} iterations=${STAGE_ITERS[$idx]}"
  done
  exit 0
fi

load_run="${START_RUN}"
checkpoint="${START_CHECKPOINT}"
stage_count="${#STAGE_LABELS[@]}"
if (( STAGE_LIMIT < stage_count )); then
  stage_count="${STAGE_LIMIT}"
fi

for ((idx = 0; idx < stage_count; idx++)); do
  label="${STAGE_LABELS[$idx]}"
  iterations="${STAGE_ITERS[$idx]}"
  run_stage "${label}" "${iterations}" "${load_run}" "${checkpoint}"
  load_run="$(basename "${RUN_DIRS[$idx]}")"
  checkpoint="model_final.pt"
done

if [[ "${SKIP_PACKAGE}" == "1" ]]; then
  echo "[PIPELINE] SKIP_PACKAGE=1; stopping after training stages."
  exit 0
fi

make_package
