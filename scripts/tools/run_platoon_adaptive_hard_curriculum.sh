#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/cnc/SSD_1T/xzw/isaac-sim}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
LOG_ROOT="${ROOT}/logs/rsl_rl/platoon_happo"

RUN_TAG="${RUN_TAG:-paper_adaptive_hard_curriculum_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_LOG="${PIPELINE_LOG:-${ROOT}/train_${RUN_TAG}.log}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${LOG_ROOT}/${RUN_TAG}_package}"
PACKAGE_TAR="${PACKAGE_TAR:-${PACKAGE_ROOT}.tar.gz}"

START_RUN="${START_RUN:-2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000}"
START_CHECKPOINT="${START_CHECKPOINT:-model_best.pt}"
NUM_ENVS="${NUM_ENVS:-64}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
EVAL_WARMUP_STEPS="${EVAL_WARMUP_STEPS:-100}"
PLOT_SMOOTH="${PLOT_SMOOTH:-100}"
ASSESS_WINDOW="${ASSESS_WINDOW:-200}"
DRY_RUN="${DRY_RUN:-0}"
STAGE_LIMIT="${STAGE_LIMIT:-5}"
START_STAGE_INDEX="${START_STAGE_INDEX:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
ALLOW_EARLY_PASS="${ALLOW_EARLY_PASS:-1}"
RESUME_STAGE_PROGRESS="${RESUME_STAGE_PROGRESS:-1}"
VALIDATION_ROLLBACK="${VALIDATION_ROLLBACK:-1}"
VALIDATION_WORSE_TOL="${VALIDATION_WORSE_TOL:-0.02}"

KIT_ARGS="${KIT_ARGS:---/rtx/verifyDriverVersion/enabled=false --/app/viewport/grid/enabled=false --/app/viewport/defaults/guide/grid/visible=false}"

if [[ "${PLATOON_PIPELINE_LOG_ACTIVE:-0}" != "1" ]]; then
  export PLATOON_PIPELINE_LOG_ACTIVE=1
  mkdir -p "$(dirname "${PIPELINE_LOG}")"
  exec >> "${PIPELINE_LOG}" 2>&1
fi

cd "${ROOT}" || exit 1
trap 'status=$?; echo "[PIPELINE] exit status=${status} at line=${LINENO}"' EXIT

# Isaac Sim's bundled python.sh expects to run outside an active conda env.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
unset _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
unset VIRTUAL_ENV PYTHONHOME

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/source/isaaclab:${ROOT}/source/isaaclab_rl:${ROOT}/source/isaaclab_tasks:${ROOT}/source/my_exts:${ISAAC_SIM_ROOT}/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"

STAGE_LABELS=(off light easy medium hard)
if [[ -n "${STAGE_LABELS_OVERRIDE:-}" ]]; then
  read -r -a STAGE_LABELS <<< "${STAGE_LABELS_OVERRIDE}"
fi
STAGE_ATTACK_LEVELS=("${STAGE_LABELS[@]}")
if [[ -n "${STAGE_ATTACK_LEVELS_OVERRIDE:-}" ]]; then
  read -r -a STAGE_ATTACK_LEVELS <<< "${STAGE_ATTACK_LEVELS_OVERRIDE}"
fi
STAGE_PASS_PROFILES=("${STAGE_LABELS[@]}")
if [[ -n "${STAGE_PASS_PROFILES_OVERRIDE:-}" ]]; then
  read -r -a STAGE_PASS_PROFILES <<< "${STAGE_PASS_PROFILES_OVERRIDE}"
fi
STAGE_MAX_FDI_POS=()
STAGE_MAX_FDI_ACC=()
STAGE_MAX_DOS_RATE=()
for _ in "${STAGE_LABELS[@]}"; do
  STAGE_MAX_FDI_POS+=("-1")
  STAGE_MAX_FDI_ACC+=("-1")
  STAGE_MAX_DOS_RATE+=("-1")
done
if [[ -n "${STAGE_MAX_FDI_POS_OVERRIDE:-}" ]]; then
  read -r -a STAGE_MAX_FDI_POS <<< "${STAGE_MAX_FDI_POS_OVERRIDE}"
fi
if [[ -n "${STAGE_MAX_FDI_ACC_OVERRIDE:-}" ]]; then
  read -r -a STAGE_MAX_FDI_ACC <<< "${STAGE_MAX_FDI_ACC_OVERRIDE}"
fi
if [[ -n "${STAGE_MAX_DOS_RATE_OVERRIDE:-}" ]]; then
  read -r -a STAGE_MAX_DOS_RATE <<< "${STAGE_MAX_DOS_RATE_OVERRIDE}"
fi

# Adaptive defaults: short no-attack sanity, then spend budget only where the metrics still need it.
STAGE_MIN_ITERS=(500 3000 5000 7000 10000)
STAGE_CHUNK_ITERS=(500 1500 2500 3500 5000)
STAGE_MAX_ITERS=(1000 15000 20000 25000 35000)

if [[ -n "${STAGE_MIN_ITERS_OVERRIDE:-}" ]]; then
  read -r -a STAGE_MIN_ITERS <<< "${STAGE_MIN_ITERS_OVERRIDE}"
fi
if [[ -n "${STAGE_CHUNK_ITERS_OVERRIDE:-}" ]]; then
  read -r -a STAGE_CHUNK_ITERS <<< "${STAGE_CHUNK_ITERS_OVERRIDE}"
fi
if [[ -n "${STAGE_MAX_ITERS_OVERRIDE:-}" ]]; then
  read -r -a STAGE_MAX_ITERS <<< "${STAGE_MAX_ITERS_OVERRIDE}"
fi
for arr_name in STAGE_MIN_ITERS STAGE_CHUNK_ITERS STAGE_MAX_ITERS STAGE_ATTACK_LEVELS STAGE_PASS_PROFILES STAGE_MAX_FDI_POS STAGE_MAX_FDI_ACC STAGE_MAX_DOS_RATE; do
  eval "arr_len=\${#${arr_name}[@]}"
  if [[ "${arr_len}" -ne "${#STAGE_LABELS[@]}" ]]; then
    echo "[ERROR] ${arr_name} must contain ${#STAGE_LABELS[@]} values, got ${arr_len}" >&2
    exit 2
  fi
done

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
if [[ -n "${COMMON_OVERRIDES_EXTRA:-}" ]]; then
  read -r -a COMMON_OVERRIDES_EXTRA_ARRAY <<< "${COMMON_OVERRIDES_EXTRA}"
  COMMON_OVERRIDES+=("${COMMON_OVERRIDES_EXTRA_ARRAY[@]}")
fi

ALL_CHUNK_DIRS=()
ALL_CHUNK_LABELS=()
FINAL_RUN_DIRS=()
FINAL_STAGE_LABELS=()
LAST_RUN_DIR=""

latest_run_for_name() {
  local run_name="$1"
  find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

stage_attack_overrides() {
  local stage_idx="$1"
  local level="${STAGE_ATTACK_LEVELS[$stage_idx]}"
  local fdi_pos="${STAGE_MAX_FDI_POS[$stage_idx]}"
  local fdi_acc="${STAGE_MAX_FDI_ACC[$stage_idx]}"
  local dos_rate="${STAGE_MAX_DOS_RATE[$stage_idx]}"
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
    if [[ "${fdi_pos}" != "-1" ]]; then
      printf '%s\n' "env.algorithm.max_fdi_pos=${fdi_pos}"
    fi
    if [[ "${fdi_acc}" != "-1" ]]; then
      printf '%s\n' "env.algorithm.max_fdi_acc=${fdi_acc}"
    fi
    if [[ "${dos_rate}" != "-1" ]]; then
      printf '%s\n' "env.algorithm.max_dos_rate=${dos_rate}"
    fi
  fi
}

stage_pass_thresholds() {
  local level="$1"
  case "${level}" in
    off)    echo "0.10 0.20 0.06 0.06 1.38 0.005 0.0001" ;;
    light)  echo "0.11 0.22 0.08 0.08 1.35 0.005 0.0001" ;;
    easy)   echo "0.14 0.28 0.12 0.12 1.25 0.010 0.0001" ;;
    medium) echo "0.18 0.34 0.16 0.15 1.18 0.015 0.0001" ;;
    hard)   echo "0.30 0.45 0.24 0.20 1.05 0.020 0.0001" ;;
    *) echo "[ERROR] unknown stage ${level}" >&2; return 2 ;;
  esac
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
  local dst="$3"
  mkdir -p "${dst}/checkpoints"
  copy_if_exists "${run_dir}/platoon_metrics.csv" "${dst}/platoon_metrics.csv"
  copy_if_exists "${run_dir}/params" "${dst}/params"
  copy_if_exists "${run_dir}/figures" "${dst}/figures"
  copy_if_exists "${run_dir}/plots" "${dst}/plots"
  copy_if_exists "${run_dir}/model_best.pt" "${dst}/checkpoints/model_best.pt"
  copy_if_exists "${run_dir}/model_final.pt" "${dst}/checkpoints/model_final.pt"
  copy_if_exists "${run_dir}/adaptive_assessment.txt" "${dst}/adaptive_assessment.txt"
  copy_if_exists "${run_dir}/git" "${dst}/git"
  echo "${label}=${run_dir}" >> "${PACKAGE_ROOT}/manifest.txt"
}

run_stage_chunk() {
  local stage_idx="$1"
  local label="$2"
  local chunk_iterations="$3"
  local load_run="$4"
  local checkpoint="$5"
  local chunk_index="$6"
  local total_before="$7"
  local run_name="${RUN_TAG}_s_${label}_c${chunk_index}"
  mapfile -t attack_overrides < <(stage_attack_overrides "${stage_idx}")
  local stage_extra_name="STAGE_EXTRA_OVERRIDES_${stage_idx}"
  local stage_extra_value="${!stage_extra_name:-}"
  local stage_extra_overrides=()
  if [[ -n "${stage_extra_value}" ]]; then
    read -r -a stage_extra_overrides <<< "${stage_extra_value}"
  fi

  echo
  echo "[PIPELINE] stage=${label} chunk=${chunk_index} chunk_iters=${chunk_iterations} total_before=${total_before} load_run=${load_run} checkpoint=${checkpoint}"
  if [[ "${#stage_extra_overrides[@]}" -gt 0 ]]; then
    echo "[PIPELINE] stage_extra_overrides=${stage_extra_overrides[*]}"
  fi
  echo "[PIPELINE] launching train.py for ${label}/chunk${chunk_index}"
  set +e
  "${ISAAC_PYTHON}" scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs "${NUM_ENVS}" \
    --headless \
    --resume \
    --load_run "${load_run}" \
    --checkpoint "${checkpoint}" \
    --run_name "${run_name}" \
    --max_iterations "${chunk_iterations}" \
    --kit_args="${KIT_ARGS}" \
    "${COMMON_OVERRIDES[@]}" \
    "${attack_overrides[@]}" \
    "${stage_extra_overrides[@]}"
  local train_status=$?
  set -e
  echo "[PIPELINE] train exit status for ${label}/chunk${chunk_index}: ${train_status}"
  if [[ "${train_status}" -ne 0 ]]; then
    exit "${train_status}"
  fi

  local run_dir
  run_dir="$(latest_run_for_name "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] could not locate completed run for ${run_name}" >&2
    exit 2
  fi
  LAST_RUN_DIR="${run_dir}"
  ALL_CHUNK_DIRS+=("${run_dir}")
  ALL_CHUNK_LABELS+=("${label}_c${chunk_index}")
  echo "[PIPELINE] completed ${label}/chunk${chunk_index}: ${run_dir}"

  if [[ -f "${run_dir}/platoon_metrics.csv" ]]; then
    python3 scripts/tools/plot_platoon_metrics.py "${run_dir}/platoon_metrics.csv" --smooth "${PLOT_SMOOTH}" || true
    python3 scripts/reinforcement_learning/rsl_rl/plot_platoon_metrics.py "${run_dir}" --rolling "${PLOT_SMOOTH}" || true
  fi
}

assess_stage() {
  local label="$1"
  local run_dir="$2"
  local total_iters="$3"
  local min_iters="$4"
  local pass_profile="$5"
  read -r pass_speed pass_gap pass_lateral pass_centerline pass_min_gap pass_reset pass_collision < <(stage_pass_thresholds "${pass_profile}")

  set +e
  python3 - "${run_dir}" "${label}" "${ASSESS_WINDOW}" "${total_iters}" "${min_iters}" \
    "${pass_speed}" "${pass_gap}" "${pass_lateral}" "${pass_centerline}" "${pass_min_gap}" "${pass_reset}" "${pass_collision}" \
    "${ALLOW_EARLY_PASS}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

run_dir = Path(sys.argv[1])
label = sys.argv[2]
window = int(sys.argv[3])
total_iters = int(sys.argv[4])
min_iters = int(sys.argv[5])
thresholds = {
    "speed_error_abs_mean": float(sys.argv[6]),
    "gap_error_abs_mean": float(sys.argv[7]),
    "lateral_error_abs_mean": float(sys.argv[8]),
    "centerline_error_abs_mean": float(sys.argv[9]),
    "min_pair_gap_mean": float(sys.argv[10]),
    "reset_on_bad_ori": float(sys.argv[11]),
    "collision_rate": float(sys.argv[12]),
}
allow_early_pass = sys.argv[13] == "1"
metrics_path = run_dir / "platoon_metrics.csv"
if not metrics_path.exists():
    print(f"[ASSESS][{label}] missing metrics: {metrics_path}")
    sys.exit(20)
df = pd.read_csv(metrics_path)
if df.empty:
    print(f"[ASSESS][{label}] empty metrics")
    sys.exit(20)
last = df.tail(window)

def mean_col(*names, default=0.0):
    for name in names:
        if name in last.columns:
            return float(last[name].mean())
    return default

m = {
    "speed_error_abs_mean": mean_col("speed_error_abs_mean"),
    "gap_error_abs_mean": mean_col("gap_error_abs_mean"),
    "lateral_error_abs_mean": mean_col("lateral_error_abs_mean"),
    "centerline_error_abs_mean": mean_col("centerline_error_abs_mean"),
    "min_pair_gap_mean": mean_col("min_pair_gap_mean", default=999.0),
    "reset_on_bad_ori": mean_col("reset_on_bad_ori", "termination_reset_on_bad_ori"),
    "collision_rate": mean_col("collision_rate"),
    "critic_grad_norm": mean_col("critic_grad_norm"),
    "shield_lateral_rate": mean_col("shield_lateral_rate"),
}
lines = [
    f"stage={label}",
    f"run_dir={run_dir}",
    f"rows={len(df)}",
    f"window={min(window, len(df))}",
    f"total_stage_iters={total_iters}",
    f"min_stage_iters={min_iters}",
    f"allow_early_pass={int(allow_early_pass)}",
]
for key in sorted(m):
    lines.append(f"{key}={m[key]:.6f}")
for key in sorted(thresholds):
    lines.append(f"threshold_{key}={thresholds[key]:.6f}")

enough = total_iters >= min_iters or allow_early_pass
pass_ok = (
    m["speed_error_abs_mean"] <= thresholds["speed_error_abs_mean"]
    and m["gap_error_abs_mean"] <= thresholds["gap_error_abs_mean"]
    and m["lateral_error_abs_mean"] <= thresholds["lateral_error_abs_mean"]
    and m["centerline_error_abs_mean"] <= thresholds["centerline_error_abs_mean"]
    and m["min_pair_gap_mean"] >= thresholds["min_pair_gap_mean"]
    and m["reset_on_bad_ori"] <= thresholds["reset_on_bad_ori"]
    and m["collision_rate"] <= thresholds["collision_rate"]
)
critical_bad = (
    m["reset_on_bad_ori"] > max(0.05, thresholds["reset_on_bad_ori"] * 5.0)
    or m["collision_rate"] > max(0.01, thresholds["collision_rate"] * 10.0)
    or m["centerline_error_abs_mean"] > max(0.55, thresholds["centerline_error_abs_mean"] * 3.0)
    or m["lateral_error_abs_mean"] > max(0.45, thresholds["lateral_error_abs_mean"] * 3.0)
    or m["min_pair_gap_mean"] < 0.90
)
if pass_ok and enough:
    status = "PASS"
    code = 0
elif critical_bad:
    status = "ABORT"
    code = 20
else:
    status = "CONTINUE"
    code = 10
lines.append(f"status={status}")
(run_dir / "adaptive_assessment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[ASSESS][{label}] status={status} total={total_iters} speed={m['speed_error_abs_mean']:.4f} gap={m['gap_error_abs_mean']:.4f} lat={m['lateral_error_abs_mean']:.4f} center={m['centerline_error_abs_mean']:.4f} min_gap={m['min_pair_gap_mean']:.4f} reset={m['reset_on_bad_ori']:.4f} collision={m['collision_rate']:.4f}")
sys.exit(code)
PY
  local assess_status=$?
  return "${assess_status}"
}

assessment_field() {
  local assessment_file="$1"
  local key="$2"
  awk -F= -v key="${key}" '$1 == key { print $2; exit }' "${assessment_file}"
}

assessment_score() {
  local assessment_file="$1"
  python3 - "${assessment_file}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
data = {}
for line in path.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        data[key] = value

def val(name, default=0.0):
    try:
        return float(data.get(name, default))
    except (TypeError, ValueError):
        return default

def ratio(metric, threshold):
    t = max(val(threshold, 1.0), 1e-9)
    return val(metric, 0.0) / t

score = 0.0
score += ratio("speed_error_abs_mean", "threshold_speed_error_abs_mean")
score += ratio("gap_error_abs_mean", "threshold_gap_error_abs_mean")
score += ratio("lateral_error_abs_mean", "threshold_lateral_error_abs_mean")
score += ratio("centerline_error_abs_mean", "threshold_centerline_error_abs_mean")
min_gap = max(val("min_pair_gap_mean", 999.0), 1e-9)
score += val("threshold_min_pair_gap_mean", 1.0) / min_gap
score += ratio("reset_on_bad_ori", "threshold_reset_on_bad_ori")
score += ratio("collision_rate", "threshold_collision_rate")
print(f"{score:.9f}")
PY
}

assessment_metrics_pass() {
  local assessment_file="$1"
  python3 - "${assessment_file}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
data = {}
for line in path.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        data[key] = value

def val(name, default=0.0):
    try:
        return float(data.get(name, default))
    except (TypeError, ValueError):
        return default

ok = (
    val("speed_error_abs_mean") <= val("threshold_speed_error_abs_mean")
    and val("gap_error_abs_mean") <= val("threshold_gap_error_abs_mean")
    and val("lateral_error_abs_mean") <= val("threshold_lateral_error_abs_mean")
    and val("centerline_error_abs_mean") <= val("threshold_centerline_error_abs_mean")
    and val("min_pair_gap_mean", 999.0) >= val("threshold_min_pair_gap_mean")
    and val("reset_on_bad_ori") <= val("threshold_reset_on_bad_ori")
    and val("collision_rate") <= val("threshold_collision_rate")
)
sys.exit(0 if ok else 1)
PY
}

score_better_than() {
  local candidate="$1"
  local best="$2"
  python3 - "${candidate}" "${best}" <<'PY'
import sys
candidate = float(sys.argv[1])
best = float(sys.argv[2])
sys.exit(0 if candidate < best else 1)
PY
}

score_worse_than() {
  local candidate="$1"
  local best="$2"
  local tolerance="$3"
  python3 - "${candidate}" "${best}" "${tolerance}" <<'PY'
import sys
candidate = float(sys.argv[1])
best = float(sys.argv[2])
tolerance = float(sys.argv[3])
sys.exit(0 if candidate > best * (1.0 + tolerance) else 1)
PY
}

run_eval() {
  local stage_idx="$1"
  local level="$2"
  local checkpoint="$3"
  local output_dir="${PACKAGE_ROOT}/evaluation/${level}"
  mapfile -t attack_overrides < <(stage_attack_overrides "${stage_idx}")

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
  mkdir -p "${PACKAGE_ROOT}/training_runs" "${PACKAGE_ROOT}/training_chunks" "${PACKAGE_ROOT}/paper_figures"

  {
    echo "run_tag=${RUN_TAG}"
    echo "started_from=${START_RUN}/${START_CHECKPOINT}"
    echo "num_envs=${NUM_ENVS}"
    echo "stage_labels=${STAGE_LABELS[*]}"
    echo "stage_attack_levels=${STAGE_ATTACK_LEVELS[*]}"
    echo "stage_pass_profiles=${STAGE_PASS_PROFILES[*]}"
    echo "stage_max_fdi_pos=${STAGE_MAX_FDI_POS[*]}"
    echo "stage_max_fdi_acc=${STAGE_MAX_FDI_ACC[*]}"
    echo "stage_max_dos_rate=${STAGE_MAX_DOS_RATE[*]}"
    echo "stage_min_iters=${STAGE_MIN_ITERS[*]}"
    echo "stage_chunk_iters=${STAGE_CHUNK_ITERS[*]}"
    echo "stage_max_iters=${STAGE_MAX_ITERS[*]}"
    echo "assess_window=${ASSESS_WINDOW}"
    echo "allow_early_pass=${ALLOW_EARLY_PASS}"
    echo "resume_stage_progress=${RESUME_STAGE_PROGRESS}"
    echo "validation_rollback=${VALIDATION_ROLLBACK}"
    echo "validation_worse_tol=${VALIDATION_WORSE_TOL}"
    echo "eval_steps=${EVAL_STEPS}"
    echo "pipeline_log=${PIPELINE_LOG}"
    echo "package_root=${PACKAGE_ROOT}"
    echo "package_tar=${PACKAGE_TAR}"
    echo
    echo "attack presets from router.py at launch:"
    sed -n '304,310p' source/my_exts/marl_platoon/algorithms/router.py
    echo
    echo "final stage runs:"
  } > "${PACKAGE_ROOT}/manifest.txt"

  for idx in "${!FINAL_RUN_DIRS[@]}"; do
    copy_run_artifacts "${FINAL_STAGE_LABELS[$idx]}" "${FINAL_RUN_DIRS[$idx]}" "${PACKAGE_ROOT}/training_runs/${FINAL_STAGE_LABELS[$idx]}"
  done

  {
    echo
    echo "all chunks:"
  } >> "${PACKAGE_ROOT}/manifest.txt"
  for idx in "${!ALL_CHUNK_DIRS[@]}"; do
    copy_run_artifacts "${ALL_CHUNK_LABELS[$idx]}" "${ALL_CHUNK_DIRS[$idx]}" "${PACKAGE_ROOT}/training_chunks/${ALL_CHUNK_LABELS[$idx]}"
  done

  copy_if_exists "${PIPELINE_LOG}" "${PACKAGE_ROOT}/pipeline.log"
  copy_if_exists "${ROOT}/debug_notes.md" "${PACKAGE_ROOT}/debug_notes.md"
  git diff > "${PACKAGE_ROOT}/git_diff_at_packaging.diff" || true

  local final_run="${FINAL_RUN_DIRS[$(( ${#FINAL_RUN_DIRS[@]} - 1 ))]}"
  local final_checkpoint="${final_run}/model_final.pt"
  if [[ ! -f "${final_checkpoint}" ]]; then
    final_checkpoint="${final_run}/model_best.pt"
  fi

  for idx in "${!STAGE_LABELS[@]}"; do
    level="${STAGE_LABELS[$idx]}"
    run_eval "${idx}" "${level}" "${final_checkpoint}"
  done

  local figure_args=()
  for idx in "${!FINAL_RUN_DIRS[@]}"; do
    figure_args+=(--run "${FINAL_STAGE_LABELS[$idx]}=${FINAL_RUN_DIRS[$idx]}")
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

echo "[PIPELINE] adaptive run_tag=${RUN_TAG}"
echo "[PIPELINE] log=${PIPELINE_LOG}"
echo "[PIPELINE] package=${PACKAGE_TAR}"
echo "[PIPELINE] start=${START_RUN}/${START_CHECKPOINT}"
echo "[PIPELINE] min_iters=${STAGE_MIN_ITERS[*]}"
echo "[PIPELINE] chunk_iters=${STAGE_CHUNK_ITERS[*]}"
echo "[PIPELINE] max_iters=${STAGE_MAX_ITERS[*]}"
echo "[PIPELINE] start_stage_index=${START_STAGE_INDEX}"

stage_count="${#STAGE_LABELS[@]}"
if (( STAGE_LIMIT < stage_count )); then
  stage_count="${STAGE_LIMIT}"
fi
if (( START_STAGE_INDEX < 0 || START_STAGE_INDEX >= stage_count )); then
  echo "[ERROR] START_STAGE_INDEX=${START_STAGE_INDEX} is outside active stage range 0..$((stage_count - 1))" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[PIPELINE] dry run only; no training will be launched."
  for ((idx = START_STAGE_INDEX; idx < stage_count; idx++)); do
    echo "[PIPELINE] dry stage ${STAGE_LABELS[$idx]} min=${STAGE_MIN_ITERS[$idx]} chunk=${STAGE_CHUNK_ITERS[$idx]} max=${STAGE_MAX_ITERS[$idx]}"
  done
  exit 0
fi

load_run="${START_RUN}"
checkpoint="${START_CHECKPOINT}"

for ((idx = START_STAGE_INDEX; idx < stage_count; idx++)); do
  label="${STAGE_LABELS[$idx]}"
  pass_profile="${STAGE_PASS_PROFILES[$idx]}"
  min_iters="${STAGE_MIN_ITERS[$idx]}"
  chunk_iters="${STAGE_CHUNK_ITERS[$idx]}"
  max_iters="${STAGE_MAX_ITERS[$idx]}"
  total_iters=0
  chunk_index=0
  stage_passed=0
  stage_best_run_dir=""
  stage_best_load_run=""
  stage_best_checkpoint=""
  stage_best_score=""

  echo
  echo "[PIPELINE] ===== adaptive stage ${label}: attack_level=${STAGE_ATTACK_LEVELS[$idx]} fdi_pos=${STAGE_MAX_FDI_POS[$idx]} fdi_acc=${STAGE_MAX_FDI_ACC[$idx]} dos=${STAGE_MAX_DOS_RATE[$idx]} pass_profile=${pass_profile} min=${min_iters} chunk=${chunk_iters} max=${max_iters} ====="
  if [[ "${load_run}" = /* ]]; then
    incoming_run_dir="${load_run}"
  else
    incoming_run_dir="${LOG_ROOT}/${load_run}"
  fi
  incoming_assessment="${incoming_run_dir}/adaptive_assessment.txt"
  if [[ -f "${incoming_assessment}" ]]; then
    incoming_stage="$(assessment_field "${incoming_assessment}" stage)"
    if [[ "${incoming_stage}" == "${label}" ]]; then
      stage_best_run_dir="${incoming_run_dir}"
      stage_best_load_run="$(basename "${incoming_run_dir}")"
      stage_best_checkpoint="${checkpoint}"
      stage_best_score="$(assessment_score "${incoming_assessment}")"
      start_total_iters="$(assessment_field "${incoming_assessment}" total_stage_iters)"
      if [[ "${RESUME_STAGE_PROGRESS}" == "1" && "${start_total_iters}" =~ ^[0-9]+$ ]]; then
        total_iters="${start_total_iters}"
      fi
      echo "[PIPELINE] seeded ${label} validation best from ${stage_best_load_run}: score=${stage_best_score} total=${total_iters}"
      if [[ "${ALLOW_EARLY_PASS}" == "1" ]] && assessment_metrics_pass "${incoming_assessment}"; then
        echo "[PIPELINE] stage ${label} already satisfies metrics; accepting ${stage_best_load_run}/${stage_best_checkpoint} without extra training."
        FINAL_RUN_DIRS+=("${stage_best_run_dir}")
        FINAL_STAGE_LABELS+=("${label}")
        load_run="${stage_best_load_run}"
        checkpoint="${stage_best_checkpoint}"
        stage_passed=1
        continue
      fi
    fi
  fi

  while (( total_iters < max_iters )); do
    remaining=$(( max_iters - total_iters ))
    current_chunk="${chunk_iters}"
    if (( current_chunk > remaining )); then
      current_chunk="${remaining}"
    fi
    chunk_index=$(( chunk_index + 1 ))
    run_stage_chunk "${idx}" "${label}" "${current_chunk}" "${load_run}" "${checkpoint}" "${chunk_index}" "${total_iters}"
    total_iters=$(( total_iters + current_chunk ))

    set +e
    assess_stage "${label}" "${LAST_RUN_DIR}" "${total_iters}" "${min_iters}" "${pass_profile}"
    assess_code=$?
    set -e
    current_assessment="${LAST_RUN_DIR}/adaptive_assessment.txt"
    current_score="$(assessment_score "${current_assessment}")"
    previous_best_score="${stage_best_score}"
    previous_best_run_dir="${stage_best_run_dir}"

    if [[ -z "${stage_best_score}" ]] || score_better_than "${current_score}" "${stage_best_score}"; then
      stage_best_run_dir="${LAST_RUN_DIR}"
      stage_best_load_run="$(basename "${LAST_RUN_DIR}")"
      stage_best_checkpoint="model_final.pt"
      stage_best_score="${current_score}"
      echo "[PIPELINE] new best for ${label}: ${stage_best_load_run}/${stage_best_checkpoint} score=${stage_best_score}"
    else
      echo "[PIPELINE] ${label}/chunk${chunk_index} score=${current_score}; best remains ${stage_best_score} at ${stage_best_load_run}/${stage_best_checkpoint}"
    fi

    if [[ "${assess_code}" -eq 0 ]]; then
      if ! assessment_metrics_pass "${stage_best_run_dir}/adaptive_assessment.txt"; then
        stage_best_run_dir="${LAST_RUN_DIR}"
        stage_best_load_run="$(basename "${LAST_RUN_DIR}")"
        stage_best_checkpoint="model_final.pt"
        stage_best_score="${current_score}"
        echo "[PIPELINE] current chunk is the first passing checkpoint for ${label}; forwarding it despite score ordering."
      fi
      echo "[PIPELINE] stage ${label} passed after ${total_iters} iterations; forwarding best checkpoint ${stage_best_load_run}/${stage_best_checkpoint}"
      FINAL_RUN_DIRS+=("${stage_best_run_dir}")
      FINAL_STAGE_LABELS+=("${label}")
      load_run="${stage_best_load_run}"
      checkpoint="${stage_best_checkpoint}"
      stage_passed=1
      break
    elif [[ "${assess_code}" -eq 20 ]]; then
      echo "[ERROR] stage ${label} hit critical bad metrics; stopping before passing a bad checkpoint forward." >&2
      exit 20
    else
      if [[ "${VALIDATION_ROLLBACK}" == "1" && -n "${previous_best_score}" ]] && score_worse_than "${current_score}" "${previous_best_score}" "${VALIDATION_WORSE_TOL}"; then
        echo "[PIPELINE] stage ${label} validation score worsened from ${previous_best_score} to ${current_score}; rolling back to preserved best checkpoint instead of stopping."
        echo "[PIPELINE] preserved best checkpoint: ${previous_best_run_dir}"
      fi
      echo "[PIPELINE] stage ${label} not good enough yet after ${total_iters} iterations; continuing from best checkpoint ${stage_best_load_run}/${stage_best_checkpoint}."
      load_run="${stage_best_load_run}"
      checkpoint="${stage_best_checkpoint}"
    fi
  done

  if [[ "${stage_passed}" -ne 1 ]]; then
    echo "[ERROR] stage ${label} reached max_iters=${max_iters} without meeting pass metrics; stopping." >&2
    exit 21
  fi
done

if [[ "${SKIP_PACKAGE}" == "1" ]]; then
  echo "[PIPELINE] SKIP_PACKAGE=1; stopping after adaptive training stages."
  exit 0
fi

make_package
