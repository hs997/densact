#!/usr/bin/env bash

# Shared storage policy for all AMS/IsaacLab experiments.
# Keep generated logs, checkpoints, videos, trajectories, temporary USD files,
# and runtime caches on the 1T data disk instead of the nearly-full system disk.

export ISAACLAB_EXPERIMENT_ROOT="${ISAACLAB_EXPERIMENT_ROOT:-${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}}"
export ISAACLAB_RUNTIME_ROOT="${ISAACLAB_RUNTIME_ROOT:-/home/cnc/SSD_1T/xzw/ams_runtime}"
export ISAACLAB_TMP_ROOT="${ISAACLAB_TMP_ROOT:-${ISAACLAB_RUNTIME_ROOT}/IsaacLab}"
export ISAACLAB_TMPDIR="${ISAACLAB_TMPDIR:-${ISAACLAB_RUNTIME_ROOT}/tmp}"
export ISAACLAB_CACHE_ROOT="${ISAACLAB_CACHE_ROOT:-${ISAACLAB_RUNTIME_ROOT}/cache}"

# These variables are consumed by Python tempfile, Isaac Sim asset conversion,
# pip and common NVIDIA/Omniverse cache writers.  AMS experiments deliberately
# override inherited /tmp and ~/.cache defaults unless the caller supplies an
# explicit value through the ISAACLAB_* variables above.
export TMPDIR="${ISAACLAB_TMPDIR}"
export TEMP="${ISAACLAB_TMPDIR}"
export TMP="${ISAACLAB_TMPDIR}"
export XDG_CACHE_HOME="${ISAACLAB_CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${ISAACLAB_CACHE_ROOT}/pip"
export CUDA_CACHE_PATH="${ISAACLAB_CACHE_ROOT}/cuda"
export __GL_SHADER_DISK_CACHE_PATH="${ISAACLAB_CACHE_ROOT}/gl"

mkdir -p \
  "${ISAACLAB_EXPERIMENT_ROOT}/logs" \
  "${ISAACLAB_RUNTIME_ROOT}" \
  "${ISAACLAB_TMP_ROOT}" \
  "${ISAACLAB_TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${__GL_SHADER_DISK_CACHE_PATH}"
