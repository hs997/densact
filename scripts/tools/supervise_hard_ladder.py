#!/usr/bin/env python3
"""Supervise full off-to-hard platoon ladder training.

The supervisor is intentionally conservative:
- it checks every five minutes;
- it never advances from a failed checkpoint by itself;
- if a pipeline stops before a full hard pass/package, it launches a more
  conservative full off-to-hard attempt from the stable baseline.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/cnc/SSD_1T/xzw/IsaacLab-main")
PIPELINE = ROOT / "scripts/tools/run_platoon_adaptive_hard_curriculum.sh"
MONITOR = ROOT / "scripts/tools/monitor_current_training.py"
STATE = ROOT / ".hard_ladder_supervisor_state.json"
SUP_LOG = ROOT / "hard_ladder_supervisor.log"
PID_POINTER = ROOT / ".current_adaptive_training.pid"
LOG_POINTER = ROOT / ".current_adaptive_training.log"
TAG_POINTER = ROOT / ".current_adaptive_training.tag"
MONITOR_PID_POINTER = ROOT / ".current_training_monitor.pid"
BASE_RUN = "2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000"
BASE_CHECKPOINT = "model_best.pt"
CHECK_INTERVAL_SEC = int(os.environ.get("HARD_LADDER_CHECK_INTERVAL_SEC", "300"))
GPU_FREE_WAIT_SEC = int(os.environ.get("HARD_LADDER_GPU_FREE_WAIT_SEC", "90"))


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    append(SUP_LOG, line)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def tail(path: Path, max_bytes: int = 500_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            parts = proc_stat.read_text(encoding="utf-8", errors="replace").split()
            if len(parts) > 2 and parts[2] == "Z":
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
                except OSError:
                    pass
                return False
        except Exception:
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pid(path: Path) -> int | None:
    text = read_text(path)
    return int(text) if text.isdigit() else None


def kill_process_group(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return


def wait_after_failed_attempt(pid: int | None) -> None:
    """Give Isaac Sim child processes time to release GPU memory before relaunching."""
    kill_process_group(pid)
    if GPU_FREE_WAIT_SEC > 0:
        log(f"cleanup_wait seconds={GPU_FREE_WAIT_SEC} old_pid={pid}")
        time.sleep(GPU_FREE_WAIT_SEC)


def load_state() -> dict:
    if not STATE.exists():
        return {"attempt": -1, "completed": False, "history": []}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"attempt": -1, "completed": False, "history": []}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def profile(attempt: int) -> dict[str, str]:
    """Return progressively more conservative full off-to-hard profiles."""
    profiles = [
        {
            "name": "full_ladder_v1",
            "labels": "off light easy_0 easy_a easy_b med_a med_b hard_a hard_b hard_c hard",
            "levels": "off light easy easy easy medium medium hard hard hard hard",
            "passes": "off light easy easy easy medium medium hard hard hard hard",
            "pos": "-1 -1 1.00 1.25 1.50 1.75 2.00 2.75 4.00 6.00 8.00",
            "acc": "-1 -1 0.18 0.25 0.35 0.45 0.50 0.75 1.30 2.20 3.00",
            "dos": "-1 -1 0.035 0.05 0.07 0.09 0.10 0.12 0.18 0.24 0.30",
            "min": "100 100 150 150 150 150 150 150 150 150 150",
            "chunk": "100 100 150 150 150 150 150 150 150 150 150",
            "max": "200 200 450 450 450 450 450 600 600 600 750",
            "window": "100",
            "worse_tol": "0.010",
            "extra": "",
        },
        {
            "name": "full_ladder_v2_finer_easy",
            "labels": "off light easy_00 easy_0 easy_1 easy_2 med_0 med_1 med_2 hard_0 hard_1 hard_2 hard_3 hard",
            "levels": "off light easy easy easy easy medium medium medium hard hard hard hard hard",
            "passes": "off light easy easy easy easy medium medium medium hard hard hard hard hard",
            "pos": "-1 -1 0.85 1.00 1.15 1.25 1.45 1.70 2.00 2.50 3.25 4.50 6.25 8.00",
            "acc": "-1 -1 0.14 0.18 0.22 0.25 0.34 0.44 0.50 0.70 1.05 1.60 2.30 3.00",
            "dos": "-1 -1 0.025 0.035 0.045 0.05 0.065 0.085 0.10 0.12 0.155 0.20 0.25 0.30",
            "min": "80 80 120 120 120 120 120 120 120 120 120 120 120 120",
            "chunk": "80 80 120 120 120 120 120 120 120 120 120 120 120 120",
            "max": "160 160 360 360 360 360 360 360 360 480 480 480 600 720",
            "window": "80",
            "worse_tol": "0.008",
            "extra": "env.algorithm.happo_action_clip=0.35",
        },
        {
            "name": "full_ladder_v3_fine_hard_no_clip",
            "labels": "off light easy_0 easy_a easy_b med_a med_b hard_a hard_b hard_c hard_d hard_e hard",
            "levels": "off light easy easy easy medium medium hard hard hard hard hard hard",
            "passes": "off light easy easy easy medium medium hard hard hard hard hard hard",
            "pos": "-1 -1 1.00 1.25 1.50 1.75 2.00 2.75 4.00 6.00 6.75 7.35 8.00",
            "acc": "-1 -1 0.18 0.25 0.35 0.45 0.50 0.75 1.30 2.20 2.50 2.75 3.00",
            "dos": "-1 -1 0.035 0.05 0.07 0.09 0.10 0.12 0.18 0.24 0.265 0.285 0.30",
            "min": "100 100 150 150 150 150 150 150 150 150 120 120 80",
            "chunk": "100 100 150 150 150 150 150 150 150 150 120 120 80",
            "max": "200 200 450 450 450 450 450 600 600 600 360 360 240",
            "window": "100",
            "worse_tol": "0.006",
            "extra": "",
            "stage_extra": {
                "12": "env.safety_shield.d_drop=1.55 env.safety_shield.forward_bias_gain=0.08 env.safety_shield.forward_bias_clip=0.03 env.safety_shield.forward_bias_min_gap=1.12 env.safety_shield.forward_bias_min_command=0.30",
            },
        },
        {
            "name": "full_ladder_v4_hard_gap_guard",
            "labels": "off light easy_0 easy_a easy_b med_a med_b hard_a hard_b hard_c hard_d hard_e hard_f hard",
            "levels": "off light easy easy easy medium medium hard hard hard hard hard hard hard",
            "passes": "off light easy easy easy medium medium hard hard hard hard hard hard hard",
            "pos": "-1 -1 1.00 1.25 1.50 1.75 2.00 2.75 4.00 6.00 6.50 7.00 7.50 8.00",
            "acc": "-1 -1 0.18 0.25 0.35 0.45 0.50 0.75 1.30 2.20 2.40 2.60 2.80 3.00",
            "dos": "-1 -1 0.035 0.05 0.07 0.09 0.10 0.12 0.18 0.24 0.255 0.275 0.290 0.30",
            "min": "100 100 150 150 150 150 150 150 150 150 100 100 100 80",
            "chunk": "100 100 150 150 150 150 150 150 150 150 100 100 100 80",
            "max": "200 200 450 450 450 450 450 600 600 600 400 400 400 400",
            "window": "100",
            "worse_tol": "0.015",
            "extra": "",
            "stage_extra": {
                "10": "env.safety_shield.d_drop=1.55 env.safety_shield.catchup_action=-0.335 env.safety_shield.catchup_lateral_limit=0.36 env.safety_shield.catchup_centerline_limit=0.25 env.safety_shield.forward_bias_gain=0.06 env.safety_shield.forward_bias_clip=0.020 env.safety_shield.forward_bias_min_gap=1.16 env.safety_shield.forward_bias_min_command=0.30",
                "11": "env.safety_shield.d_drop=1.60 env.safety_shield.catchup_action=-0.325 env.safety_shield.catchup_lateral_limit=0.34 env.safety_shield.catchup_centerline_limit=0.24 env.safety_shield.forward_bias_gain=0.08 env.safety_shield.forward_bias_clip=0.025 env.safety_shield.forward_bias_min_gap=1.18 env.safety_shield.forward_bias_min_command=0.30",
                "12": "env.safety_shield.d_drop=1.64 env.safety_shield.catchup_action=-0.318 env.safety_shield.catchup_lateral_limit=0.32 env.safety_shield.catchup_centerline_limit=0.23 env.safety_shield.forward_bias_gain=0.09 env.safety_shield.forward_bias_clip=0.030 env.safety_shield.forward_bias_min_gap=1.20 env.safety_shield.forward_bias_min_command=0.30",
                "13": "env.safety_shield.d_drop=1.68 env.safety_shield.catchup_action=-0.312 env.safety_shield.catchup_lateral_limit=0.30 env.safety_shield.catchup_centerline_limit=0.22 env.safety_shield.forward_bias_gain=0.10 env.safety_shield.forward_bias_clip=0.035 env.safety_shield.forward_bias_min_gap=1.22 env.safety_shield.forward_bias_min_command=0.30",
            },
        },
        {
            "name": "full_ladder_v5_hard_e_gap_guard",
            "labels": "off light easy_0 easy_a easy_b med_a med_b hard_a hard_b hard_c hard_d hard_e0 hard_e1 hard_f0 hard_f1 hard",
            "levels": "off light easy easy easy medium medium hard hard hard hard hard hard hard hard hard",
            "passes": "off light easy easy easy medium medium hard hard hard hard hard hard hard hard hard",
            "pos": "-1 -1 1.00 1.25 1.50 1.75 2.00 2.75 4.00 6.00 6.40 6.70 7.00 7.30 7.65 8.00",
            "acc": "-1 -1 0.18 0.25 0.35 0.45 0.50 0.75 1.30 2.20 2.35 2.50 2.60 2.72 2.86 3.00",
            "dos": "-1 -1 0.035 0.05 0.07 0.09 0.10 0.12 0.18 0.24 0.250 0.265 0.275 0.285 0.292 0.30",
            "min": "100 100 150 150 150 150 150 150 150 150 100 100 100 100 100 80",
            "chunk": "100 100 150 150 150 150 150 150 150 150 100 100 100 100 100 80",
            "max": "200 200 450 450 450 450 450 600 600 600 400 400 400 400 400 400",
            "window": "100",
            "worse_tol": "0.012",
            "extra": "",
            "stage_extra": {
                "10": "env.safety_shield.d_drop=1.58 env.safety_shield.catchup_action=-0.330 env.safety_shield.catchup_lateral_limit=0.34 env.safety_shield.catchup_centerline_limit=0.24 env.safety_shield.forward_bias_gain=0.07 env.safety_shield.forward_bias_clip=0.022 env.safety_shield.forward_bias_min_gap=1.18 env.safety_shield.forward_bias_min_command=0.30",
                "11": "env.safety_shield.d_drop=1.68 env.safety_shield.catchup_action=-0.310 env.safety_shield.catchup_lateral_limit=0.32 env.safety_shield.catchup_centerline_limit=0.22 env.safety_shield.forward_bias_gain=0.09 env.safety_shield.forward_bias_clip=0.027 env.safety_shield.forward_bias_min_gap=1.25 env.safety_shield.forward_bias_min_command=0.30",
                "12": "env.safety_shield.d_drop=1.78 env.safety_shield.catchup_action=-0.295 env.safety_shield.catchup_lateral_limit=0.30 env.safety_shield.catchup_centerline_limit=0.21 env.safety_shield.forward_bias_gain=0.11 env.safety_shield.forward_bias_clip=0.035 env.safety_shield.forward_bias_min_gap=1.30 env.safety_shield.forward_bias_min_command=0.30",
                "13": "env.safety_shield.d_drop=1.88 env.safety_shield.catchup_action=-0.280 env.safety_shield.catchup_lateral_limit=0.28 env.safety_shield.catchup_centerline_limit=0.20 env.safety_shield.forward_bias_gain=0.13 env.safety_shield.forward_bias_clip=0.040 env.safety_shield.forward_bias_min_gap=1.34 env.safety_shield.forward_bias_min_command=0.30",
                "14": "env.safety_shield.d_drop=1.96 env.safety_shield.catchup_action=-0.270 env.safety_shield.catchup_lateral_limit=0.27 env.safety_shield.catchup_centerline_limit=0.19 env.safety_shield.forward_bias_gain=0.15 env.safety_shield.forward_bias_clip=0.045 env.safety_shield.forward_bias_min_gap=1.38 env.safety_shield.forward_bias_min_command=0.30",
                "15": "env.safety_shield.d_drop=2.05 env.safety_shield.catchup_action=-0.260 env.safety_shield.catchup_lateral_limit=0.26 env.safety_shield.catchup_centerline_limit=0.18 env.safety_shield.forward_bias_gain=0.17 env.safety_shield.forward_bias_clip=0.050 env.safety_shield.forward_bias_min_gap=1.42 env.safety_shield.forward_bias_min_command=0.30",
            },
        },
        {
            "name": "full_ladder_v6_hard_gap_brake_guard",
            "labels": "off light easy_0 easy_a easy_b med_a med_b hard_a hard_b hard_c hard_d0 hard_d1 hard_e0 hard_e1 hard_f0 hard_f1 hard_g hard",
            "levels": "off light easy easy easy medium medium hard hard hard hard hard hard hard hard hard hard hard",
            "passes": "off light easy easy easy medium medium hard hard hard hard hard hard hard hard hard hard hard",
            "pos": "-1 -1 1.00 1.25 1.50 1.75 2.00 2.75 4.00 6.00 6.20 6.40 6.55 6.70 6.90 7.15 7.50 8.00",
            "acc": "-1 -1 0.18 0.25 0.35 0.45 0.50 0.75 1.30 2.20 2.28 2.35 2.42 2.50 2.58 2.67 2.84 3.00",
            "dos": "-1 -1 0.035 0.05 0.07 0.09 0.10 0.12 0.18 0.24 0.245 0.250 0.258 0.265 0.272 0.280 0.290 0.30",
            "min": "100 100 150 150 150 150 150 150 150 150 80 80 80 80 80 80 80 80",
            "chunk": "100 100 150 150 150 150 150 150 150 150 80 80 80 80 80 80 80 80",
            "max": "200 200 450 450 450 450 450 600 600 600 320 320 320 320 320 320 320 320",
            "window": "80",
            "worse_tol": "0.010",
            "extra": "",
            "stage_extra": {
                "10": "env.safety_shield.d_crit=0.95 env.safety_shield.d_drop=1.60 env.safety_shield.catchup_action=-0.325 env.safety_shield.catchup_lateral_limit=0.34 env.safety_shield.catchup_centerline_limit=0.24 env.safety_shield.forward_bias_gain=0.07 env.safety_shield.forward_bias_clip=0.022 env.safety_shield.forward_bias_min_gap=1.25 env.safety_shield.forward_bias_min_command=0.30",
                "11": "env.safety_shield.d_crit=1.00 env.safety_shield.d_drop=1.70 env.safety_shield.catchup_action=-0.300 env.safety_shield.catchup_lateral_limit=0.32 env.safety_shield.catchup_centerline_limit=0.22 env.safety_shield.forward_bias_gain=0.09 env.safety_shield.forward_bias_clip=0.028 env.safety_shield.forward_bias_min_gap=1.32 env.safety_shield.forward_bias_min_command=0.30",
                "12": "env.safety_shield.d_crit=1.06 env.safety_shield.d_drop=1.85 env.safety_shield.catchup_action=-0.275 env.safety_shield.catchup_lateral_limit=0.30 env.safety_shield.catchup_centerline_limit=0.21 env.safety_shield.forward_bias_gain=0.11 env.safety_shield.forward_bias_clip=0.034 env.safety_shield.forward_bias_min_gap=1.40 env.safety_shield.forward_bias_min_command=0.30",
                "13": "env.safety_shield.d_crit=1.10 env.safety_shield.d_drop=2.00 env.safety_shield.catchup_action=-0.250 env.safety_shield.catchup_lateral_limit=0.28 env.safety_shield.catchup_centerline_limit=0.20 env.safety_shield.forward_bias_gain=0.13 env.safety_shield.forward_bias_clip=0.040 env.safety_shield.forward_bias_min_gap=1.48 env.safety_shield.forward_bias_min_command=0.30",
                "14": "env.safety_shield.d_crit=1.12 env.safety_shield.d_drop=2.15 env.safety_shield.catchup_action=-0.230 env.safety_shield.catchup_lateral_limit=0.27 env.safety_shield.catchup_centerline_limit=0.19 env.safety_shield.forward_bias_gain=0.15 env.safety_shield.forward_bias_clip=0.046 env.safety_shield.forward_bias_min_gap=1.55 env.safety_shield.forward_bias_min_command=0.30",
                "15": "env.safety_shield.d_crit=1.14 env.safety_shield.d_drop=2.30 env.safety_shield.catchup_action=-0.215 env.safety_shield.catchup_lateral_limit=0.26 env.safety_shield.catchup_centerline_limit=0.18 env.safety_shield.forward_bias_gain=0.17 env.safety_shield.forward_bias_clip=0.052 env.safety_shield.forward_bias_min_gap=1.62 env.safety_shield.forward_bias_min_command=0.30",
                "16": "env.safety_shield.d_crit=1.16 env.safety_shield.d_drop=2.45 env.safety_shield.catchup_action=-0.200 env.safety_shield.catchup_lateral_limit=0.25 env.safety_shield.catchup_centerline_limit=0.17 env.safety_shield.forward_bias_gain=0.19 env.safety_shield.forward_bias_clip=0.058 env.safety_shield.forward_bias_min_gap=1.70 env.safety_shield.forward_bias_min_command=0.30",
                "17": "env.safety_shield.d_crit=1.18 env.safety_shield.d_drop=2.60 env.safety_shield.catchup_action=-0.185 env.safety_shield.catchup_lateral_limit=0.24 env.safety_shield.catchup_centerline_limit=0.16 env.safety_shield.forward_bias_gain=0.21 env.safety_shield.forward_bias_clip=0.064 env.safety_shield.forward_bias_min_gap=1.78 env.safety_shield.forward_bias_min_command=0.30",
            },
        },
        {
            "name": "full_ladder_v7_hard_speed_gap_balance",
            "labels": "off light easy_0 easy_a easy_b med_a med_b hard_a hard_b hard_c hard_d0 hard_d1 hard_e0 hard_e1 hard_f0 hard_f1 hard_g hard",
            "levels": "off light easy easy easy medium medium hard hard hard hard hard hard hard hard hard hard hard",
            "passes": "off light easy easy easy medium medium hard hard hard hard hard hard hard hard hard hard hard",
            "pos": "-1 -1 1.00 1.25 1.50 1.75 2.00 2.75 4.00 6.00 6.20 6.40 6.55 6.70 6.90 7.15 7.50 8.00",
            "acc": "-1 -1 0.18 0.25 0.35 0.45 0.50 0.75 1.30 2.20 2.28 2.35 2.42 2.50 2.58 2.67 2.84 3.00",
            "dos": "-1 -1 0.035 0.05 0.07 0.09 0.10 0.12 0.18 0.24 0.245 0.250 0.258 0.265 0.272 0.280 0.290 0.30",
            "min": "100 100 150 150 150 150 150 150 150 150 80 80 80 80 80 80 80 80",
            "chunk": "100 100 150 150 150 150 150 150 150 150 80 80 80 80 80 80 80 80",
            "max": "200 200 450 450 450 450 450 600 600 600 320 400 400 400 400 400 400 400",
            "window": "80",
            "worse_tol": "0.012",
            "extra": "",
            "stage_extra": {
                "10": "env.safety_shield.d_crit=0.94 env.safety_shield.d_drop=1.58 env.safety_shield.catchup_action=-0.330 env.safety_shield.catchup_lateral_limit=0.34 env.safety_shield.catchup_centerline_limit=0.24 env.safety_shield.forward_bias_gain=0.08 env.safety_shield.forward_bias_clip=0.026 env.safety_shield.forward_bias_min_gap=1.22 env.safety_shield.forward_bias_min_command=0.30",
                "11": "env.safety_shield.d_crit=0.95 env.safety_shield.d_drop=1.60 env.safety_shield.catchup_action=-0.320 env.safety_shield.catchup_lateral_limit=0.32 env.safety_shield.catchup_centerline_limit=0.22 env.safety_shield.forward_bias_gain=0.105 env.safety_shield.forward_bias_clip=0.034 env.safety_shield.forward_bias_min_gap=1.24 env.safety_shield.forward_bias_min_command=0.30",
                "12": "env.safety_shield.d_crit=0.98 env.safety_shield.d_drop=1.72 env.safety_shield.catchup_action=-0.295 env.safety_shield.catchup_lateral_limit=0.30 env.safety_shield.catchup_centerline_limit=0.21 env.safety_shield.forward_bias_gain=0.120 env.safety_shield.forward_bias_clip=0.040 env.safety_shield.forward_bias_min_gap=1.32 env.safety_shield.forward_bias_min_command=0.30",
                "13": "env.safety_shield.d_crit=1.02 env.safety_shield.d_drop=1.85 env.safety_shield.catchup_action=-0.270 env.safety_shield.catchup_lateral_limit=0.28 env.safety_shield.catchup_centerline_limit=0.20 env.safety_shield.forward_bias_gain=0.140 env.safety_shield.forward_bias_clip=0.046 env.safety_shield.forward_bias_min_gap=1.40 env.safety_shield.forward_bias_min_command=0.30",
                "14": "env.safety_shield.d_crit=1.06 env.safety_shield.d_drop=2.00 env.safety_shield.catchup_action=-0.245 env.safety_shield.catchup_lateral_limit=0.27 env.safety_shield.catchup_centerline_limit=0.19 env.safety_shield.forward_bias_gain=0.160 env.safety_shield.forward_bias_clip=0.052 env.safety_shield.forward_bias_min_gap=1.48 env.safety_shield.forward_bias_min_command=0.30",
                "15": "env.safety_shield.d_crit=1.10 env.safety_shield.d_drop=2.15 env.safety_shield.catchup_action=-0.225 env.safety_shield.catchup_lateral_limit=0.26 env.safety_shield.catchup_centerline_limit=0.18 env.safety_shield.forward_bias_gain=0.180 env.safety_shield.forward_bias_clip=0.058 env.safety_shield.forward_bias_min_gap=1.56 env.safety_shield.forward_bias_min_command=0.30",
                "16": "env.safety_shield.d_crit=1.14 env.safety_shield.d_drop=2.32 env.safety_shield.catchup_action=-0.205 env.safety_shield.catchup_lateral_limit=0.25 env.safety_shield.catchup_centerline_limit=0.17 env.safety_shield.forward_bias_gain=0.200 env.safety_shield.forward_bias_clip=0.064 env.safety_shield.forward_bias_min_gap=1.66 env.safety_shield.forward_bias_min_command=0.30",
                "17": "env.safety_shield.d_crit=1.18 env.safety_shield.d_drop=2.50 env.safety_shield.catchup_action=-0.190 env.safety_shield.catchup_lateral_limit=0.24 env.safety_shield.catchup_centerline_limit=0.16 env.safety_shield.forward_bias_gain=0.220 env.safety_shield.forward_bias_clip=0.070 env.safety_shield.forward_bias_min_gap=1.76 env.safety_shield.forward_bias_min_command=0.30",
            },
        },
        {
            "name": "full_ladder_v8_hard_b_final",
            "labels": "off light easy_0 easy_a easy_b med_a med_b hard_a hard",
            "levels": "off light easy easy easy medium medium hard hard",
            "passes": "off light easy easy easy medium medium hard hard",
            "pos": "-1 -1 1.00 1.25 1.50 1.75 2.00 2.75 4.00",
            "acc": "-1 -1 0.18 0.25 0.35 0.45 0.50 0.75 1.30",
            "dos": "-1 -1 0.035 0.05 0.07 0.09 0.10 0.12 0.18",
            "min": "100 100 150 150 150 150 150 150 150",
            "chunk": "100 100 150 150 150 150 150 150 150",
            "max": "200 200 450 450 450 450 450 600 600",
            "window": "80",
            "worse_tol": "0.010",
            "extra": "",
        },
    ]
    return profiles[min(attempt, len(profiles) - 1)]


def completed(log_text: str) -> bool:
    return (
        "[PIPELINE] stage hard passed" in log_text
        and "[PIPELINE] package tar:" in log_text
        and "[PIPELINE] exit status=0" in log_text
    )


def failure_summary(log_text: str) -> str:
    patterns = [
        r"\[ERROR\].*",
        r"\[ASSESS\]\[[^\]]+\] status=(?:ABORT|CONTINUE).*",
        r"\[PIPELINE\] exit status=\d+.*",
        r"Traceback \(most recent call last\).*",
        r"FileNotFoundError:.*",
    ]
    lines: list[str] = []
    for line in log_text.splitlines()[-400:]:
        if any(re.search(p, line) for p in patterns):
            lines.append(line)
    return " | ".join(lines[-8:]) if lines else "no explicit failure marker found"


def ensure_monitor() -> None:
    pid = read_pid(MONITOR_PID_POINTER)
    if pid_alive(pid):
        return
    proc = subprocess.Popen(
        ["bash", "-c", f"while true; do {MONITOR} || true; sleep 300; done"],
        cwd=ROOT,
        stdout=(ROOT / "train_monitor_loop.out").open("ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    MONITOR_PID_POINTER.write_text(f"{proc.pid}\n", encoding="utf-8")
    log(f"monitor_started pid={proc.pid}")


def launch_attempt(attempt: int) -> None:
    p = profile(attempt)
    labels = p["labels"].split()
    run_tag = f"paper_full_hard_auto_a{attempt + 1}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline_log = ROOT / f"train_{run_tag}.log"
    env = os.environ.copy()
    env.update(
        {
            "RUN_TAG": run_tag,
            "PIPELINE_LOG": str(pipeline_log),
            "START_RUN": BASE_RUN,
            "START_CHECKPOINT": BASE_CHECKPOINT,
            "STAGE_LIMIT": str(len(labels)),
            "START_STAGE_INDEX": "0",
            "STAGE_LABELS_OVERRIDE": p["labels"],
            "STAGE_ATTACK_LEVELS_OVERRIDE": p["levels"],
            "STAGE_PASS_PROFILES_OVERRIDE": p["passes"],
            "STAGE_MAX_FDI_POS_OVERRIDE": p["pos"],
            "STAGE_MAX_FDI_ACC_OVERRIDE": p["acc"],
            "STAGE_MAX_DOS_RATE_OVERRIDE": p["dos"],
            "STAGE_MIN_ITERS_OVERRIDE": p["min"],
            "STAGE_CHUNK_ITERS_OVERRIDE": p["chunk"],
            "STAGE_MAX_ITERS_OVERRIDE": p["max"],
            "ASSESS_WINDOW": p["window"],
            "PLOT_SMOOTH": "50",
            "ALLOW_EARLY_PASS": "1",
            "RESUME_STAGE_PROGRESS": "1",
            "VALIDATION_ROLLBACK": "1",
            "VALIDATION_WORSE_TOL": p["worse_tol"],
            "COMMON_OVERRIDES_EXTRA": p["extra"],
        }
    )
    for stage_idx, overrides in p.get("stage_extra", {}).items():
        env[f"STAGE_EXTRA_OVERRIDES_{stage_idx}"] = overrides
    out = (ROOT / f"/tmp/{run_tag}.setsid").open("ab") if False else subprocess.DEVNULL
    proc = subprocess.Popen(
        [str(PIPELINE)],
        cwd=ROOT,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_POINTER.write_text(f"{proc.pid}\n", encoding="utf-8")
    LOG_POINTER.write_text(f"{pipeline_log}\n", encoding="utf-8")
    TAG_POINTER.write_text(f"{run_tag}\n", encoding="utf-8")
    state = load_state()
    state.update({"attempt": attempt, "run_tag": run_tag, "pid": proc.pid, "log": str(pipeline_log), "profile": p["name"], "completed": False})
    state.setdefault("history", []).append({"time": now(), "attempt": attempt, "run_tag": run_tag, "profile": p["name"]})
    save_state(state)
    log(f"launched attempt={attempt + 1} profile={p['name']} pid={proc.pid} tag={run_tag}")
    ensure_monitor()


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    log(f"supervisor_started interval={CHECK_INTERVAL_SEC}s")
    ensure_monitor()
    state = load_state()
    if not pid_alive(read_pid(PID_POINTER)):
        launch_attempt(max(int(state.get("attempt", -1)) + 1, 0))

    while True:
        state = load_state()
        pid = read_pid(PID_POINTER)
        log_path = Path(read_text(LOG_POINTER))
        if pid_alive(pid):
            log(f"check status=RUNNING pid={pid} tag={read_text(TAG_POINTER)}")
        else:
            log_text = tail(log_path)
            if completed(log_text):
                state["completed"] = True
                state["completed_at"] = now()
                save_state(state)
                log(f"completed tag={state.get('run_tag')} log={log_path}")
                return 0
            summary = failure_summary(log_text)
            log(f"check status=STOPPED_UNFINISHED pid={pid} summary={summary}")
            wait_after_failed_attempt(pid)
            next_attempt = int(state.get("attempt", -1)) + 1
            launch_attempt(next_attempt)
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
