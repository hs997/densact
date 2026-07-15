#!/usr/bin/env python3
"""Lightweight monitor for the current adaptive platoon training run."""

from __future__ import annotations

import csv
import datetime as dt
import os
import subprocess
from pathlib import Path


ROOT = Path("/home/cnc/SSD_1T/xzw/IsaacLab-main")
LOG_POINTER = ROOT / ".current_adaptive_training.log"
PID_POINTER = ROOT / ".current_adaptive_training.pid"
MONITOR_LOG = ROOT / "train_monitor_current.log"
ALERT_LOG = ROOT / "train_monitor_alerts.log"


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def newest_run_dir(log_text: str) -> Path | None:
    candidates: list[Path] = []
    for line in log_text.splitlines():
        if "/logs/rsl_rl/platoon_happo/" not in line:
            continue
        start = line.find("/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/")
        if start < 0:
            continue
        token = line[start:].split()[0].strip(",:)")
        path = Path(token)
        if path.name.endswith(".pt") or path.name.endswith(".csv") or path.name in {"figures", "plots"}:
            path = path.parent
        if path.exists() and path.is_dir():
            candidates.append(path)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    root = ROOT / "logs/rsl_rl/platoon_happo"
    tag = read_text(ROOT / ".current_adaptive_training.tag")
    dirs = [p for p in root.glob(f"*{tag}*") if p.is_dir()] if tag else []
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def tail(path: Path, max_bytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def mean_last_metrics(csv_path: Path, n: int = 200) -> dict[str, float]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    rows = rows[-n:]
    keys = [
        "speed_error_abs_mean",
        "gap_error_abs_mean",
        "lateral_error_abs_mean",
        "centerline_error_abs_mean",
        "min_pair_gap_mean",
        "termination_reset_on_bad_ori",
        "reset_on_bad_ori",
        "collision_rate",
        "critic_grad_norm",
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = []
        for row in rows:
            try:
                vals.append(float(row.get(key, "")))
            except ValueError:
                pass
        if vals:
            out[key] = sum(vals) / len(vals)
    out["rows"] = float(len(rows))
    return out


def gpu_summary() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip()
        return out
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi unavailable: {exc}"


def main() -> int:
    alerts: list[str] = []
    log_path = Path(read_text(LOG_POINTER))
    pid_text = read_text(PID_POINTER)
    pid = int(pid_text) if pid_text.isdigit() else None

    if not log_path.exists():
        alerts.append(f"log missing: {log_path}")
    if pid is None:
        alerts.append(f"pid pointer invalid: {pid_text!r}")
    elif not pid_alive(pid):
        alerts.append(f"pipeline pid not alive: {pid}")

    log_text = tail(log_path)
    if any(token in log_text for token in ["Traceback (most recent call last)", "FileNotFoundError", "[ERROR]", "There was an error running python"]):
        alerts.append("recent log contains error marker")

    run_dir = newest_run_dir(log_text)
    metrics = mean_last_metrics(run_dir / "platoon_metrics.csv") if run_dir else {}
    if not run_dir:
        alerts.append("could not locate current run directory")
    elif not metrics:
        alerts.append(f"metrics missing or empty under {run_dir}")
    else:
        reset = metrics.get("termination_reset_on_bad_ori", metrics.get("reset_on_bad_ori", 0.0))
        collision = metrics.get("collision_rate", 0.0)
        if metrics.get("centerline_error_abs_mean", 0.0) > 0.55:
            alerts.append(f"centerline_error_abs_mean high: {metrics['centerline_error_abs_mean']:.4f}")
        if metrics.get("lateral_error_abs_mean", 0.0) > 0.45:
            alerts.append(f"lateral_error_abs_mean high: {metrics['lateral_error_abs_mean']:.4f}")
        if metrics.get("min_pair_gap_mean", 999.0) < 0.90:
            alerts.append(f"min_pair_gap_mean too low: {metrics['min_pair_gap_mean']:.4f}")
        if reset > 0.05:
            alerts.append(f"reset_on_bad_ori high: {reset:.4f}")
        if collision > 0.01:
            alerts.append(f"collision_rate high: {collision:.4f}")

    status = "ALERT" if alerts else "OK"
    metric_text = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())) if metrics else "no_metrics"
    line = f"{now()} status={status} pid={pid} run_dir={run_dir} gpu={gpu_summary()} metrics=({metric_text})"
    append(MONITOR_LOG, line)
    if alerts:
        append(ALERT_LOG, line + " alerts=" + "; ".join(alerts))
    return 1 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
