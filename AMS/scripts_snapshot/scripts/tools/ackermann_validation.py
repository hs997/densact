#!/usr/bin/env python3
"""Fail-closed, content-addressed validation of isolated Ackermann evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


SEEDS = ((42, 3407), (3407, 9917), (777, 17011))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f'nonfinite {key}')
    return value


def inspect(output, spec):
    """Check raw rows as well as summaries; a nonempty summary is insufficient."""
    output = Path(output)
    with (output / 'eval_summary.csv').open() as handle:
        summaries = list(csv.DictReader(handle))
    if len(summaries) != 1:
        raise ValueError('exactly one checkpoint per process is required')
    summary = summaries[0]
    if Path(summary['checkpoint_path']).resolve() != Path(spec['checkpoint']).resolve():
        raise ValueError('checkpoint mismatch')
    if number(summary, 'eval_rows') != spec['steps'] or number(summary, 'summary_rows') != spec['steps']:
        raise ValueError('incomplete summary or unexpected warmup')
    step_path = output / ('eval_steps_' + Path(spec['checkpoint']).stem + '.csv')
    with step_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != spec['steps']:
        raise ValueError('incomplete step CSV')
    expected = {
        'num_envs': spec['num_envs'], 'num_agents': 5,
        'attack_enabled': int(spec['enabled']),
        'attack_max_fdi_pos': spec['pos'], 'attack_max_fdi_acc': spec['acc'],
        'attack_max_dos_rate': spec['dos'], 'shield_expert_blend_ratio': 0,
    }
    if spec['enabled']:
        expected.update(attack_strength_scale=1, attack_clean_env_fraction=0,
                        attack_action_dim=2, attack_action_fdi_longitudinal_only=1)
    metrics = {
        'completion': 'platoon_completion_ever_rate',
        'instant': 'platoon_completion_instant_rate',
        'dwell': 'platoon_completion_dwell_rate',
        'collision': 'collision_rate', 'road_any': 'road_boundary_any_rate',
        'road_termination': 'termination_road_boundary_exceeded',
    }
    maxima = dict.fromkeys(metrics, 0.0)
    for step, row in enumerate(rows, 1):
        if number(row, 'eval_step') != step:
            raise ValueError('missing, duplicate or reordered steps')
        for key, value in expected.items():
            if not math.isclose(number(row, key), value, abs_tol=1e-6):
                raise ValueError(f'protocol mismatch at step {step}: {key} != {value}')
        if spec['enabled'] and row['attack_level'] != spec['level']:
            raise ValueError('attack level mismatch')
        for label, key in metrics.items():
            value = number(row, key)
            if not 0 <= value <= 1:
                raise ValueError(f'invalid rate: {key}')
            maxima[label] = max(maxima[label], value)
        for robot in range(1, 6):
            number(row, f'trajectory_progress_robot_{robot}_mean')
        number(row, 'eval_total_reward_mean')
    # Ever-completion persists across the terminal reset in the evaluator.
    completion = number(rows[-1], metrics['completion'])
    summary_metrics = {
        'platoon_completion_ever_rate_final': completion,
        'platoon_completion_instant_rate_peak': maxima['instant'],
        'platoon_completion_dwell_rate_peak': maxima['dwell'],
    }
    for key, value in summary_metrics.items():
        if not math.isclose(number(summary, key), value, abs_tol=1e-6):
            raise ValueError(f'summary/raw disagreement: {key}')
    return dict(maxima, completion=completion, name=output.name,
                attack_enabled=int(spec['enabled']),
                checkpoint_sha256=spec['checkpoint_sha256'],
                env_seed=spec['env_seed'], attack_seed=spec['attack_seed'])


def load_verified(output):
    output = Path(output)
    stamp = json.loads((output / 'verified.json').read_text())
    for filename, sha in stamp['artifacts'].items():
        if digest(output / filename) != sha:
            raise ValueError(f'changed artifact: {output / filename}')
    if digest(stamp['spec']['checkpoint']) != stamp['spec']['checkpoint_sha256']:
        raise ValueError('checkpoint contents changed after evaluation')
    return inspect(output, stamp['spec']), stamp['spec']


def clean_passed(row):
    return (all(row[k] == 1 for k in ('completion', 'instant', 'dwell'))
            and all(row[k] == 0 for k in ('collision', 'road_any', 'road_termination', 'attack_enabled')))


def gate(root, threshold):
    root = Path(root)
    clean, clean_spec = load_verified(root / 'validation/clean_env42')
    clean_ok = clean_passed(clean)
    attacks = []
    if clean_ok:
        attack_specs = []
        for env_seed, attack_seed in SEEDS:
            row, spec = load_verified(root / f'validation/attack_env{env_seed}_seed{attack_seed}')
            if (spec['env_seed'], spec['attack_seed']) != (env_seed, attack_seed) or not spec['enabled']:
                raise ValueError('wrong or duplicated attack seeds')
            if spec['checkpoint_sha256'] != clean_spec['checkpoint_sha256']:
                raise ValueError('clean/attack checkpoint mismatch')
            if spec.get('sources') != clean_spec.get('sources'):
                raise ValueError('source code changed between clean and attack evaluations')
            attack_specs.append((spec['level'], spec['pos'], spec['acc'], spec['dos']))
            attacks.append(row)
        if len(set(attack_specs)) != 1:
            raise ValueError('attack budgets differ between seeds')
    safe = clean_ok and len(attacks) == 3 and all(
        r[k] == 0 for r in attacks for k in ('collision', 'road_any', 'road_termination'))
    score = min((r['completion'] for r in attacks), default=0)
    report = dict(passed=safe and score >= threshold, threshold=threshold,
                  clean_ok=clean_ok, safe=safe, attack_ok=safe and score >= threshold,
                  attack_completion_min=score,
                  attack_completion_mean=sum(r['completion'] for r in attacks) / 3 if attacks else 0,
                  rows=[clean, *attacks])
    write_json(root / 'validation_gate.json', report)
    return report


def run(args):
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        raise ValueError('missing evaluator command')
    source_root = Path(__file__).resolve().parents[2]
    sources = sorted((source_root / 'source/my_exts/marl_platoon').rglob('*.py'))
    sources += [source_root / 'scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py', Path(__file__)]
    spec = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
                steps=args.steps, num_envs=args.num_envs, env_seed=args.env_seed,
                attack_seed=args.attack_seed, enabled=args.enabled == 'true',
                level=args.level, pos=args.pos, acc=args.acc, dos=args.dos,
                command=command, sources={str(p.relative_to(source_root)): digest(p) for p in sources})
    output = args.output
    try:
        _, previous = load_verified(output)
        if previous == spec:
            print(f'[VALIDATION] reuse verified {output}', flush=True)
            return
    except (OSError, ValueError, KeyError):
        pass
    if output.exists() and any(output.iterdir()):
        archive = output.with_name(output.name + f'.superseded_{time.time_ns()}')
        output.rename(archive)
        print(f'[VALIDATION] preserved unmatched/partial artifacts: {archive}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'request.json', spec)
    busy = subprocess.run(['pgrep', '-af', r'scripts/reinforcement_learning/.*/(train|play|eval_happo_platoon)\.py'],
                          capture_output=True, text=True)
    # Ignore this runner/its ancestors: their argv contains the future evaluator command.
    ancestors = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    conflicts = [line for line in busy.stdout.splitlines() if int(line.split()[0]) not in ancestors]
    if conflicts:
        raise RuntimeError('another Isaac workload is active: ' + '\n'.join(conflicts))
    print(f'[VALIDATION] start {output}', flush=True)
    with (output / 'evaluation.log').open('w') as log:
        # A stuck simulator cannot silently occupy the GPU for days.
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                rc = process.wait(timeout=args.timeout)
            except BaseException:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    if rc != 0:
        raise RuntimeError(f'evaluator exit={rc}; see {output / "evaluation.log"}')
    row = inspect(output, spec)
    artifacts = {p.name: digest(p) for p in output.glob('*.csv')}
    write_json(output / 'verified.json', dict(spec=spec, metrics=row, artifacts=artifacts))
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    for name in ('output', 'checkpoint'):
        r.add_argument('--' + name, type=Path, required=True)
    for name in ('steps', 'num-envs', 'env-seed', 'attack-seed'):
        r.add_argument('--' + name, type=int, required=True)
    for name in ('pos', 'acc', 'dos'):
        r.add_argument('--' + name, type=float, required=True)
    r.add_argument('--enabled', choices=('true', 'false'), required=True)
    r.add_argument('--level', required=True)
    r.add_argument('--timeout', type=int, default=1800)
    r.add_argument('command', nargs=argparse.REMAINDER)
    c = sub.add_parser('clean')
    c.add_argument('output', type=Path)
    g = sub.add_parser('gate')
    g.add_argument('root', type=Path)
    g.add_argument('threshold', type=float)
    args = parser.parse_args()
    if args.action == 'run':
        run(args)
    elif args.action == 'clean':
        raise SystemExit(0 if clean_passed(load_verified(args.output)[0]) else 8)
    else:
        report = gate(args.root, args.threshold)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report['passed'] else 8)


if __name__ == '__main__':
    main()
