#!/usr/bin/env python3
"""Complete final-model regressions, then run a separate, uniform six-method cohort.

Old results remain immutable. All new training starts from the common trajectory
BC checkpoint, not from a retention/attack-trained AMS checkpoint. Training seeds
are independent conditional on that shared warm start, not independent BC fits.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics

import torch
import yaml

from ackermann_validation import SEEDS, digest, load_verified, write_json
from continue_ackermann_retention import checkpoint_state, actor_learning_rate, training_overrides
from revalidate_ackermann_retention import STRICT_EVAL
from run_ackermann_verified_followup import (
    Followup, LEVELS, METHODS, ROOT, method_overrides, note, same_evaluation, verify_hard_b,
)


def immutable_json(path, value):
    value = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing identity differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)


def import_evaluation(source, target, sources):
    """Copy raw verified evidence, retaining its original command and provenance."""
    row, spec = load_verified(source)
    if spec['sources'] != sources:
        raise ValueError(f"Evaluation source mismatch: {source}")
    if target.exists():
        _, old = load_verified(target)
        if old != spec:
            raise ValueError(f"Refuse to replace evaluation: {target}")
        return
    target.mkdir(parents=True)
    stamp = json.loads((source / 'verified.json').read_text())
    for name in [*stamp['artifacts'], 'verified.json', 'request.json', 'evaluation.log']:
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    write_json(target / 'imported_from.json', dict(source=str(source),
               source_verified_sha256=digest(source / 'verified.json')))
    load_verified(target)


def scalar_equal(left, right):
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    except (ValueError, TypeError):
        return str(left).lower() == str(right).lower()


def audit_new_training(run, parent, level, updates, label, overrides, bc):
    """Check effective runtime protocol, method identity, optimizer and BC anchor."""
    before, after = checkpoint_state(parent), checkpoint_state(run / 'model_final.pt')
    if int(after['update_count']) - int(before['update_count']) != updates:
        raise ValueError('Training update count does not match the requested continuation')
    expected = dict(item.split('=', 1) for item in overrides)
    lr = float(expected['env.algorithm.happo_actor_lr'])
    if actor_learning_rate(after) != lr or float(after['cfg']['actor_lr']) != lr:
        raise ValueError('Restored optimizer learning rate differs from protocol')
    saved = yaml.load((run / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    for key, value in expected.items():
        if key.startswith('env.algorithm.'):
            if not scalar_equal(saved['algorithm'][key.rsplit('.', 1)[-1]], value):
                raise ValueError(f'Training algorithm override not effective: {key}')
    _, teacher, factor, shared, mode, ams = METHODS[label]
    for key, value in dict(use_happo_factor=factor, share_actor=shared,
                           actor_update_mode=mode, ams_enabled=ams).items():
        if after['cfg'][key] != value:
            raise ValueError(f'Wrong runtime method: {key}')
    changed = any(not torch.equal(old[k], new[k])
                  for old, new in zip(before['actors'], after['actors']) for k in old)
    if not changed:
        raise ValueError('No actor parameter changed; retain this run for diagnosis')
    references = after.get('retention_reference_actors', [])
    bc_actors = checkpoint_state(bc)['actors']
    if len(references) != len(bc_actors) or any(
            ref is None or any(not torch.equal(ref[k], actor[k]) for k in actor)
            for ref, actor in zip(references, bc_actors)):
        raise ValueError('Retention reference is not the common BC actor')
    with (run / 'platoon_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != updates:
        raise ValueError('Missing training metric rows')
    enabled = level[0] != 'off'
    required = dict(attack_enabled=int(enabled), shield_expert_blend_ratio=0,
                    ams_enabled=int(ams), attack_strength_scale=1,
                    attack_clean_env_fraction=.5, attack_max_fdi_pos=level[2],
                    attack_max_fdi_acc=level[3], attack_max_dos_rate=level[4])
    for row in rows:
        for key, value in required.items():
            if not math.isclose(float(row[key]), value, abs_tol=1e-6):
                raise ValueError(f'Effective training protocol differs: {key}')
        if row['attack_level'] != level[1] or row['attack_mode'] != 'profile':
            raise ValueError('Wrong training attack profile')
        for key in ('bc_loss', 'reference_kl', 'retention_loss', 'ams_q_loss', 'reward_env_mean'):
            if not math.isfinite(float(row[key])):
                raise ValueError(f'Nonfinite training metric: {key}')
    perturbation = max(float(r['action_attack_delta_abs_mean']) for r in rows)
    if enabled and perturbation <= 0:
        raise ValueError('No observed perturbation in attack training')
    return dict(checkpoint=str(run / 'model_final.pt'), checkpoint_sha256=digest(run / 'model_final.pt'),
                parent_sha256=digest(parent), updates_before=int(before['update_count']),
                updates_after=int(after['update_count']), actual_actor_lr=lr,
                actor_parameters_changed=changed, common_bc_reference_verified=True,
                effective_protocol_verified=True, metrics_sha256=digest(run / 'platoon_metrics.csv'),
                action_attack_delta_abs_mean_max=perturbation)


class CompletionQueue(Followup):
    def __init__(self, args):
        self.args, self.output, self.current = args, args.output.resolve(), {}
        self.evaluation_cache, self.render_cache = {}, {}
        self.output.mkdir(parents=True, exist_ok=True)
        self.environment = dict(os.environ)
        for key in ('PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV'):
            self.environment.pop(key, None)
        self.environment['PYTHONPATH'] = ':'.join(str(ROOT / p) for p in (
            'source/isaaclab', 'source/isaaclab_rl', 'source/isaaclab_tasks', 'source/my_exts'))
        self.environment['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        files = sorted((ROOT / 'source/my_exts/marl_platoon').rglob('*.py')) + [ROOT / p for p in (
            'scripts/tools/run_ackermann_completion_queue.py', 'scripts/tools/continue_ackermann_retention.py',
            'scripts/tools/revalidate_ackermann_retention.py', 'scripts/tools/ackermann_validation.py',
            'scripts/tools/run_ackermann_verified_followup.py', 'scripts/tools/run_medium_algorithm_comparison.sh',
            'scripts/tools/run_platoon_spatial_attack_ladder.sh', 'scripts/tools/plot_ackermann_recovery_stage.py',
            'scripts/tools/plot_ackermann_six_attack_matrix.py', 'scripts/tools/experiment_env.sh',
            'scripts/reinforcement_learning/rsl_rl/train.py', 'scripts/reinforcement_learning/rsl_rl/play.py',
            'scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py')]
        self.sources = {str(p.relative_to(ROOT)): digest(p) for p in files}
        self.evaluation_sources = {k: v for k, v in self.sources.items()
            if k.startswith('source/my_exts/marl_platoon/') or k in (
                'scripts/tools/ackermann_validation.py', 'scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py')}
        self.render_sources = {k: v for k, v in self.sources.items()
            if k.startswith('source/my_exts/marl_platoon/') or k in (
                'scripts/reinforcement_learning/rsl_rl/play.py', 'scripts/tools/run_platoon_spatial_attack_ladder.sh')}
        self.references, self.commons, self.checkpoints = [], [], []
        for index in range(3):
            stage = args.reference.resolve() / f'candidate_{index}/hard_b'
            checkpoint = verify_hard_b(stage)
            render = json.loads((stage / 'evidence/off/render_complete.json').read_text())
            self.references.append(stage)
            self.commons.append(render['signature']['common'])
            self.checkpoints.append(checkpoint)
            for name in ['clean_env42', *[f'attack_env{e}_seed{a}' for e, a in SEEDS]]:
                _, spec = load_verified(stage / 'validation' / name)
                if spec['sources'] != self.evaluation_sources:
                    raise ValueError('Reference evaluation sources changed')
        if any(common != self.commons[0] for common in self.commons):
            raise ValueError('Final model regression protocols differ across seeds')
        self.common = self.commons[0]
        self.algorithm = yaml.load((self.checkpoints[0].parent / 'params/env.yaml').read_text(),
                                   Loader=yaml.BaseLoader)['algorithm']
        bc = args.bc_checkpoint.resolve()
        report = json.loads(bc.with_suffix('.json').read_text())
        if Path(report['output']).resolve() != bc or report['samples'] <= 0:
            raise ValueError('Invalid real-trajectory BC provenance')
        bc_state = checkpoint_state(bc)
        if bc_state['update_count'] != 0 or bc_state['actor_optimizers']:
            raise ValueError('Expected the common BC warm start with fresh optimizers')
        manifest = dict(reference=str(args.reference.resolve()),
            final_checkpoints=[dict(path=str(p), sha256=digest(p)) for p in self.checkpoints],
            regression_common=self.common, levels=LEVELS, sources=self.sources,
            matrix=dict(methods=METHODS, train_seeds=[e for e, _ in SEEDS], eval_seeds=SEEDS,
                        stage_updates=args.stage_updates, rounds=1, clean_acquisition_updates=240,
                        actor_lr=5e-7, train_envs=32, eval_envs=16, eval_steps=3500,
                        expert_ratio=0, attack_strength=1, attack_clean_fraction=.5,
                        retention_bc_coef=4., retention_kl_coef=.8,
                        bc_checkpoint=str(bc), bc_sha256=digest(bc), bc_report=report,
                        trajectory_sha256=digest(report['source']),
                        seed_scope='independent adaptation seeds conditional on one common BC fit',
                        prior_results_reused_for_training=False))
        immutable_json(self.output / 'queue_manifest.json', manifest)
        for path in files:
            target = self.output / 'source_snapshot' / path.relative_to(ROOT)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def run(self, command, log, env=None, timeout=7200):
        if self.args.preflight_only:
            raise RuntimeError(f'Preflight found an unverified cache; would run: {command[:3]}')
        # Fail before launching if any recorded source has changed during a long queue.
        if any(digest(ROOT / name) != sha for name, sha in self.sources.items()):
            raise ValueError('Source drift during queue; no further process launched')
        super().run(command, log, env, timeout)

    def evaluate(self, label, checkpoint, level, output, env_seed=42, attack_seed=3407):
        key = (label, digest(checkpoint), tuple(level), env_seed, attack_seed)
        cached = self.evaluation_cache.get(key)
        if cached and cached != output and not output.exists():
            import_evaluation(cached, output, self.evaluation_sources)
        row = super().evaluate(label, checkpoint, level, output, env_seed, attack_seed)
        self.evaluation_cache[key] = output
        return row

    def evidence(self, label, checkpoint, level, destination, verified_dir, *, diagnostic_clean=False):
        key = (label, digest(checkpoint), tuple(level), diagnostic_clean)
        cached = self.render_cache.get(key)
        directory = destination / level[0]
        if cached and cached != directory and not (directory / 'render_complete.json').exists():
            old = json.loads((cached / 'render_complete.json').read_text())
            _, spec = load_verified(verified_dir)
            sig = old['signature']
            if (sig['sources'] == self.render_sources and sig['common'] == self.common
                    and same_evaluation(sig['evaluation_spec'], spec)
                    and sig['evaluation_spec']['sources'] == spec['sources']):
                for name, sha in old['artifacts'].items():
                    if digest(cached / name) != sha:
                        raise ValueError('Cached render artifacts changed')
                    target = directory / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and digest(target) != sha:
                        raise ValueError('Refuse to replace another rendered artifact')
                    if not target.exists():
                        shutil.copy2(cached / name, target)
                write_json(directory / 'imported_evidence.json', dict(source=str(cached),
                           source_render_sha256=digest(cached / 'render_complete.json')))
        super().evidence(label, checkpoint, level, destination, verified_dir,
                         diagnostic_clean=diagnostic_clean)
        self.render_cache[key] = directory

    def import_reference(self, index):
        stage, checkpoint = self.references[index], self.checkpoints[index]
        root = self.output / f'final_model_regressions/final_ladders/s{index}'
        for level in (LEVELS[0], LEVELS[-1]):
            names = ['clean_env42'] if level[0] == 'off' else [f'attack_env{e}_seed{a}' for e, a in SEEDS]
            for name in names:
                destination = root / level[0] / 'validation' / name
                import_evaluation(stage / 'validation' / name, destination, self.evaluation_sources)
                seed = (42, 3407) if name == 'clean_env42' else next(
                    (e, a) for e, a in SEEDS if name == f'attack_env{e}_seed{a}')
                # The inherited evaluator independently compares the entire command,
                # checkpoint and configuration; preflight cannot launch a simulator.
                self.evaluate('happo_meta_ams', checkpoint, level, destination, *seed)
            original = stage / 'evidence' / level[0]
            stamp = json.loads((original / 'render_complete.json').read_text())
            sig = stamp['signature']
            if sig['checkpoint_sha256'] != digest(checkpoint) or sig['common'] != self.common \
                    or sig['level'] != list(level) or sig['method'] != 'happo_meta_ams' \
                    or sig['diagnostic_clean'] or any(sig['sources'].get(k) != v for k, v in self.render_sources.items()):
                raise ValueError('Reference render protocol differs')
            target = root / level[0]
            for name, sha in stamp['artifacts'].items():
                if digest(original / name) != sha:
                    raise ValueError('Reference render artifact changed')
                out = target / name
                if out.exists():
                    if digest(out) != sha:
                        raise ValueError('Imported render artifact changed')
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original / name, out)
            shutil.copy2(original / 'render_complete.json', target / 'original_render_complete.json')
            verified = target / 'validation' / names[0]
            _, spec = load_verified(verified)
            if sig['evaluation_spec'] != spec:
                raise ValueError('Reference render is bound to another evaluation')
            immutable_json(target / 'evidence_complete.json', dict(
                signature=dict(checkpoint_sha256=digest(checkpoint), spec=spec),
                artifacts=stamp['artifacts'], imported_from=str(original),
                source_render_sha256=digest(original / 'render_complete.json')))

    def regressions(self):
        for index, checkpoint in enumerate(self.checkpoints):
            self.import_reference(index)
            self.export('happo_meta_ams', checkpoint,
                        self.output / f'final_model_regressions/final_ladders/s{index}')
        directory = self.output / 'final_model_regressions'
        self.run([self.args.host_python, 'scripts/tools/plot_ackermann_six_attack_matrix.py',
                  str(directory / 'final_ladders')], directory / 'plotting.log')
        outcomes = {f's{i}/{level[0]}': json.loads((directory /
            f'final_ladders/s{i}/{level[0]}/assessment.json').read_text())['passed']
            for i in range(3) for level in LEVELS}
        write_json(directory / 'complete.json', dict(all_gates_passed=all(outcomes.values()), gates=outcomes,
                   assessment_only=True, training_performed=False))
        note(f'- 三个最终模型全强度回归完成，all_gates_passed={all(outcomes.values())}；`{directory}`。'
             '对照训练从独立BC起点开始，不使用任何失败回归模型晋级。')


class UniformMatrix(CompletionQueue):
    def __init__(self, queue, index):
        self.__dict__.update(queue.__dict__)
        self.args = copy.copy(queue.args)
        self.args.seed_index = index
        self.args.rounds = 1
        self.args.actor_lr = 5e-7
        self.args.train_envs = 32
        self.output = queue.output / f'uniform_six/s{index}'
        self.output.mkdir(parents=True, exist_ok=True)
        self.current = {}
        self.evaluation_cache, self.render_cache = {}, {}
        self.method_labels = tuple(METHODS)
        # Explicit saved algorithm parameters, shared physical/safety protocol.
        # Training-only switches are overridden again for strict evaluation.
        values = dict(item.split('=', 1) for item in queue.common)
        values.update({f'env.algorithm.{k}': v for k, v in self.algorithm.items()})
        values.update(STRICT_EVAL)
        values.update({'env.algorithm.happo_actor_lr': '5e-7',
                       'env.algorithm.happo_retention_bc_coef': '4.0',
                       'env.algorithm.happo_retention_kl_coef': '0.8'})
        self.common = [f'{k}={v}' for k, v in values.items()]
        immutable_json(self.output / 'matrix_manifest.json', dict(seed_index=index,
            seed=SEEDS[index][0], methods=METHODS, common=self.common, sources=self.sources,
            stage_updates=self.args.stage_updates, rounds=1,
            bc_checkpoint=str(self.args.bc_checkpoint.resolve()), bc_sha256=digest(self.args.bc_checkpoint)))

    def train(self, label, parent, level, stage, updates, round_index):
        if round_index != 0:
            raise ValueError('This fixed-budget cohort has exactly one candidate per level')
        if parent.resolve() != self.args.bc_checkpoint.resolve():
            if actor_learning_rate(checkpoint_state(parent)) != self.args.actor_lr:
                raise ValueError('Parent optimizer rate differs from the uniform protocol')
        extra = training_overrides(self.common, self.algorithm, level, self.args.actor_lr,
                                   SEEDS[self.args.seed_index][1])
        extra += method_overrides(label)
        extra += [f'env.algorithm.enable_attack={str(level[0] != "off").lower()}']
        tag = f'{self.args.output.name}_s{self.args.seed_index}_{label}_{stage.name}'
        signature = dict(parent_sha256=digest(parent), method=label, level=list(level), updates=updates,
            seed=SEEDS[self.args.seed_index][0], overrides=extra, sources=self.sources,
            bc_sha256=digest(self.args.bc_checkpoint), train_envs=32)
        marker, request = stage / 'training_complete.json', stage / 'training_request.json'
        if request.exists() and json.loads(request.read_text()) != signature:
            raise ValueError('Training request changed; refusing overwrite')
        if marker.exists():
            recorded = json.loads(marker.read_text())
            if recorded['signature'] != signature:
                raise ValueError('Completed training identity differs')
            candidate = Path(recorded['audit']['checkpoint'])
            audit = audit_new_training(candidate.parent, parent, level, updates, label, extra, self.args.bc_checkpoint)
            if audit != recorded['audit']:
                raise ValueError('Completed training artifacts changed')
            return candidate
        candidates = list((ROOT / 'logs/rsl_rl/platoon_happo').glob(f'*_{tag}_{label}/model_final.pt'))
        if request.exists():
            if len(candidates) != 1:
                raise RuntimeError(f'Partial training retained; refusing blind repeat: {stage}')
            candidate = candidates[0]
        else:
            stage.mkdir(parents=True, exist_ok=True)
            write_json(request, signature)
            alias = ROOT / 'logs/rsl_rl/platoon_happo' / f'{tag}_parent'
            alias.mkdir(exist_ok=True)
            link = alias / 'model_final.pt'
            if link.exists() or link.is_symlink():
                if link.resolve() != parent.resolve():
                    raise ValueError('Parent alias conflict')
            else:
                link.symlink_to(parent.resolve())
            resume = stage / 'resume_runs.txt'
            resume.write_text(f'{label}={alias}\n')
            environment = dict(self.environment, TAG=tag, PIPELINE_LOG=str(stage / 'training_driver.log'),
                RESULT_ROOT=str(stage), RESULT_TAR=str(stage.with_suffix('.tar.gz')),
                PROGRESS_NOTES_PATH=str(ROOT.parent / 'ams_debug_notes.md'),
                NUM_ENVS='32', EVAL_NUM_ENVS='16', SEED=str(signature['seed']),
                MAX_ITERATIONS=str(updates), EVAL_EVERY=str(updates), EVAL_STEPS='3500',
                SKIP_EVAL='1', FRESH_ENV_PER_CHECKPOINT='1', PARALLEL_TRAIN_JOBS='1',
                COMPARE_ALGOS=label, RESUME_RUNS_FILE=str(resume), METHOD_VARIANT='legacy',
                ENABLE_ATTACK=str(level[0] != 'off').lower(), ATTACK_PROFILE_LABEL=level[0],
                ATTACK_LEVEL=level[1], ATTACK_MODE='profile', ATTACK_MAX_FDI_POS=str(level[2]),
                ATTACK_MAX_FDI_ACC=str(level[3]), ATTACK_MAX_DOS_RATE=str(level[4]),
                SAFETY_CATCHUP_ACTION='0.4', COMMON_EXTRA_OVERRIDES=' '.join(extra),
                HAPPO_META_EXTRA_OVERRIDES='', HAPPO_META_TRAIN_EXTRA_OVERRIDES=' '.join(extra),
                HAPPO_META_EVAL_EXTRA_OVERRIDES='', PLOT_PYTHON=self.args.host_python)
            self.status('training', algorithm=label, seed=signature['seed'], level=level[0],
                        parent=str(parent), updates=updates, output=str(stage))
            self.run(['bash', 'scripts/tools/run_medium_algorithm_comparison.sh'],
                     self.output / 'training_process.log', environment, timeout=14400)
            runs = dict(line.split('=', 1) for line in (stage / 'training/runs.txt').read_text().splitlines())
            candidate = Path(runs[label]) / 'model_final.pt'
        audit = audit_new_training(candidate.parent, parent, level, updates, label, extra, self.args.bc_checkpoint)
        (stage / 'training').mkdir(exist_ok=True)
        (stage / 'training/runs.txt').write_text(f'{label}={candidate.parent}\n')
        write_json(marker, dict(signature=signature, audit=audit))
        note(f'- 统一对照s{self.args.seed_index}/{label}/{level[0]}完成{updates}次真实更新；'
             f'方法开关、BC参考、有效攻击及学习率审计通过；`{stage}`。')
        return candidate


def summarize_training_seeds(root):
    """Training-seed means are the replicates; do not count nine evals as nine fits."""
    tables = []
    for index in range(3):
        with (root / f's{index}/six_algorithm_attack_matrix.csv').open() as handle:
            table = {(row['algorithm'], row['attack_level']): row for row in csv.DictReader(handle)}
        expected = {(method, level[0]) for method in METHODS for level in LEVELS}
        if set(table) != expected:
            raise ValueError('Incomplete six-method seed matrix')
        tables.append(table)
    rows = []
    for method in METHODS:
        for level in LEVELS:
            records = [table[method, level[0]] for table in tables]
            means = [float(row['whole_platoon_completion_rate']) for row in records]
            rows.append(dict(method=method, level=level[0], training_seed_count=3,
                completion_mean=statistics.mean(means),
                completion_training_seed_sample_std=statistics.stdev(means),
                completion_worst_evaluation=min(float(row['completion_seed_min']) for row in records),
                completion_s0=means[0], completion_s1=means[1], completion_s2=means[2],
                collision_max=max(float(row['collision_rate']) for row in records),
                road_boundary_max=max(float(row['road_boundary_violation_rate']) for row in records),
                passed_seed_count=sum(row['gate_passed'] == 'True' for row in records),
                seed_scope='adaptation seeds conditional on shared BC fit'))
    with (root / 'three_training_seed_summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--bc-checkpoint', type=Path,
        default=ROOT / 'logs/ackermann_bc_warmstart/real_trajectory_bc/model_bc.pt')
    parser.add_argument('--stage-updates', type=int, default=40)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.stage_updates <= 0:
        parser.error('stage-updates must be positive')
    args.sim = ROOT.parent / 'isaac-sim'
    args.host_python = '/home/cnc/Downloads/ENTER/bin/python3'
    args.eval_envs, args.reuse_root = 16, None
    with (ROOT / 'logs/ams_ackermann_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = CompletionQueue(args)
        matrices = [UniformMatrix(queue, i) for i in range(3)]
        if args.preflight_only:
            for i in range(3):
                queue.import_reference(i)
            queue.status('preflight_passed', imported_evaluations=12, imported_evidence_sets=6,
                         pending_evaluations=63, pending_evidence_sets=21,
                         queued_training='six methods x three adaptation seeds, common BC initialization')
            return
        note(f'- 启动最终模型回归→统一六算法三种子队列；复用12份严格评估和6套视频，'
             f'新增63份评估/21套视频后进入全新统一BC起点训练；`{queue.output}`。')
        try:
            queue.regressions()
            for index, matrix in enumerate(matrices):
                queue.status('uniform_matrix', seed_index=index, output=str(matrix.output))
                matrix.matrix()
                matrix.status('complete', seed_index=index)
            summarize_training_seeds(queue.output / 'uniform_six')
            queue.status('complete')
            note(f'- 回归及统一六算法三种子队列完成；`{queue.output}`。')
        except BaseException as exc:
            queue.status('stopped_at_error', error=repr(exc), previous=queue.current)
            note(f'- 补齐队列停止，已完成训练/评估保留且不盲目重跑：{exc!r}；`{queue.output}`。')
            raise


if __name__ == '__main__':
    main()
