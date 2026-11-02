#!/usr/bin/env python3
"""Build a reproducible paper supplement from the completed, gated six-method cohort.

Collection reads original artifacts only. Rebuild mode needs just the exported
CSV files, protocol, and representative rollout, not Isaac Sim or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ['mappo', 'happo_no_meta', 'happo_meta_ams', 'harl_mappo_shared', 'harl_haa2c', 'harl_hatrpo']
LABELS = dict(zip(METHODS, ['MAPPO', 'HAPPO (no Meta)', 'Meta-AMS', 'Shared MAPPO', 'HAA2C', 'HATRPO']))
LEVELS = ['off', 'light', 'easy_0', 'easy_a', 'easy_b', 'med_a', 'med_b', 'hard_a', 'hard_b']
PAIRS = [(42, 3407), (3407, 9917), (777, 17011)]
COLORS = ['#4477aa', '#ee6677', '#228833', '#ccbb44', '#66ccee', '#aa3377']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_evidence(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise ValueError(f'Refuse to replace different evidence: {target}')
    else:
        shutil.copy2(source, target)


def exact_sign_flip(differences):
    """Two-sided |mean difference| test, conditional on paired sign exchangeability."""
    differences = tuple(float(v) for v in differences)
    if not differences or any(not math.isfinite(v) for v in differences):
        raise ValueError('Expected finite paired differences')
    observed = statistics.mean(differences)
    distribution = []
    for signs in itertools.product((-1, 1), repeat=len(differences)):
        value = statistics.mean(s * d for s, d in zip(signs, differences))
        distribution.append(dict(signs=','.join(map(str, signs)), statistic=value,
                                 extreme=abs(value) >= abs(observed) - 1e-12))
    return observed, sum(r['extreme'] for r in distribution) / len(distribution), distribution


def holm_adjust(pvalues):
    if any(not 0 <= p <= 1 for p in pvalues):
        raise ValueError('Invalid p value')
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, previous = [0.] * len(pvalues), 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(order) - rank) * pvalues[i]))
        adjusted[i] = previous
    return adjusted


def collect(source, out):
    import sys
    sys.path.insert(0, str(source.parent.parent / 'scripts/tools'))
    from ackermann_validation import load_verified
    protocol = json.loads((source / 'queue_manifest.json').read_text())
    expected_sources = {k:v for k,v in protocol['sources'].items()
        if k.startswith('source/my_exts/marl_platoon/') or k in (
            'scripts/tools/ackermann_validation.py', 'scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py')}
    attack_levels = {item[0]:item for item in protocol['levels']}
    assert json.loads((source / 'followup_status.json').read_text())['state'] == 'complete'
    for name, expected in protocol['sources'].items():
        origin = source / 'source_snapshot' / name
        if sha(origin) != expected:
            raise ValueError(f'Source snapshot changed: {origin}')
        copy_evidence(origin, out / 'evidence/source_snapshot' / name)
    copy_evidence(source / 'queue_manifest.json', out / 'data/queue_manifest.json')
    inputs = {str(source / 'queue_manifest.json'): sha(source / 'queue_manifest.json')}
    evaluations, stages, selections, rewards = [], [], [], []
    for index in range(3):
        branch = source / f'uniform_six/s{index}'
        outcomes_path = branch / 'method_outcomes.json'
        outcomes = json.loads(outcomes_path.read_text())
        copy_evidence(outcomes_path, out / f'data/original/s{index}_outcomes.json')
        copy_evidence(branch / 'six_algorithm_attack_matrix.csv', out / f'data/original/s{index}_matrix.csv')
        for method in METHODS:
            outcome = outcomes[method]
            total_updates, accepted_updates, interactions = 0, 0, 0
            for level_index, level in enumerate(LEVELS[1:], 1):
                stage = branch / f'stages/{method}/{level}_r1'
                marker = stage / 'training_complete.json'
                if not marker.exists():
                    continue
                recorded = json.loads(marker.read_text())
                audit, sig = recorded['audit'], recorded['signature']
                run = Path(audit['checkpoint']).parent
                if sha(run / 'model_final.pt') != audit['checkpoint_sha256'] or sha(run / 'platoon_metrics.csv') != audit['metrics_sha256']:
                    raise ValueError(f'Training artifact changed: {run}')
                gate = json.loads((stage / 'validation_gate.json').read_text())
                with (run / 'platoon_metrics.csv').open() as f:
                    raw_rewards = list(csv.DictReader(f))
                assert len(raw_rewards) == sig['updates'] == audit['updates_after'] - audit['updates_before']
                assert [int(row['env_step']) for row in raw_rewards] == list(range(32, 32 * (len(raw_rewards) + 1), 32))
                assert all(int(row['num_envs']) == 32 for row in raw_rewards)
                block_interactions = int(raw_rewards[-1]['env_step']) * 32
                for j, row in enumerate(raw_rewards, 1):
                    rewards.append(dict(seed_index=index, train_seed=PAIRS[index][0], method=method,
                        level=level, stage_update=j, attempted_update=total_updates+j,
                        checkpoint_update=int(row['update']), reward_env_mean=float(row['reward_env_mean']),
                        bc_loss=float(row['bc_loss']), reference_kl=float(row['reference_kl']),
                        ams_weight=float(row['ams_advantage_weight'])))
                total_updates += sig['updates']
                interactions += block_interactions
                accepted_updates += sig['updates'] if gate['passed'] else 0
                stages.append(dict(seed_index=index, train_seed=PAIRS[index][0], method=method, level=level,
                    updates=sig['updates'], environment_transitions=block_interactions,
                    clean_ok=gate['clean_ok'], gate_passed=gate['passed'],
                    candidate_sha256=audit['checkpoint_sha256'], parent_sha256=audit['parent_sha256'],
                    attack_completion_min=gate.get('attack_completion_min', 0),
                    source=str(stage), training_marker_sha256=sha(marker)))
                for filename in ('training_complete.json', 'validation_gate.json'):
                    copy_evidence(stage / filename, out / f'evidence/training/s{index}/{method}/{level}' / filename)
                inputs[str(marker)] = sha(marker)
            fallback = outcome['checkpoint_sha256'] == protocol['matrix']['bc_sha256']
            retained = 'BC' if fallback else LEVELS[accepted_updates // 40]
            selections.append(dict(seed_index=index, train_seed=PAIRS[index][0], method=method,
                attempted_updates=total_updates, retained_updates=accepted_updates,
                attempted_environment_transitions=interactions, retained_level=retained,
                stopped_level=outcome['stopped_level'] or '', training_status=outcome['status'],
                bc_fallback=fallback, checkpoint=outcome['checkpoint'], checkpoint_sha256=outcome['checkpoint_sha256']))
            ladder = branch / 'final_ladders' / method
            assert (ladder / 'ladder_complete.json').exists()
            for level in LEVELS:
                names = ['clean_env42'] if level == 'off' else [f'attack_env{e}_seed{a}' for e, a in PAIRS]
                assessment = json.loads((ladder / level / 'assessment.json').read_text())
                for name in names:
                    origin = ladder / level / 'validation' / name
                    row, spec = load_verified(origin)
                    assert spec['checkpoint_sha256'] == outcome['checkpoint_sha256']
                    assert spec['sources'] == expected_sources
                    assert (spec['steps'], spec['num_envs']) == (3500, 16)
                    assert (spec['level'], spec['pos'], spec['acc'], spec['dos']) == tuple(attack_levels[level][1:5])
                    assert (spec['env_seed'], spec['attack_seed']) in PAIRS
                    assert name == ('clean_env42' if level == 'off' else f"attack_env{spec['env_seed']}_seed{spec['attack_seed']}")
                    target = out / f'evidence/evaluations/s{index}/{method}/{level}/{name}'
                    for filename in ('verified.json', 'eval_summary.csv'):
                        copy_evidence(origin / filename, target / filename)
                    evaluations.append(dict(seed_index=index, train_seed=PAIRS[index][0], method=method,
                        level=level, env_seed=spec['env_seed'], attack_seed=spec['attack_seed'],
                        completion=row['completion'], instant_peak=row['instant'], dwell_peak=row['dwell'],
                        collision=row['collision'], road_any=row['road_any'], road_termination=row['road_termination'],
                        checkpoint_sha256=spec['checkpoint_sha256'], gate_passed=assessment['passed'],
                        source=str(origin), verified_sha256=sha(origin / 'verified.json')))
                    inputs[str(origin / 'verified.json')] = sha(origin / 'verified.json')
            print(f'Collected s{index}/{method}', flush=True)
    assert len(evaluations) == 450 and len(stages) == 104 and len(selections) == 18 and len(rewards) == 4160
    for name, data in [('evaluations', evaluations), ('training_stages', stages), ('selection_outcomes', selections), ('training_rewards', rewards)]:
        write_csv(out / f'data/{name}.csv', data)
    representative = source / 'uniform_six/s0/final_ladders/happo_meta_ams/hard_b'
    render = json.loads((representative / 'evidence_complete.json').read_text())
    for name in ('trajectory_env0.csv', 'topdown_true_xy.png', 'video/rl-video-step-0.mp4'):
        assert sha(representative / name) == render['artifacts'][name]
        copy_evidence(representative / name, out / 'evidence/representative_s0_hard_b' / name)
    copy_evidence(representative / 'evidence_complete.json', out / 'evidence/representative_s0_hard_b/evidence_complete.json')
    dump(out / 'data/input_manifest.json', dict(source_root=str(source), input_hashes=inputs,
         source_snapshot_hashes=protocol['sources'], representative_rule='Fixed s0 hard-b render; no best-rollout selection',
         included_raw_scope='All 450 eval summaries and verification manifests; full per-step originals remain at source paths'))


def analyze(out):
    frame = pd.read_csv(out / 'data/evaluations.csv')
    selections = pd.read_csv(out / 'data/selection_outcomes.csv').fillna('')
    seed_means, rows = {}, []
    for method in METHODS:
        for level in LEVELS:
            values, worst, passes = [], [], []
            for index in range(3):
                subset = frame[(frame.method == method) & (frame.level == level) & (frame.seed_index == index)]
                assert len(subset) == (1 if level == 'off' else 3)
                assert subset.checkpoint_sha256.nunique() == 1
                mean = float(subset.completion.mean())
                seed_means[index, method, level] = mean
                values.append(mean); worst.append(float(subset.completion.min()))
                assert subset.gate_passed.nunique() == 1
                passes.append(bool(subset.gate_passed.iloc[0]))
            chosen = selections[selections.method == method]
            rows.append(dict(method=method, level=level, mean=statistics.mean(values),
                training_seed_sample_sd=statistics.stdev(values), worst_evaluation=min(worst),
                passed_seed_count=sum(passes), s0=values[0], s1=values[1], s2=values[2],
                unique_retained_checkpoints=int(chosen.checkpoint_sha256.nunique()),
                bc_fallback_count=int(chosen.bc_fallback.sum())))
    write_csv(out / 'data/full_ladder_summary.csv', rows)
    tests, permutations = [], []
    for method in METHODS:
        if method == 'happo_meta_ams':
            continue
        differences = [100 * (seed_means[i, 'happo_meta_ams', 'hard_b'] - seed_means[i, method, 'hard_b']) for i in range(3)]
        delta, p, distribution = exact_sign_flip(differences)
        tests.append(dict(comparator=method, matched_training_seeds=3, mean_difference_pp=delta,
            difference_sample_sd_pp=statistics.stdev(differences), difference_s0_pp=differences[0],
            difference_s1_pp=differences[1], difference_s2_pp=differences[2], exact_two_sided_p=p,
            all_pairs_positive=all(v > 0 for v in differences), permutations=len(distribution)))
        permutations.extend(dict(comparator=method, **item) for item in distribution)
    for item, corrected in zip(tests, holm_adjust([t['exact_two_sided_p'] for t in tests])):
        item['holm_adjusted_p'] = corrected
        item['reject_at_0_05'] = corrected < .05
    write_csv(out / 'data/hard_b_paired_comparisons.csv', tests)
    write_csv(out / 'data/sign_flip_distribution.csv', permutations)
    metrics = dict(training_seed_count=3, eval_pairs_per_attack=3, completed_training_blocks=104,
        total_updates=int(selections.attempted_updates.sum()),
        total_environment_transitions=int(selections.attempted_environment_transitions.sum()),
        safety_max={k:float(frame[k].max()) for k in ['collision', 'road_any', 'road_termination']},
        hard_b=[r for r in rows if r['level']=='hard_b'], tests=tests,
        inference='Post-hoc, conditional on a shared BC fit and paired label/sign exchangeability; pipeline outcomes including fallback',
        not_independent_samples=['evaluation seed repetitions', 'individual vehicles', 'time steps', 'copies of the same retained BC checkpoint'])
    dump(out / 'data/analysis_results.json', metrics)
    return rows, tests, selections, seed_means


def tex_escape(value):
    return str(value).replace('_', r'\_').replace('%', r'\%')


def table(path, caption, label, columns, header, body, note='', wide=False):
    env = 'table*' if wide else 'table'
    text = '\\begin{'+env+'}[t]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
    text += '\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
    text += ' & '.join(header) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + r' \\' for row in body)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    if note:
        text += '\n\\vspace{2pt}\\begin{minipage}{\\linewidth}\\footnotesize '+note+'\\end{minipage}\n'
    text += '\\end{'+env+'}\n'
    path.write_text(text)


def make_tables(out, rows, tests, selections):
    directory=out/'tables'; directory.mkdir(exist_ok=True)
    hard = {r['method']:r for r in rows if r['level']=='hard_b'}
    table(directory/'hard_b.tex', 'Final retained policies at hard-b: completion in percent.', 'tab:hardb', 'lrrr',
          ['Method', 'Mean $\\pm$ SD', 'Worst', 'Pass'],
          [[LABELS[m], f"{100*hard[m]['mean']:.2f} $\\pm$ {100*hard[m]['training_seed_sample_sd']:.2f}",
            f"{100*hard[m]['worst_evaluation']:.2f}",f"{hard[m]['passed_seed_count']}/3"] for m in METHODS],
          'SD is across three training-seed means, not nine independent fits. Pass counts final evaluation gates. HATRPO retains the common BC model in all three runs; shared MAPPO does so in two runs.')
    lookup={(r['method'],r['level']):r for r in rows}
    table(directory/'all_levels.tex', 'Mean whole-platoon completion (percent) across the complete ladder.', 'tab:ladder', 'l'+'r'*9,
          ['Method', *[tex_escape(l) for l in LEVELS]],
          [[LABELS[m], *[f"{100*lookup[m,l]['mean']:.2f}" for l in LEVELS]] for m in METHODS],
          'Each attack cell averages three training-seed means, each obtained from three fixed environment/attack seed pairs. The clean cell uses one evaluation pair per training seed.', wide=True)
    body=[]
    for m in METHODS:
        chosen=selections[selections.method==m].sort_values('seed_index')
        body.append([LABELS[m], *[f"{int(r.attempted_updates)}/{int(r.retained_updates)}" for r in chosen.itertuples()],
                     '/'.join(tex_escape(r.retained_level) for r in chosen.itertuples()),int(chosen.bc_fallback.sum())])
    table(directory/'budgets.tex', 'Realized training budgets and retained stages.', 'tab:budget', 'lrrrlr',
          ['Method','s0 A/R','s1 A/R','s2 A/R','Retained stage s0/s1/s2','BC'],body,
          'A: attempted updates, including rejected candidates; R: updates on the retained lineage. BC: number of runs retaining the common warm start. Each block uses 40 updates, 32 environments and 32 rollout steps per update. Equal per-stage caps do not imply equal realized budgets or equal wall-clock computation.',wide=True)
    table(directory/'statistics.tex','Exploratory paired hard-b comparisons: Meta-AMS minus comparator.', 'tab:tests','lrrr',
          ['Comparator', '$\\Delta$ (pp)', 'Exact $p$', 'Holm $p$'],
          [[LABELS[t['comparator']],f"{t['mean_difference_pp']:.2f}",f"{t['exact_two_sided_p']:.2f}",f"{t['holm_adjusted_p']:.2f}"] for t in tests],
          'Two-sided exact sign-flip tests enumerate all eight sign assignments of three matched seed differences. Five post-hoc comparisons form one Holm family. This tests the gated procedure conditional on exchangeability; it does not isolate an AMS module effect.')
    protocol=json.loads((out/'data/queue_manifest.json').read_text())
    body=[]
    for m in METHODS:
        _, teacher, factor, shared, mode, ams=protocol['matrix']['methods'][m]
        body.append([LABELS[m],int(teacher),int(ams),int(factor),int(shared),mode.upper()])
    table(directory/'methods.tex','Method switches in the executed repository implementations.', 'tab:methods','lrrrrl',
          ['Method','Meta','AMS','Factor','Shared','Update'],body,
          'These are the repository implementations selected by the recorded flags, not a claim of unmodified external HARL code. Architecture/sharing differs between some rows.')


def save_figure(fig, directory, stem):
    fig.savefig(directory/(stem+'.pdf'), bbox_inches='tight')
    fig.savefig(directory/(stem+'.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)


def make_figures(out, rows, selections, seed_means):
    directory=out/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    lookup={(r['method'],r['level']):r for r in rows}
    for field, title, stem in [('mean','Mean completion (%)','completion_mean'),('worst_evaluation','Worst evaluation completion (%)','completion_worst')]:
        matrix=np.array([[100*lookup[m,l][field] for l in LEVELS] for m in METHODS])
        fig,ax=plt.subplots(figsize=(10,4))
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        ax.set(xticks=range(9),xticklabels=LEVELS,yticks=range(6),yticklabels=[LABELS[m] for m in METHODS],title=title)
        for i in range(6):
            for j in range(9):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='black' if matrix[i,j]>60 else 'white',fontsize=9)
        fig.colorbar(im,ax=ax,label='%');fig.tight_layout();save_figure(fig,directory,stem)
    fig,ax=plt.subplots(figsize=(8.8,4.3))
    for i,m in enumerate(METHODS):
        values=np.array([100*seed_means[s,m,'hard_b'] for s in range(3)])
        ax.bar(i,values.mean(),color=COLORS[i],alpha=.22,width=.65)
        ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='_',capsize=5,color='black')
        for s,offset in enumerate([-.16,0,.16]):
            fallback=bool(selections[(selections.method==m)&(selections.seed_index==s)].bc_fallback.iloc[0])
            ax.scatter(i+offset,values[s],marker='x' if fallback else ['o','s','^'][s],color=COLORS[i],s=48,zorder=5)
    ax.set(xticks=range(6),xticklabels=[LABELS[m] for m in METHODS],ylabel='Hard-b completion (%)',ylim=(-5,105),title='Three training-seed means; bars = mean, whiskers = sample SD')
    ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.2)
    fig.text(.12,.01,'Crosses denote retention of the common BC checkpoint; error bars are not confidence intervals.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save_figure(fig,directory,'hard_b_training_seeds')
    fig,ax=plt.subplots(figsize=(8.8,4.4))
    for s,offset in enumerate([-.24,0,.24]):
        chosen=selections[selections.seed_index==s].set_index('method').loc[METHODS]
        positions=np.arange(6)+offset
        ax.barh(positions,chosen.attempted_updates,height=.22,color=COLORS[s],alpha=.35,label=f's{s} attempted')
        ax.barh(positions,chosen.retained_updates,height=.22,facecolor='none',edgecolor=COLORS[s],hatch='///')
    ax.set(yticks=range(6),yticklabels=[LABELS[m] for m in METHODS],xlabel='Policy updates',xlim=(0,340),title='Attempted budget (solid) and retained lineage (hatched)')
    ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8);ax.grid(axis='x',alpha=.2);fig.tight_layout();save_figure(fig,directory,'training_budgets')
    rewards=pd.read_csv(out/'data/training_rewards.csv')
    fig,ax=plt.subplots(figsize=(9,3.5))
    for s in range(3):
        subset=rewards[(rewards.method=='happo_meta_ams')&(rewards.seed_index==s)].sort_values('attempted_update')
        ax.plot(subset.attempted_update,subset.reward_env_mean,color=COLORS[s],alpha=.15,lw=.6)
        for j,level in enumerate(LEVELS[1:]):
            stage=subset[subset.level==level]
            ax.plot(stage.attempted_update,stage.reward_env_mean.rolling(5,min_periods=1).mean(),color=COLORS[s],label=f's{s}' if j==0 else None)
    for j,level in enumerate(LEVELS[1:]):
        ax.axvline(j*40,color='gray',lw=.5,alpha=.4)
        ax.text(j*40+20,1.01,level,transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
    ax.set(xlabel='Cumulative attempted updates',ylabel='Recorded environment reward',xlim=(0,320))
    ax.legend(ncol=3);ax.grid(alpha=.2);fig.tight_layout();save_figure(fig,directory,'meta_ams_training_reward')
    trajectory=pd.read_csv(out/'evidence/representative_s0_hard_b/trajectory_env0.csv')
    assert len(trajectory)==17500 and trajectory.robot_id.nunique()==5
    fig,axes=plt.subplots(3,1,figsize=(8,7),gridspec_kw={'height_ratios':[1.2,1,1]})
    x=np.linspace(float(trajectory.x.min()),float(trajectory.x.max()),500)
    z=np.clip((x+7.75)/6.,0,1);y=-.065+(-1.160+.065)*(10*z**3-15*z**4+6*z**5)
    axes[0].plot(x,y,'k--',label='Shared spatial reference',lw=1.5)
    for robot in range(1,6):
        vehicle=trajectory[trajectory.robot_id==robot].sort_values('step')
        # Omit only the final simulator reset when drawing a continuous path.
        jumps=np.flatnonzero(np.abs(np.diff(vehicle.x.to_numpy()))>2)
        if len(jumps):vehicle=vehicle.iloc[:jumps[0]+1]
        color=COLORS[robot-1]
        axes[0].plot(vehicle.x,vehicle.y,color=color,lw=1,label=f'Vehicle {robot}')
        axes[0].scatter(vehicle.x.iloc[0],vehicle.y.iloc[0],color=color,s=18)
        axes[1].plot(vehicle.sim_time_s,vehicle.reference_progress,color=color)
        axes[2].plot(vehicle.sim_time_s,vehicle.y,color=color)
    axes[0].set(xlabel='World x (m)',ylabel='World y (m)')
    axes[0].legend(ncol=3,fontsize=8)
    axes[1].axhline(.99,color='black',ls=':',lw=1)
    axes[1].set(xlabel='Time (s)',ylabel='Path progress',ylim=(-.04,1.08))
    axes[2].axhline(-1.160,color='black',ls=':',lw=1)
    axes[2].set(xlabel='Time (s)',ylabel='World y (m)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Recorded s0 / hard-b rendered rollout: five vehicles, one spatial path',fontsize=11)
    fig.tight_layout();save_figure(fig,directory,'five_vehicle_lane_change')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,help='Completed queue root; omit with --rebuild')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.rebuild:
        if args.source:parser.error('Use --source or --rebuild, not both')
    elif args.source:
        collect(args.source.resolve(),out)
    else:parser.error('--source is required for initial collection')
    rows,tests,selections,means=analyze(out)
    make_tables(out,rows,tests,selections)
    make_figures(out,rows,selections,means)
    (out/'scripts').mkdir(exist_ok=True)
    target=out/'scripts'/Path(__file__).name
    if target.resolve()!=Path(__file__).resolve():shutil.copy2(__file__,target)
    dump(out/'data/generated_artifacts.json',{str(p.relative_to(out)):sha(p) for directory in ['data','tables','figures']
        for p in (out/directory).rglob('*') if p.is_file() and p.name!='generated_artifacts.json'})
    print(json.dumps(dict(output=str(out),comparisons=tests),indent=2),flush=True)


if __name__=='__main__':
    main()
