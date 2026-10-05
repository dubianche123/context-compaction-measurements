"""Agent actions in the first three accepted steps after each summary is first used.

Covers finished GLM S/A/K runs and the DeepSeek S/A pairs. Read-only: no model,
tool or provider calls. Each run's L0 is opened with the SessionLog reader of the
runtime that wrote it, in its own subprocess. Accepted steps and summary
generations come from the saved native summary-consumption record. The first
tool call of every accepted step is classified by the frozen v2 first-action
classifier (preparations/skeep-20261001/branch_classifier_v2.py), the same rule
used for the branch experiments.

Run from the repository root:
  .venv/bin/python results/harness-diagnosis-20260927/post-switch-behavior-20261004/analyze.py [--earlier CACHE]
Only this directory is written. Command text is not saved. Every run's step labels (categories only) are also
written to run-labels-cache.json; for a run whose L0 is no longer on disk (the first repetition's logs are
archived, monitoring/disk-cleanup-20261004/RESTORE.md), --earlier reuses its labels from such a cache.
"""
from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
from pathlib import Path
import random
import re
import statistics as st
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GLM_RT = 'results/glm-repetition-tiered-20260930/work/runtime'
KEEP_RT = 'results/skeep-20261001/work/runtime'
JOBS = ROOT / GLM_RT / 'output/harbor-jobs'
LOOKBACK = 20  # reread window, in accepted steps


def discover():
    runs = []

    def official(path):
        o = json.loads(Path(path).read_text())
        vr = o.get('verifier_result')
        return None if not vr else (vr.get('rewards') or {}).get('reward')

    def add(model, policy, task, thr, rep, runtime, native, off):
        reward = official(off)
        if reward is not None:
            runs.append(dict(model=model, policy=policy, task=task, thr=thr, rep=rep, runtime=runtime,
                             private=str(Path(native).parent), reward=reward))

    for d in glob.glob(f'{JOBS}/glm-window-sweep-20260927/main/long-011-formal-summary12288/'
                       'window-main-long-011-formal-summary12288-*-b00[123]-[sa]*'):
        m = re.search(r'summary12288-(.+)-(b00\d)-([sa])(\d+)$', d)
        nat, off = glob.glob(d + '/*/axiom-private/result.json'), glob.glob(d + '/*/result.json')
        if m and nat and off:
            t, b, a, w = m.groups()
            add('glm', a.upper(), t, int(w), b, GLM_RT, nat[0], off[0])
    cells = json.loads((ROOT / 'monitoring/glm-skeep-endpoint-summary-20261003.json').read_text())['cells']
    for c in cells:
        t, b, w = re.search(r'skeep-001-(.+)-(b00\d)-k(\d+)', c['job']).groups()
        add('glm', 'K', t, int(w), b, KEEP_RT, c['native_result'], c['official_result'])
    for d in glob.glob(f'{JOBS}/deepseek-flash-window-sweep-20260928/main/long-011-formal-summary12288/'
                       'deepseek-window-main-long-011-formal-summary12288-*-b001-[sa]*'):
        t, a, w = re.search(r'summary12288-(.+)-b001-([sa])(\d+)$', d).groups()
        if t == 'coq-block-bound':
            continue  # branch source only, no A partner
        nat, off = glob.glob(d + '/*/axiom-private/result.json'), glob.glob(d + '/*/result.json')
        if nat and off:
            add('deepseek', a.upper(), t, int(w), 'b001', GLM_RT, nat[0], off[0])
    return sorted(runs, key=lambda r: (r['model'], r['policy'], r['task'], r['thr'], r['rep']))


def extract(spec):
    """Accepted steps in order, each with its accepted assistant message (first tool only)."""
    sys.path.insert(0, str(ROOT / spec['runtime']))
    from axiom_shortloop.session_log import SessionLog

    private = Path(spec['private'])
    path = private / 'summary-consumption.json'
    consumption = (json.loads(path.read_text()) if path.exists()
                   else json.loads((private / 'result.json').read_text())['summary_consumption'])
    lp = private / 'l0/session.jsonl'
    with lp.open() as stream:
        header = json.loads(stream.readline())
    log = SessionLog.open(lp.parent, session_id=header['session_id'])
    if log.has_truncated_tail:
        raise ValueError(f'truncated L0: {lp}')
    purpose, step_of, messages, appended = {}, {}, {}, {}
    last, orphans, duplicates = None, 0, 0
    # Native acceptance appends the assistant message after its completed attempt.
    for event in log.events:
        kind, data = event['type'], event['data']
        if kind == 'model/attempt' and data.get('phase') == 'started':
            purpose[data['attempt_id']] = data.get('purpose')
            step_of[data['attempt_id']] = data.get('step_id')
        elif kind == 'model/attempt' and data.get('phase') == 'terminal':
            if purpose.get(data['attempt_id']) == 'conversation' and data.get('status') == 'completed':
                last = step_of[data['attempt_id']]
        elif kind == 'message/append':
            message = data.get('message') or {}
            if message.get('role') == 'assistant' and (data.get('source') or {}).get('kind') == 'assistant':
                if last is None:
                    orphans += 1
                    continue
                duplicates += last in messages
                messages[last] = message
                appended[last] = event['seq']
                last = None
    # accepted_step_ids are stored in string order (":100:" before ":97:"); order
    # steps by the L0 sequence of their accepted assistant message instead.
    order = lambda step_id: appended.get(step_id, float('inf'))
    cycles = sorted(consumption['foreground_cycles']['cycles'], key=lambda c: c['generation'])
    flat = [s for c in cycles for s in sorted(c['accepted_step_ids'], key=order)]
    if [order(s) for s in flat] != sorted(order(s) for s in flat):
        raise ValueError('accepted steps are not chronological across generations')
    steps = []
    for cycle in cycles:
        for position, step_id in enumerate(sorted(cycle['accepted_step_ids'], key=order), 1):
            message = messages.get(step_id)
            calls = [c for c in ((message or {}).get('tool_calls') or []) if isinstance(c, dict)]
            commands = []
            for call in calls:
                function = call.get('function') if isinstance(call.get('function'), dict) else {}
                commands.append([function.get('name'), function.get('arguments')])
            steps.append(dict(generation=cycle['generation'], position=position,
                              cycle_steps=len(cycle['accepted_step_ids']), has_message=message is not None,
                              first=None if message is None else dict(role='assistant', tool_calls=calls[:1]),
                              commands=commands))
    return dict(steps=steps, orphans=orphans, duplicates=duplicates,
                missing=sum(not s['has_message'] for s in steps))


def classifier():
    spec = importlib.util.spec_from_file_location(
        'branch_classifier_v2', ROOT / 'preparations/skeep-20261001/branch_classifier_v2.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def label_steps(run, extracted, v2):
    """Category of each step's first tool call, and whether it rereads known paths."""
    def classify(name, arguments):
        call = dict(type='function', function=dict(name=name, arguments=arguments))
        return v2.classify_action(dict(role='assistant', tool_calls=[call]))

    rows, history = [], []
    for step in extracted['steps']:
        category = v2.classify_action(step['first'])['category'] if step['has_message'] else 'missing'
        labels = [classify(*c) for c in step['commands']]
        paths = labels[0].get('static_paths') if labels else None
        recent = {p for s in history[-LOOKBACK:] for p in s}
        rows.append(dict(generation=step['generation'], position=step['position'],
                         cycle_steps=step['cycle_steps'], category=category,
                         reread=bool(category == 'inspect_only' and paths and set(paths) <= recent)))
        history.append({p for label in labels for p in (label.get('static_paths') or [])})
    return rows


READ = {'inspect_only', 'evidence_read'}


def run_measures(rows):
    events = [i for i, r in enumerate(rows) if r['generation'] > 0 and r['position'] == 1]
    base = [r for r in rows if r['position'] >= 4]
    m = dict(events=len(events), baseline_steps=len(base))
    if not events:
        return m
    first = [rows[i] for i in events]
    for c in ('inspect_only', 'execution', 'work', 'evidence_read', 'poll', 'no_tool', 'unknown'):
        m['step1_' + c] = sum(r['category'] == c for r in first) / len(first)
        if base:
            m['base_' + c] = sum(r['category'] == c for r in base) / len(base)
    m['step1_reread'] = sum(r['reread'] for r in first) / len(first)
    if base:
        m['base_reread'] = sum(r['reread'] for r in base) / len(base)
    full = [rows[i:i + 3] for i in events if rows[i]['cycle_steps'] >= 3]
    m['events_with_three_steps'] = len(full)
    if full:
        m['three_any_work'] = sum(any(r['category'] == 'work' for r in w) for w in full) / len(full)
        m['three_all_read'] = sum(all(r['category'] in READ for r in w) for w in full) / len(full)
        m['three_any_evidence'] = sum(any(r['category'] == 'evidence_read' for r in w) for w in full) / len(full)
        m['three_read_steps'] = sum(sum(r['category'] in READ for r in w) for w in full) / len(full)
    return m


def bootstrap(values, seed=20261004, draws=10000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(values) for _ in values) for _ in range(draws))
    return [round(means[int(0.025 * draws)], 3), round(means[int(0.975 * draws) - 1], 3)]


def summarize(runs):
    keys = ['step1_inspect_only', 'step1_execution', 'step1_work', 'step1_evidence_read', 'step1_reread',
            'base_inspect_only', 'base_execution', 'base_work', 'base_evidence_read', 'base_reread',
            'three_any_work', 'three_all_read', 'three_any_evidence', 'three_read_steps']
    groups = collections.defaultdict(list)
    for r in runs:
        if r['measures']['events']:
            groups[(r['model'], r['policy'], r['thr'])].append(r)
            groups[(r['model'], r['policy'], 'all')].append(r)
    table = {}
    for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(runs=len(rs), events=sum(r['measures']['events'] for r in rs))
        for k in keys:
            vals = [r['measures'][k] for r in rs if k in r['measures']]
            row[k] = round(st.mean(vals), 3) if vals else None
        table['/'.join(map(str, g))] = row
    # Paired S-A differences (same model, task, threshold, repetition; both with events)
    pairs = collections.defaultdict(dict)
    for r in runs:
        if r['policy'] in 'SA' and r['measures']['events']:
            pairs[(r['model'], r['task'], r['thr'], r['rep'])][r['policy']] = r['measures']
    paired = {}
    for model in ('glm', 'deepseek'):
        for thr in (32, 64, 128, 'all'):
            sel = [v for (mo, t, w, b), v in pairs.items()
                   if mo == model and (thr == 'all' or w == thr) and len(v) == 2]
            if not sel:
                continue
            out = dict(pairs=len(sel))
            for k in ('step1_inspect_only', 'step1_execution', 'step1_work', 'step1_reread',
                      'three_any_work', 'three_all_read', 'three_read_steps'):
                diffs = [v['S'][k] - v['A'][k] for v in sel if k in v['S'] and k in v['A']]
                if diffs:
                    out[k] = dict(mean=round(st.mean(diffs), 3), ci95=bootstrap(diffs), n=len(diffs))
            paired[f'{model}/{thr}'] = out
    # Post-switch minus baseline, per policy (within-run differences)
    shift = {}
    for g in [k for k in groups if k[2] == 'all'] + [k for k in groups if k[2] == 64]:
        rs = groups[g]
        out = {}
        for c in ('inspect_only', 'execution', 'work', 'reread'):
            diffs = [r['measures']['step1_' + c] - r['measures']['base_' + c]
                     for r in rs if 'base_' + c in r['measures']]
            if diffs:
                out[c] = dict(mean=round(st.mean(diffs), 3), ci95=bootstrap(diffs), n=len(diffs))
        shift['/'.join(map(str, g))] = out
    return table, paired, shift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--one')
    parser.add_argument('--earlier')
    args = parser.parse_args()
    if args.one:
        print(json.dumps(extract(json.loads(args.one))))
        return
    v2 = classifier()
    runs = discover()
    key = lambda r: (r['model'], r['policy'], r['task'], r['thr'], r['rep'])
    earlier = {key(c): c for c in json.loads(Path(args.earlier).read_text())['runs']} if args.earlier else {}
    cache, reused = [], 0
    for run in runs:
        if (Path(run['private']) / 'l0/session.jsonl').exists():
            spec = json.dumps(dict(runtime=run['runtime'], private=run['private']))
            extracted = json.loads(subprocess.check_output([sys.executable, __file__, '--one', spec], text=True))
            entry = dict(orphans=extracted['orphans'], duplicates=extracted['duplicates'],
                         missing=extracted['missing'], rows=label_steps(run, extracted, v2))
        else:
            old = earlier[key(run)]
            if old['private'] != str(Path(run['private']).relative_to(ROOT)):
                raise SystemExit(f'cached labels come from another run: {key(run)}')
            entry = {k: old[k] for k in ('orphans', 'duplicates', 'missing', 'rows')}
            reused += 1
        rows = entry['rows']
        cache.append(dict(**{k: run[k] for k in ('model', 'policy', 'task', 'thr', 'rep')},
                          private=str(Path(run['private']).relative_to(ROOT)), **entry))
        run.update(steps=len(rows), orphans=entry['orphans'], duplicates=entry['duplicates'],
                   missing=entry['missing'], measures=run_measures(rows),
                   categories=dict(collections.Counter(r['category'] for r in rows)))
        run['labels'] = rows
    (HERE / 'run-labels-cache.json').write_text(json.dumps(dict(
        schema='post-switch-run-labels/v1', note='step categories only, no command text; '
        f'{reused} runs reused from {args.earlier}', runs=cache)) + '\n')
    table, paired, shift = summarize(runs)
    with (HERE / 'step-labels.jsonl').open('w') as stream:
        for run in runs:
            head = {k: run[k] for k in ('model', 'policy', 'task', 'thr', 'rep')}
            for i, row in enumerate(run.pop('labels'), 1):
                stream.write(json.dumps(dict(**head, step=i, **row)) + '\n')
    for run in runs:
        run['private'] = str(Path(run['private']).relative_to(ROOT))
    report = dict(schema='post-switch-first-three-steps/v1', classifier='skeep-first-action/blind-secondary-v2.1',
                  definitions=dict(
                      event='first accepted step of each summary generation (first use of that summary)',
                      step1='category of the first tool call in the first accepted step after first use',
                      three='first three accepted steps of the generation; events whose generation ended earlier are excluded',
                      baseline='accepted steps at position 4 or later within any generation',
                      reread=f'first tool call is inspect_only and every static path it reads appeared in a tool call of the previous {LOOKBACK} accepted steps (lexical paths only)',
                      aggregation='events averaged within run; runs weighted equally; paired S-A differences by model/task/threshold/repetition; bootstrap over runs or pairs'),
                  runs=runs, table=table, paired_s_minus_a=paired, post_switch_minus_baseline=shift)
    (HERE / 'post-switch-behavior.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(table=table, paired=paired, shift=shift), indent=1))


if __name__ == '__main__':
    main()
