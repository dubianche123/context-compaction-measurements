"""Steps a summary must be used to repay its cost, against the steps it was used (read-only; no model calls).

Run from the repository root:
  python3 results/harness-diagnosis-20260927/compaction-payback-20261004/analyze.py [--from-data]

--from-data reprices the API runs in the frozen runs.json from their recorded native
requests, preserving payback.json's self-hosted replay, async and decode data.

API runs: the S runs of the frozen snapshot (prelim-numbers-20261003/runs.json). For each adopted summary,
  one-time cost  = its summarization requests (every attempt with usage) + the uncached input of the first agent
                   request after the switch beyond the run's median, charged at the uncached-minus-cached price;
  saving / step  = removed context priced as cached input; removed context = reported input of the last agent
                   request before the switch + its reported output - reported input of the first request after. The
                   reported output approximates the response as carried in later inputs, and the last tool result is
                   not counted, so this is an approximation, not a bound. output.txt also reports the plain input
                   drop (without the last output) as a sensitivity check;
  repay steps    = one-time cost / saving per step;
  steps used     = accepted agent steps in the summary's cycle (runtime record), until the next summary or run end.
Self-hosted replays (results/colab-mechanism-20261005, public CSVs): repay steps are the amortized linear
projections (mean over repetitions); steps used come from the same summary's cycle in its GLM source run. For A on
Qwen3.8, output.txt reports the measured ten-step comparisons (against S and against the same steps without a
summary); a repayment figure that combines them with S's saving per step is kept only as a labeled estimate.
The Qwen3-32B MLSys supplement (tables/qwen3-mlsys-n1) adds a later session's 81,664-token decode calibration, which
makes Coq64 and Retro64 computable (their rows replace the earlier null rows; the other points keep the frozen
calibration), and one measured 20-step A/S/L wall comparison at Vig64 with the original tool gaps.
Writes payback.json and output.txt next to this file.
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DIAG, 'prelim-numbers-20261003'))
from metrics import PRICE, PRICE_AS_OF, PRICE_SOURCES  # noqa: E402

G = 'results/glm-repetition-tiered-20260930/work/runtime/output/harbor-jobs'
RUN_DIRS = {'glm': f'{G}/glm-window-sweep-20260927/main/long-011-formal-summary12288/window-main-long-011-formal-summary12288-{{task}}-{{rep}}-s{{thr}}',
            'deepseek': f'{G}/deepseek-flash-window-sweep-20260928/main/long-011-formal-summary12288/deepseek-window-main-long-011-formal-summary12288-{{task}}-{{rep}}-s{{thr}}'}
GLM_SOURCE = f'{G}/glm-window-sweep-20260927/main/long-011-formal-summary12288'
COLAB = 'results/colab-mechanism-20261005'
REPLAYS = 'preparations/colab-mechanism-20261005/work/replays-metadata.json'
DEPLOYMENTS = {'qwen38': ('Qwen/Qwen3.8-27B', 'QWEN38_FORMAL_V1_PREFIX.csv', 'QWEN38_FORMAL_V1_PRICES.csv'),
               'qwen3': ('Qwen/Qwen3-32B', 'QWEN3_FORMAL_PREFIX.csv', 'QWEN3_FORMAL_PRICES.csv')}
MLSYS = f'{COLAB}/tables/qwen3-mlsys-n3'
UPDATED = {'qwen3': f'{MLSYS}/QWEN3_MLSYS_FORMAL_PREFIX.csv'}
DECODE_ADDED = {'qwen3': f'{MLSYS}/QWEN3_MLSYS_DECODE.csv'}
ASYNC = [('qwen38', f'{COLAB}/QWEN38_FORMAL_V1_ASYNC.csv'), ('qwen38', f'{COLAB}/QWEN38_R3_COQ_ASYNC.csv'),
         ('qwen3', f'{MLSYS}/QWEN3_MLSYS_ASYNC.csv')]


def native(d):
    return json.load(open(glob.glob(d + '/*/axiom-private/result.json')[0]))


def usage(q):
    u = q.get('usage') or {}
    i = u.get('input_tokens', 0) or 0
    c = u.get('cached_tokens') or 0
    return i, i - c, c, u.get('output_tokens', 0) or 0


def cycles_by_id(R):
    cyc = R['summary_consumption']['foreground_cycles']['cycles']
    return {c['compaction_id']: (c['accepted_logical_step_count'], c is cyc[-1]) for c in cyc if c.get('compaction_id')}


def api_summaries(model, R):
    """One record per adopted summary of a sync run, in run order."""
    pu, pc, po = PRICE[model]
    qs = sorted(R['requests'], key=lambda q: q['call_index'])
    groups, cur = [], []
    for i, q in enumerate(qs):
        if q['purpose'] == 'compaction':
            cur.append(i)
        elif q['purpose'] == 'conversation' and cur:
            groups.append(cur)
            cur = []
    comps = R['audit']['compactions']
    if len(groups) != len(comps):
        raise SystemExit(f'compaction request groups ({len(groups)}) do not match audit records ({len(comps)})')
    ok = lambda q: q['purpose'] == 'conversation' and q.get('status') == 'completed' and q.get('usage')
    firsts, pairs = set(), []
    for g in groups:
        before = next((qs[j] for j in range(g[0] - 1, -1, -1) if ok(qs[j])), None)
        after = next((qs[j] for j in range(g[-1] + 1, len(qs)) if ok(qs[j])), None)
        pairs.append((before, after))
        if after is not None:
            firsts.add(after['call_index'])
    med_u = st.median(usage(q)[1] for q in qs if ok(q) and q['call_index'] not in firsts)
    cyc = cycles_by_id(R)
    out = []
    for g, (before, after), c in zip(groups, pairs, comps):
        if c.get('outcome') != 'summarized' or before is None or after is None or c['compaction_id'] not in cyc:
            continue
        summ = sum((usage(qs[j])[1] * pu + usage(qs[j])[2] * pc + usage(qs[j])[3] * po) / 1e6 for j in g if qs[j].get('usage'))
        prefix = max(0, usage(after)[1] - med_u) * (pu - pc) / 1e6
        drop = usage(before)[0] + usage(before)[3] - usage(after)[0]
        plain = usage(before)[0] - usage(after)[0]
        used, last = cyc[c['compaction_id']]
        rec = dict(summary_usd=summ, prefix_usd=prefix, removed_tokens=drop, input_drop_tokens=plain,
                   steps_used=used, cut_by_run_end=last)
        if drop > 0:
            rec.update(saving_usd_per_step=drop * pc / 1e6, repay_steps=(summ + prefix) / (drop * pc / 1e6))
        if plain > 0:
            rec['repay_steps_input_drop'] = (summ + prefix) / (plain * pc / 1e6)
        out.append(rec)
    return out


def fl(x):
    return float(x) if x not in ('', None) else None


def colab():
    meta = {c['point_id']: c for c in json.load(open(REPLAYS))['cells'] if c.get('state') == 'eligible'}
    used = {}
    for p, c in meta.items():
        used[p] = cycles_by_id(native(f"{GLM_SOURCE}/{c['source_job']}"))[c['selection']['compaction_id']]
    points, decode = [], {}
    for dep, (model, prefix_csv, price_csv) in DEPLOYMENTS.items():
        rows = list(csv.DictReader(open(f'{COLAB}/{prefix_csv}')))
        if dep in UPDATED:
            new = list(csv.DictReader(open(UPDATED[dep])))
            rows = [r for r in rows if r['point_id'] not in {n['point_id'] for n in new}] + new
        for p in sorted({r['point_id'] for r in rows}):
            for cond in 'SK':
                rs = [r for r in rows if r['point_id'] == p and r['condition'] == cond]
                pay = [fl(r['amortized_linear_payback_steps']) for r in rs]
                pay = [x for x in pay if x is not None]
                sav = [fl(r['amortized_per_step_savings_s']) for r in rs if fl(r['amortized_per_step_savings_s']) is not None]
                lim = [fl(r['limit_foreground_20_reconstructed_s']) for r in rs]
                share = [(fl(r['limit_prefill_20_s']) - fl(r['prefill_20_s']) + fl(r['first_request_prefill_extra_s'])) /
                         (fl(r['limit_foreground_20_reconstructed_s']) - fl(r['amortized_foreground_20_reconstructed_s']))
                         for r in rs if None not in lim]
                points.append(dict(deployment=dep, model=model, point_id=p, condition=cond, repetitions=len(rs),
                                   switch_h=int(rs[0]['switch_h_prompt_tokens']),
                                   summary_s=st.mean(fl(r['summary_engine_s']) for r in rs),
                                   first_request_extra_s=st.mean(fl(r['first_request_prefill_extra_s']) for r in rs),
                                   saving_s_per_step=st.mean(sav) if sav else None,
                                   repay_steps=st.mean(pay) if pay else None,
                                   prefill_share_of_saving=st.mean(share) if share else None,
                                   crossing_within_20=sorted({r['break_even_request'] for r in rs}),
                                   replays_crossed_within_20=sum(r['break_even_request'].isdigit() for r in rs),
                                   steps_used=used[p][0], cut_by_run_end=used[p][1]))
        price = list(csv.DictReader(open(f'{COLAB}/{price_csv}')))
        xs = [int(r['context_tokens']) for r in price]
        ys = [1000 * fl(r['output_token_s']) for r in price]
        later = []
        if dep in DECODE_ADDED:
            add = list(csv.DictReader(open(DECODE_ADDED[dep])))
            later = [int(add[0]['prompt_tokens'])]
            xs.append(later[0])
            ys.append(1000 * st.mean(fl(r['per_decode_interval_s']) for r in add))
        fit = st.linear_regression(xs, ys)
        decode[dep] = dict(model=model, context_tokens=xs, ms_per_output_token=ys, later_session_context_tokens=later,
                           slope_ms_per_1k=1000 * fit.slope, intercept_ms=fit.intercept)
    async_points = []
    for dep, f in ASYNC:
        rows = list(csv.DictReader(open(f)))
        p = rows[0]['point_id']
        s = next(x for x in points if x['deployment'] == dep and x['point_id'] == p and x['condition'] == 'S')
        inc = st.mean(fl(r['overlap_foreground_increase_s']) for r in rows)
        rec = dict(deployment=dep, point_id=p, condition='A', overlap_slowdown_s=inc,
                   summary_wait_s=st.mean(fl(r['sync_wait_available_to_hide_s']) for r in rows),
                   net_gain_s=st.mean(fl(r['net_gain_s']) for r in rows),
                   a_minus_l_foreground_s=st.mean(fl(r['async_foreground_request_elapsed_mean_s']) -
                                                  fl(r['baseline_h_foreground_request_elapsed_mean_s']) for r in rows),
                   steps_used=s['steps_used'], cut_by_run_end=s['cut_by_run_end'])
        if rows[0].get('a_vs_l_break_even_request'):
            # measured 20-step wall runs with the original tool gaps, one row per repetition: crossings are measured,
            # not estimated; totals are means of the per-repetition values
            rec.update(steps_compared=20, l_wall_s=st.mean(fl(r['l_j20_wall_s']) for r in rows),
                       a_minus_l_wall_s=st.mean(fl(r['a_minus_l_j20_s']) for r in rows),
                       s_minus_l_wall_s=st.mean(fl(r['s_minus_l_j20_s']) for r in rows),
                       a_crossings_vs_l=[int(r['a_vs_l_break_even_request']) for r in rows],
                       s_crossings_vs_l=[int(r['s_vs_l_break_even_request']) for r in rows], repetitions=len(rows))
        else:
            rec.update(steps_compared=10,
                       combined_estimate_repay_steps=(inc + s['first_request_extra_s']) / s['saving_s_per_step'])
        async_points.append(rec)
    return points, async_points, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-data", action="store_true",
                        help="Reprice frozen API runs; preserve processed self-hosted data")
    args = parser.parse_args()
    snap = [r for r in json.load(open(os.path.join(DIAG, 'prelim-numbers-20261003/runs.json'))) if r['policy'] == 'S']
    api = []
    for r in snap:
        if args.from_data:
            recorded = json.load(open(r['native']))
        else:
            d = glob.glob(RUN_DIRS[r['model']].format(task=r['task'], rep=r['rep'], thr=r['thr']))
            if len(d) != 1:
                raise SystemExit(f"run directory not found: {r['model']} {r['task']} {r['thr']} {r['rep']}")
            recorded = native(d[0])
        for rec in api_summaries(r['model'], recorded):
            rec.update(model=r['model'], task=r['task'], thr=r['thr'], rep=r['rep'])
            api.append(rec)
    if args.from_data:
        previous = json.load(open(os.path.join(HERE, 'payback.json')))
        points, async_points, decode = (previous[k] for k in ('replay', 'replay_async', 'decode'))
    else:
        points, async_points, decode = colab()
    data = dict(schema='compaction-payback/v2', api=api, replay=points, replay_async=async_points, decode=decode,
                notes=dict(api='S runs of the frozen runs.json snapshot; official USD list prices; DeepSeek peak, no discounts',
                           pricing=dict(currency='USD', unit='USD per million tokens', as_of=PRICE_AS_OF,
                                        vectors=PRICE, sources=PRICE_SOURCES),
                           replay='amortized linear projection; steps used from the GLM source run',
                           cut_by_run_end='the last summary of a run; its use was cut short by the run end'))
    json.dump(data, open(os.path.join(HERE, 'payback.json'), 'w'), indent=1)
    L = []
    say = lambda *a: L.append(' '.join(str(x) for x in a))
    say('API S runs:', len({(a['model'], a['task'], a['thr'], a['rep']) for a in api}), 'summaries:', len(api),
        'without removed context:', sum('repay_steps' not in a for a in api), '(summaries cut by the run end are left out below)')
    for model in ('glm', 'deepseek'):
        for thr in (32, 64, 128, None):
            a = [x for x in api if x['model'] == model and (thr is None or x['thr'] == thr) and 'repay_steps' in x and not x['cut_by_run_end']]
            if not a:
                continue
            byrun = {}
            for x in a:
                byrun.setdefault((x['task'], x['rep'], x['thr']), []).append(x['repay_steps'] < x['steps_used'])
            shares = [sum(v) / len(v) for v in byrun.values()]
            plain = [x for x in a if 'repay_steps_input_drop' in x]
            say(f"{model:8s} {str(thr or 'all'):>4s} runs={len(byrun):2d} summaries={len(a):4d} repay med {st.median(x['repay_steps'] for x in a):7.2f}"
                f" | used med {st.median(x['steps_used'] for x in a):5.1f} | repaid before next {sum(x['repay_steps'] < x['steps_used'] for x in a)}/{len(a)}"
                f" | runs where most repaid {sum(v > 0.5 for v in shares)}/{len(shares)}, per-run share {min(shares):.2f}-{max(shares):.2f}"
                f" | prefix share of one-time {100 * st.median(x['prefix_usd'] / (x['prefix_usd'] + x['summary_usd']) for x in a):.0f}%"
                f" | input-drop check: repaid {sum(x['repay_steps_input_drop'] < x['steps_used'] for x in plain)}/{len(a)}")
    for dep in DEPLOYMENTS:
        s = [x for x in points if x['deployment'] == dep and x['condition'] == 'S']
        v = [x for x in s if x['repay_steps'] is not None]
        nl = [x for x in v if not x['cut_by_run_end']]
        say(f"{dep}: S points {len(s)}, computable {len(v)}; repay median {st.median(x['repay_steps'] for x in v):.1f}"
            f"; summary {min(x['summary_s'] for x in s):.1f}-{max(x['summary_s'] for x in s):.1f} s"
            f"; repaid before next {sum(x['repay_steps'] < x['steps_used'] for x in nl)}/{len(nl)} (run-end cycles left out)"
            f"; prefill share of S+K saving median {100 * st.median(x['prefill_share_of_saving'] for x in s + [y for y in points if y['deployment'] == dep and y['condition'] == 'K'] if x['prefill_share_of_saving'] is not None):.1f}%"
            f" max {100 * max(x['prefill_share_of_saving'] for x in points if x['deployment'] == dep and x['prefill_share_of_saving'] is not None):.1f}%")
        say(f"  decode slope {decode[dep]['slope_ms_per_1k']:.4f} ms per output token per 1K context; intercept {decode[dep]['intercept_ms']:.2f} ms"
            + (f" (fit includes the later-session point at {decode[dep]['later_session_context_tokens'][0]} tokens)" if decode[dep]['later_session_context_tokens'] else ''))
        sk = [x for x in points if x['deployment'] == dep and x['repay_steps'] is not None]
        say(f"  S and K replays whose reconstructed time fell below L's within 20 steps: "
            f"{sum(x['replays_crossed_within_20'] for x in sk)}/{sum(x['repetitions'] for x in sk)} ({len(sk)} computable point-conditions)")
    common = {x['point_id'] for x in points if x['deployment'] == 'qwen3' and x['repay_steps'] is not None}
    for dep in DEPLOYMENTS:
        for cond in 'SK':
            v = [x['repay_steps'] for x in points if x['deployment'] == dep and x['condition'] == cond and x['point_id'] in common]
            say(f"common {len(common)} points {dep} {cond}: repay median {st.median(v):.1f}")
    for a in async_points:
        head = (f"A {a['deployment']} {a['point_id'][9:]}: overlap slowdown {a['overlap_slowdown_s']:.2f} s, summary wait {a['summary_wait_s']:.1f} s, "
                f"A faster than S by {a['net_gain_s']:.1f} s, A minus no-summary baseline over {a['steps_compared']} steps {a['a_minus_l_foreground_s']:+.2f} s; ")
        if 'a_crossings_vs_l' in a:
            say(head + f"measured {a['repetitions']} times with the original tool gaps, means: L {a['l_wall_s']:.1f} s, A-L {a['a_minus_l_wall_s']:+.1f} s "
                f"({100 * a['a_minus_l_wall_s'] / a['l_wall_s']:+.1f}%), S-L {a['s_minus_l_wall_s']:+.1f} s ({100 * a['s_minus_l_wall_s'] / a['l_wall_s']:+.1f}%); "
                f"first step below L per run: A {a['a_crossings_vs_l']}, S {a['s_crossings_vs_l']}; used {a['steps_used']}")
        else:
            say(head + f"combined estimate (not a measured crossing) {a['combined_estimate_repay_steps']:.1f} steps, used {a['steps_used']}")
    open(os.path.join(HERE, 'output.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
