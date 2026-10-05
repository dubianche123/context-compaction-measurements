"""Foreground time and summary waits of S runs, as used by Section 4.1 and Figure 3a (read-only; no model calls).

Run from the repository root with the repository .venv Python:
  .venv/bin/python results/harness-diagnosis-20260927/prelim-numbers-20261003/wait_metrics.py \
      --earlier results/harness-diagnosis-20260927/prelim-numbers-20261003/review-wait-metrics-20261003.json
Reads the S runs of runs.json (written by collect.py) and writes wait-metrics.json in this directory. Each run's
L0 is opened in its own subprocess with the SessionLog reader of the runtime that wrote it, and foreground time
and waits come from that runtime's experiments.adoption_continuation_report decoders, the same ones used for the
2026-10-03 review file review-wait-metrics-20261003.json. The L0 logs of the first repetition were archived off this
machine on 2026-10-04 (monitoring/disk-cleanup-20261004/RESTORE.md); for a run whose L0 is no longer on disk, the row
computed earlier from the same log with the same decoders is reused from --earlier. Runs written by an older
collect.py carry no result paths; --earlier also supplies those.
"""
import argparse
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def one(native_result):
    native = Path(native_result)
    runtime = Path(str(native).split("/output/harbor-jobs/")[0])
    sys.path.insert(0, str(runtime))
    from axiom_shortloop.session_log import SessionLog
    from experiments.adoption_continuation_report import _awaits, _foreground

    result = json.loads(native.read_text())
    lp = native.parent / "l0/session.jsonl"
    with lp.open() as stream:
        header = json.loads(stream.readline())
    log = SessionLog.open(lp.parent, session_id=header["session_id"])
    truncated = bool(log.has_truncated_tail)
    events = list(log.events)
    foreground = _foreground(events, result.get("turns") or [])
    foreground.pop("turns", None)
    waits = _awaits(events, truncated=truncated)
    for name in ("ordinary", "hard", "unknown"):
        waits[name].pop("events", None)
    walls = [q.get("provider_wall_duration_seconds") for q in result.get("requests", []) if q.get("purpose") == "compaction"]
    known = [w for w in walls if isinstance(w, (int, float))]
    fg = foreground["duration_ms"]
    ordinary = waits["ordinary"]["duration_ms"]
    every = None if None in (ordinary, waits["hard"]["duration_ms"]) or waits["unknown"]["observed_count"] else \
        ordinary + waits["hard"]["duration_ms"]
    return dict(native_result=str(native), l0_path=str(lp), l0_truncated=truncated, foreground=foreground, waits=waits,
                ordinary_wait_fraction=ordinary / fg if fg and ordinary is not None else None,
                all_wait_fraction=every / fg if fg and every is not None else None,
                native_elapsed_seconds=result.get("elapsed_seconds"),
                summary_physical_request_count=len(walls), summary_physical_request_wall_seconds=sum(known),
                summary_physical_request_wall_missing=len(walls) - len(known),
                summary_physical_request_wall_median=st.median(known) if known else None,
                summary_physical_request_walls=known)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one")
    parser.add_argument("--earlier")
    args = parser.parse_args()
    if args.one:
        print(json.dumps(one(args.one)))
        return
    runs = [r for r in json.loads((HERE / "runs.json").read_text()) if r["policy"] == "S"]
    earlier = {}
    if args.earlier:
        for row in json.loads(Path(args.earlier).read_text())["rows"]:
            earlier[(row["model"], row["task"], row["thr"], row["rep"])] = row
    rows, reused = [], 0
    for r in sorted(runs, key=lambda r: (r["model"], r["thr"], r["task"], r["rep"])):
        key = (r["model"], r["task"], r["thr"], r["rep"])
        native = r.get("native") or earlier[key]["native_result"]
        if (Path(native).parent / "l0/session.jsonl").exists():
            row = json.loads(subprocess.check_output([sys.executable, str(__file__), "--one", native], text=True))
        else:
            old = earlier[key]
            if Path(old["native_result"]).resolve() != Path(native).resolve():  # earlier rows hold absolute paths
                raise SystemExit(f"earlier row comes from another result: {key}")
            row = {k: v for k, v in old.items() if k not in ("model", "policy", "task", "thr", "rep", "reward", "exception")}
            row["summary_physical_request_walls"] = [
                q.get("provider_wall_duration_seconds") for q in json.loads(Path(native).read_text()).get("requests", [])
                if q.get("purpose") == "compaction" and isinstance(q.get("provider_wall_duration_seconds"), (int, float))]
            reused += 1
        rows.append(dict(model=r["model"], policy=r["policy"], task=r["task"], thr=r["thr"], rep=r["rep"],
                         reward=r["reward"], exception=r["exception"], **row))
    groups = []
    by = defaultdict(list)
    for row in rows:
        by[(row["model"], row["thr"])].append(row)
    for (model, thr), sel in sorted(by.items()):
        ordinary = [x["ordinary_wait_fraction"] for x in sel if x["ordinary_wait_fraction"] is not None]
        every = [x["all_wait_fraction"] for x in sel if x["all_wait_fraction"] is not None]
        groups.append(dict(model=model, threshold=thr, n=len(sel), known=len(ordinary),
                           ordinary_wait_fraction_median=st.median(ordinary) if ordinary else None,
                           ordinary_wait_fraction_max=max(ordinary) if ordinary else None,
                           all_wait_fraction_median=st.median(every) if every else None,
                           all_wait_fraction_max=max(every) if every else None,
                           hard_wait_seconds=sum(x["waits"]["hard"]["duration_ms_known_sum"] for x in sel) / 1000,
                           unknown_wait_events=sum(x["waits"]["unknown"]["observed_count"] for x in sel)))
    walls = defaultdict(list)
    for row in rows:
        walls[row["model"]] += row.pop("summary_physical_request_walls")
    out = dict(schema="wait-metrics/v1", source_snapshot=str(HERE / "runs.json"),
               scope=f"S runs of the snapshot; {reused} rows reused from {args.earlier} because their L0 is archived",
               definitions=dict(
                   foreground="native turn/state terminal durations; excludes host assessment and cleanup",
                   wait="native agent/compaction-await durations, ordinary and hard separately; contained in foreground",
                   summary_request_wall="physical compaction request wall durations of these S runs"),
               groups=groups, summary_request_wall_by_model={m: dict(n=len(v), median=st.median(v)) for m, v in walls.items()},
               rows=rows)
    (HERE / "wait-metrics.json").write_text(json.dumps(out, indent=1) + "\n")
    for g in groups:
        print(f"{g['model']:8s} S{g['threshold']}: n={g['n']} ordinary wait share median {100 * g['ordinary_wait_fraction_median']:.1f}% "
              f"max {100 * g['ordinary_wait_fraction_max']:.1f}%; hard waits {g['hard_wait_seconds']:.0f} s")
    for m, v in out["summary_request_wall_by_model"].items():
        print(f"{m:8s} summary requests: n={v['n']} median {v['median']:.1f} s")


if __name__ == "__main__":
    main()
