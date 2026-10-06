"""Context trajectories of GLM Coq runs for the paper's first figure; no provider calls.

Run from the repository root with the repository .venv Python:
  .venv/bin/python results/paper-mlsys2027-draft/figures/context-trajectories/build_figure.py
Each run's L0 is opened in its own subprocess with the SessionLog reader and the
summary_consumption attempt validator of the runtime that wrote it. Accepted steps,
summary generations and first summary use come from the saved native
summary-consumption record. The first tool call after each first use is classified
by the frozen v2 first-action classifier. Only this directory is written; the
pgfplots fragment is pasted into main.tex so that the manuscript stays self-contained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "results/harness-diagnosis-20260927/prelim-numbers-20261003"))
from metrics import PRICE, PRICE_AS_OF, PRICE_CURRENCY, PRICE_SOURCES  # noqa: E402

GLM = "results/glm-repetition-tiered-20260930/work/runtime"
KEEP = "results/skeep-20261001/work/runtime"
JOBS = GLM + "/output/harbor-jobs/glm-window-sweep-20260927"
S = JOBS + "/main/long-011-formal-summary12288/window-main-long-011-formal-summary12288-coq-block-bound-{rep}-{arm}"
K = KEEP + "/output/harbor-jobs/glm-window-sweep-20260927/skeep-001/window-skeep-001-coq-block-bound-{rep}-{arm}"
L = JOBS + "/main/long-011/n1m-deadline5400-b001/window-main-long-011-d5400-coq-block-bound-b001-n1m"
SOURCES = [
    dict(column=0, label="S64", policy="S", threshold=64000, runtime=GLM, job=S.format(rep="b001", arm="s64")),
    dict(column=0, label="A64", policy="A", threshold=64000, runtime=GLM, job=S.format(rep="b001", arm="a64")),
    dict(column=0, label="K64", policy="K", threshold=64000, runtime=KEEP, job=K.format(rep="b001", arm="k64")),
    dict(column=0, label="L", policy="L", threshold=None, runtime=GLM, job=L),
    dict(column=1, label="S32, 2nd run", policy="S", threshold=32000, runtime=GLM, job=S.format(rep="b002", arm="s32")),
    dict(column=1, label="K32, 1st run", policy="K", threshold=32000, runtime=KEEP, job=K.format(rep="b001", arm="k32")),
    dict(column=1, label="K32, 2nd run", policy="K", threshold=32000, runtime=KEEP, job=K.format(rep="b002", arm="k32")),
    dict(column=1, label="K32, 3rd run", policy="K", threshold=32000, runtime=KEEP, job=K.format(rep="b003", arm="k32")),
]
COLORS = dict(S="polS", A="polA", K="polK", L="polL")


def request_usage(trial):
    """Read only recorded billing usage, without rebuilding trajectory events."""
    totals = dict(uncached_input=0, cached_input=0, output=0)
    missing = 0
    for request in json.loads((ROOT / trial / "axiom-private/result.json").read_text()).get("requests", []):
        usage = request.get("usage") or {}
        missing += not usage.get("input_tokens")
        inp, cached = usage.get("input_tokens") or 0, usage.get("cached_tokens") or 0
        totals["uncached_input"] += inp - cached
        totals["cached_input"] += cached
        totals["output"] += usage.get("output_tokens") or 0
    return totals, missing


def reprice(data):
    """Reprice stored token totals; migrate older caches from their recorded requests."""
    pu, pc, po = PRICE["glm"]
    for run in data["runs"]:
        if "usage_tokens" not in run:
            run["usage_tokens"], run["requests_without_usage"] = request_usage(run["trial"])
        usage = run["usage_tokens"]
        cost = (pu * usage["uncached_input"] + pc * usage["cached_input"] + po * usage["output"]) / 1e6
        run["usd_per_100_steps"] = 100 * cost / run["accepted_steps"]
        run.pop("cny_per_100_steps", None)
    data["schema"] = "context-trajectories/v2"
    data["cost"] = dict(currency=PRICE_CURRENCY, per_million_tokens=PRICE["glm"],
                        as_of=PRICE_AS_OF, source=PRICE_SOURCES["glm"],
                        includes="all agent and summary requests")


def extract(index):
    source = SOURCES[index]
    sys.path.insert(0, str(ROOT / source["runtime"]))
    from axiom_shortloop.session_log import SessionLog
    from experiments.summary_consumption import _attempt

    trial = [p for p in glob.glob(str(ROOT / source["job"] / "*")) if Path(p, "axiom-private").is_dir()]
    if len(trial) != 1:
        raise ValueError(f"expected one trial directory: {source['job']}")
    private = Path(trial[0], "axiom-private")
    official = json.loads(Path(trial[0], "result.json").read_text())
    consumption = json.loads((private / "summary-consumption.json").read_text())
    lp = private / "l0/session.jsonl"
    with lp.open() as stream:
        header = json.loads(stream.readline())
    log = SessionLog.open(lp.parent, session_id=header["session_id"])
    if log.has_truncated_tail:
        raise ValueError(f"Truncated selected L0: {lp}")
    cycles = consumption["foreground_cycles"]["cycles"]
    by_step = {}
    for cycle in cycles:
        for step in cycle["accepted_step_ids"]:
            if step in by_step:
                raise ValueError("Accepted step assigned to multiple generations")
            by_step[step] = cycle
    attempt_events = defaultdict(list)
    purpose, step_of, messages, last = {}, {}, {}, None
    for event in log.events:
        kind, data = event["type"], event["data"]
        if kind == "model/attempt":
            attempt_events[data["attempt_id"]].append(event)
            if data.get("phase") == "started":
                purpose[data["attempt_id"]], step_of[data["attempt_id"]] = data.get("purpose"), data.get("step_id")
            elif purpose.get(data["attempt_id"]) == "conversation" and data.get("status") == "completed":
                last = step_of[data["attempt_id"]]
        elif kind == "message/append" and (data.get("message") or {}).get("role") == "assistant":
            # Native acceptance appends the assistant message after its completed attempt.
            if last is not None:
                messages[last] = data["message"]
                last = None
    dispatch = {d["attempt_id"]: d for d in consumption["dispatches"]
                if d["verification"] == "verified" and d["mapping_status"] == "complete"}
    candidates = defaultdict(list)
    for identity, events in attempt_events.items():
        attempt = _attempt(events)
        if (attempt["purpose"] == "conversation"
                and attempt["step_id"] in by_step
                and attempt["verification"] == "verified"
                and attempt["terminal_status"] == "completed"
                and attempt["finish_reason"] in {"stop", "tool_calls", "function_call"}
                and identity in dispatch):
            started = next(e for e in events if e["data"].get("phase") == "started")
            candidates[attempt["step_id"]].append((attempt, started, dispatch[identity]))
    if set(candidates) != set(by_step) or any(len(v) != 1 for v in candidates.values()):
        raise ValueError("Accepted-step coverage or unique accepted-attempt mapping failed")
    first_use = {j["first_consumption"]["step_id"]: j for j in consumption["jobs"]
                 if j.get("first_consumption") is not None}
    points = []
    for ordinal, (attempt, started, sent) in enumerate(
            sorted((v[0] for v in candidates.values()), key=lambda x: x[2]["seq"]), 1):
        d = started["data"]
        calibrated = d["compaction_policy"]["calibrated_input_tokens"]
        if not isinstance(calibrated, int):
            raise ValueError("Missing calibrated input measurement")
        job = first_use.get(attempt["step_id"])
        message = messages.get(attempt["step_id"])
        calls = (message or {}).get("tool_calls") or []
        points.append(dict(step=ordinal, calibrated_input_tokens=calibrated,
                           provider_reported_input_tokens=attempt["usage"]["input_tokens"],
                           first_consumption=bool(job),
                           first_action=dict(role="assistant", tool_calls=calls[:1]) if job and message else None,
                           retained_tokens_at_proposal=(job.get("proposal", {}).get("retained_projection_tokens")
                                                        if job else None)))
    usage, missing_usage = request_usage(Path(trial[0]).relative_to(ROOT))
    reward = ((official.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return dict(**source, trial=str(Path(trial[0]).relative_to(ROOT)), reward=reward,
                accepted_steps=len(points), first_consumption_count=len(first_use),
                unplotted_first_consumption=len(set(first_use) - set(candidates)),
                usage_tokens=usage,
                requests_without_usage=missing_usage, points=points)


def classify_first_actions(data):
    spec = importlib.util.spec_from_file_location(
        "branch_classifier_v2", ROOT / "preparations/skeep-20261001/branch_classifier_v2.py")
    v2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v2
    spec.loader.exec_module(v2)
    for run in data["runs"]:
        for p in run["points"]:
            action = p.pop("first_action", None)
            p["first_action_category"] = v2.classify_action(action)["category"] if action else None


def coords(points, keep=lambda p: True):
    return " ".join(f"({p['step']},{p['calibrated_input_tokens'] / 1000:.1f})" for p in points if keep(p))


def write_tex(data):
    observed = [p["calibrated_input_tokens"] for r in data["runs"] for p in r["points"]]
    if min(observed) < 2000 or max(observed) > 512000:
        raise ValueError("Shared log-axis bounds would truncate observed data")
    xmax = [math.ceil(max(r["accepted_steps"] for r in data["runs"] if r["column"] == c) / 50) * 50 for c in (0, 1)]
    rows = [[r for r in data["runs"] if r["column"] == c] for c in (0, 1)]
    lines = [
        "% Generated by figures/context-trajectories/build_figure.py; data in the same directory.",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=2 by 4,horizontal sep=0.5cm,vertical sep=0.62cm,"
        r"x descriptions at=edge bottom,y descriptions at=edge left},",
        r"scale only axis,width=0.43\textwidth,height=1.5cm,",
        r"ymode=log,log basis y=2,ymin=2,ymax=512,ytick={2,8,32,128,512},yticklabels={2,8,32,128,512},",
        r"scaled ticks=false,tick label style={font=\scriptsize},axis line style={black!45},",
        r"tick style={black!45},grid=major,grid style={black!8},",
        r"title style={at={(0,1)},anchor=south west,font=\scriptsize,inner sep=1.5pt},",
        r"xlabel={Agent step},xlabel style={font=\small},clip=true,enlargelimits=false]",
    ]
    for i in range(4):
        for c in (0, 1):
            run = rows[c][i]
            outcome = "passed" if run["reward"] == 1 else "failed"
            title = f"{run['label']}: {run['accepted_steps']} steps, {outcome}"
            if c == 0:
                bound = "$\\ge$" if run["requests_without_usage"] else ""
                title += f", {bound}{run['usd_per_100_steps']:.2f} USD/100 steps"
            color = COLORS[run["policy"]]
            ticks = "xtick={0,100,200,300,400}" if c == 0 else "xtick={0,100,200,300,400}"
            lines.append(r"\nextgroupplot[" + f"xmin=0,xmax={xmax[c]},{ticks}," + r"title={" + title + "}]")
            if run["threshold"]:
                t = run["threshold"] // 1000
                lines.append(r"\addplot[black!55,dashed,line width=0.45pt] coordinates {(0," + str(t) + ")(" + str(xmax[c]) + "," + str(t) + ")};")
            lines.append(r"\addplot[" + color + r",line width=0.6pt,no marks] coordinates {" + coords(run["points"]) + "};")
            read = lambda p: p["first_consumption"] and p["first_action_category"] == "inspect_only"
            other = lambda p: p["first_consumption"] and p["first_action_category"] != "inspect_only"
            if any(other(p) for p in run["points"]):
                lines.append(r"\addplot[" + color + r",only marks,mark=o,mark size=1.35pt,line width=0.45pt,mark options={fill=white}] coordinates {" + coords(run["points"], other) + "};")
            if any(read(p) for p in run["points"]):
                lines.append(r"\addplot[" + color + r",only marks,mark=*,mark size=1.35pt,line width=0.45pt] coordinates {" + coords(run["points"], read) + "};")
    lines += [
        r"\end{groupplot}",
        r"\node[font=\small,anchor=south] at ([yshift=0.38cm]group c1r1.north) {(a) Coq at 64K: one run of each policy};",
        r"\node[font=\small,anchor=south] at ([yshift=0.38cm]group c2r1.north) {(b) Coq at 32K: S and all three K runs};",
        r"\node[rotate=90,font=\small,anchor=south] at ([xshift=-0.7cm,yshift=-0.31cm]group c1r2.south west) {Context size (K tokens, log scale)};",
        r"\end{tikzpicture}",
        r"\caption{Context size at each agent step in GLM Coq runs, as the runtime estimates it for",
        r"comparison with the threshold (dashed); the estimate runs about 15\% above the provider's reported input.",
        r"Circles mark the first step that uses a new summary; filled circles, a first action that only read files",
        r"or listings. (a)~The first S, A, and K runs at 64K and the first L run, which never reached its 1M limit;",
        r"all four passed, and $\ge$ marks a lower bound where a failed request reported no usage. (b)~At 32K, the",
        r"S run that compacted and all three K runs; the other S32 run failed after 12 steps. Clusters of circles in",
        r"the K runs are summaries replaced after a single step, mostly when the two kept steps were large",
        r"(Section~\ref{sec:behavior}).}",
        r"\label{fig:trajectories}",
        r"\end{figure*}",
    ]
    (HERE / "context-trajectories-pgfplots.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", type=int)
    parser.add_argument("--from-data", action="store_true")
    args = parser.parse_args()
    if args.one is not None:
        print(json.dumps(extract(args.one)))
        return
    if args.from_data:
        data = json.loads((HERE / "context-trajectories-data.json").read_text())
    else:
        runs = [json.loads(subprocess.check_output([sys.executable, str(__file__), "--one", str(i)], text=True))
                for i in range(len(SOURCES))]
        data = dict(schema="context-trajectories/v1",
                    selection="Coq b001 S64/A64/K64 and the b001 L run; Coq S32 b002 and K32 b001-b003. "
                              "S32 b001 ended after 12 steps and is omitted.",
                    context_source="model/attempt started.compaction_policy.calibrated_input_tokens",
                    first_action_rule="skeep-first-action/blind-secondary-v2.1, first tool call",
                    runs=runs)
        classify_first_actions(data)
    reprice(data)
    (HERE / "context-trajectories-data.json").write_text(json.dumps(data, indent=1) + "\n")
    write_tex(data)
    print(json.dumps([{k: r[k] for k in ("label", "accepted_steps", "reward", "first_consumption_count",
                                         "unplotted_first_consumption", "usd_per_100_steps")} for r in data["runs"]],
                     indent=1))


if __name__ == "__main__":
    main()
