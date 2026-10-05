"""Extra reading steps per 100 agent steps after switches (Section 4.3); reads post-switch-behavior.json only.

Run from the repository root:
  python3 results/harness-diagnosis-20260927/post-switch-behavior-20261004/extra_reads.py
Per run: read steps among the first three steps after a switch, minus three times the run's usual read share, times
the run's switches per 100 agent steps; then the mean over the runs of each model, policy and threshold.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "post-switch-behavior.json").read_text())
    groups = defaultdict(list)
    for r in data["runs"]:
        m = r["measures"]
        if "three_read_steps" in m and "base_inspect_only" in m and r["steps"]:
            extra = (m["three_read_steps"] - 3 * m["base_inspect_only"]) * 100 * m["events"] / r["steps"]
            groups[(r["model"], r["policy"], r["thr"])].append(extra)
    for (model, policy, thr), xs in sorted(groups.items()):
        print(f"{model} {policy}{thr}: {st.mean(xs):.2f} extra read steps per 100 agent steps (mean of {len(xs)} runs)")


if __name__ == "__main__":
    main()
