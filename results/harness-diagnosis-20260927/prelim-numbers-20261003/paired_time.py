"""Paired elapsed time of GLM S/A runs for Section 4.2 (read-only; no model calls).

Run from the repository root after collect.py and wait_metrics.py:
  python3 results/harness-diagnosis-20260927/prelim-numbers-20261003/paired_time.py
A pair is the S and A run of the same task, threshold, and repetition. The interval is a percentile bootstrap over
pairs with a fixed seed. The wait share is the ordinary summary wait over foreground time of the pair's S run, from
wait-metrics.json, the same measure as Figure 3a.
"""
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    runs = json.loads((HERE / "runs.json").read_text())
    waits = {(r["model"], r["task"], r["thr"], r["rep"]): r["ordinary_wait_fraction"]
             for r in json.loads((HERE / "wait-metrics.json").read_text())["rows"]}
    for model in ("glm", "deepseek"):
        by = {}
        for r in runs:
            if r["model"] == model and r["policy"] in ("S", "A"):
                by.setdefault((r["task"], r["thr"], r["rep"]), {})[r["policy"]] = r
        pairs = [v for _, v in sorted(by.items()) if len(v) == 2]
        diff = [(p["A"]["elapsed"] - p["S"]["elapsed"]) / 3600 for p in pairs]
        rng = random.Random(20261005)
        boot = sorted(st.median(rng.choices(diff, k=len(diff))) for _ in range(10000))
        share = [waits[(model, p["S"]["task"], p["S"]["thr"], p["S"]["rep"])] for p in pairs]
        share = [x for x in share if x is not None]
        print(f"{model}: {len(pairs)} complete pairs; A finished sooner in {sum(d < 0 for d in diff)}; "
              f"A minus S elapsed median {st.median(diff):+.2f} h (95% bootstrap {boot[249]:+.2f} to {boot[9749]:+.2f}); "
              f"S ordinary wait share median {100 * st.median(share):.1f}% (n={len(share)})")


if __name__ == "__main__":
    main()
