# Context compaction measurements: processed data and analysis scripts

Processed measurements and scripts behind the figures and tables of
"When Does Context Compaction Pay in Long-Horizon Agents?" (Yuyao Wang).
Every file keeps its path from the research repository, so the scripts run unchanged from this directory.

## Rebuild the figures and tables (Python 3.11 or later, no dependencies)

    python3 results/paper-mlsys2027-draft/figures/overview/build_figure.py                           # Figure 1 (schematic, no data)
    python3 results/paper-mlsys2027-draft/figures/context-trajectories/build_figure.py --from-data   # Figure 2
    python3 results/paper-mlsys2027-draft/figures/waiting-actions/build_figure.py                    # Figure 3
    python3 results/paper-mlsys2027-draft/figures/selfhosted-timeline/build_figure.py                # Figure 4
    python3 results/paper-mlsys2027-draft/figures/cost-parts/build_figure.py                         # Figure 5
    python3 results/paper-mlsys2027-draft/figures/cost/build_figure.py                               # Figure 6
    python3 results/paper-mlsys2027-draft/figures/payback/build_figure.py                            # Figure 7
    python3 results/paper-mlsys2027-draft/figures/pass-table/build_table.py                          # Table 4
    python3 results/harness-diagnosis-20260927/prelim-numbers-20261003/paired_time.py             # paired S/A elapsed time, Section 4.2
    python3 results/harness-diagnosis-20260927/post-switch-behavior-20261004/extra_reads.py       # extra reading steps, Section 4.3

Each builder writes a pgfplots or LaTeX fragment and its plotted data next to itself; the fragments shipped here are
the ones in the paper.

Monetary results use the official USD API list prices checked on 2026-10-06, not currency conversions:
GLM-5.3-Flash is $0.15/$0.03/$0.50 and DeepSeek-Flash is $0.30/$0.006/$1.20 per million
uncached-input/cached-input/output tokens. DeepSeek uses peak rates. Discounts are excluded; these are
standardized costs, not invoices. Prices and source URLs are recorded in `metrics.py`.

## Contents

- `results/harness-diagnosis-20260927/prelim-numbers-20261003/runs.json`: one record per API run (model, policy, task, threshold, repetition,
  official reward, ending, elapsed time, agent steps, compactions, and uncached-input, cached-input and output tokens
  of agent and summarization requests). `metrics.py` holds the fixed price vectors and the cost definitions.
- `results/harness-diagnosis-20260927/deepseek-limitonly-numbers-20261004/deepseek-l-runs.json`: the same for DeepSeek's L runs.
- `results/harness-diagnosis-20260927/prelim-numbers-20261003/wait-metrics.json`: foreground time and summary waits of the S runs.
- `results/harness-diagnosis-20260927/post-switch-behavior-20261004/`: category of the first tool call of every accepted step, per run
  (`step-labels.jsonl`), and the summary used in the paper (`post-switch-behavior.json`).
- `results/harness-diagnosis-20260927/compaction-payback-20261004/payback.json`: per-summary repayment estimates (API runs) and replay results.
- `results/harness-diagnosis-20260927/claude-cross-annotation-20261003/`, `monitoring/glm-b002-branch-replication-20261003/`: branch contrasts.
- `results/colab-mechanism-20261005/`: scalar tables of the self-hosted replays on Qwen3.8-27B and Qwen3-32B.

## Not included

Raw run records (session logs, provider ledgers, transcripts) are not distributed: they contain task content and
model output. The tasks are Terminal-Bench tasks taken from the public task repository at commit 452bf30. The
scripts listed below produced the processed data from those records and are included to document the definitions;
they need the raw records and the agent runtime to run.

- `results/harness-diagnosis-20260927/prelim-numbers-20261003/collect.py`
- `results/harness-diagnosis-20260927/prelim-numbers-20261003/wait_metrics.py`
- `results/harness-diagnosis-20260927/deepseek-limitonly-numbers-20261004/collect.py`
- `results/harness-diagnosis-20260927/post-switch-behavior-20261004/analyze.py`
- `results/harness-diagnosis-20260927/compaction-payback-20261004/analyze.py`
- `preparations/skeep-20261001/branch_classifier_v2.py`

## License

The scripts (`*.py`) are released under the MIT License (`LICENSE`). The data, including the figure and table
fragments, are released under CC BY 4.0 (`LICENSE-DATA`).
