# Contributions

This repository is a fork of the [wineslab `ns-o-ran-gym`](https://github.com/wineslab/ns-o-ran-gym)
original. This file records what was added on top of that upstream, for the BME / Nokia Bell Labs
project **"Hybrid SON and Multi-Agent AI for Autonomous Optimization of Future Mobile Networks"**.

All figures below are derived directly from git, comparing this branch (`main`) against the tracked
upstream (`upstream/main`). They were produced with:

```
git diff --name-only --diff-filter=A upstream/main..main   # files created
git diff --name-only --diff-filter=M upstream/main..main   # files modified
git diff --numstat upstream/main..main                     # per-file +/- line counts
git ls-files | wc -l                                       # total tracked files
```

## Table 1 — Files created from scratch

| File | Lines | What it does |
|---|---:|---|
| `analysis/load_metric.py` | 231 | Core read-only load-metric module, imported by the other analysis scripts. `build_load_view()` assembles one row per (cell, timestamp) for NR cells 2–8 from the SQLite datalake; `compute_load()` produces a compound load score in [0,1] from PRB / UE-count / volume / quality components. Run as a script it validates whether the compound metric ranks cells differently from raw PRB utilization. |
| `analysis/load_metric_diagnose.py` | 189 | Diagnostic over 7 baseline seeds: component collinearity matrices, a spectral-efficiency candidate term, an AUC predictive test (predictor at t vs degradation at t+D), and a weight grid search. Read-only; does not modify `load_metric.py`. |
| `analysis/EXPERIMENT_RESULTS.md` | 164 | Documentation (not code): accumulating record of experiment results from V28 onward, with a run-UUID registry and per-experiment result tables. |
| `analysis/load_metric_sweep.py` | 151 | Multi-seed sweep of the load metric over the Phase-0 baseline runs; reports hotspot tracking, temporal variation, rank churn, and component dominance. Includes `coerced_copy()`, which copies each database to scratch and repairs TEXT `-inf` SINR values before analysis. Truncated episodes are excluded loudly, not averaged in. |
| `analysis/PHASE1_FINDINGS.md` | 148 | Documentation (not code): write-up of the Phase-1 compound load metric — datalake KPM column inventory, the rejection of the SINR histogram as a quality term, the idle-cell audit, and the final metric weights with their stated limits. |
| `compare_runs.py` | 136 | Before/after comparison CLI for two runs (given by UUID). Prints a delta table (data delivered, total backlog, signal-loss events) computed via `load_metric`, and can optionally emit a styled HTML card. |
| `analysis/load_metric_inspect.py` | 130 | Manual-inspection tool comparing two candidate weightings (A vs B) of the load score across 7 seeds; prints per-cell context for the highest-scoring and disagreement cases so a human can judge the ranking. No statistics beyond counting. |
| `analysis/cio_conditions_analysis.py` | 95 | Cross-condition analysis for the CIO experiments (C0–C4). Computes per-cell episode means (load under two weightings), handover counts, RLF-row counts, delivered volume, and backlog from a run's `database.db` and its handover trace. |
| `examples/cio_experiment.py` | 83 | Driver that runs one CIO experiment condition (a static per-cell dB bias) at 21 UEs, seed 555, by reusing `EnergySavingEnv` with the control file swapped to the CIO lever. This is the driver behind conditions C0–C4. |
| `examples/cio_headroom.py` | 83 | The V30 headroom-scan variant of the CIO driver. Functionally identical to `examples/cio_experiment.py` except the scenario config uses `ues: [5]` (35 UEs) instead of `[3]`. Note: its docstring was not updated and still describes the 21-UE C0–C4 conditions from the sibling file. |
| `tests/test_server.py` | 54 | Standalone test bridge script to validate the `ZmqStateDatabase` class, test time-series delta retrieval, and verify the synchronous bidirectional ns-3 handshake. |
| `bridge/zmq_database.py` | 101 | Defines a ZeroMQ server class that acts as an in-memory database and communication bridge to exchange JSON-formatted state updates and control actions between the ns3 simulator and a Python environment. |

## Table 2 — Files modified

| File | +added / −removed | What was changed |
|---|---:|---|
| `src/nsoran/datalake.py` | +64 / −4 | Three additions to the ingestion path: (1) `_coerce_value()` and its use to coerce CSV string values to each column's declared type before insert, so `-inf` outage markers and empty fields store as proper REAL/NULL instead of TEXT; (2) a `(timestamp, ueImsiComplete)` collision policy (`collision_keep_min`) that, for `gnb_cu_cp`, keeps the row with the worst L3 serving SINR — replacing the stored row when a strictly-worse one arrives — so an outage marker always survives ingestion; (3) a read-only `has_kpms(timestamp)` existence check used by the env. |
| `src/environments/scenario_configurations/es_use_case.json` | +54 / −17 | Reformatted from single-line arrays to multi-line, and added one new key `e2nrEnabled: [1]`. The added key is the only semantic change; the remaining diff is whitespace reformatting. |
| `src/nsoran/ns_env.py` | +41 / −12 | Restructured `step()` to handle the episode-boundary step: when `last_timestamp` has been advanced past the last KPM period (no rows exist, so `read_kpms()` returns None), it now returns a valid terminal 5-tuple with a zero observation instead of crashing. Also sorts the `cu-up` / `cu-cp` / `du` file globs by ascending cellId (`extract_cellId`) so cross-file collisions dedup deterministically. |
| `src/environments/es_env.py` | +5 / −2 | In the RLF-counter path, replaced `df['L3 serving SINR'].replace(-np.inf, 0)` with `pd.to_numeric(..., errors='coerce')` so outage rows keep their `-inf` value and are counted as an RLF by the downstream `< -5` test. Added comment documents the "Option B" RLF definition. |
| `.gitignore` | +3 / −0 | Added `*.bak`, `build_log*.txt` and `output/` ignore patterns. |
| `pyproject.toml` | 19 | Build system configuration to allow editable (`pip install -e .`) package installation, resolving local `ns_o_ran_gym` module import paths cleanly across the environment. |

## Summary

- **Total lines added:** 1,817 (of which 1,565 are in the 12 new files; 252 in modified files). 83 lines removed.
- **Files created:** 12
- **Files modified:** 6
- **Files untouched:** 25 (of 41 tracked files total)

## Line-level authorship

`git blame <file>` shows line-level authorship for anything in the repository, including which
lines in the modified files above came from upstream versus this fork.
