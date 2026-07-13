# Phase 1 Findings — The Compound Load Metric

**Date:** 2026-07-13
**Scope:** working-Phase-1 (compound load metric) of the MLB plan. Thesis-section draft.
All numbers below were re-verified against artifacts on 2026-07-13; where a claim comes
from a single audited run, that run is named. Code: `analysis/load_metric.py` (metric),
`analysis/load_metric_sweep.py` (7-seed sweep), `analysis/load_metric_diagnose.py`
(collinearity/AUC/grid-search), `analysis/load_metric_inspect.py` (manual inspection of
weighting candidates).

Audited runs referenced:
- `ee4dc79c-69d5-47e5-8f2c-95ffa5a7568c` — seed-555 baseline run used for the column
  audit, SINR-bin audit, and dedup audit.
- Baseline sweep seeds 555, 556, 560, 561, 562, 563, 564 (7 complete episodes;
  557–559 excluded — they died to the V19 crash, since fixed).

---

## 1. Column inventory: what the datalake actually delivers

The gym ingests four KPM streams per NR cell (2–8) into SQLite: `du`, `gnb_cu_cp`,
`gnb_cu_up`, plus LTE cell 1 (`lte_cu_cp`, `lte_cu_up`). Not all columns carry data.
Verified on run `5d523414` (2,079 rows per table, 99 timestamps × 21 UEs):

**Usable (populated, cell-level or per-UE):**
- `du`: `dlprbusage` (PRB %), `qosflowpdcppduvolumedl_filter` (DL PDCP volume/period),
  `drbbuffersizeqos` (DL buffer backlog), `rruprbuseddl`, MCS distributions, TB counts,
  L1M.RS-SINR bins (see §2 for why they are still unusable), per-UE throughput
  (`drbuethpdlueid`). Cell-level columns are repeated identically on every UE row of a
  cell — take `first`, never sum (asserted in `load_metric.build_load_view`).
- `gnb_cu_cp`: `numactiveues` (cell-repeated, = ueMap.size() at report time),
  `l3_serving_sinr` (per-UE serving SINR — the quality signal Phase 1 uses),
  per-UE neighbour SINRs (`l3 neigh SINR *` — populated but so far **unused**; see §5).

**Always NULL, and why:**
- `gnb_cu_up`: both NR data columns (`qosflowpdcppduvolumedl_filterueid…`,
  `drbpdcppdunbrdlqosueid…`) are 0/2,079 non-NULL. **Blank at source** — the raw
  `cu-up-cell-{2..8}.txt` rows carry empty fields for every column after
  `ueImsiComplete`. NR DL volume must come from `du.qosflowpdcppduvolumedl_filter`
  instead (it does).
- `lte_cu_up`: the four per-UE columns (`…txbytes`, `…txdlpackets`,
  `…pdcpthroughput`, `…pdcplatency`) are 0/2,079 non-NULL while the two cell-level
  columns are fully populated. **Key-name space mismatch** at `datalake.py:51-54`: the
  schema keys contain a space — `"DRB.PdcpSduVolumeDl_Filter.UEID (txBytes)"` — while
  the CSV header has none — `"DRB.PdcpSduVolumeDl_Filter.UEID(txBytes)"`. `insert_data`
  filters the row dict to schema keys, so the mismatched columns are silently dropped
  and remain NULL. (One-character fix if LTE per-UE volume is ever needed; harmless to
  MLB, which acts on NR cells 2–8.)

## 2. The SINR-histogram rejection

Nokia's compound-load directive uses the per-cell SINR histogram as the quality term.
The simulator does export one (`L1M.RS-SINR.Bin34/46/58/70/82/94/127`), but it was
rejected after audit. Evidence (run `ee4dc79c` unless noted):

1. **The bins are transport-block-weighted PHY SINR, not a UE distribution.** They are
   incremented once per received transport block in `MmWavePhyTrace::UpdateTraces`
   (`mmwave-phy-trace.cc:305-333`, called from the per-TB RX trace,
   `RxPacketTraceEnbCallback`). A cell counts only when it transmits: at the first
   indication tick, cell 2 had 8 active UEs with serving SINRs from 4.1 to 35.6 dB and
   **all seven bins zero** — no DL traffic had flowed yet.
2. **Exclusive 6 dB bins.** Edges: ≤−6, −6..0, 0..6, 6..12, 12..18, 18..24, >24 dB.
   Each TB increments exactly one bin; 6 dB granularity is coarser than the effects
   MLB needs to see (the DynamicTtt selector's whole dynamic range for TTT is 3–20 dB
   of *difference*).
3. **Absent or near-empty exactly where quality matters.** 77 of 693 present
   (cell, timestamp) pairs have all-zero bins, plus 52 pairs absent from `du` entirely
   → **129/693 ≈ 18.6 % of cell-timesteps carry no usable bin data**. The starvation is
   concentrated on lightly-loaded cells: cells 3 and 6 averaged only **16.6 and 7.0
   counts/tick** vs cell 2's 1,135.7. A 7-seed re-check (sweep DBs) shows the same
   shape: 9.7 % of present cell-timesteps all-zero, 15.0 % of the full grid unusable.

A quality term built on these bins would be silent or noise-dominated precisely on the
cells MLB most needs to rank (lightly-loaded offload targets). **Deviation from the
Nokia spec, recorded:** the quality component instead uses per-UE `l3_serving_sinr`
aggregated per cell with `min` (worst user), linearly mapped from [−5, 30] dB into
[0, 1] (`compute_load`, `load_metric.py:139-141`).

## 3. The idle-cell audit: absence means idle, not lost

A cell can be absent from `du` at a timestamp. Before treating that as "zero load" the
ingestion path was audited end-to-end (artifact:
`cio_verification/scratchpad/dedup_audit.py`, run `ee4dc79c`):

- Raw `du-cell-*.txt` rows: **2,079. Rows in the db `du` table: 2,079.** Nothing dropped.
- **0** (timestamp, imsi) pairs reported by more than one cell in the raw du files →
  the V21 dedup path had nothing to discard in this run.
- **0** (cell, ts) pairs with raw rows but no db rows; **0** undercounted pairs.
- The 52 absent (cell, ts) pairs (cells 3: 2, 4: 12, 6: 38) have **no rows in the raw
  files either** — the cell genuinely served nobody. `build_load_view` therefore encodes
  absence as an idle row (zeros for volume/count columns, NaN SINR), not as missing data.

## 4. Final weights, their justification, and their honest limits

**Chosen weighting (Phase-1 final, provisional):**
`prb 0.15, ues 0.40, vol 0.10, qual 0.35` (the "W_B" of
`load_metric_inspect.py:26`).

> **Code note:** `load_metric.py`'s `DEFAULT_WEIGHTS` still carries the pre-decision
> starting point (prb .40/ues .20/vol .20/qual .20). The chosen weights must be passed
> explicitly (`compute_load(view, weights=...)`). Left as-is deliberately: the C0–C3
> CIO analyses of 2026-07-12/13 were run under the defaults, and changing the default
> mid-experiment would fork the numbers. Reconcile before Phase-3 code freezes.

**Why these weights — three converging lines, no single decisive one:**
1. **Nokia's directive** that load be a compound measure (PRB + users + throughput +
   quality), not raw PRB.
2. **An AUC grid search** (`load_metric_diagnose.py`) over predictors of near-term
   degradation, which assigned **zero weight to PRB unprompted** — raw PRB utilization
   predicted upcoming degradation at AUC ≈ 0.508, i.e. chance; the best single- or
   combined-predictor AUC found anywhere in the grid was **0.607**.
3. **Manual inspection of disagreement cases** (`load_metric_inspect.py`, artifact
   `cio_verification/scratchpad/inspect_out.txt`): where W_A and W_B rank differently,
   the W_B pick is the cell a human operator would offload (many UEs + bad worst-user
   SINR, vs a PRB-saturated cell serving one happy user).

**What the weights are validated as — and not:**
- Validated: an **identifier of cells a human would offload**, judged by inspection
  across 7 seeds.
- NOT validated: a **predictor of degradation**. No optimum is identifiable from
  failure-free baseline data — the baseline contains almost no degradation events to
  predict (best AUC 0.607 barely above chance). The weights are **provisional**; if
  Phase-3 training shows the reward mis-ranking cells, re-derive them from data that
  actually contains failures (e.g. biased/perturbed runs), not from the baseline.

## 5. Open items carried into Phase 3

1. **Offload-target reachability is never verified.** The metric ranks how loaded a
   cell is; neither it nor the CIO lever checks that the UEs being pushed off can
   actually reach the intended target at usable SINR. The per-UE **neighbour SINR
   columns in `gnb_cu_cp` exist and are populated but unused** — they are the natural
   input for a reachability check (and for MRO later).
2. **The dedup collision path is untested under stress.** The V21 worst-SINR-wins
   policy was verified on baseline runs where collisions are rare (0 in the audited
   run). Cell-shutoff (ES) and RLF-heavy regimes — exactly what Phase-3 training will
   produce — will exercise it far harder; audit again on the first biased/ES training
   runs.
3. **Weights default mismatch** (§4 code note): unify `DEFAULT_WEIGHTS` with the chosen
   weighting before training code reads it implicitly.
4. **Load migrates, it does not vanish — and the metric's top pick is not the right
   offload target** (V26/V27, `VERIFICATION_LOG.md`). Offloading cell 2 cost ~6 % of
   delivered volume; offloading the metric's top-ranked cell (cell 3, V27) cost
   ~17 % — its high score was *productive* load (it was the top volume producer) —
   and total DL backlog ended no better than the cell-2 offload (both ~40 % above
   the no-bias baseline). The metric measures how busy a cell is, not whether
   offloading it helps the network. Any "minimize load" reward built on this metric
   alone will teach the agent to tank throughput. The reward must trade load balance
   against delivered volume, and MLB must set CIO on all cells jointly.
