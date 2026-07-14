# EXPERIMENT RESULTS — Nokia SON/MARL Project

**Purpose:** the single accumulating record of every experiment's numbers, structured so
they can be lifted directly into thesis tables. Every number is traceable to a run UUID
under `ns-o-ran-gym/output/<uuid>/`. Claims about *how* each number was verified live in
`~/oran-project/VERIFICATION_LOG.md` (claim → command → output → verdict); this file
holds the results themselves.

**Append convention:** one `## V<N> — <title>` section per experiment, newest at the
bottom, matching the V-numbering of the verification log. Each section records: date,
setup (seed, UE count, build profile), a run-UUID table, the result tables, and a
verdict/decisions block. Add every new run to the Run Registry below. Never edit past
sections except to fix a demonstrated transcription error (note the correction in place).

Experiments V13–V27 predate this file; their numbers are recorded in
`VERIFICATION_LOG.md` and `analysis/PHASE1_FINDINGS.md`, and the Phase-0 10-seed
baseline in `PROJECT_STATUS.md` §5. New experiments (V28+) are recorded here.

---

## Run Registry

| Run UUID | Experiment | Seed | UEs | Build | Condition | Trace md5 (`CellIdStatsHandover.txt`) |
|---|---|---|---|---|---|---|
| `d7d1d669-d658-4178-adf5-b97a1d7259fc` | V24 baseline (pre-dates this file) | 555 | 21 | debug | no CIO | `38e5f1bb9a45c702a6a7d4ce3e5b2ade` |
| `ab06637e-543e-450d-a4be-dee2299d97c4` | V28 | 555 | 21 | optimized | no CIO | `38e5f1bb9a45c702a6a7d4ce3e5b2ade` |
| `c321ab7c-d61b-4eaa-90cd-59061c7d36d0` | V29 | 555 | 21 | optimized | no CIO, parallel ×3 | `38e5f1bb9a45c702a6a7d4ce3e5b2ade` |
| `abb2b80f-9e24-4577-90ea-d7119cc61435` | V29 | 556 | 21 | optimized | no CIO, parallel ×3 | `f78f837b0c020eeef22c7a64e316a5a1` |
| `022f459a-aab9-480c-8dba-d2bd7c126081` | V29 | 557 | 21 | optimized | no CIO, parallel ×3 | `8a3b3ba446173ebdf8a91aacffaddbb3` |
| `44dadaab-51d6-4eea-90cb-b6fcf53a040b` | V30 scan + H0 | 555 | 35 | optimized | no CIO | — |
| `da046e0c-47d2-4ca8-be84-4db32b41275f` | V30 scan | 555 | 63 | optimized | no CIO | — |
| `5e89652b-3c20-4b29-bf4f-bba31fba96b2` | V30 H1 | 555 | 35 | optimized | cell 8 −3 dB | — |
| `7cb3494f-8c92-488d-a411-bd757fda71c1` | V30 H2 | 555 | 35 | optimized | cell 8 −6 dB | — |

---

## V28 — Optimized ns-3 build: 2.8× wall-time speedup, physics byte-identical

**Date:** 2026-07-14
**Setup:** `./ns3 configure -d optimized` (profile confirmed via `./ns3 show profile`);
seed 555, 21 UEs (`ues=3`), zero CIO, scenario-three, 10 s episode.
**Run:** `ab06637e-543e-450d-a4be-dee2299d97c4` (log: `~/oran-project/opt_test.log`).

| Metric | Debug build (V24 baseline, `d7d1d669`) | Optimized build (`ab06637e`) |
|---|---|---|
| Wall time / episode | 2799 s (46m39s) | **987 s (16m27s)** |
| Speedup | — | **2.8×** |
| `CellIdStatsHandover.txt` md5 | `38e5f1bb9a45c702a6a7d4ce3e5b2ade` | `38e5f1bb9a45c702a6a7d4ce3e5b2ade` |

**Verdict:** 2.8× speedup with a **byte-identical** handover trace vs the debug
baseline. Physics unchanged; **all prior results remain comparable** across build
profiles. The optimized build is now the default for all experiments.

## V29 — Parallel seed execution: 3 concurrent runs do not interfere; ~8× combined throughput

**Date:** 2026-07-14
**Setup:** 3 concurrent scenario-three runs, seeds 555/556/557, optimized build, 21 UEs,
zero CIO. Logs: `~/oran-project/par_555.log` / `par_556.log` / `par_557.log`.

| Seed | Run UUID | Trace md5 |
|---|---|---|
| 555 | `c321ab7c-d61b-4eaa-90cd-59061c7d36d0` | `38e5f1bb9a45c702a6a7d4ce3e5b2ade` |
| 556 | `abb2b80f-9e24-4577-90ea-d7119cc61435` | `f78f837b0c020eeef22c7a64e316a5a1` |
| 557 | `022f459a-aab9-480c-8dba-d2bd7c126081` | `8a3b3ba446173ebdf8a91aacffaddbb3` |

- **Wall time:** 17m07s for all three concurrent vs 16m27s for one alone (V28).
- **CPU:** user time 50m28s across the batch — ~3 cores saturated, as expected for
  3 single-threaded processes.
- **Non-interference:** seed 555's trace md5 is still `38e5f1bb…` — identical to the
  solo optimized run (V28) and to the debug baseline (V24). Parallel runs do not
  perturb each other.
- **RAM:** peak ~5.7 GB of 15 GB at 21 UEs × 3 runs.
- **Scope/limits:** ns-3 is single-threaded; the concurrency is **per-seed, not
  per-run**. Untested: 6 concurrent seeds, and RAM headroom at 35/63 UEs.

**Verdict:** combined with V28, seed throughput goes from ~1 seed per 47 min to
~3 seeds per 17 min (**~8×**).

## V30 — Headroom experiment: MLB only has room to work in the congested-but-not-saturated regime

**Date:** 2026-07-14
**Motivation:** at 21 UEs, every CIO intervention in V26/V27 lost on delivered volume.
If doing nothing is a strong policy, an agent has nothing to learn. So we scanned the
load axis.
**Driver:** `examples/cio_headroom.py` (committed with this file). Logs:
`~/oran-project/headroom_35.log`, `headroom_H0.log`, `headroom_H1.log`, `headroom_H2.log`.
**Runtime note:** 21 UEs ≈ 16 min; 35 UEs ≈ 55 min; 63 UEs = 100m48s (optimized build) —
scaling is worse than linear in UE count.

### Scan: no-CIO baselines at three UE counts (seed 555)

| UEs | `ues` param | Run UUID | Load spread (max−min) | Busiest cell | Verdict |
|---|---|---|---|---|---|
| 21 | 3 | `d7d1d669-d658-4178-adf5-b97a1d7259fc` | 0.51 (0.10–0.61) | 3 | Too light — no congestion to fix |
| 35 | 5 | `44dadaab-51d6-4eea-90cb-b6fcf53a040b` | 0.31 | 8 | Hotspot AND an idle cell — usable |
| 63 | 9 | `da046e0c-47d2-4ca8-be84-4db32b41275f` | 0.16 | 2 | Uniformly saturated — nowhere to offload |

**Key observation at 63 UEs:** the network is not unbalanced, it is **over capacity**.
Every cell carries a 3–20 MB backlog; there is no idle cell to offload to. Two cells
show `sinr_min = -inf` (real outages). Load balancing cannot fix a capacity shortfall.

**Load-metric weakness, confirmed in a second regime (consistent with V27):** at 63 UEs
the load metric ranked cell 2 (0.484) above cell 8 (0.469) — but cell 8 carried a
19.7 MB backlog vs cell 2's 6.2 MB. The metric weights user count at 0.40 and cell 2 had
more users. Under uniform congestion the metric fails to identify the cell in worst
trouble. This is the same weakness V27 exposed at 21 UEs, appearing in a new regime.

### The 35-UE regime — per-cell, no CIO (H0, `44dadaab-…`)

| Cell | n_ues | PRB % | Buffer | sinr_min |
|---|---|---|---|---|
| 2 | 8.97 | 47.65 | 5.25 MB | 7.05 |
| 3 | 5.73 | 70.65 | 9.85 MB | 8.57 |
| 4 | 3.42 | 6.84 | 0.12 MB | −inf |
| 5 | 2.84 | 22.93 | 0.61 MB | 11.73 |
| 6 | 2.49 | 0.07 | 4 bytes | 17.79 |
| 7 | 4.05 | 19.31 | 2.73 MB | 11.41 |
| 8 | 7.57 | 62.57 | 14.20 MB | 8.30 |

Cell 8 is the hotspot; cell 6 is essentially idle. A victim and a destination — the
condition MLB requires.

### The headroom test — bias applied to cell 8 (seed 555, 35 UEs)

| Condition | Run UUID | Data delivered | Total backlog | RLF | Mean sinr_min | Handovers |
|---|---|---|---|---|---|---|
| H0 — no CIO | `44dadaab-51d6-4eea-90cb-b6fcf53a040b` | 123.4 MB | 3,244.2 MB | 4 | 10.44 | 178 |
| H1 — −3 dB cell 8 | `5e89652b-3c20-4b29-bf4f-bba31fba96b2` | 136.6 MB (+10.7%) | 2,281.9 MB (−29.7%) | 0 | 10.08 | 155 |
| H2 — −6 dB cell 8 | `7cb3494f-8c92-488d-a411-bd757fda71c1` | 136.8 MB (+10.9%) | 2,253.8 MB (−30.5%) | 2 | 9.69 | 180 |

**Verdict: headroom exists.** A hand-set bias improves throughput, backlog and dropped
calls **simultaneously — no trade-off**. This is the first unambiguous win in the
project, and it is the opposite of the 21-UE result (where every intervention cost
throughput).

**Network-wide effect, not just the biased cell.** Offloading cell 8 unclogged cells
that were never touched:
- cell 2 buffer: 5.25 MB → 1.52 MB
- cell 4 buffer: 124,888 B → 24 B
- cell 5 buffer: 614,205 B → 16,408 B

**−6 dB is NOT better than −3 dB.** Same volume and backlog, but 2 RLFs vs 0, worse
mean SINR, and 180 handovers vs 155. The gentle push wins. Consistent with the 21-UE
finding that only −3 dB beat baseline on backlog. The reward must penalise RLFs and
over-aggression, or the agent will overshoot.

**Limits:** single seed (555), single episode per condition — point estimates, not
distributions.

### Decisions recorded (V30)

1. **The experimental scenario moves from 21 UEs to 35 UEs (`ues=5`).** 21 has nothing
   to fix; 63 has nothing to fix it with. This is a defensible, evidence-backed choice —
   the full scan is recorded above so it cannot be mistaken for tuning.
2. **The Phase-0 baseline is now the wrong scenario.** It was measured at 21 UEs. A new
   10-seed baseline at 35 UEs is required before training. (Cheap now: ~3 seeds per hour
   with the optimized build + parallelism.)
3. **The load metric's weakness is confirmed in a second regime** (V27 at 21 UEs, V30 at
   63 UEs): it tracks how busy a cell is, not how much trouble it is in. Keep it as an
   observation; do not use it as the reward.
4. **Reward design is now much better constrained.** In the 35-UE regime, throughput ↑,
   backlog ↓ and RLF ↓ all move together — they are not in conflict. The reward can
   optimise all three without pricing a trade-off, but must penalise RLF to prevent
   overshoot.
