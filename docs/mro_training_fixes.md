# MRO reward and action: what we built, what we measured, and what's still open

Companion to `docs/mlb_training_fixes.md`, same purpose: a numbers-backed record
of what was built, what was measured, and what remains unresolved for the MRO
(Mobility Robustness Optimization) agent -- the second SON function under
Architecture A. Full term-by-term reward design and its ns-3 source audit are
in `docs/mro_reward_design.pdf`; this document covers what happened after that
design was implemented.

---

## What was built

- **`MroZmqEnv`** (`src/environments/mro_zmq_env.py`) -- standalone, over the
  same ZeroMQ bridge `MlbZmqEnv` uses.
- **Five-term reward**: `HandoverQuality`, `PingPong`, `HandoverRate`,
  `OutageExposure`, `ActionSmoothness`. `HandoverQuality` replaces the
  reward-design brief's literal RRC handover-success rate (traced: the mmWave
  secondary-cell handover path never goes through the RRC state machine that
  can report failure, so a literal success rate would read a constant 1.0 in
  this simulator). `OutageExposure` replaces a literal RLF rate
  (`LteUeRrc::DoNotifyRadioLinkFailure` is confirmed dead code -- an empty
  stub, never called).
- **Action**: a per-cell handover-margin OFFSET in dB (`Box(-3, 3, (7,))`),
  sent as `ho_margin_db`, ADDED to (not replacing) the network's base
  `HoSinrDifference` attribute.
- **ns-3 wiring** (`feature/mro-agent` on `ns-o-ran-ns3-mmwave`):
  `LteEnbRrc::m_hysteresisMarginOffset` (a per-cell map mirroring
  `m_cellIndividualOffset`'s own shape), `GetEffectiveSinrThreshold()`
  replacing all 10 direct reads of `m_sinrThresholdDifference` across
  `TttBasedHandover` / `ThresholdBasedSecondaryCellHandover` /
  `ThresholdBasedInterRatHandover`, `SetHandoverMarginOffset()` (clamped to
  [-3, +3] dB), and a new `ho_margin_db` branch in
  `LteEnbNetDevice::ApplyControlPayload`. Compiles clean (0 errors, 3
  pre-existing warnings unrelated to this change).

**Why standalone rather than a subclass of `MlbZmqEnv`**: `mlb_zmq_env.py` was
under active, concurrent development (the PPO -> SAC migration and the
reward rate-normalization fix landed while this work was in progress -- see
below). Duplicating the small amount of ns-3/ZeroMQ lifecycle boilerplate was
a deliberate, temporary tradeoff to avoid two people editing the same file at
once, to be collapsed into a shared base once both environments stabilise.

## Smoke test

`examples/train_mro_smoke.py`, same CFG and argparse defaults as
`train_mlb_smoke.py`. 16 timesteps, 2 PPO rollouts, `ep_len_mean=10`,
`ep_rew_mean=-16.1` (a real, non-degenerate number from the 5-term reward),
exit code 0. Confirms the full pipeline -- including the new `ho_margin_db`
wiring -- runs end to end against real ns-3. Does not prove the action has a
real physical effect; that needed a dedicated check (next section).

## Does the margin action do anything real? (`examples/mro_margin_experiment.py`)

Two fixed-action arms (no RL agent), same seed (`RngRun=555`,
`vary_rng_run_per_episode=False`), `ues=3`, `simTime=10`:

| arm | margin offset | ZMQ-reported HO | ZMQ-reported PP | **ns-3's own trace file** |
|---|---|---|---|---|
| neutral | 0 dB | 77 | 29 | **85** |
| loosen | -3 dB | 82 | 25 | **91** |

**Result: PASS, with honest caveats.** Loosening the trigger margin genuinely
increased real handover activity in ns-3's own `UeHandoverStartStats.txt`
(+7%), cross-checked independently of the derived Python-side numbers -- the
wiring has a real, correctly-directional physical effect, not just an
inert-and-safe one.

Caveats, not glossed over:
- n=1 seed per arm. A +7% effect from a single run is a sanity check, not a
  statistically powered result.
- `PingPong` counterintuitively DROPPED (29 -> 25) even though raw handovers
  rose. Not a contradiction -- ping-pong is a specific return-within-0.8s
  pattern, not the same thing as overall churn -- but worth keeping in mind
  once real weight-tuning starts.
- The ZMQ-reported totals under-count ns-3's own file by ~9 in both arms
  (77 vs 85, 82 vs 91). Both arms only logged 9 of the expected 10 steps: the
  final control step hits the environment's time-limit-truncation branch,
  which deliberately returns `reward=0.0` with no `reward_terms` (correct for
  training -- that terminal observation is bootstrap-only). This is why the
  experiment was designed to trust ns-3's own file as ground truth rather than
  the derived count.

## Baseline weight measurement, round 1 (`examples/mro_baseline_experiment.py`)

5 seeded do-nothing (all-zero-action) episodes, same config as the smoke test
(`ues=3`, `simTime=10`, default `controlInterval` = 1.0 s), following MLB's
own weight-derivation method: weights from each term's DETRENDED standard
deviation (per-seed linear trend removed before pooling), not hand-picked.

| term | n | mean | detrended sd | derived weight |
|---|---|---|---|---|
| quality | 40/45 | 0.579 | 0.230 | 1.00 (anchor) |
| pingpong | 45 | 0.136 | 0.074 | 3.11 |
| handover_rate | 45 | 0.483 | 0.146 | 1.57 |
| outage | 45 | 0.000 | 0.000 | **undefined** |

`OutageExposure` was exactly 0.0 -- mean, sd, every one of 45 steps across all
5 seeds. No UE in any of these runs ever dropped below -5 dB SINR.

## Two problems found before investing further tuning effort

Both surfaced by comparing this work against what landed on `origin/main`
in the meantime (`c576d6c`, `13be2a8`, `696010f`, `083550e`):

1. **`ues=3` is too light a network.** Mirrors MLB's own documented lesson
   (`ues=3` -> 21 UEs is a "healthy network" with no real congestion
   traction; MLB's own locked training config moved to `ues=5`). The SAC
   pilot's locked CFG (`train_mlb_sac_pilot.py`) already reflects this:
   `ues=5`, `simTime=30`. Round 1's baseline was run at the wrong regime for
   `OutageExposure` to ever show signal.

2. **`PingPong` and `HandoverRate` are not rate-normalized.** `c576d6c` found
   that MLB's own `PingPong` (`pingpong_count / active_ues`) is a COUNT PER
   CONTROL STEP in disguise, not a rate -- it silently rescales if
   `controlInterval` changes, unlike every other term, which is a level or
   ratio. MRO's `PingPong` was copied from MLB's pre-fix code and has the
   identical bug; `HandoverRate` (new, MRO-only) is structurally the same
   shape and has never been checked for this. This is not cosmetic: the
   colleague's locked SAC config runs at `controlInterval=0.1`, not the 1.0 s
   default this baseline was measured at, so the bug would actively distort
   both terms the moment MRO adopts the shared config. `HandoverQuality` and
   `OutageExposure` are pure ratios/fractions, immune to this class of bug,
   same as MLB's non-`PingPong` terms.

## Status / next steps

- [x] Rebase `feature/mro-agent` onto `origin/main` (pulls in the
      rate-normalization pattern, SAC pilot infra, the `sys.executable`
      portability fix, and the `/dev/null` trace-file redirects)
- [x] Fix `PingPong` / `HandoverRate` rate-normalization in `mro_zmq_env.py`,
      mirroring `c576d6c`'s `handover_window_s`-based approach exactly; add
      the equivalent T-invariance test. Both terms now share one
      `HANDOVER_RATE_REF_S` scale factor (they count the same event type
      over the same reported window); a `REWARD_FORMULA_VERSION` stamp was
      added at the same time, mirroring MLB's own. At the reference window
      (1.0 s, what every number in this document was measured at) the fix is
      an exact no-op -- confirmed by the full suite passing unchanged
      (12/12 MRO, 37/37 combined with MLB) -- so round 1's baseline table
      above is NOT invalidated by this fix, only by the `ues=3` regime
      problem below. New test
      `test_pingpong_and_handover_rate_are_rates_not_per_step_counts`,
      mutation-tested (confirmed it fails when the scale factor is dropped).
- [x] Re-run the baseline weight measurement ONCE, at the colleague's locked
      SAC config (`ues=5`, `simTime=30`, `controlInterval=0.1`, RLC AM,
      `/dev/null` trace redirects) instead of the smoke-test config. Full
      writeup in `20260910_Dani_mro_reward_implementation.pdf` Section 4.4-5;
      summary below. 2 of 5 seeds crashed mid-episode, cause unexplained (see
      the crash-rate checklist item further down); pooled stats are 1,372
      steps, not five clean equal-weight runs.
- [x] Resolve the `OutageExposure` / `ActionSmoothness` weight edge cases with
      a stated rule rather than case by case, per
      `20260915_Dani_RewardWeightMethodologyReport.pdf` Section 5 ("rare
      terms" / "silent terms") -- see the round-2 results table below and
      `DEFAULT_W_OUTAGE`/`DEFAULT_W_SMOOTH`'s code comments in
      `mro_zmq_env.py` for the full rationale.
- [x] Audit every MRO reward term for boundedness (methodology report
      Section 5.1, zero simulator cost). Found one real gap: the
      `handover_rate_scale` factor (shared by `PingPong`/`HandoverRate`) was
      only guarded against `handover_window_s <= 0.0`, not against a small
      *positive* window -- an unguarded report of e.g. 0.001 s would have
      inflated both terms 1000x. Fixed with `HANDOVER_WINDOW_FLOOR_S` (floors
      the window at 1% of `HANDOVER_RATE_REF_S`, capping the scale factor at
      100x); new test `test_handover_rate_scale_is_capped_for_a_pathologically_small_window`,
      mutation-tested. The other four terms are all ratios/fractions or
      clipped-action deltas, confirmed mathematically bounded by construction.
- [x] Run the adversarial `OutageExposure` stress test -- a fixed +3 dB
      "sticky" margin on every cell (more restrictive, not less: the
      too-late-handover mechanism that should actually drive outage), 10
      seeds pooled across two parallel batches. See "`OutageExposure`
      adversarial stress test (final)" below. Confirms the fixed-guard
      decision with real evidence instead of only an analogy to `BadSignal`.
- [x] Measure `ActionSmoothness` from the real policy class instead of a
      hand-approximated Monte Carlo distribution -- a freshly-initialised,
      untrained SB3 SAC policy (the project's own locked hyperparameters)
      acting against a synthetic-but-representative KPI stream, no ns-3
      needed (see `examples/mro_actionsmoothness_probe.py`). Real sd = 3.19,
      about 3x the old guess (sd ~1.06) -- `DEFAULT_W_SMOOTH` updated from
      0.26 to 0.086. Still not a measurement under REAL training rollouts
      (no MRO training run has happened yet), so still provisional in that
      narrower sense -- a measurement of the real policy class, not of real
      training behaviour.
- [x] Added a trajectory-ranking sanity check
      (`examples/mro_weight_ranking_check.py`), closing the methodology
      report's Section 5.3 gap: confirms the final weight set ranks the
      sticky (too-restrictive-margin) arm WORSE than do-nothing, as a real
      MRO failure mode must. PASS, with a real but modest per-step margin --
      see below, not glossed over.
- [ ] Build `train_mro_sac_pilot.py` mirroring `train_mlb_sac_pilot.py` (Arm C
      hyperparameters, `DiskGuard`/`MemGuard`, `CriticProbe`/`StepLogger`)
- [ ] Run MRO's own SAC pilot against the same "does the critic climb" bar
- [ ] Investigate the round-2 mid-run crash rate (2/5 seeds, unexplained) at
      `ues=5`/`controlInterval=0.1` before trusting a large training run on
      this config. NOTE: a related, narrower issue was observed live during
      the outage-stress run -- one seed's ns-3 process logged a non-zero
      exit code at the very end of a run that otherwise collected its full
      299/299 steps. `MroZmqEnv.step()`'s end-of-episode branch does not
      currently distinguish a clean `simTime` finish from a crash (both hit
      the same `next_kpis is None` path), so this went unflagged in the
      data itself -- harmless here (full matching row count), but worth
      fixing (an `info["crashed"]`/exit-code flag) before trusting this for
      real training, where a silent mid-episode crash would look identical
      to a legitimate truncation to the training loop.
- [ ] Re-measure `PingPong`/`HandoverRate` statistics from real training
      rollouts once the SAC pilot has run, per the methodology report's
      Section 5.2 ("do not trust a do-nothing-baseline weight forever")

## Baseline weight measurement, round 2 -- final (the real training config)

5 seeded do-nothing episodes at the locked SAC configuration (`ues=5`,
`simTime=30`, `controlInterval=0.1`, RLC AM, `/dev/null` trace redirects),
1,372 pooled control steps after the rate-normalization fix above:

| term | n | mean | detrended sd | derived weight | **weight in code** |
|---|---|---|---|---|---|
| signal_quality | 1076/1372 | 0.873 | 0.273 | 1.00 (anchor) | **1.00** |
| pingpong | 1372 | 0.119 | 0.181 | 1.51 | **1.51** |
| handover_rate | 1372 | 0.427 | 0.343 | 0.80 | **0.80** |
| outage | 1372 | 0.0001 | 0.0017 | 159.7 (not usable) | **1.00 (fixed guard)** |
| smoothness | -- (silent under do-nothing) | -- | 1.06 (Monte Carlo) | ~0.26 | superseded, see below |

`OutageExposure` is confirmed alive at this config (fired on 0.01% of
UE-steps, resolving round 1's open question), but its literal equalised
weight (159.7) is not usable -- matching its variance to `HandoverSignalQuality`
would make the reward swing enormously on the rare step it fires, the same
failure shape MLB's own `BadSignal` term hit (never equalised, kept at its
default instead). Per the methodology report's stated rule for rare terms,
`DEFAULT_W_OUTAGE` is fixed at 1.00 -- a modest guard weight in the same range
as the other four, not the 160x-scaled literal value. The sections below
replace the "pending the adversarial test" caveat this used to carry with a
real result.

## `OutageExposure` adversarial stress test (final)

`examples/mro_outage_stress_experiment.py`: a fixed **+3 dB "sticky" margin**
on every cell, every step -- not an arbitrary stress hack. A MORE restrictive
margin makes `GetEffectiveSinrThreshold` harder to satisfy, so a UE moving
away from its serving cell stays latched onto it LONGER than it should --
the textbook too-late-handover MRO failure mode, and exactly the mechanism
that should drive real SINR outage. The untested opposite extreme from
`mro_margin_experiment.py`'s already-validated "loosen" arm (easier
handovers, irrelevant to outage).

10 seeds, same locked config as round 2, run as two batches of 5 (see
"Parallel execution" below), pooled with `examples/mro_outage_stress_pool.py`:

| | n | outage mean | detrended sd | fired | derived weight |
|---|---|---|---|---|---|
| batch 1 (seeds 3001-3005) | 1495 | 0.0001 | 0.0013 | 3/1495 (0.201%) | 214.3 |
| batch 2 (seeds 3006-3010) | 1495 | 0.0001 | 0.0015 | 4/1495 (0.268%) | 186.1 |
| **pooled (10 seeds)** | **2990** | **0.0001** | **0.0014** | **7/2990 (0.234%)** | **198.7** |

**Result: the mechanism is real, the fixed-guard decision was correct.**
The firing rate rose from 0.01% (do-nothing) to 0.234% (sticky) -- about
**20x** more frequent, confirming the stress genuinely exercises the term,
not just the config. But even under a condition that reliably makes outage
fire 20x more often, the literal equalised weight (198.7) lands in the same
"clearly unusable" range as every measurement of this term so far (159.7 in
round 2, 214.3 and 186.1 per batch). This is now a validated conclusion, not
an assumption: no amount of realistic stress on this term produces a usable
variance-matched weight, which is exactly the "rare term" failure mode the
methodology report predicted, now confirmed rather than only analogised to
`BadSignal`.

**Trajectory-ranking check** (`examples/mro_weight_ranking_check.py`, closing
the methodology report's Section 5.3 gap): applies the full candidate weight
set (`w_quality=1.00, w_pingpong=1.51, w_churn=0.80, w_outage=1.0`) to both
the pooled sticky-arm data and the round-2 do-nothing baseline, and checks
the sign.

| condition | n | mean reward | sd |
|---|---|---|---|
| do-nothing | 1372 | 0.1628 | 0.472 |
| sticky +3 dB | 2990 | 0.1232 | 0.495 |

**PASS** -- the sticky arm scores worse than do-nothing (gap +0.0396), as a
real MRO failure mode must. Stated plainly: the margin is real but modest at
the per-step level (well under a tenth of a standard deviation), not a
dramatic separation -- consistent with this project's established practice
of reporting a PASS with its actual effect size rather than only the sign.

**Parallel execution.** Both batches ran with `mro_outage_stress_experiment.py
--workers 5` (added this session): each seed as an independent OS process on
its own ZeroMQ port, cutting wall-clock from a multi-hour sequential chain to
roughly one seed's worth of time per batch. Two real bugs were caught and
fixed before trusting it for the real run:
1. A double-`SIGINT` race (two independently-issued interrupts, one via a
   now-removed shell-script wrapper) left 5 orphaned ns-3 run directories
   unpurged -- a second interrupt arriving while a worker's own
   `finally: env.close()` was still unwinding aborted that cleanup
   mid-block. Fixed by dispatching the forwarded `SIGINT` to every worker
   **exactly once**, guarded against a repeat.
2. `MroZmqEnv` defaults to rebuilding ns-3 on every construction
   (`build_ns3=True`). Harmless run one at a time, but with 5 concurrent
   workers it meant 5 simultaneous `ns3 configure` + cmake invocations
   against the SAME build directory and CMake cache -- caught live on the
   first real parallel attempt (didn't crash outright, but disturbed the
   build badly enough that the next build needed to redo 1644/1644 steps
   instead of a no-op). Fixed by passing `build_ns3=False` in `run_seed()`
   -- same pattern `train_mlb_sac_pilot.py`'s own parallel workers already
   use (build once, construct workers with `build_ns3=False`).

Both fixes verified with short, real (not synthetic) ns-3 runs before
trusting them for the real 10-seed measurement above.

## `ActionSmoothness`, measured from the real policy class

`examples/mro_actionsmoothness_probe.py`: a freshly-initialised, **untrained**
SB3 SAC policy (the project's own locked hyperparameters -- `gamma`,
`learning_rate`, `batch_size`, `ent_coef` from `train_mlb_sac_pilot.py`'s
`ARM_C`; `net_arch` left at SB3's own default, since no MRO-specific
architecture is locked anywhere yet) acting stochastically against a
synthetic-but-representative KPI stream. No ns-3 needed: `ActionSmoothness`
is a pure function of the action sequence, so what matters is what this
policy class, at this initialisation, actually samples step over step, not
how closely the KPIs resemble reality. 20 episodes x up to 300 steps, 5,980
total steps.

| | mean | detrended sd |
|---|---|---|
| old Monte Carlo guess | 1.99 | 1.06 |
| **real measurement** | **7.24** | **3.19** |

The real untrained SAC policy is roughly 3x more erratic than the old
hand-approximated guess assumed -- a randomly-initialised, entropy-driven
SAC policy tends to saturate near the action bounds early in training, more
than the earlier guess accounted for. `w_smooth = anchor_sd(0.273) /
detrended_sd(3.19) = 0.086`, replacing the old `0.26`.

Still provisional in one narrower sense: this measures the real policy
*class* MRO will train with, not real training *behaviour*, since no MRO
training run has happened yet. Re-derive from real rollout logs after the
SAC pilot, per the methodology report's Section 5.2.

## Current state

**Reward weights are set in code** (`DEFAULT_W_*` in `mro_zmq_env.py`):
`w_quality=1.00`, `w_pingpong=1.51`, `w_churn=0.80` (measured, round 2),
`w_outage=1.0` (fixed guard, now validated against a real 10-seed stress
test), `w_smooth=0.086` (measured from the real untrained SAC policy class).
None of the five have been checked against real MRO training rollouts yet --
see the open items above (SAC pilot script, the crash-rate investigation,
and re-measuring `PingPong`/`HandoverRate` once real training data exists)
before starting a full training run.

## ActionSmoothness dropped from the reward (REWARD_FORMULA_VERSION 1 -> 2)

The first full MRO SAC pilot (8000 timesteps, 9000 gradient updates, critic
calibration clean -- see the pilot report) showed why the `w_smooth=0.086`
measurement above couldn't work as a shaping weight: it was calibrated
against an UNTRAINED policy's random noise, not against a real learning
signal. In the real pilot, `ActionSmoothness` was the only one of the five
terms easy enough to improve that the agent fixated on it -- smoothness
collapsed roughly 5x the moment gradient updates started, while `quality`,
`handover_rate`, `pingpong` and `outage` never moved, early to late. The
term's original justification also traces to an MLB-specific failure mode
(an undamped controller oscillating 3 dB/step, flipping direction 7/10
steps) imported by analogy -- never confirmed as an MRO problem of its own
before being added to the design.

**Change:** `ActionSmoothness` is no longer in the weighted sum. It is still
computed and logged every step (`reward_terms["smoothness"]`) -- useful as a
diagnostic for the next point below -- just no longer rewarded or penalised.
`REWARD_FORMULA_VERSION` bumped 1 -> 2; any reward measured under v1 is not
comparable to one measured under v2. Regression test:
`test_action_smoothness_is_logged_but_not_rewarded` in
`tests/test_mro_zmq_env.py`, mutation-tested (reverting the drop makes it
fail, confirmed, then restored).

Next steps per the pilot report: investigate alternative actions besides the
margin offset (the loosen/tighten asymmetry found during that investigation
suggests the current lever may not be the right one), and rerun the SAC
pilot with the four-term reward before drawing further conclusions.

## The real blocker: MRO's margin was wired into almost nothing (fixed)

The loosen/tighten asymmetry flagged above turned out not to be a lever
problem but a wiring problem. Under Ra'ed's locked config
(`handoverMode=DynamicTtt`, `tttFromBiasedSinr=1`, `handoverHysteresisDb=0`),
`LteEnbRrc::TttBasedHandover`'s PRIMARY trigger branch (the one that fires
for the overwhelming majority of handovers) decided purely on
`HysteresisAllowsHandover()`, which reads only `m_handoverHysteresisDb` and
the CIO-biased SINR -- never `m_hysteresisMarginOffset`, MRO's own per-cell
offset. `GetEffectiveSinrThreshold()` (which DOES fold in MRO's offset) was
only consulted in one narrow secondary branch: re-evaluating an
already-scheduled handover whose target cell changed mid-TTT-wait. In
practice this meant MRO's action could almost never change what the
simulator actually did -- fully explaining why `quality` / `handover_rate` /
`pingpong` / `outage` stayed dead flat across all 8000 timesteps of the
first pilot, independent of the `ActionSmoothness` problem above. Full trace
and validation: `mro_investigation_log.md`.

**Fix** (`src/lte/model/lte-enb-rrc.cc`, `TttBasedHandover`'s primary
branch): require `sinrDifference > GetEffectiveSinrThreshold(maxSinrCellId)`
alongside the existing `HysteresisAllowsHandover()` check before declaring a
handover needed. `sinrDifference` is the raw (non-CIO-biased) SINR gap,
already an in-scope parameter at this call site. This also restores the
standard 3GPP-style 3dB base hysteresis (`m_sinrThresholdDifference`,
ns-3's `HoSinrDifference` attribute) to the do-nothing baseline itself --
that branch had no SINR-gap requirement at all before this fix, which is why
every do-nothing baseline measured before this point in this document is now
stale.

Validated at increasing scale (1 -> 3 -> 15 seed x arm combinations, final
batch at the full locked config: 5 seeds x {tighten +3dB, do-nothing, loosen
-3dB}, simTime=30, 1,495 steps/arm): `handover_rate` and `pingpong` are
monotonic in margin direction in every single seed-arm combination, zero
exceptions, loosen's pingpong rate ~23x do-nothing's; `outage` is
monotonically DECREASING as margin loosens (the symmetric too-late-handover
mechanism) -- both classic 3GPP mobility-robustness failure modes now
independently, cleanly reproduced by the same fix.

**Weights re-derived a 3rd time** from the fixed-wiring do-nothing baseline
(same detrended-sd method): `w_pingpong` 1.51 -> **9.39**, `w_churn` 0.80 ->
**1.21**, `w_outage` kept at the fixed guard 1.0 (same rare-event rationale
as before). The trajectory-ranking check (`mro_weight_ranking_check.py`) now
PASSES for both failure modes -- tighten: mean reward 0.1320 vs do-nothing's
0.1526 (gap +0.0206); loosen: -0.2323 vs 0.1526 (gap +0.3849, decisive).
Under the OLD (pre-fix) weights, loosen had ranked BETTER than do-nothing
(+0.2355 vs +0.2190) -- exactly backwards. These weights are only valid
together with the wiring fix; do not use them against an unpatched binary.

These weights are now `MroZmqEnv`'s defaults (`DEFAULT_W_PINGPONG`,
`DEFAULT_W_CHURN`). `REWARD_FORMULA_VERSION` is unchanged at 2 (a weight
retune, not a formula change, per this project's own versioning convention).

## Second full SAC pilot (post-wiring-fix): trained, but worse than baseline

With the wiring fix and new weights both in place, a full SAC pilot was
trained for real: `n_envs=8`, `gradient_steps=24` (3.00 updates/sample),
`learning_starts=5000`, `total_timesteps=12400`, seed 555. Ran 11h42m,
completed cleanly (22,200 gradient updates, no crashes, no guard trips).

The fix clearly works as designed -- MRO's action moved real KPIs during
training for the first time in this project's history (`handover_rate`
and `pingpong` both shifted measurably at the `learning_starts`
boundary, where they had always stayed dead flat before). But the
critic's Q-value diverged continuously throughout training (1.9 -> 79.3,
no plateau -- a textbook SAC overestimation pattern), and the resulting
policy drifted toward MORE handovers and MORE ping-pong, not fewer --
the same too-early-handover failure mode this project's own stress test
validated as bad.

**Day-3 deterministic evaluation (5 seeds, matching the baseline's own
seeds, same locked config, both datasets re-scored under the current
weights for a fair comparison) confirms this is a real regression, not
training-time noise:**

| condition | mean reward | quality | pingpong | handover_rate | outage |
|---|---|---|---|---|---|
| do-nothing baseline | +0.1526 | 0.9357 | 0.0021 | 0.1216 | 0.00038 |
| trained policy (deterministic) | -0.1265 | 0.8537 | 0.0315 | 0.1990 | 0.00050 |

The trained policy loses on all 5 seeds. pingpong is ~15x higher,
handover_rate ~64% higher, quality lower. **This checkpoint is not
deployable.**

**Takeaway:** the wiring fix was necessary but not sufficient. The
remaining problem is almost certainly the unbounded critic overestimation
seen throughout training, not the reward design or the action space
(both were separately validated as correct via the fixed-wiring
trajectory-ranking check). Next session should investigate: the
gradient_steps/n_envs update ratio (24/8=3.00, sustained for a long run
-- a known SAC destabiliser), the auto entropy coefficient's starting
value (0.1, possibly too high for this action space, encouraging the
policy to chase the overestimated Q into bad regions early and never
recover), and whether reward clipping/normalization would help bound the
critic's targets. Full details, plots and the per-phase training
trajectory: `mro_investigation_log.md`, `output/fig_reward_kpis.pdf`,
`output/fig_action_q.pdf`.

## Third pilot attempt: decoupled gradient_steps + warm start

Literature search confirmed the leading hypothesis directly (Hussing et
al. 2024, arXiv:2403.05996: high update-to-data ratio causes exactly this
kind of Q divergence). Relaunched with `gradient_steps` decoupled from
`n_envs` (kept at a conservative absolute 8, not scaled to 24+ as
`n_envs` rose to 12) and a warm start: the replay buffer is pre-filled
with 3600 transitions from a new traditional rule-based MRO controller
(`examples/mro_rule_based_controller.py`, modeled on the literature's
closed-loop CIO/margin-vs-threshold pattern already used in commercial
MRO) before SAC's own random-exploration phase, via a new
`--warm_start_steps` flag on `train_mro_sac_pilot.py`. A real bug was
caught and fixed during smoke-testing (`_store_transition` needs
`model._last_obs` primed before being called standalone -- see
`mro_investigation_log.md`'s 10-18 entry for the full isolation trace).
No dedicated empirical burst-size reproduction was run before committing
to this relaunch (time budget); the fix is literature-justified.

**Result: the fix worked.** Critic Q stayed bounded with roughly-linear
growth throughout 18,000 timesteps (final Q=11.3, vs the diverged run's
Q=79.3 and still accelerating at the same budget) -- no divergence.
Day-3 deterministic evaluation (5 seeds vs the do-nothing baseline):
mean reward +0.1228 vs baseline's +0.1526, gap +0.0298 -- still a loss,
but **89% smaller than the diverged run's +0.2791 gap**, and the trained
policy wins outright on 1 of 5 seeds for the first time in this
project's history. pingpong (the clearest signature of the "loosen/
too-early" failure mode) dropped from ~15x baseline to ~2.6x baseline.
Not yet a full pass; the most promising next step is simply more
training budget now that the stability problem is solved. Full numbers:
`mro_investigation_log.md`'s 2026-10-19 entries,
`output/fig_combined_v3.pdf`, `output/mro_v3_trained_policy_eval.csv`.
