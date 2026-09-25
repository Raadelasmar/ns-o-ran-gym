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
- [ ] Build `train_mro_sac_pilot.py` mirroring `train_mlb_sac_pilot.py` (Arm C
      hyperparameters, `DiskGuard`/`MemGuard`, `CriticProbe`/`StepLogger`)
- [ ] Run MRO's own SAC pilot against the same "does the critic climb" bar
- [ ] Run the adversarial `OutageExposure` check (force congestion onto one
      cell, mirroring how MLB's `BadSignal` numerator was proven alive)
      before fully trusting `DEFAULT_W_OUTAGE`'s fixed guard value
- [ ] Investigate the round-2 mid-run crash rate (2/5 seeds, unexplained) at
      `ues=5`/`controlInterval=0.1` before trusting a large training run on
      this config
- [ ] Re-measure `PingPong`/`HandoverRate`/`ActionSmoothness` statistics from
      real training rollouts once the SAC pilot has run, per the methodology
      report's Section 5.2 ("do not trust a do-nothing-baseline weight
      forever") -- `ActionSmoothness` especially, since its current weight is
      a Monte Carlo estimate, not a measurement

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
| smoothness | -- (silent under do-nothing) | -- | 1.06 (Monte Carlo) | ~0.26 | **0.26 (provisional)** |

`OutageExposure` is confirmed alive at this config (fired on 0.01% of
UE-steps, resolving round 1's open question), but its literal equalised
weight (159.7) is not usable -- matching its variance to `HandoverSignalQuality`
would make the reward swing enormously on the rare step it fires, the same
failure shape MLB's own `BadSignal` term hit (never equalised, kept at its
default instead). Per the methodology report's stated rule for rare terms,
`DEFAULT_W_OUTAGE` is fixed at 1.00 -- a modest guard weight in the same range
as the other four, not the 160x-scaled literal value -- pending the
adversarial test above.

`ActionSmoothness` cannot be measured under an all-zero baseline action by
construction (zero delta every step), so its weight is a Monte Carlo estimate
from an untrained-PPO-like action distribution, run through the same
anchor-equalisation formula as the other terms. **Provisional, not a
measurement** -- to be re-derived from real rollout logs after the SAC pilot.

**Reward weights ARE now set in code** (`DEFAULT_W_*` in `mro_zmq_env.py`),
but two of five (`outage`, `smooth`) are documented fixed/provisional values
rather than measurements, and none of the five have been checked against a
real training rollout yet -- see the open items above before starting a full
training run.
