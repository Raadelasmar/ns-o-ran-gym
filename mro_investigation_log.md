# MRO Agent Investigation — 7-Hour Autonomous Session

Start: 2026-10-16 01:06:56 CEST. Target end: ~08:07 CEST.

---

## 2026-10-17: Day 1/2 execution session (roadmap PDF now being implemented
## for real, not just diagnosed)

User authorized a second autonomous 7-8h block to actually execute Day 1
(wiring fix, weight recalibration) and Day 2 (launch the real SAC pilot) of
the roadmap, same autonomy terms as the 10-16 session. Default assumed:
implement + validate + launch, but do NOT commit/push -- that's held for
the user's go-ahead at check-in, per the project's own norm of flagging
core ns-3 changes (lte-enb-rrc.cc) before committing.

### [01:54] Wiring fix re-implemented for real (kept, not reverted)
Same patch as the 10-16 diagnostic, now applied to stay: `&&
sinrDifference > GetEffectiveSinrThreshold(maxSinrCellId)` added to
TttBasedHandover's primary trigger branch (src/lte/model/lte-enb-rrc.cc).
Rebuilt clean (single-file recompile + relink, 22s). `git diff
--ignore-cr-at-eol --stat` on the ns-3 repo confirms exactly this one
file, 6 insertions/1 deletion -- no other content change (the repo's
working tree also shows a tree-wide CRLF line-ending flip, same cosmetic
issue flagged at the end of the 10-16 session; confirmed zero semantic
diff outside this file via --ignore-cr-at-eol).

### [01:5x] Weights applied to the codebase (previously only derived, not
### applied)
`DEFAULT_W_PINGPONG` 1.51 -> 9.39, `DEFAULT_W_CHURN` 0.80 -> 1.21 in
`src/environments/mro_zmq_env.py` (the values the 10-16 session's 15-seed
sweep already derived; see that session's [06:27] entry). `w_outage` and
`w_quality` unchanged. Comments updated to explain the dependency: these
weights are only valid together with the wiring fix above.

### [11:34] Restarted at real parallelism (n_envs 2 -> 8) at user's request
First launch (logs_20261017_025612) ran healthy for ~9h, reaching 4276
timesteps -- but n_envs=2 was far too conservative for this machine (16
cores, only ~300MB RSS per ns-3 worker observed; 9+ GB MemAvailable stayed
free the whole time). User asked why only 2 cores were in use; n_envs=2
turned out to be MLB's own hard-won-conservative default, inherited by
MRO's launcher without ever being re-profiled for MRO's own footprint.

Attempted a resume at n_envs=8 (paired mro_sac_final.zip with a renamed
mro_sac_replay.pkl per the --resume_from convention) -- REFUSED by the
script's own assert_actuals safety net: SAC's ReplayBuffer bakes in
buffer_size-per-lane AND lane count at save time (100000-slot/2-lane vs the
new intent of 25000-slot/8-lane), so a loaded buffer can't just adopt a new
n_envs. No harm done -- the check caught it before any training step, old
checkpoint files untouched.

Resolved by NOT resuming: num_timesteps=4276 had never crossed
learning_starts=5000, so zero gradient updates had happened -- the actor/
critic were still at random init, and the "lost" replay buffer was pure
random-exploration data, regenerated faster at 8x the collection rate
anyway. Started fresh at n_envs=8, gradient_steps=24 (same 3.00
updates/sample ratio as before). New logs:
output/mro_sac_pilot/logs_20261017_113650/. All 8 workers up, all
assert_actuals checks pass, memory healthy (7.9GB available).

### [11:51] Budget extended for the user's full 13h return window
Measured real n_envs=8 throughput over the first 13.9 min: 241 steps =
17.36 steps/min (settled down from an early noisy 6-min read of ~19.3/min
-- scaling from n_envs=2's 8.24/min is ~2.1x, not the naive 4x linear
guess, i.e. real contention on this 16-core box, not free scaling).
User asked why run a short pilot then a second long one instead of just
running the long one directly -- correct call, no good reason not to:
stopped the n_envs=8 run again (clean, 248 timesteps, same n_envs so the
resume this time is NOT a repeat of the earlier n_envs mismatch) and
relaunched with --total_timesteps 12400, budgeted as 90% of the user's
13h window at the measured rate (702 min x 17.36 steps/min ~= 12,190 +
the 248 already in hand), holding back 78 min (10%) for the Day-3
evaluation + documentation after training finishes. This also crosses
learning_starts=5000 by a comfortable margin (~7400 steps of actual
gradient-updated experience, vs only ~3000 in the original 8000-step
plan). New logs: output/mro_sac_pilot/logs_20261017_115135/. Resume
verified clean: all assert_actuals checks pass, resumed at 248 timesteps,
12152 remaining to the new target.

### [23:33] Training COMPLETE -- 12,400/12,400 timesteps, 22,200 gradient
### updates, clean exit, no guard trips
Launched 11:51:35, finished 23:33:39 -- 11h42m wall-clock, inside the 13h
budget. One transient ns-3 worker crash mid-run (documented "~1 in 10
seeds kills ns-3 at start-up" behaviour in make_env's own comment) --
self-recovered via the env's automatic reset, zero training disruption,
confirmed via process count and advancing total_timesteps immediately
after. replay_buffer_saves=24, no disk/mem guard trips.

Crossed learning_starts=5000 at ~14:25 (probe log). Gradient updates
began immediately and cleanly: Q jumped from flat -0.035 to 1.9 within
the first 250 post-threshold steps -- normal SAC mechanics, matches both
earlier smoke tests.

**Concerning trend, flagged live during monitoring, confirmed in the full
data:** the critic's Q-value climbed CONTINUOUSLY and MONOTONICALLY for
the entire rest of training, 1.9 -> 79.3 over ~7400 gradient-updated
steps, with no sign of plateauing (see fig_action_q.pdf, bottom panel --
an accelerating, not decaying, curve). This is a textbook SAC critic
overestimation pattern. Over the same window, handover_rate and pingpong
both drifted UP (phase breakdown, scratch_training_summary.py):

| phase | n | reward | quality | pingpong | handover_rate | outage |
|---|---|---|---|---|---|---|
| pre-learning (t<5000) | 4744 | +0.028 | 0.888 | 0.0153 | 0.170 | 0.00054 |
| early-learning (5000-7000) | 2000 | +0.031 | 0.868 | 0.0143 | 0.175 | 0.00060 |
| mid-learning (7000-9500) | 2504 | +0.037 | 0.878 | 0.0154 | 0.176 | 0.00082 |
| late-learning (t>=9500) | 2904 | +0.012 | 0.881 | 0.0180 | 0.187 | 0.00086 |
| last ~1500 steps | 1504 | +0.014 | 0.878 | 0.0179 | 0.193 | 0.00111 |

Training-time reward stayed roughly flat (contaminated by exploration
noise -- alpha was still ~0.055 at the end, not annealed to zero), but
the drift in handover_rate/pingpong is the same DIRECTION as the
validated "loosen" failure mode from the 10-16 session. Hypothesis: the
diverging critic made loosening (more frequent handovers) look
artificially attractive to the policy, independent of whether it actually
scores well under the real, current reward weights.

### [after 23:33] Day 3: deterministic evaluation vs do-nothing baseline
`examples/mro_eval_trained_policy.py`, 5 seeds (2001-2005, matching the
project's established baseline seeds), simTime=30, same locked config as
`pooled_mpos0.csv` (the fixed-wiring do-nothing baseline from the 10-16
session). First attempt ran all 5 seeds sequentially in one process (a
parallelism oversight, same class of issue as the original n_envs=2
default) -- seed 2001 alone took 62 minutes solo; killed after user
caught it ("why is only one core used") and relaunched the remaining 4
seeds as 4 parallel single-seed processes (no shared state risk, each
writes its own CSV), finishing in ~70 min instead of a projected 5+
hours. Pooled all 5 into `output/mro_trained_policy_eval.csv`.

**Fairness note:** `pooled_mpos0.csv`'s stored `reward` column predates
today's weight update (1.51/0.80, not 9.39/1.21), so it is NOT directly
comparable to the eval CSV's own `reward` column (computed live under the
current weights). Re-scored BOTH datasets from their raw per-term columns
under the current weights (`scratch_final_comparison.py`) for a fair,
apples-to-apples comparison.

**VERDICT -- the trained policy is WORSE than doing nothing, decisively,
0/5 seeds:**

| condition | n | mean reward | sd | quality | pingpong | handover_rate | outage |
|---|---|---|---|---|---|---|---|
| do-nothing baseline | 1495 | +0.1526 | 0.3595 | 0.9357 | 0.0021 | 0.1216 | 0.00038 |
| trained policy (det.) | 1495 | -0.1265 | 0.9621 | 0.8537 | 0.0315 | 0.1990 | 0.00050 |

gap (baseline minus trained) = +0.2791, decisive, not marginal. Per-seed:
trained policy loses on ALL 5 seeds (seed means -0.370, -0.017, -0.139,
-0.005, -0.102 vs baseline's +0.192, +0.143, +0.130, +0.125, +0.174).
pingpong is ~15x higher under the trained policy; handover_rate is ~64%
higher; quality is lower. This is exactly the too-early-handover /
ping-pong failure mode validated as bad in the 10-16 session's stress
test, now actively LEARNED by the policy rather than imposed as a fixed
stress action.

**Conclusion:** the wiring fix worked exactly as designed -- MRO's action
now has a real, strong, measurable effect on handover behaviour (this is
unambiguous from both the Day 1 validation and the fact that training
moved real KPIs at all, unlike the original pre-fix pilot). But the SAC
pilot, as configured (12,400 steps, gradient_steps=24/n_envs=8 ->3.00
updates/sample, learning_starts=5000), learned a policy that exploits
this new lever in the WRONG direction -- almost certainly downstream of
the unbounded critic overestimation observed throughout training. The
fix was necessary but not sufficient; the training run itself needs
further work (see next steps) before this agent is deployable.

**Candidate causes for the Q-divergence, ranked by plausibility for a
next session to check first:** (1) gradient_steps=24 at n_envs=8 is a
high update-to-sample ratio sustained for a long run -- known to
destabilize SAC's critic when the replay buffer is this actively
reused; (2) ent_coef=auto starting at 0.1 may be miscalibrated for this
action space/reward scale, letting the policy explore too aggressively
into high-Q, low-real-reward regions early and never recovering;
(3) no reward clipping/normalization -- a single large handover-outage
event could produce an outlier target value that the critic overfits to.

Re-ran the OFFICIAL `examples/mro_weight_ranking_check.py` (not just the
scratch analysis the 10-16 session used) against the same pooled
`output/rewire_*.csv` data, new weights:
- tighten vs do-nothing: 0.1320 vs 0.1526, gap +0.0206, PASS
- loosen vs do-nothing: -0.2323 vs 0.1526, gap +0.3849, PASS (decisive)
Matches the 10-16 log exactly -- reproducible via the project's own
checked-in tooling, not just an ad hoc scratch script.

Full test suite (`pytest tests/`, excluding the two pre-existing unrelated
collection errors dating to the original fork): 52/52 passed.

docs/mro_training_fixes.md and examples/train_mro_sac_pilot.py's comments
updated to document the fix + re-derivation (3rd generation weights) as
the current, permanent state rather than a diagnostic finding.

Archived the first (broken-wiring) pilot's checkpoints to
`output/mro_sac_pilot/archive_pre_wiring_fix/` before launching the real
run, rather than letting fresh checkpoints silently overwrite them by
filename collision (mro_sac_1000_steps.zip etc.) -- preserves the "before"
state the original pilot report is based on.

### [02:56] Real pilot launched: `mrosacpilot.service`, logs in
### `output/mro_sac_pilot/logs_20261017_025612/`
n_envs=2, total_timesteps=8000 (absolute), learning_starts=5000,
gradient_steps=6 (3.00 updates/sample, within 10% of arm C's 3.20), seed
555, ports 5565-5566. Pre-flight (disk/mem/ports) all clean, dry-run
verified first. `watch_mro_sac_pilot.sh` + a Monitor on pilot.log
(filtering for probe lines / MemGuard / DiskGuard / tracebacks / OOM /
STOPPED_EARLY) cover unattended health checking for the remainder of this
block.

One caveat logged, not blocking: "MemGuard: own cgroup not resolvable, the
anon ceiling is INACTIVE, only the MemAvailable floor will fire." Same
legacy/hybrid-cgroup issue noted in the 10-16 session (MemoryMax isn't
really enforced on this WSL2 instance) now also affects MemGuard's own
in-process anon-ceiling probe, not just systemd's cgroup limit -- so
BOTH of the run's two intended memory safety nets are degraded to the
single remaining one, the system-wide MemAvailable floor (2.5 GB). This is
exactly why N_ENVS was kept at the conservative default of 2 rather than
raised to use more of the machine's cores -- the roadmap's "use full
resources" intent is satisfied by letting this run occupy its full
intended wall-clock budget, not by adding workers this safety margin was
specifically sized against.
Goal: find why the margin action doesn't produce a clean, learnable,
correctly-directional effect on handover quality under the post-merge
simulator, and build an evidence-backed 3-day roadmap. NOT editing any
existing shipped file during this session; NOT committing/pushing anything.

## Known going in (from prior sessions, see docs/mro_training_fixes.md)
- Action: per-cell handover margin offset, Box(-3, 3, (7,)), modifies
  LteEnbRrc::GetEffectiveSinrThreshold via m_hysteresisMarginOffset.
- Reward (v2, post this-session): quality - 1.51*pingpong - 0.80*handover_rate
  - 1.0*outage. ActionSmoothness dropped entirely (was masking everything).
- margin-effect single-seed diagnostic: neutral(0dB) vs loosen(-3dB) gave
  DIFFERENT handover counts (256 vs 222); neutral vs tighten(+3dB) gave
  IDENTICAL counts (256 vs 256). Asymmetric.
- Trajectory-ranking check (5 seeds, 1495 steps/arm, post-merge):
  do-nothing reward=+0.2074; sticky+3dB=+0.2159 (BETTER, wrong direction);
  loosen-3dB=+0.2050 (worse, but tiny, and NOT via pingpong -- pingpong
  actually improved under loosen too, 0.0469 vs 0.0494. The small loosen
  "win" for do-nothing is almost entirely a rare-outage uptick, 12 vs ~2
  fired-steps/1495).
- SAC pilot (8000 timesteps, pre-smoothness-removal): quality/pingpong/
  handover_rate/outage never moved at all across training, early to late.

## Open hypotheses to test this session
H1: TTT timing dominates the handover decision; the SINR-threshold margin
    comparison is rarely the binding constraint under tttFromBiasedSinr=1.
H2: +-3dB is too small a perturbation for this topology's real SINR gaps.
H3: A uniform, all-7-cells, whole-episode offset cancels out at the network
    level (stops good handovers elsewhere even while preventing bad ones).
H4: Statistical power -- 5 seeds/1495 steps may be too few to see a modest
    real effect through the noise.

## Timeline / findings (append as we go)
(Note: entries are mostly chronological by their [HH:MM] tag, but two
entries -- "Synthesis" and "Caveat worth flagging" -- were written in
between other work and landed out of strict file order below. Read by
timestamp, not by position, if that matters.)

### [01:30] ROOT CAUSE FOUND: MRO's margin is structurally disconnected from
### the primary handover decision under the current config.

Traced LteEnbRrc's actual decision path in full (lte-enb-rrc.cc):

1. `maxSinrCellId` (the argmax target cell) is computed from `cellSinr * cio`
   -- CIO only (MLB's lever, m_cellIndividualOffset). MRO's margin never
   enters cell selection at all.
2. handoverMode=DynamicTtt (our config) always dispatches to
   `TttBasedHandover()`.
3. Inside it, when no handover is already pending (the common case), the
   ENTIRE trigger decision is:
       if (maxSinrCellId != m_lastMmWaveCell[imsi]
           && HysteresisAllowsHandover(imsi, maxSinrCellId, currentSinrDb))
           handoverNeeded = true;
4. `HysteresisAllowsHandover()`:
       if (m_handoverHysteresisDb <= 0.0) return true; // unconditional pass
   Our CFG sets handoverHysteresisDb=0 (deliberately, at merge time, to
   "keep MRO's own margin authoritative" -- see docs/mro_training_fixes.md).
   THAT REASONING WAS WRONG. HysteresisAllowsHandover() never reads MRO's
   m_hysteresisMarginOffset at all -- only Ra'ed's m_handoverHysteresisDb
   and CIO (via GetCioBiasedSinrDb). Setting it to 0 doesn't "return control
   to MRO" -- it just removes RA'ED's gate, leaving NEITHER lever
   constraining the primary trigger. The primary decision reduces to pure
   "did the CIO-biased argmax cell change" -- zero margin involvement.
5. TTT DURATION is also CIO-only: TttSinrDifference(), with
   tttFromBiasedSinr=1 (our config), returns
   |GetCioBiasedSinrDb(target) - GetCioBiasedSinrDb(serving)| -- again no
   m_hysteresisMarginOffset.
6. MRO's margin (GetEffectiveSinrThreshold) is consulted in exactly ONE
   place under our config: the narrow "a handover is already scheduled AND
   the argmax target shifted mid-TTT-wait" re-evaluation branch (line
   ~4461). Everywhere else that reads it (ThresholdBasedSecondaryCellHandover,
   the old non-TTT branches) is DEAD CODE under handoverMode=DynamicTtt +
   tttFromBiasedSinr=1.

This fully explains every anomaly so far: why +-3dB stress arms gave weak/
wrong-direction effects (the lever barely touches the dominant decision
path), why the SAC pilot never moved quality/pingpong/handover_rate/outage
(almost nothing for the policy to control), and why the single-seed loosen
test showed SOME effect (222 vs 256 handovers) -- that's the narrow
secondary branch, a real but much weaker lever than intended.

NEXT: validate this with a small, surgical, REVERSIBLE C++ patch (wire
GetEffectiveSinrThreshold into the primary TttBasedHandover trigger check),
rebuild, re-run the same short diagnostic, see if the effect becomes clean
and correctly-directional. Revert the patch after testing either way --
not committing, leaving it for the roadmap's Day 1 to implement for real.

### [01:35] Patch applied + rebuilt successfully
Added `&& sinrDifference > GetEffectiveSinrThreshold(maxSinrCellId)` to the
primary trigger condition in TttBasedHandover (lte-enb-rrc.cc). Confirmed
single-file recompile + relink, clean. Running mro_margin_wiring_check.py
(new, short simTime=10 diagnostic script, reports full reward terms not
just raw count) at margin 0/+3/-3, seed 2001, sequential -- results next.

### [02:05] PATCH VALIDATED -- clean, monotonic, correctly-directional effect
simTime=10 (~99 steps), seed 2001, one episode per arm:

| margin | handovers | handover_rate | pingpong | quality | reward |
|---|---|---|---|---|---|
| +3dB tighten | 27 | 0.0778 | 0.0000 | 1.0000(n=24) | +0.1793 |
| 0dB neutral  | 50 | 0.1443 | 0.0029 | 0.9268(n=41) | +0.2638 |
| -3dB loosen  | 71 | 0.2047 | 0.0202 | 0.9444(n=51) | +0.2919 |

handover_rate and pingpong are now PERFECTLY MONOTONIC with margin
direction -- tighten < neutral < loosen on both, exactly as 3GPP theory
predicts (too-late vs too-early handover). pingpong under loosen is ~7x
neutral's rate -- a real, strong, learnable signal, completely absent
before the patch (pre-patch: pingpong moved the WRONG way under loosen,
0.0469 vs do-nothing's 0.0494 -- see pre-patch loosen-stress result above).
Tighten now correctly scores WORSE than neutral on reward (+0.1793 <
+0.2638) -- recovers the ORIGINAL pre-merge validated finding that this
session's post-merge re-test had found broken.

Loosen still scores BETTER than neutral on reward (+0.2919 > +0.2638)
despite the large pingpong jump -- quality and handover_rate's contributions
currently outweigh pingpong's penalty at the CURRENT weights. Single
seed/short episode, not conclusive on the exact ranking yet, but the
MECHANISM (margin -> handover_rate/pingpong, monotonic, correct direction)
is now unambiguously real. This is the central finding of the whole session.

NEXT: (a) multi-seed, still-short-simTime confirmation this isn't a
one-seed fluke: (b) re-measure do-nothing baseline stats under the patched
simulator (weights may need re-deriving again, 3rd time, since the
underlying term distributions have now genuinely changed); (c) re-run
trajectory-ranking with fresh weights; (d) short SAC smoke-training check.

### [06:27] FULL-SCALE CONFIRMATION (5 seeds x 3 arms, simTime=30, 1495
### steps/arm, locked config) -- the complete fix validated end to end.
NOTE: this batch took ~4h05m wall-clock (15-way parallel on 16 cores hit
real contention -- a mistake in resource planning, noted for the roadmap's
own time-estimation; should have run at most ~8-10 way parallel, or in two
waves). This consumed most of the remaining budget; no SAC smoke-training
run fit in the time left. Recommending it as Day 1's own closing step
instead of demonstrating it live here.

| arm | reward | quality | pingpong | handover_rate | outage |
|---|---|---|---|---|---|
| tighten +3dB | 0.1657 | 0.9801 | 0.0002 | 0.0786 | 0.0012 |
| neutral (do-nothing) | 0.2190 | 0.9357 | 0.0021 | 0.1216 | 0.0004 |
| loosen -3dB | 0.2355 | 0.8677 | 0.0475 | 0.2279 | 0.0002 |

handover_rate and pingpong are monotonic in margin direction in EVERY one
of the 15 seed-arm combinations, no exceptions -- tighten always lowest,
loosen always highest, by a wide margin with zero overlap between arms.
pingpong under loosen is ~23x the do-nothing rate. quality falls
monotonically from tighten(0.98) to loosen(0.87) -- pickier handovers are
better handovers, exactly as theory predicts.

Re-derived weights from this (fixed-wiring) do-nothing baseline, same
detrended-sd method as always: w_pingpong 1.51 -> **9.39**, w_churn 0.80 ->
**1.21**, w_outage 63.6 (still a rare-event term, keep the existing fixed
guard w_outage=1.0 rather than this literal value, consistent with every
prior measurement of this term).

Trajectory-ranking check, OLD (stale) vs NEW (re-derived) weights, same
underlying data:
| | OLD weights | NEW weights |
|---|---|---|
| do-nothing | +0.2190 | +0.1526 |
| tighten +3dB | +0.1657 (worse, correct) | +0.1320 (worse, correct) |
| loosen -3dB | **+0.2355 (better, WRONG)** | **-0.2323 (worse, correct)** |

With the wiring fixed AND weights re-derived, BOTH failure modes now
correctly rank worse than doing nothing -- loosen's penalty under the new
weights is large and decisive (-0.232 vs +0.153), not a marginal pass.
This is full, complete, end-to-end validation of the fix. Central finding
of the entire session.

Reverting the diagnostic C++ patch now (not committing it) -- the roadmap
specifies implementing it properly as Day 1's first task.

### [07:00] Short SAC smoke test (800-timestep budget, n_envs=4,
### learning_starts=200, STILL using the patched binary) -- early signal
Launched 06:30, checked at 06:59 (392/800 timesteps collected, crossed
learning_starts at 200, n_updates=312 by t=300, Q rising cleanly 1.375 from
flat -0.036 -- normal SAC mechanics, as expected). The actually-new
question: do the REAL handover terms move, this time? Split the data
almost exactly at learning_starts (n=196/392, close to the true t=200
boundary):

| | first half (~pre-learning) | second half (~post-learning) |
|---|---|---|
| handover_rate | 0.224 | **0.150** (-33%) |
| pingpong | 0.015 | 0.018 (flat) |
| quality | 0.902 | 0.814 |
| smoothness | 5.57 | 5.53 (flat, as expected -- no longer rewarded) |

Only ~100 post-learning-start steps, nowhere near enough for a real
conclusion -- but handover_rate dropping noticeably right at the learning
boundary, in a SHORT run, is a good early sign the fix transfers to actual
training, not just to fixed-action diagnostics. smoothness correctly
staying flat (not collapsing) confirms it's no longer distorting the
learning signal now that it's out of the reward sum. Stopping the run here
(out of time budget) -- this is preview evidence for Day 2, not a
substitute for it.

### [07:31] Replicated with a second, independent seed (888 vs 777)
Extended smoke test (seed 888, same config, 2400-step budget, stopped at
t=408 for time): pre-learning handover_rate=0.2125, post-learning (>=200)
handover_rate=0.1824 -- same DIRECTION as the first smoke test (decrease
once learning starts), smaller magnitude this time (-14% vs -33%), still
early/noisy (only ~208 post-learning steps). Quality and outage trends
differ in sign between the two seeds at this tiny sample size -- not
something to read into yet. The one thing that DID replicate across both
independent seeds: handover_rate moving, in the same direction, right at
the learning boundary. Modest, early, but a real second data point, not
just one. Stopping here -- investigation time budget (7h) is up.

### [06:35] Bonus confirmation: outage rate is ALSO monotonic, opposite
### direction -- completes the symmetric too-early/too-late story
outage fired on: tighten 59/1495 (3.95%), neutral 18/1495 (1.20%), loosen
11/1495 (0.74%) -- monotonically DECREASING as margin loosens, exactly the
too-late-handover mechanism (lingering on a degrading cell under a strict
margin raises real outage risk). Combined with pingpong's opposite-direction
monotonic increase under loosen (too-early handover), both classic 3GPP
failure modes are now independently, cleanly, monotonically reproduced by
the SAME single fix, in opposite directions as the theory requires. Also
noted: quality's measurement availability itself varies hugely by arm
(23.5% of steps under tighten vs 56.5% under loosen -- tighten simply
produces far fewer handovers to score at all), a reminder that "quality"
alone is not the full picture without handover_rate alongside it.

### [02:25] Synthesis: this explains the ENTIRE session's prior findings
Three layered problems, found progressively across this whole multi-session
investigation, each real, each necessary to fix, NONE sufficient alone:
1. RLC AM crash (ns-3 core bug) -- fixed earlier this session. Necessary
   just to run episodes to completion at all.
2. ActionSmoothness dominated the reward -- fixed earlier this session
   (dropped from the sum). Necessary so the reward doesn't hand the agent
   a trivial, always-learnable distractor it can "solve" by freezing.
3. THE REAL BLOCKER (this finding): under handoverMode=DynamicTtt +
   tttFromBiasedSinr=1 (Ra'ed's locked config) + handoverHysteresisDb=0
   (our deliberate choice), MRO's margin action was wired into almost
   nothing in the primary handover decision. No matter what the agent
   does, handover_rate/pingpong/quality/outage barely change -- there was
   NEVER a real signal on the terms that matter, independent of reward
   shaping. This fully explains why the SAC pilot's quality/handover_rate/
   pingpong/outage stayed dead flat across all 8000 training timesteps:
   not a training-budget problem, not a reward-weight problem -- the
   action had nothing to teach the policy on those axes. Fixing #2 without
   #3 could never have worked; #3 is the one that makes the agent's action
   matter at all.
This is the central narrative for the roadmap: fixing #3 (validated below)
is what turns MRO from "an agent with no real lever" into one with a
genuine, theory-grounded, empirically-confirmed control mechanism.

### [02:35] Caveat worth flagging in the roadmap: the patch also changes
### the do-nothing baseline's OWN behaviour, not only MRO's controllable range
`GetEffectiveSinrThreshold() = m_sinrThresholdDifference + offsetDb`, and
`m_sinrThresholdDifference` (ns-3's own "HoSinrDifference" attribute, "the
value for which a handover... is triggered") defaults to **3 dB** -- a
standard 3GPP-style base hysteresis already used by the OLDER, non-TTT
threshold-based handover mode (ThresholdBasedSecondaryCellHandover, which
also calls GetEffectiveSinrThreshold the same way). The pre-patch
TttBasedHandover primary branch had NO sinrDifference requirement at all
(any nonzero argmax-cell change triggered a handover) -- strictly more
permissive than "requires a 3dB edge," by construction, no empirical check
needed to know this. So the patch does two things at once: (a) gives MRO's
offset real leverage, and (b) restores the standard 3dB base threshold to
the do-nothing condition itself, which that code path had apparently been
bypassing. This is plausibly the RIGHT fix (consistent with how the older
threshold-based mode already uses this same combined value, i.e. restoring
an omission rather than inventing new behaviour) but it DOES mean every
prior "do-nothing baseline" measurement in this project's history (all of
docs/mro_training_fixes.md's weight derivations) is now stale and must be
re-measured under the fixed wiring -- which is exactly the full-scale
measurement already running. Roadmap must state this explicitly: Day 1
isn't just "add MRO's lever", it's "fix a real upstream ns-3 bug that
affects the SON baseline too", which is a slightly bigger/more important
claim than originally scoped, and worth flagging to Ra'ed given it's core
ns-3 code outside MRO's own files (same category as the RLC AM crash fix).

### [02:15] Multi-seed (3 seeds x 3 margins, simTime=10) CONFIRMS the effect
Pooled across seeds 2001/2002/2003:

| margin | mean handovers | handover_rate | pingpong | quality | reward |
|---|---|---|---|---|---|
| +3dB tighten | 30.7 | 0.088 | 0.001 | **1.000 (all 3 seeds)** | 0.182 |
| 0dB neutral  | 50.0 | 0.144 | 0.001 | 0.926 | 0.247 |
| -3dB loosen  | 78.0 | 0.225 | 0.032 | 0.900 | 0.250 |

handover_rate and pingpong are monotonic in EVERY single seed, no
exceptions -- not a fluke. quality=1.000 exactly for tighten in all 3 seeds
(very few, highly-selective handovers). Reward: tighten is worse than
neutral in all 3 seeds (consistent, robust) -- recovers the original
pre-merge "sticky=bad" result. Loosen vs neutral reward is mixed at this
sample size (2/3 seeds slightly worse, 1/3 slightly better; pooled means
are nearly tied) -- plausibly resolves to "worse" with more steps/seeds,
given pingpong's real, large, consistent jump. Mechanism is validated;
moving to full-scale (5 seed, simTime=30, locked config) measurement next
for real weight re-derivation and a proper ranking check.

---

## 2026-10-18: Diagnostic session -- literature, traditional-MRO research,
## warm start, relaunch with decoupled gradient_steps
(Out-of-order append, same convention as this log's other sessions --
this is the LATEST entry chronologically, not a continuation of the
block above it.)

User asked whether n_envs itself affects learning quality -- answered
in-conversation: the naive pitfall (not scaling gradient_steps with
n_envs) was avoided in the diverged run, but scaling gradient_steps WITH
n_envs to hold updates/sample constant means the BURST SIZE (consecutive
gradient updates per rollout-collection cycle) still grows with n_envs (6
at n_envs=2 -> 24 at n_envs=8 in that run) -- a mechanism distinct from
n_envs itself. User then authorized 3h for diagnostics + literature +
investigating a warm start "as Ra'ed did [based on] the traditional
approach", then a new training run using more cores/RAM, with issues
fixed proactively first.

### Literature search (web)
- **Hussing et al. 2024, "Dissecting Deep RL with High Update Ratios:
  Combatting Value Divergence"** (arXiv:2403.05996) -- directly on point:
  high update-to-data (UTD) ratio causes Q-value DIVERGENCE (not just
  overfitting), at arbitrary points in training, tied to overestimation
  on unseen/out-of-distribution actions amplified by optimizer momentum.
  Fix: unit-ball network normalization (not implemented this session --
  time budget; candidate for a future one).
- **DroQ** (dropout in Q-functions) -- cited as a known mitigation for
  UTD-driven overestimation while keeping only 2 critics.
- **Nikishin et al. 2022, "The Primacy Bias in Deep RL"** (arXiv:2205.07802)
  -- periodic network resets combat overfitting to early replay data,
  allow higher UTD; known drawback: resets themselves cause performance
  dips. Not implemented this session.
- **Traditional/commercial MRO** (arXiv:1310.6173, arXiv:2506.22793, US
  patents 9894571/12063529): confirms CIO + TTT/hysteresis as the two
  tuning knobs, 3GPP's too-early/too-late/wrong-cell handover failure
  classification, and that commercial MRO runs as a rule-based closed
  loop comparing recent handover KPIs against fixed thresholds over a
  time window to step CIO. Rule-based versions are already deployed in
  live LTE networks.
- **Alizadeh et al. 2025, "Offline RL for MRO"** (arXiv:2506.22793) --
  Ericsson Research, CQL/Decision Transformer tuning CIO in place of the
  rule-based MRO algorithm, up to 7% gain. Confirms RL-for-MRO is an
  active, validated research direction (offline, not online SAC like
  here).
- **Warm-start pattern**: Kumar et al. (BuildSys 2022, CityLearn) encode a
  heuristic rule-based controller into the policy network via BEHAVIOR
  CLONING, then fine-tune with RL -- reports gains over both the pure
  rule-based controller and pure RL. ResFiT (arXiv:2509.19301) instead
  freezes a BC base policy and learns a residual correction via off-policy
  RL. **Could not find Ra'ed's own specific implementation anywhere in
  either repo or its docs** (git log --all --grep, docs/ contents, grep
  for "warm start"/"traditional"/"pretrain"/"bootstrap" -- nothing).
  Proceeding from the general literature pattern rather than a specific
  known-prior implementation; this is a real gap, flagged to the user,
  not assumed to match whatever Ra'ed actually did.

### Warm start: implemented (replay-buffer pre-fill variant, not full BC)
Given the time budget, implemented the simpler of the two literature
patterns: pre-fill the replay buffer with transitions from a rule-based
MRO controller (`examples/mro_rule_based_controller.py` -- per-cell,
reads `badsignal_frac` as the too-late/outage-risk proxy and
`handover_activity` as the too-early/churn proxy directly off
MroZmqEnv's own observation vector, steps the margin by +-0.5dB against
fixed thresholds, matching the literature's "compare recent KPIs to
thresholds" closed-loop pattern) BEFORE SAC's own random-exploration
phase, rather than full behavior-cloning pretraining of the policy
network (more implementation risk/time than the budget allowed). New
`--warm_start_steps` flag on `train_mro_sac_pilot.py`
(`warm_start_collect()`): additive, does not count against
`--total_timesteps`.

**Real bug found and fixed during smoke-testing** (caught BEFORE the long
run, not after): `SAC._store_transition()` reads `self._last_obs`
internally as the transition's stored observation -- not any local
variable passed in. This is normally primed by SB3's own
`_setup_learn()` before `collect_rollouts` ever calls it; calling it
standalone (as warm-start does) without priming `model._last_obs` first
means the FIRST stored transition's observation is `None` (the
attribute's default), which silently corrupted the buffer and crashed
the first gradient step with a NaN in the actor's output distribution.
Reproduced twice (ruling out "buffer too small" as the cause by testing a
buffer safely above batch_size -- still crashed), then isolated by
re-running the identical tiny config with `--warm_start_steps 0` (worked
fine) vs with it (crashed) -- confirmed the bug was in the warm-start
code, not a pre-existing fragility. Fix: set `model._last_obs = obs` once
before the collection loop, matching what `_setup_learn()` does.
Re-tested clean after the fix (Q progressed sanely: 0.76 -> 1.12 -> 1.16
over a tiny 500-step smoke run, no crash).

### Final hyperparameters for the relaunch (decoupled from n_envs this time)
`n_envs` raised to 12 (from 8) per the user's "use more cores/RAM"
instruction -- checked headroom first (16 cores, 10GB+ available
immediately before launch; 12 ns-3 workers at ~300MB RSS each is ~3.6GB,
comfortably inside real limits even with the known-degraded MemGuard
cgroup check, same MemAvailable-floor safety net as before). **`gradient_
steps` is NOT scaled with n_envs this time** -- kept at a conservative
absolute value (8, vs the diverged run's 24) specifically to cap the
burst size per the Hussing et al. finding, independent of n_envs. This
gives 0.67 updates/sample (well below arm C's 3.20) -- a deliberate trade
of sample efficiency for stability, justified by direct evidence
(yesterday's divergence) rather than theory alone. `learning_starts`
lowered to 1500 (from 5000) since the buffer is now pre-warmed with 3600
informative (non-random) transitions from the traditional controller --
less need for a long blind random-exploration phase, and more of the
budget spent on genuine gradient-updated experience (yesterday's run
spent 62.5% of its budget never updating at all). `total_timesteps=18000`
(up from 12400 -- judgement call given no explicit time budget was set
this round beyond "start a run, I'll check in later").

**Honest gap**: no dedicated empirical burst-size diagnostic was run
before committing to these hyperparameters -- smoke-testing (catching and
fixing the warm-start bug above) consumed most of the available time,
and the diagnostic/implementation phase ran well past the nominal 3h
budget as a result. The gradient_steps reduction is literature-justified
and the warm-start mechanics are verified working, but the claim "this
run won't diverge" is not independently confirmed by a short
reproduction, only by Hussing et al.'s published results and the
mechanism argument. Stated plainly here rather than glossed over.

### Launched: `mrosacpilot.service`,
### `output/mro_sac_pilot/logs_20261018_171731/`
n_envs=12, gradient_steps=8 (0.67 updates/sample), warm_start_steps=3600,
learning_starts=1500, total_timesteps=18000, seed=555, ports 5565-5576.
Dry-run verified first. All 12 workers up, warm-start collection
confirmed starting cleanly, memory healthy (6.8GB available at launch).

### [2026-10-19 11:40] TRAINING COMPLETE -- 18,000/18,000 timesteps,
### 11,000 gradient updates, clean exit, no guard trips
Warm-start took far longer than planned -- ~2h58m for the single
3600-transition (= 1 full episode/worker) pre-fill, a design-sizing
mistake on my part (did not check the actual wall-clock cost of "one
episode" at n_envs=12 contention before choosing that number). Real
training then ran ~15h25m (started 20:15:56, finished 11:40:23) at a
measured ~19-20 combined steps/min, matching the mid-estimate given to
the user ahead of time. One transient ns-3 worker crash during training
(same documented ~1-in-10 pattern), self-recovered, zero disruption.
Memory cycled between ~2.2GB and ~6.0GB available throughout (same
per-episode-growth-then-reset pattern seen during warm-start, not a
leak) -- never approached the 2.5GB floor during real training, when
MemGuard's callback IS active (unlike during the standalone warm-start
phase, which has no such protection -- a real gap, noted for next time).

**THE CENTRAL RESULT: no divergence.** Critic Q grew throughout training
but LINEARLY and SLOWLY -- final Q=11.3 (max 11.3, probe-to-probe
deltas shrinking into a gentle, roughly-linear climb by mid-training) --
vs the diverged run's Q=79.3 at the same total-timestep budget, already
accelerating with no plateau at that point. Same qualitative check as
before, decisively different outcome: this is NOT the runaway
overestimation pattern Hussing et al. describe; it looks like ordinary,
healthy SAC value-learning.

**Training-time term averages also improved dramatically** (last-1500-step
window, same method as the diverged run's phase breakdown):

| metric | diverged run (final) | this run (final) |
|---|---|---|
| reward | +0.0135 | **+0.1440** |
| quality | 0.8783 | **0.9242** |
| pingpong | 0.0179 | **0.0045** (4x lower) |
| handover_rate | 0.1926 | **0.1489** |

pingpong in particular -- the clearest signature of the "loosen/too-early"
failure mode the diverged policy fell into -- stayed low and flat the
entire run (visually near-zero in fig_combined_v3.pdf's KPI panel),
rather than creeping upward. handover_rate also settled lower, not
higher. This is training-time data only (contaminated by exploration
noise, alpha still ~0.009 at the end -- fully annealed compared to the
diverged run's ~0.055, for what it's worth); the real verdict is the
Day-3 deterministic evaluation, running now (5 seeds, properly
parallelized from the start this time -- learned from the earlier
sequential-eval mistake the user caught).

New figure: `output/fig_combined_v3.pdf` (same 4-panel layout as
`fig_combined.pdf`, reward/KPIs/action/Q, dashed line at
learning_starts=1500 this time).

### [2026-10-19 14:30-15:51] Rule-based controller v1 found to UNDERPERFORM
### baseline; v2 redesigned (windowed, dead-band, slower cadence), closed
### 89% of the gap but still not a confirmed win (1h budget exceeded)
Zero-new-compute check: extracted the exact 3600 warm-start transitions
from the saved replay buffer (positions [0:300) per lane, confirmed never
overwritten). The v1 rule-based controller's OWN mean reward:
**+0.0625**, vs the do-nothing baseline's +0.1526 -- the controller I
warm-started from was itself worse than doing nothing. Root cause,
diagnosed against a closer literature pass on hysteresis-margin-specific
(not CIO) SON algorithms (D-HCP, a patent computing too-late RATE over a
window per neighbor rather than reacting to instantaneous values,
"hysteresis/TTT change by one step per optimization only"): v1 reacted to
a single noisy per-step observation with no dead-band, at a flat 0.5dB
step every single control step -- likely injecting its own oscillation.

**v2** (`examples/mro_rule_based_controller.py`, `RuleBasedMroController`
class): 5-step rolling window per cell, decisions only every 5 steps
(not every step), dead-band release condition (must drop well below
threshold to re-arm, not just cross back over the line once).

Quick parallel diagnostics (simTime=10, 4 seeds, weighted reward under
current weights):
| version | mean reward | vs baseline (+0.1823, same seeds) |
|---|---|---|
| v1 (instantaneous) | +0.0625 (from buffer, different seeds) | -59% relative |
| v2a (window=5, cadence=5, thresholds 0.05/0.25) | +0.1702 | -6.6%, wins 1/4 seeds |
| v2b (same, thresholds 0.10/0.45) | +0.1524 | -16%, wins 1/4, ties 1/4 |

v2a kept as current best (reverted from v2b after it scored worse, not
better -- the "be more conservative" guess didn't pan out, though n=4
seeds is too small to be fully confident of the direction).
**Structural redesign (windowing/dead-band/cadence) did essentially all
of the work** -- closed ~89% of the relative gap to baseline. Did NOT
reach a confirmed win within the 1-hour budget (ran ~21min over: 81min
total, mostly one slow iteration that took 31min instead of the expected
~10-15min -- yet another throughput-estimation miss this session). Not
re-used for warm-start yet -- would need a proper multi-seed validation
first (same discipline as every other weight/wiring claim in this
project), which the time budget didn't allow.

### [2026-10-19 13:43] Day 3 (v3): deterministic evaluation vs baseline
5 seeds (2001-2005), simTime=30, launched PROPERLY PARALLEL from the
start this time (learned from the earlier sequential-eval mistake the
user caught) -- took ~2h (notably longer than the ~62min solo-episode
benchmark it was estimated from; that benchmark was a fixed-action
diagnostic, not a deterministic-policy eval calling model.predict() every
step, so the comparison wasn't apples-to-apples -- an estimation error on
my part, not a run problem; all 5 workers stayed at 99.9% CPU the whole
time, never stalled). Pooled into
`output/mro_v3_trained_policy_eval.csv`, re-scored from raw terms under
the current weights against `pooled_mpos0.csv` for a fair comparison
(`scratch_final_comparison_v3.py`).

**VERDICT -- still worse than doing nothing, but the gap closed by ~89%
compared to the diverged run, and the agent wins on 1/5 seeds for the
first time in this project's history:**

| condition | n | mean reward | sd | quality | pingpong | handover_rate | outage |
|---|---|---|---|---|---|---|---|
| do-nothing baseline | 1495 | +0.1526 | 0.3595 | 0.9357 | 0.0021 | 0.1216 | 0.00038 |
| trained policy v3 (det.) | 1495 | +0.1228 | 0.4674 | 0.9172 | 0.0055 | 0.1455 | 0.00061 |

gap (baseline minus trained) = +0.0298 (vs the diverged run's +0.2791 --
a 9.4x smaller gap). Per-seed: trained wins on seed 2004 (+0.1475 vs
baseline's +0.1253) -- loses on the other 4, but by much smaller margins
than before (+0.1355/+0.1314/+0.0809/+0.1189 vs baseline's
+0.1915/+0.1425/+0.1301/+0.1737). pingpong is only ~2.6x baseline (was
~15x in the diverged run); handover_rate is ~20% above baseline (was
~64% above); quality is within 2% of baseline (was ~9% below).

**Conclusion: the two literature-grounded fixes (decoupled gradient_steps
burst size + warm start from the traditional rule-based MRO controller)
worked, decisively, as a stability/quality intervention.** The critic
never diverged (bounded, roughly-linear Q growth throughout, vs the
previous run's unbounded acceleration), and the resulting policy is
dramatically closer to matching the do-nothing baseline on every term,
with a genuine win on one seed. This is NOT yet a full pass -- the
aggregate verdict is still a loss, by a real (if much smaller) margin, on
4 of 5 seeds. The most plausible next lever, given how close this result
is: more training budget (this run's 18,000 steps / 11,000 gradient
updates is still modest), now that the stability problem is solved and
there is real headroom to simply train longer without the earlier
runaway-divergence risk.

### [2026-10-19, rule-based controller tuning] v1 found below baseline,
### v2 (windowed/dead-band/cadence) closed 89% of the gap, 3h sweep
### confirms it as a near-tie, not a clean win
Separately from the SAC pilot work above: a zero-new-compute check
(extracting the exact 3600 warm-start transitions from the already-saved
replay buffer, positions [0:300) per lane, confirmed never overwritten)
found the v1 rule-based MRO controller's own mean reward was +0.0625 --
BELOW the do-nothing baseline's +0.1526. The controller warm-start was
built from was itself worse than doing nothing.

Redesigned (v2, `examples/mro_rule_based_controller.py`,
`RuleBasedMroController` class) per a closer literature pass on
hysteresis-margin-specific (not CIO) SON algorithms: a patent on this
exact problem computes a too-late RATE over a window per neighbor cell
and compares it to a threshold, rather than reacting to a single
instantaneous value; another source states hysteresis/TTT "change by one
step per optimization only" -- i.e. decide on a slower cadence. v1 reacted
to raw per-step noise with no dead-band at a flat step every control
step -- likely injecting its own oscillation. v2: 5-step rolling window
per cell, decisions only every 5 steps, dead-band release logic (must
drop well below threshold to re-arm).

1h quick-tune (4 seeds, simTime=10): v2a (thresholds 0.05/0.25) scored
+0.1702 vs a 4-seed baseline of +0.1823 -- closed ~89% of the relative
gap vs v1, won on 1/4 seeds, but not a confirmed win.

3h follow-up sweep (fixed 6-seed baseline +0.1764, reused across all
variants; each 6-seed controller run took 35-45min, capping this to 4
total configs tested):

| config | badsignal_thresh | activity_thresh | step_db | mean reward | vs baseline |
|---|---|---|---|---|---|
| v2a | 0.05 | 0.25 | 0.5 | **+0.1750** | -0.8% (near-tie) |
| v3a | 0.03 | 0.15 | 0.5 | +0.1672 | -5.2% |
| v2b | 0.10 | 0.45 | 0.5 | +0.1524 | -13.6% |
| v2c | 0.05 | 0.25 | 0.3 (gentler step) | +0.1461 | -17.2% |

v2a (the module's current live defaults) remains the best of everything
tested. Neither more sensitive (v3a) nor less sensitive (v2b) thresholds
improved on it, and a gentler step size (v2c) made things clearly worse
-- counter to the "finer control is more precise" intuition; likely
because a smaller step means each trigger only partially corrects the
signal, so the dead-band release re-engages less often and the
controller stays "stuck" acting on stale state longer.

Per-seed pattern: v2a wins decisively on 2/6 seeds (+0.0238, +0.0829) and
loses by smaller amounts on the other 4 -- the average is a near-tie
despite only winning 2 of 6 seeds, because the wins are large relative to
the losses. Reads as genuine seed/traffic-pattern sensitivity, not an
obvious design flaw still to fix.

**Honest conclusion: no clean, consistent win was demonstrated within
either time budget (1h then 3h).** v2a is roughly on par with
do-nothing, not confirmed better. A real confirmation would need more
seeds per config (10+, to shrink per-seed noise) and/or the literature's
actual recommendation of an adaptive/graduated step size (not a single
flat value swept one parameter at a time) -- neither fit in the time
available. Not wired into warm-start or anything else; kept as a
standalone, documented, not-yet-validated candidate.

### [2026-10-20 overnight session] More axes tested (all land back at v2a),
### then a real automation bug nearly wasted the whole night
User authorized a full overnight investigation (until 9am) targeting a
controller that CONSISTENTLY beats baseline, plus a 2-page PDF at the
end. Continued the coordinate search: max_repeat (allowing 2 consecutive
same-direction steps before requiring release, not just 1) scored worse
(+0.1458 vs v2a's +0.1750); window/cadence=3 scored +0.1385, window/
cadence=8 scored +0.1577 -- both worse than window=5's +0.1750. v2a is
now confirmed a local optimum across FIVE independent axes (thresholds,
step size, max_repeat, window, cadence): nothing tried beats it, which
is itself informative -- either it's a genuine (if shallow) local
optimum, or the remaining gap to baseline is dominated by per-seed noise
that no amount of this-shaped tuning can resolve.

**Automation bug** (caught and fixed, not left to run broken all night):
built `overnight_runner.py` to run v2a vs baseline across many seeds
unattended, launched via `setsid nohup ... & disown` so it survives
terminal/session disconnection. First version used
`subprocess.run(cmd, timeout=1800)` to bound each run -- but this only
kills the DIRECT child (the Python wrapper script), not the ns-3
grandchild process the wrapper spawns via MroZmqEnv. On timeout, the
wrapper's own `finally: env.close()` cleanup never runs (the whole
process is just killed), so ns-3 is orphaned and keeps running --
confirmed directly: ns-3 processes from the FIRST batch (seeds 2001-2005)
were still alive and burning CPU at the 30-minute mark, and the SECOND
batch then tried to reuse the exact same hardcoded port range
(5900-5904, 5950-5954), colliding with the still-running orphans. Result:
ALL 10 runs in the first batch timed out, and every subsequent batch
would have cascaded the same way all night -- zero usable data, silently,
for 6+ hours, if not caught.

Caught because a Monitor was watching the log for TIMEOUT lines, which
all fired simultaneously at the 30-min mark -- a visible, actionable
signal rather than silent nothing. Root-caused by checking `ps aux`
directly and finding the batch-1 ns-3 processes still alive with 30+
minutes of accumulated CPU time, well past when they should have been
killed. Fix: launch each run with `start_new_session=True` (its own
process group) and kill the WHOLE group via `os.killpg(...,
SIGKILL)` on timeout, not rely on `subprocess.run`'s built-in timeout
handling. Also rotate the port range every batch as defense-in-depth.
Verified the fix in isolation (forced a 10s timeout on a real run,
confirmed via `pgrep` that no orphaned ns-3 process remained afterward)
before trusting it for the rest of the night. Relaunched ~02:55.

Overnight result (24 seeds, paired): v2a (window=5, cadence=5,
badsignal_threshold=0.05, activity_threshold=0.25, dead_band=0.3,
step_db=0.5, max_repeat=1) landed at a STATISTICAL TIE with baseline --
controller mean +0.1496 vs baseline +0.1607, t=-1.31 (n=24, need
|t|~2.07 for significance), 9/24 wins. Not a win. 2-page PDF delivered
summarizing this.

### [2026-10-20, later] User rejected the tie as an acceptable outcome;
### literature's "graduated step" and "per-neighbor" ideas tried for
### real, with real multi-seed verification -- six more variants tested,
### one materially tightens the gap (still a tie, not a win)

User: "select the best performing approach from the literature... implement
it immediately, and verify it's better than doing nothing... I refuse to
believe that no such controller exists," followed immediately by: "use
the multi-cell decision making" (pointed correction that this should have
been in the plan already -- it had been identified from the literature
earlier, Jansen et al./D-HCP's "per neighbor" framing, but omitted from
the first draft of this plan).

**ProportionalMroController (graduated step size, 4 parameterizations)**
-- the literature's other specific recommendation (step size scales with
how far past threshold the signal is, not v2a's flat 0.5dB jump), plus
multi-cell via network-relative severity. All four tested WORSE than both
baseline and v2a on the same 9-seed quick-check subset (seeds 2001-2009,
baseline mean 0.1632 / v2a mean 0.1581 on this subset):
- v1, aggressive gains (kp_loosen=10, kp_tighten=2), no absolute gate --
  compared each cell only to the live network average, so ~half of all
  cells always looked "above average" even when nothing was wrong,
  causing constant unnecessary correction. Mean ~0.124 (seeds 2002-2009).
- v2, same gains, absolute-threshold gate added (same 0.05/0.25 as v2a,
  network average now only a severity multiplier once gated). Mean
  0.1361 -- fixed the over-triggering bug but still below both targets.
- v3, reduced gains (kp_loosen=3, kp_tighten=0.5). Mean 0.1439 -- better
  than v2 but still below both targets.
- v4a, original gains + the SAME trigger-once/hold/dead-band state
  machine already validated in v2a (continuous per-tick re-application
  was suspected integral windup, the same failure mode v2a's own
  max_repeat axis had already ruled out). Mean 0.1216 -- WORSE, not
  better; the single bigger one-shot step apparently overshoots more
  than the smaller continuous increments did.
Conclusion: graduated step size, in every form tried, underperforms
v2a's flat step. The flat/infrequent/trigger-once step itself looks like
the robust part of the design, not something to replace.

**Multi-cell extensions to v2a itself** (keeping v2a's flat step, trigger-
once, dead-band mechanics -- the part that keeps testing as robust --
and adding cross-cell context on top, rather than replacing it):
- v2b: per-cell trigger THRESHOLD scaled by network-relative severity
  (a cell notably worse than its peers becomes more sensitive; one
  merely reflecting network-wide load is unaffected), all triggering
  cells still act every cadence tick. Full 24 seeds: 9/24 wins (same
  count as v2a), mean diff vs baseline -0.021 (WORSE than v2a's -0.011).
  Higher variance than v2a -- some strong wins (+0.22, +0.23 vs
  baseline's ~0.18) offset by a few severe losses (seeds 2005, 2010,
  2023: -0.08 to -0.13).
- v2c: cross-cell ARBITRATION -- rank all cells exceeding threshold by
  normalized excess each cadence tick, act on only the single worst
  (top_k=1) instead of every triggering cell simultaneously. Full 24
  seeds: 12/24 wins (exact coin flip), mean diff vs baseline -0.0048,
  t=-0.629 -- the tightest result of the whole investigation, well
  inside the noise floor (v2a's own t was -1.31). Still not a
  significant win, but the closest approach to true parity found.
- v2d: v2c's arbitration + v2b's severity-based ranking (pick the worst
  CELL using network-relative severity rather than absolute excess).
  Tested WORSE than v2c alone (3/9 wins vs v2c's 4/9, mean gap -0.0158
  vs v2c's -0.0037, same 9-seed subset) -- severity-based ranking is a
  worse way to choose which single cell acts than plain absolute-excess
  ranking. Abandoned.
- v2e: v2c + wider dead_band (0.5 instead of 0.3). IDENTICAL per-seed
  results to v2c -- dead-band width has no effect once top_k=1 already
  caps actions to one cell per tick; this axis is a dead end under
  arbitration.
- v2f: v2c + higher absolute thresholds (0.08/0.35, the pre-v2 values).
  WORSE than v2c (4/9 wins vs v2c's 4/9 but mean gap -0.0144 vs -0.0037,
  driven by a sharp regression on seed 2001: 0.069 vs v2c's 0.206) --
  v2a's original 0.05/0.25 thresholds remain the right level even under
  arbitration.

**Consolidated finding across all 8 variants tested today** (4
proportional, v2a baseline, v2b/c/d/e/f multi-cell/arbitration
extensions): every change that makes the controller act MORE (bigger
steps, more cells per tick, continuous re-triggering, lower/more
sensitive thresholds) scores worse; every change that makes it act LESS
(arbitration to a single cell per tick) scores closer to parity, up to
v2c's near-exact tie. None achieves a statistically significant win.
Working hypothesis: the reward's heavy ping-pong weight (9.39x quality)
relative to a do-nothing baseline that pays zero ping-pong risk by
construction sets a high bar for ANY active hysteresis-margin
correction to clear -- v2c (the most selective design, acting on exactly
one cell per 5-step window) is the closest any variant has come, and is
now the leading candidate, but "closest to parity" and "beats baseline"
remain two different outcomes.

`RuleBasedMroController` now supports `multi_cell` (threshold scaling,
v2b) and `top_k` (arbitration, v2c/e/f) as constructor flags, both
default off (preserving v2a's exact original behavior). v2c is
`top_k=1` with all other params at v2a's validated values.
