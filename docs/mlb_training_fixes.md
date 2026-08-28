# MLB training pipeline: what we fixed, what we measured, and what will bite you

This began as a note on why PPO on `MlbZmqEnv` (live ns-3 over ZMQ) was too slow
to train. It has since grown to cover the reward's validation in the training
regime, the post-mortem on the first real pilot, and how the training algorithm
was chosen. Figures are measured on this machine (12 logical / 6 physical cores,
~8 GB RAM available) rather than estimated, unless labelled otherwise.

---

## What changed

In `src/environments/mlb_zmq_env.py`, there is a new `KPI_WINDOW_S = 0.1`
constant that documents the counter-reset chain: `m_macVolumeCellSpecific` is
written at `mmwave-enb-net-device.cc:1212`, and the per-UE sources are zeroed by
`ResetPhyTracesForRntiCellId` at `:1377`. The default for `control_period_s`
moved from `1.0` to `KPI_WINDOW_S`, and the two comment blocks that argued for
1.0 were rewritten rather than left contradicting the code. There is a new
`purge_sim_path_on_close` parameter, defaulting to `False`, along with
`_purge_sim_path()`. Finally, `close()` now purges and also reaps the child
process: it runs on every `reset()`, and without a `wait()` a long run left one
zombie per episode. That last one was a latent bug rather than anything on the
original list.

`examples/train_mlb_smoke.py` sets `purge_sim_path_on_close=True`.

`tests/test_mlb_zmq_env.py` asserts `control_period_s == KPI_WINDOW_S`, and its
stale "T_control = 1.0 s" comment was corrected. The suite has since grown to 22
tests there and 11 in `tests/test_step_logger.py`, all passing. Neither needs
ns-3 or ZeroMQ, so they run in seconds off stubs.

Purging defaults to off, with training opting in, because
`examples/cio_zmq_experiment.py` calls `env.close()` and then writes its figures
into `env.sim_path`. A default-on purge would have silently broken that plotting
script. The guards were verified directly: it refuses non-UUID directory names,
anything not directly inside `output_folder`, symlinks, and `sim_path=None`.

---

## The new Backlog magnitude

Recomputed from real logged KPIs rather than synthetic ones, in
`output/41b101c9-.../20260814_202537_kpis.csv`:

| | mean | max | steps where backlog > balance's [0,1] range |
|---|---|---|---|
| Old (÷1.0 s) | 0.8766 s | 3.0662 | 3 / 9 |
| New (÷0.1 s) | 0.0877 s | 0.3066 | 0 / 9 |

That is exactly 10.00x, and the reward is now properly scaled, with Backlog no
longer swamping Balance. This is what was driving the smoke test's
`ep_rew_mean = -3.6`.

---

## Why `enableTraces: [0]` was not added

It was requested, and this document previously recommended it, but it turns out
to be the same class of silent KPI killer as the E2 flags.

The chain runs like this. `enableTraces` gates `MmWaveHelper::EnableTraces()`,
which is the only place `RxPacketTraceUe/EnbCallback` are connected
(`mmwave-helper.cc:3116-3141`). Those callbacks are the only callers of
`MmWavePhyTrace::UpdateTraces()`, which is the only writer of
`m_macVolumeUeSpecific` (`mmwave-phy-trace.cc:262`). That same `MmWavePhyTrace`
instance is the device's `E2DuCalculator` (`mmwave-helper.cc:2208`), which is
the source of `volume_bytes` and `prb_utilization`.

The evidence was there in the earlier measurements and was misread. The
traces-off rollout returned `reward == 1.000` on every step, with balance guarded
on `max_prb == 0` and backlog 0 on `volume == 0`, and that was put down to
traffic warm-up. The real CSV above, same seed but with traces on, carries
~1.1 MB/step from t=1.0, so it was not warm-up: the counters were dead. Traces
are not a logging luxury here, they are the KPI pipeline. The earlier claim that
"10 simulated seconds is pure warm-up" was wrong and should be disregarded.

The consequences are that the ~11% speedup is not available, so per-step cost
stays at ~88 s, and disk usage returns to ~5 MB per simulated second, which makes
the purge fix mandatory rather than optional.

### Three flags to leave at their defaults

| Flag | Default | What breaks if changed |
|---|---|---|
| `enableTraces` | `1` | `volume_bytes` and `prb_utilization` never increment, silently |
| `enableE2FileLogging` | `1` | Selects a live RIC connection instead of offline files; ns-3 died at step 5 of 10 |
| `e2cuCp` / CU-CP reporting | on | `m_l3sinrMap` never filled; all 7 `sinr_db_mean` features pin to -40 dB, silently |

All three rationales are recorded in the training CFG so nobody has to re-derive
them later.

---

## Cost decomposition

| Phase | Time | Frequency |
|---|---|---|
| `construct` (configure + build check) | 9.90 s cold / 2.18 s warm | once per process |
| `start_sim` (mkdir + Popen + ZMQ bind) | 0.0008 s | per episode |
| `first_kpi_wait` (launch to first KPI) | 0.124 s | per episode |
| `close` (teardown) | 0.0008 s | per episode |
| steady-state per step | 79.06 s (traces off) / ~88 s (traces on) | per step |

Fixed per-episode overhead totals 0.126 s, about 0.016% of a step. The original
hypothesis that ns-3 startup dominates is wrong: there is nothing meaningful to
amortize, and longer episodes do not help. Wall time scales with simulated
seconds rather than RL steps, since ns-3 simulates 1 s of this scenario in about
88 s of wall clock.

---

## The two ns-3 knobs this document used to list as pending

Both are now done and in the pushed code, though the names differ from what this
document originally proposed. What it called `zmqEndpoint`, a full
`tcp://host:port` string, shipped instead as `zmqPort`, a `UintegerValue`
(`uint16_t`) defaulting to 5555. `controlInterval` kept its name and shipped as a
`DoubleValue` defaulting to 1.0 with a minimum of 0.01.

`MlbZmqEnv.__init__` now sets `scenario_configuration['zmqPort']` from its own
`zmq_port`. Those used to be two separate settings, and a caller that set only
one of them got a silent deadlock: Python listening on one port, ns-3 blocked
forever on another, both processes alive at roughly 0% CPU with no error and no
log line anywhere. The environment now owns the port and passes it down, so the
two cannot disagree.

The build race this document predicted did happen. Every `SubprocVecEnv` worker
constructed its own `MlbZmqEnv`, which calls `setup_sim()`, which calls
`configure_and_build_ns3()`, so N workers ran concurrent `./ns3 build` against
one CMake cache and one lock file. The fix is a `build_ns3` parameter, defaulting
to `True`. `train_mlb_parallel.py` builds once in the parent with a throwaway
environment, then constructs every worker with `build_ns3=False`. Measured at 2
workers that gives 1.86x throughput with results byte-identical to sequential, so
parallelism does not perturb the simulation.

One caution from the original text still stands: `control_period_s` tracks
`indicationPeriodicity` (0.1 s), not T_control. Do not wire one from the other.

---

## Projected throughput for 100k steps

These projections date from 2026-08-14 and assume step count is the budget. It is
not, for the reasons in "What the pilot cost, and why it did not learn" below.
They are kept here for the cost model rather than for the conclusion.

| Config | 100k steps |
|---|---|
| Today, 1 worker | ~102 days |
| + 6 workers | ~17 days |
| + 6 workers + T_control 0.2 | ~3.4 days |

In practice the ceiling is 5 workers rather than 6. The machine has 6 physical
cores and ns-3 is single-threaded at roughly 0.5 GB RSS. Measured cost is about
285 s per sample at 5 workers under RLC AM, and each 35-UE run directory is about
250 MB, which is why purging is mandatory rather than optional.

---

## Precondition: resolved

The reward varies across a live rollout, and the reward itself is now validated
in the training regime. A damped balancing controller beat do-nothing on the
physical KPIs, delivering 87.8 Mbps against 84.2 and reaching a Jain index of
0.583 against 0.425, while a flip-flop controller was correctly ranked worst
with 461 handover starts against neutral's 240.

Getting there took one wrong turn worth recording. An undamped arm (K=12)
improved every physical KPI but churned 1.36x neutral, and it scored a
statistical tie rather than a win. The reward was not blind to the improvement;
it was charging a ping-pong bill that slightly exceeded it. Dumping the offsets
the controller actually applied showed why: 7 sign flips in 10 steps, and 3.12 dB
of movement per step. It was pulling users into a cell, watching that cell's PRB
rise, then shoving them back out on the next step. Adding damping (K=8, slew
limit 1.5 dB/step, deadband 0.04, EMA 0.5) brought that down to 0.67 dB/step and
ping-pongs from 8.9 to 5.7 per step. The lesson is that the bounced users are the
ping-pongs: an oscillating controller manufactures the churn it then gets
penalised for.

---

## What the pilot cost, and why it did not learn

A 16.5 hour pilot on 5 workers ran clean. Zero crashes, zero stalls, checkpoints
landed, all five reward terms live in the sum. It also learned nothing, and the
cause was not the environment.

The value function never got near its own scale. SB3 bootstraps at truncation,
which makes the task continuing, so the value target is
`r/(1-gamma) = 0.470/0.01 = 47.0`. Loading the pilot checkpoints and measuring V
directly, the critic ended at 6.52, about 14% of scale, climbing roughly 0.5 per
iteration. That is around 90 more iterations just to reach the right magnitude.
Its sd was 0.25 against a target sd of 4.79, so it was close to a constant
function. That also explains why `value_loss` rose from 9.27 to 17.5 instead of
falling: it was chasing a target that kept growing.

The gradient budget was the other half of it. With `n_steps=16` and
`batch_size=80` the whole rollout is a single batch, giving 10 gradient steps per
iteration and 130 in the entire pilot. A normal PPO run does 1e4 to 1e5.

Three other explanations were tested and ruled out. Observation scaling was not
it: over 299,385 real cell-steps the scaled-field SD ratio is 11.8x, not the
21.8 million that had been feared, and clipping never binds. Irreducible reward
noise was not it either, since a leave-one-episode-out ridge predicts the
discounted return at EV +0.60 out of sample. Nor was there an unmodellable clock:
`buf_MB` alone predicts the return at EV +0.588, and `buffer_bytes` is already
in the observation.

### Picking an algorithm

Racing algorithms on the real simulator is not affordable, so we built a
synthetic environment matched to the real reward statistics and ran the
comparison there at roughly 1e5 steps per second. The match is close: untrained
mean reward +0.4697 against a real +0.470, sd 0.3399 against 0.344, and
within-episode autocorrelation +0.7197 against +0.716. Behaviour cloning with
the exact SB3 network (64x64 Tanh) reaches 100% of the gap, so anything an RL
agent leaves on the table here is credit assignment and sample efficiency rather
than capacity or observability.

Results are given as a percentage of the neutral-to-optimal gap, measured from
neutral because a policy that merely outputs zero scores 0.4827 for free.

| arm | 20k | 50k | 60k | 100k | 200k |
|---|---|---|---|---|---|
| PPO, pilot config | -16.9% | 19.2% | 21.2% | 31.3% | 24.6% |
| PPO, gamma .95 / n=128 | 5.1% | 21.7% | 25.7% | 44.5% | **56.7%** |
| PPO, lr 1e-3 | 8.5% | 25.3% | 27.6% | - | - |
| **SAC, gradient_steps=16** | **26.8%** | **42.2%** | **49.8%** | - | - |
| SAC, gradient_steps=64 | -4.5% | - | - | - | - |

PPO does not get there at 50k, landing at 19% to 25% in all three
configurations. SAC with `gradient_steps=16` is about three times more sample
efficient, reaching at 20k what tuned PPO needs roughly 65k to 70k for, and it is
the only arm that gets near half the gain inside a plausible budget. On the real
simulator at 285 s per sample with 5 workers, that is about 33 to 40 days to 50%
of the gain, against never for the pilot configuration.

Two results ran against expectation. More gradient steps per sample actively
hurt: gs=64 scores below doing nothing at -4.5%, and with `learning_starts=500`
it fell to -0.435, far below untrained. A higher learning rate did not help
either, giving 27.6% against 25.7% at 60k, which is within noise, so step size is
not what binds. SAC is also unstable early, with both seeds collapsing to about
-0.16 around 5k samples before recovering; on the real simulator 5k samples is
about 33 days in.

The synthetic has real limits. It has no ns-3 dynamics at all: no handover
hysteresis or TTT, no ping-pong, no RLC queue physics, no per-cell capacity
differences, and its load process is stationary linear-Gaussian with a reward
that collapses to a single imbalance scalar rather than five interacting terms.
`KAPPA = 0.35`, how much load one dB of differential CIO moves in a single step,
was chosen rather than measured, and it directly controls per-step
credit-assignment difficulty, so the absolute budgets are sensitive to it in a way
the reward statistics are not. Treat the ranking of the arms as far more robust
than the sample counts. A pass here is necessary but not sufficient; a failure
would have been decisive.

---

## Two ways to misread a training run

Both of these were hit on the pilot, and both are easier to avoid up front than
to undo once you have started reasoning about numbers you have already seen.

The first is correlating reward against raw queue bytes. Backlog is a drain time,
queued bytes divided by the rate they leave at, so raw bytes are the wrong
quantity to judge it by. On the pilot, 226 of 690 steps had reward going up while
raw buffer bytes rose, which is 32.8% against a 25% chance rate and reads like
reward hacking. It was not. On exactly those steps the backlog term fell, because
the drain rate was rising faster than the queue. Judging a drain-time penalty by
raw bytes manufactures a false positive.

The second is testing for learning without clustering.
`vary_rng_run_per_episode=True` redraws the ns-3 seed every episode, so episodes
are not independent samples of a policy. On the pilot a per-episode regression
gave t(23) = 2.19 and looked significant, while the clustered test over the 5
episode waves gave t(3) = 1.49 and did not. `std=1.0` and `clip_fraction=0`
confirmed the policy had never moved at all. Report the clustered number, and
read it next to the SB3 log's `std` and `clip_fraction`.

The general rule behind both: reward is the training signal, KPIs are the
verdict. Never evaluate on reward alone. This project has twice been bitten by a
term improving while the network got worse.

---

## Why the logging is in-process

Reward terms used to be reconstructed after the fact from ns-3's output files,
and that broke twice, so `src/callbacks/step_logger.py` now records them at the
source instead. Reconstruction also could not be made complete. Out-of-process
tailing captured 715 of roughly 1040 steps, about 69%, and it never captured the
last step of any episode, 25 out of 25, because the run directory is purged at the
episode boundary before the next poll comes round. That last step is the worst one
in the episode, so the tailed mean was biased about 0.07 high. No poll interval
fixes this. Only in-process logging does.

---

## A testing trap worth knowing about

The project is installed editable, with a finder hook
(`__editable__.ns_o_ran_gym-0.1.0.pth`) that hardcodes the real `src` path and
overrides `sys.path`. The `sys.path.insert` at the top of the test files is
therefore decorative. You cannot test a modified copy of the environment by
juggling paths: it will silently import the original and report a false pass.
Test the real file, or uninstall the editable package first.
