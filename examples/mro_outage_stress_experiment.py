"""Measure OutageExposure's real scale under a condition that actually stresses
it -- not the do-nothing baseline it was structurally silent on.

docs/mro_training_fixes.md's round-2 baseline measured OutageExposure as
alive (0.0001 mean, fired on 0.01% of UE-steps) but its literal equalised
weight (159.7) was unusable: a term that rare swinging the reward that hard
on the one step it fires is the same failure shape MLB's own BadSignal term
hit. The shipped DEFAULT_W_OUTAGE (1.0) is a fixed guard value chosen BY
ANALOGY to BadSignal's own precedent, not derived from a real measurement of
OutageExposure under conditions where it actually fires meaningfully.

This script is that measurement. The physical mechanism MRO's margin action
can realistically use to manufacture real outage is well-grounded, not an
arbitrary stress hack: margin_offset = +MARGIN_LIMIT_DB (+3 dB) on every cell
makes GetEffectiveSinrThreshold MORE restrictive -- handovers trigger only on
a much bigger target-cell SINR advantage, so a UE moving away from its
serving cell stays latched onto it LONGER than it should. That is the
textbook "too-late handover" MRO failure mode: lingering on a degrading
serving cell before finally handing over is exactly what should drive real
SINR outage. This mirrors mro_margin_experiment.py's already-validated
"loosen" arm (-3 dB, confirmed +7% real ns-3 handover activity) but at the
OPPOSITE, never-yet-tested extreme -- loosening makes handovers easier
(irrelevant to outage), stickiness makes them harder (the actual outage
mechanism).

Same seeded-episode / incremental-write method as mro_baseline_experiment.py,
same locked SAC config (ues=5, simTime=30, controlInterval=0.1, RLC AM,
/dev/null trace redirects) -- so this arm's numbers for quality/pingpong/
handover_rate are directly comparable to the round-2 do-nothing baseline on
record, and the trajectory-ranking check below can use them together.

PARALLEL EXECUTION (--workers)
-------------------------------
Each seed is an independent episode (its own RngRun) -- nothing couples one
seed's simulation to another's -- so running several at once is purely a
resource question, not a correctness one. Checked before relying on this:
one ns-3 process at this config (ues=5, controlInterval=0.1) costs ~1 CPU
core and ~230 MB RES; on a 16-core / 10+ GB-free machine, 5 concurrent seeds
costs ~5 cores and ~1.2 GB, comfortably inside both budgets. Re-check
`nproc` / `free -h` before setting --workers above what headroom allows.

run_seed() passes build_ns3=False -- this is NOT optional with --workers > 1.
MroZmqEnv defaults to rebuilding ns-3 on every construction; with several
concurrent workers that means several `ns3 configure` + cmake invocations
hitting the SAME build directory and CMake cache at once. Caught live during
this script's first real parallel run: it didn't crash outright, but
disturbed the build badly enough that the next build needed to redo
1644/1644 steps instead of a no-op -- verified clean afterwards, but not a
risk worth re-taking. Build the binary once (e.g. `./ns3 build` directly)
before running this script with --workers > 1.

Concurrency is implemented with multiprocessing (spawn context), not
threads: each worker needs its own real OS-level ns-3 subprocess and its own
ZeroMQ port (port = --zmq_port + slot index, so concurrent workers never
contend for the same socket -- see MlbZmqEnv's own comment on what a shared
port silently does: two sides alive at ~0% CPU, no error, forever). Workers
are scheduled onto a fixed-size pool of ports/slots (min(--workers,
n_seeds)), so --workers can be set lower than the seed count to cap resource
use; a finished slot immediately picks up the next pending seed.

Every worker runs the EXACT same run_seed() code path as the --workers=1
(fully sequential, default) case -- this flag only changes how many OS
processes call it at once, never what it does. Results are written back to
ONE shared CSV from the main process only, incrementally as each seed
finishes (flushed + fsync'd), the same crash-survival discipline every other
MRO experiment script already follows -- whichever mode produced the row.

Stopping a concurrent run (Ctrl-C / SIGINT) forwards exactly ONE SIGINT to
every still-running worker, so each one's own `finally: env.close()` runs
(kills its ns-3 child, purges its sim_path) instead of leaving orphaned
processes and stuck ports behind. Sending a SECOND SIGINT while workers are
already mid-cleanup is deliberately NOT re-forwarded -- an earlier run of
this experiment, stopped via two overlapping SIGINTs from two separate
commands, had a worker's cleanup interrupted mid-`finally`, which is exactly
the race this guards against.

Usage:
    python examples/mro_outage_stress_experiment.py --n_seeds 5 --seed_start 3001
    python examples/mro_outage_stress_experiment.py --n_seeds 5 --workers 5
"""
import argparse
import csv
import multiprocessing as mp
import os
import queue
import signal
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import CELLS, MARGIN_LIMIT_DB, MroZmqEnv

import numpy as np

# Identical to mro_baseline_experiment.py's CFG -- see that module's
# docstring for why each field is set this way. Kept as an independent copy
# for the same file-independence reason the rest of this project's
# experiment scripts follow.
CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1],
    "controlInterval": [0.1],
    # Ra'ed's sim-baseline handover-stability switches (merged into
    # feature/mro-agent 2026-10-09), adopted for MRO with ONE deliberate
    # exception -- see train_mro_sac_pilot.py's CFG for the full per-flag
    # rationale. handoverHysteresisDb stays at 0 (NOT his v8 value of 3):
    # HysteresisAllowsHandover() gates MRO's primary handover-trigger
    # decision under handoverMode=DynamicTtt and only reduces to "no gate,
    # previous behaviour" when this is <= 0 -- turning it on would silently
    # stop MRO's own margin action from controlling that decision.
    "channelConditionUpdatePeriodMs": [0], "handoverSinrFilterTauMs": [400],
    "handoverHysteresisDb": [0], "tttFromBiasedSinr": [1],
    "controlPhaseOffsetS": [0.001],
    "ns3::MmWavePhyTrace::OutputFilename": ["/dev/null"],
    "ns3::MmWavePhyTrace::DlPhyTransmissionFilename": ["/dev/null"],
    "ns3::MmWaveMacTrace::SchedInfoOutputFilename": ["/dev/null"],
    "ns3::MmWaveBearerStatsConnector::MmWaveSinrOutputFilename": ["/dev/null"],
    "ns3::CoreNetworkStatsCalculator::X2FileName": ["/dev/null"],
    "ns3::PhyStatsCalculator::DlRsrpSinrFilename": ["/dev/null"],
    "ns3::PhyStatsCalculator::UlInterferenceFilename": ["/dev/null"],
    "ns3::LteRlcAm::BufferSizeFilename": ["/dev/null"],
}

TERMS = ("quality", "pingpong", "handover_rate", "outage")


def run_seed(seed: int, args, zmq_port: int) -> list:
    """One fixed-sticky-margin episode; returns a list of per-step term dicts.

    zmq_port is an explicit parameter, not read off `args`, so a parallel
    caller can hand each concurrently-running seed its own port -- see the
    module docstring's PARALLEL EXECUTION section. The sequential
    (--workers=1) path passes args.zmq_port here, so single-worker behaviour
    is byte-for-byte what it was before this parameter existed.
    """
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    cfg["simTime"] = [args.simtime]

    env = MroZmqEnv(ns3_path=args.ns3_path, scenario_configuration=cfg,
                    output_folder=args.output_folder, optimized=False,
                    zmq_port=zmq_port,
                    vary_rng_run_per_episode=False,
                    purge_sim_path_on_close=True,
                    # MroZmqEnv defaults to rebuilding ns-3 on every
                    # construction (build_ns3=True). Harmless run one at a
                    # time, but with --workers > 1 every concurrent worker
                    # independently triggered a full `ns3 configure` + cmake
                    # build against the SAME build directory and CMake
                    # cache at once -- caught live during this script's
                    # first real parallel run, and it disturbed the build
                    # badly enough that the next build needed to redo
                    # 1644/1644 steps instead of a no-op. False here because
                    # the binary this script needs is already built and
                    # verified before any seed runs; same pattern
                    # train_mlb_sac_pilot.py's own parallel workers use
                    # (build once in the parent, construct workers with
                    # build_ns3=False) -- see MlbZmqEnv.setup_sim()'s
                    # docstring for the identical rationale.
                    build_ns3=False)

    action = np.full(len(CELLS), MARGIN_LIMIT_DB, dtype=np.float32)  # +3 dB, sticky
    rows = []
    try:
        obs, info = env.reset(seed=seed)
        print(f"  seed {seed}: reset ok, rng_run={info.get('rng_run')}", flush=True)
        step_idx = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            if "reward_terms" not in info:
                break
            terms = info["reward_terms"]
            row = {"seed": seed, "step": step_idx, "reward": reward}
            for t in TERMS:
                row[t] = terms[t]
            rows.append(row)
            step_idx += 1
            if terminated or truncated:
                break
    except Exception as exc:
        print(f"  seed {seed}: CRASHED ({exc!r}), discarding this seed's rows", flush=True)
        return []
    finally:
        env.close()
    print(f"  seed {seed}: {len(rows)} steps collected", flush=True)
    return rows


def _worker_entry(seed: int, args, zmq_port: int, result_queue) -> None:
    """multiprocessing target: run one seed, put its result on the queue.

    Module-level (not a closure) because the spawn start method must be able
    to pickle and re-import the target function in the child process.
    """
    rows = run_seed(seed, args, zmq_port)
    result_queue.put((zmq_port, seed, rows))


def run_all_seeds(seeds: list, args):
    """Yields (seed, rows) as each seed finishes.

    args.workers <= 1: fully sequential, in-process, identical order and
    behaviour to every run of this script before --workers existed.

    args.workers > 1: up to min(args.workers, len(seeds)) seeds run
    concurrently as separate OS processes, each on its own ZeroMQ port
    (args.zmq_port + slot index). A finished slot immediately picks up the
    next pending seed, so this also supports --workers < len(seeds) as a
    resource cap, not just "all seeds at once".
    """
    if args.workers <= 1:
        for seed in seeds:
            yield seed, run_seed(seed, args, args.zmq_port)
        return

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    n_workers = min(args.workers, len(seeds))
    pending = list(seeds)
    running: dict = {}          # port -> (Process, seed)
    shutdown = {"requested": False}

    def launch(port: int) -> None:
        if not pending or shutdown["requested"]:
            return
        seed = pending.pop(0)
        p = ctx.Process(target=_worker_entry, args=(seed, args, port, result_queue))
        p.start()
        running[port] = (p, seed)
        print(f"  seed {seed}: launched on port {port} (pid {p.pid})", flush=True)

    def handle_sigint(signum, frame):
        # Forward exactly ONCE. A second SIGINT arriving while workers are
        # already unwinding their own finally:env.close() must NOT be
        # re-sent -- that race is what left 5 orphaned run directories
        # behind the first time this was tried with two overlapping,
        # independently-issued SIGINTs. See the module docstring.
        if shutdown["requested"]:
            return
        shutdown["requested"] = True
        print(f"\nInterrupted -- forwarding SIGINT once to {len(running)} running "
              f"worker(s); each will run its own cleanup before exiting.", flush=True)
        for p, seed in running.values():
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)

    old_handler = signal.signal(signal.SIGINT, handle_sigint)
    try:
        for port in (args.zmq_port + i for i in range(n_workers)):
            launch(port)

        while running or (pending and not shutdown["requested"]):
            if not running:
                break
            try:
                port, seed, rows = result_queue.get(timeout=1.0)
            except queue.Empty:
                # Reap any worker that exited WITHOUT reporting a result --
                # an unexpected crash bypassing run_seed's own
                # except Exception, or a worker that just received the
                # forwarded SIGINT above. Without this, one such process
                # would hang this loop forever waiting for a queue entry
                # that will never arrive.
                dead = [p_ for p_, (proc, _) in running.items() if not proc.is_alive()]
                for p_ in dead:
                    proc, dead_seed = running.pop(p_)
                    print(f"  seed {dead_seed}: worker exited without reporting "
                          f"(exit code {proc.exitcode}), treating as crashed/interrupted",
                          flush=True)
                    yield dead_seed, []
                    launch(p_)
                continue
            running.pop(port, None)
            yield seed, rows
            launch(port)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        for p, seed in list(running.values()):
            p.join(timeout=15)


def detrended_sd(rows: list, term: str) -> tuple:
    """Identical method to mro_baseline_experiment.py's own detrended_sd."""
    by_seed = {}
    for r in rows:
        if r[term] is None:
            continue
        by_seed.setdefault(r["seed"], []).append((r["step"], r[term]))

    raw_values, residuals = [], []
    for seed, pairs in by_seed.items():
        if len(pairs) < 2:
            continue
        steps = np.array([p[0] for p in pairs], dtype=float)
        values = np.array([p[1] for p in pairs], dtype=float)
        raw_values.extend(values.tolist())
        slope, intercept = np.polyfit(steps, values, 1)
        residuals.extend((values - (slope * steps + intercept)).tolist())

    raw_values = np.array(raw_values)
    residuals = np.array(residuals)
    return (float(raw_values.std()) if raw_values.size else float("nan"),
            float(residuals.std()) if residuals.size else float("nan"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=3001)
    ap.add_argument("--simtime", type=int, default=30)
    ap.add_argument("--zmq_port", type=int, default=5555)
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many seeds concurrently, each its own OS "
                         "process on its own ZeroMQ port (--zmq_port + slot "
                         "index). Default 1 = fully sequential (original "
                         "behaviour). See the module docstring's PARALLEL "
                         "EXECUTION section before raising this.")
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--csv_out", default=path.join(path.dirname(__file__), "..", "output",
                                                    "mro_outage_stress_terms.csv"))
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print(f"Running {len(seeds)} fixed-sticky-margin (+{MARGIN_LIMIT_DB} dB) seeds: {seeds} "
          f"({'sequential' if args.workers <= 1 else f'{min(args.workers, len(seeds))} concurrent workers'})",
          flush=True)
    all_rows = []
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "step", "reward"] + list(TERMS))
        writer.writeheader()
        f.flush()
        for seed, rows in run_all_seeds(seeds, args):
            all_rows.extend(rows)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
            print(f"  seed {seed}: {len(rows)} rows written to {args.csv_out}", flush=True)

    if not all_rows:
        sys.exit("No usable rows collected -- every seed crashed or produced no steps.")
    print(f"\nWrote {len(all_rows)} total rows to {args.csv_out}", flush=True)

    print("\n=== per-term statistics (pooled across seeds, sticky +3 dB arm) ===")
    print(f"{'term':<16} {'n':>6} {'mean':>10} {'raw_sd':>10} {'detrended_sd':>13}")
    stats = {}
    for term in TERMS:
        values = [r[term] for r in all_rows if r[term] is not None]
        raw_sd, det_sd = detrended_sd(all_rows, term)
        mean = float(np.mean(values)) if values else float("nan")
        stats[term] = {"mean": mean, "raw_sd": raw_sd, "detrended_sd": det_sd, "n": len(values)}
        print(f"{term:<16} {len(values):>6} {mean:>10.4f} {raw_sd:>10.4f} {det_sd:>13.4f}")

    n_fired = sum(1 for r in all_rows if r["outage"] and r["outage"] > 0.0)
    print(f"\noutage fired on {n_fired}/{len(all_rows)} steps "
          f"({100*n_fired/len(all_rows):.3f}%) under the sticky arm")
    print("(round-2 do-nothing baseline: fired on 0.01% of UE-steps -- compare the rate, "
          "not just the derived weight, to judge whether this arm genuinely stressed the term)")

    print("\n=== derived outage weight (anchored to round-2's HandoverSignalQuality sd) ===")
    ANCHOR_SD = 0.273  # round-2 baseline, locked -- see docs/mro_training_fixes.md
    det_sd = stats["outage"]["detrended_sd"]
    w = (ANCHOR_SD / det_sd) if det_sd and det_sd > 0 else float("nan")
    print(f"  w_outage = anchor_sd({ANCHOR_SD}) / detrended_sd({det_sd:.4f}) = {w:.4f}")
    print("  (compare against the shipped fixed-guard value, DEFAULT_W_OUTAGE=1.0, and against "
          "round-1's unusable literal value, 159.7, before deciding whether to replace either)")
