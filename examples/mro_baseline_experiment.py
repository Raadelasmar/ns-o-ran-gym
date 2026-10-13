"""Measure MRO's reward-term statistics from seeded do-nothing baselines.

Same purpose and method as MLB's own weight-tuning (docs/mlb_training_fixes.md
/ mlb_zmq_env.py's DEFAULT_W_* comment block): run several SEEDED, all-zero-
action ("do nothing") episodes, log every reward term per step, and derive
weights from each term's DETRENDED standard deviation -- not the raw one, and
not hand-picked. Detrending matters for the same reason it did for MLB's
Backlog: a term that drifts with the step index (queues/handover activity
building up over an episode) gets its raw spread inflated by that drift alone,
which would punish it for merely running longer rather than for anything an
action could plausibly cause.

ActionSmoothness is deliberately NOT measured here. Under an all-zero action
every step, its delta is always exactly 0 -- a do-nothing baseline cannot see
its natural scale at all, since that term is purely a property of the ACTION
sequence, not of network physics. See the sibling Monte Carlo calculation
(no ns-3 needed) that estimates it from an untrained-PPO-like action
distribution instead.

Usage:
    python examples/mro_baseline_experiment.py --seeds 1001,1002,1003,1004,1005
    python examples/mro_baseline_experiment.py --n_seeds 5 --seed_start 2001
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
from environments.mro_zmq_env import CELLS, MroZmqEnv

import numpy as np

# ROUND 2. Same locked config as the colleague's train_mlb_sac_pilot.py's
# ARM_C run, NOT the smoke-test config round 1 used -- see
# docs/mro_training_fixes.md "Two problems found before investing further
# tuning effort". ues=5 (35 UEs, not 21) and simTime=30 (not 10) because
# ues=3 was too light a network for OutageExposure to ever fire, mirroring
# MLB's own documented lesson. controlInterval=0.1 (not the 1.0 default)
# because that is the colleague's locked SAC config, and the whole point of
# fixing PingPong/HandoverRate's rate-normalization (see mro_zmq_env.py's
# HANDOVER_RATE_REF_S) was to make this config safe to measure in. The
# trace-file redirects are ns-3 default-attribute overrides on existing
# trace-source classes -- no rebuild needed, same as the colleague's own use.
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
    """One do-nothing episode; returns a list of per-step term dicts.

    zmq_port is explicit, not read off args, so a parallel caller can hand
    each concurrently-running seed its own port -- see run_all_seeds() and
    mro_outage_stress_experiment.py's PARALLEL EXECUTION docs for the full
    rationale (including why build_ns3=False is not optional with
    --workers > 1: several concurrent `ns3 configure` invocations against
    the same build directory corrupted a build earlier this session).
    """
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    cfg["simTime"] = [args.simtime]

    env = MroZmqEnv(ns3_path=args.ns3_path, scenario_configuration=cfg,
                    output_folder=args.output_folder, optimized=False,
                    zmq_port=zmq_port,
                    build_ns3=False,
                    vary_rng_run_per_episode=False,   # this seed, not a redraw
                    purge_sim_path_on_close=True)      # pure measurement, keep no run dirs

    action = np.zeros(len(CELLS), dtype=np.float32)
    rows = []
    try:
        obs, info = env.reset(seed=seed)
        print(f"  seed {seed}: reset ok, rng_run={info.get('rng_run')}", flush=True)
        step_idx = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            if "reward_terms" not in info:
                break  # time-limit end; see the module docstring on this branch
            terms = info["reward_terms"]
            row = {"seed": seed, "step": step_idx, "reward": reward}
            for t in TERMS:
                row[t] = terms[t]  # None where the term was unavailable/dropped
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
    """multiprocessing target: run one seed, put its result on the queue."""
    rows = run_seed(seed, args, zmq_port)
    result_queue.put((zmq_port, seed, rows))


def run_all_seeds(seeds: list, args):
    """Yields (seed, rows) as each seed finishes. Identical mechanism to
    mro_outage_stress_experiment.py's run_all_seeds -- see that module's
    PARALLEL EXECUTION docstring for the full rationale, including the
    single-dispatch SIGINT forwarding and why a worker that exits without
    reporting is reaped rather than left to hang the loop.
    """
    if args.workers <= 1:
        for seed in seeds:
            yield seed, run_seed(seed, args, args.zmq_port)
        return

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    n_workers = min(args.workers, len(seeds))
    pending = list(seeds)
    running: dict = {}
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
    """(raw_sd, detrended_sd) for one term, pooled across all seeds.

    Detrending is done PER SEED (fit step -> value, subtract the fit, pool the
    residuals across seeds) so a trend that happens to run in opposite
    directions on different seeds cannot cancel out and hide a real per-episode
    drift -- the same per-world treatment MLB's own Backlog analysis used.
    """
    by_seed = {}
    for r in rows:
        if r[term] is None:
            continue
        by_seed.setdefault(r["seed"], []).append((r["step"], r[term]))

    raw_values = []
    residuals = []
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
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated explicit seed list, overrides --n_seeds")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=2001)
    ap.add_argument("--simtime", type=int, default=30)
    ap.add_argument("--zmq_port", type=int, default=5555)
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many seeds concurrently, each its own OS "
                         "process on its own ZeroMQ port (--zmq_port + slot "
                         "index). Default 1 = sequential. See "
                         "mro_outage_stress_experiment.py's module docstring "
                         "before raising this.")
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--csv_out", default=path.join(path.dirname(__file__), "..", "output",
                                                    "mro_baseline_terms.csv"))
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print(f"Running {len(seeds)} do-nothing baseline seeds: {seeds} "
          f"({'sequential' if args.workers <= 1 else f'{min(args.workers, len(seeds))} concurrent workers'})",
          flush=True)
    # Written incrementally, one seed at a time, flushed and fsync'd after
    # each -- a run that gets interrupted (Ctrl-C, machine shutdown) keeps
    # every seed that had already finished. This USED to hold everything in
    # memory and write once at the very end: a real interruption (this
    # happened) lost every completed seed's ~hour of real ns-3 time because
    # nothing had touched disk yet. Real ns-3 time is too expensive to gamble
    # on the process surviving to the last line.
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

    print("\n=== per-term statistics (pooled across seeds) ===")
    print(f"{'term':<16} {'n':>6} {'mean':>10} {'raw_sd':>10} {'detrended_sd':>13}")
    stats = {}
    for term in TERMS:
        values = [r[term] for r in all_rows if r[term] is not None]
        raw_sd, det_sd = detrended_sd(all_rows, term)
        mean = float(np.mean(values)) if values else float("nan")
        stats[term] = {"mean": mean, "raw_sd": raw_sd, "detrended_sd": det_sd, "n": len(values)}
        print(f"{term:<16} {len(values):>6} {mean:>10.4f} {raw_sd:>10.4f} {det_sd:>13.4f}")

    print("\n=== derived weights (equalise detrended sd, quality anchored at 1.0) ===")
    anchor = stats["quality"]["detrended_sd"]
    for term in TERMS:
        sd = stats[term]["detrended_sd"]
        w = (anchor / sd) if sd and sd > 0 else float("nan")
        print(f"  w_{term:<14} = {w:.4f}")
    print("\n(ActionSmoothness weight is NOT derived here -- see the Monte Carlo "
          "estimate run separately, since this term has no signal under an "
          "all-zero action.)")
