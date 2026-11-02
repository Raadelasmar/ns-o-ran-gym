"""Diagnostic (2026-10-27): what do too_late_rate/churn_rate actually look
like in real episodes, and how often do they cross TOO_LATE_RATE_THRESHOLD /
CHURN_RATE_THRESHOLD = 0.3?

Motivated by a direct finding: evaluating a freshly-trained MRO SAC policy
(deterministic actions, clearly non-trivial -- e.g. [-1.08, -2.99], [1.89,
2.82] on real in-episode observations) produced a per-step reward trajectory
BYTE-IDENTICAL to the pure rule baseline (v3_rule, action=[0,0]) across all
15/15 seeds, all 99/99 steps each. Traced to source
(_pairwise_margin_targets, mro_zmq_env.py) and confirmed:
_pairwise_payload_from_action sends an ho_margin_pairs entry for a directed
pair ONLY if that pair's rolling too-late/churn rate exceeds its 0.3
threshold -- so the agent's gain (BASE + its action delta) is multiplied by
`rate - threshold`, which is <=0 for every pair whenever no role has EVER
fired. If that is true for essentially every pair in essentially every
episode under this scenario, BASE_TOO_LATE_GAIN_DB/BASE_CHURN_GAIN_DB (and
therefore the entire agent action) are a structural no-op -- action has
literally never reached ns-3 as a nonzero hysteresis offset.

This script does NOT touch the threshold. It just runs real episodes
(do_nothing action -- confirmed elsewhere to produce statistically
indistinguishable handover/mobility dynamics from v3_rule, since neither's
gain is ever actually applied) and logs the RAW too_late_rate/churn_rate
dict for every (pair, step), reading env._pair_rates directly after each
step (the same attribute _pairwise_payload_from_action reads from). From
the empirical distribution, a threshold can be picked that the roles can
actually reach, instead of 0.3 which this session's own prior diagnostic
(mro_v3_throughput_diagnostic.py) already found 0 active pairs on, for 80/80
steps, under both conditions it tested.

Usage:
    python examples/mro_v3_role_rate_diagnostic.py --seeds 9101,9102,9103,9104,9105
"""
import argparse
import csv
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import MroZmqEnv, ALL_CELL_PAIRS_DIRECTED, ALL_CELL_PAIRS_UNDIRECTED  # noqa: E402

import numpy as np  # noqa: E402

CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1], "controlInterval": [0.1],
    "channelConditionUpdatePeriodMs": [0], "handoverSinrFilterTauMs": [400],
    "handoverHysteresisDb": [0], "tttFromBiasedSinr": [1], "controlPhaseOffsetS": [0.001],
    "ns3::MmWavePhyTrace::OutputFilename": ["/dev/null"],
    "ns3::MmWavePhyTrace::DlPhyTransmissionFilename": ["/dev/null"],
    "ns3::MmWaveMacTrace::SchedInfoOutputFilename": ["/dev/null"],
    "ns3::MmWaveBearerStatsConnector::MmWaveSinrOutputFilename": ["/dev/null"],
    "ns3::CoreNetworkStatsCalculator::X2FileName": ["/dev/null"],
    "ns3::PhyStatsCalculator::DlRsrpSinrFilename": ["/dev/null"],
    "ns3::PhyStatsCalculator::UlInterferenceFilename": ["/dev/null"],
    "ns3::LteRlcAm::BufferSizeFilename": ["/dev/null"],
}

DO_NOTHING = np.array([-1.0, -1.0], dtype=np.float32)


def run_seed(seed: int, args, zmq_port: int) -> list:
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    cfg["simTime"] = [args.simtime]

    env = MroZmqEnv(ns3_path=args.ns3_path, scenario_configuration=cfg,
                    output_folder=args.output_folder, optimized=False,
                    zmq_port=zmq_port, build_ns3=False,
                    vary_rng_run_per_episode=False, purge_sim_path_on_close=True)

    rows = []
    try:
        obs, info = env.reset(seed=seed)
        print(f"  seed {seed}: reset ok, rng_run={info.get('rng_run')}", flush=True)
        step_idx = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(DO_NOTHING)
            if "reward_terms" not in info:
                break
            too_late_rate, churn_rate = env._pair_rates
            for pair in ALL_CELL_PAIRS_DIRECTED:
                r = too_late_rate.get(pair, 0.0)
                if r > 0.0:
                    rows.append({"seed": seed, "step": step_idx, "kind": "too_late",
                                "pair": f"{pair[0]}->{pair[1]}", "rate": r})
            for pair in ALL_CELL_PAIRS_UNDIRECTED:
                r = churn_rate.get(pair, 0.0)
                if r > 0.0:
                    rows.append({"seed": seed, "step": step_idx, "kind": "churn",
                                "pair": f"{pair[0]}-{pair[1]}", "rate": r})
            step_idx += 1
            if step_idx % 20 == 0:
                print(f"  seed {seed}: step {step_idx}, {len(rows)} nonzero-rate samples so far",
                      flush=True)
            if terminated or truncated:
                break
    except Exception as exc:
        print(f"  seed {seed}: CRASHED ({exc!r})", flush=True)
        return []
    finally:
        env.close()
    print(f"  seed {seed}: {step_idx} steps, {len(rows)} nonzero-rate samples", flush=True)
    return rows


def _worker_entry(seed, args, zmq_port, result_queue) -> None:
    rows = run_seed(seed, args, zmq_port)
    result_queue.put((zmq_port, seed, rows))


def run_condition(seeds: list, args, workers: int, per_seed_timeout_s: float,
                  deadline_unix_s: float):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    n_workers = min(workers, len(seeds))
    pending = list(seeds)
    running: dict = {}

    def launch(port: int) -> None:
        if not pending or time.time() >= deadline_unix_s:
            return
        seed = pending.pop(0)
        p = ctx.Process(target=_worker_entry, args=(seed, args, port, result_queue))
        p.start()
        running[port] = (p, seed, time.time())
        print(f"  seed {seed}: launched on port {port} (pid {p.pid})", flush=True)

    for port in (args.zmq_port + i for i in range(n_workers)):
        launch(port)

    while running:
        try:
            port, seed, rows = result_queue.get(timeout=1.0)
        except queue.Empty:
            now = time.time()
            timed_out = [p_ for p_, (proc, sd, started) in running.items()
                        if now - started > per_seed_timeout_s]
            for p_ in timed_out:
                proc, dead_seed, _ = running.pop(p_)
                print(f"  seed {dead_seed}: TIMED OUT, interrupting (pid {proc.pid})", flush=True)
                os.kill(proc.pid, signal.SIGINT)
                proc.join(timeout=15)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
                yield dead_seed, []
                launch(p_)
            dead = [p_ for p_, (proc, sd, started) in running.items()
                   if p_ not in timed_out and not proc.is_alive()]
            for p_ in dead:
                proc, dead_seed, _ = running.pop(p_)
                yield dead_seed, []
                launch(p_)
            continue
        running.pop(port, None)
        yield seed, rows
        launch(port)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(9101, 9106)))
    ap.add_argument("--simtime", type=int, default=30)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--per_seed_timeout_s", type=float, default=3300)
    ap.add_argument("--total_budget_s", type=float, default=10800)
    ap.add_argument("--zmq_port", type=int, default=7000)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--results_dir", default=path.join(path.dirname(__file__), "..", "output",
                                                        "mro_v3_role_rate_diagnostic"))
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    deadline = t0 + args.total_budget_s

    print("=" * 78, flush=True)
    print(f"role-rate diagnostic: simtime={args.simtime}, {len(seeds)} seeds "
          f"({seeds[0]}..{seeds[-1]}), {args.workers} workers", flush=True)
    print("=" * 78, flush=True)

    csv_path = path.join(args.results_dir, "nonzero_rates.csv")
    all_rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "step", "kind", "pair", "rate"])
        writer.writeheader()
        for seed, rows in run_condition(seeds, args, args.workers, args.per_seed_timeout_s,
                                        deadline):
            all_rows.extend(rows)
            for r in rows:
                writer.writerow(r)
            f.flush()
            print(f"  >>> running total: {len(all_rows)} nonzero-rate samples from "
                  f"{len(set(r['seed'] for r in all_rows))} seeds so far", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"DONE. {(time.time()-t0)/60:.1f} min elapsed. {len(all_rows)} total nonzero-rate "
          f"samples written to {csv_path}", flush=True)
    for kind in ("too_late", "churn"):
        vals = [r["rate"] for r in all_rows if r["kind"] == kind]
        print(f"\n{kind}: {len(vals)} nonzero samples across {len(seeds)} seeds", flush=True)
        if vals:
            import collections
            hist = collections.Counter(round(v, 2) for v in vals)
            for rate in sorted(hist):
                print(f"    rate={rate:.2f}: {hist[rate]} samples "
                      f"{'>= 0.3 threshold' if rate >= 0.3 else ''}", flush=True)
            print(f"    max rate ever observed: {max(vals):.2f}", flush=True)
        else:
            print("    NEVER nonzero -- this role never fired even once in any episode", flush=True)
    print("=" * 78, flush=True)
