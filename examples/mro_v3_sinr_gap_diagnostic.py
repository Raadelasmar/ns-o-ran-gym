"""Diagnostic (2026-10-27): follow-up to mro_v3_role_rate_diagnostic.py.

That script found too_late_rate is NEVER nonzero across 990 real steps (10
seeds, simtime=10) -- i.e. the qualifying condition itself (a UE's candidate
cell beats its serving cell's SINR by >= TOO_LATE_SINR_MARGIN_DB=3.0 dB)
never fires even once, not just "fires too rarely to cross the 0.3 rate
threshold". Before touching that threshold (which would be moot if the
underlying boolean never fires), this measures two prerequisites directly,
per step, raw and ungated by any threshold:

  - multi_cell_ue_steps: how many (seed, step, imsi) have >1 cell in
    env's internal sinr matrix at all (a UE with only 1 entry can never
    qualify, since the condition needs a servingvs-candidate COMPARISON)
  - max_gap_db: for every UE with >=2 cells measured, the largest
    (candidate_sinr - serving_sinr) value that step, whatever its sign

This tells us whether 3.0 dB is simply too strict (gaps happen but top out
around 1-2 dB) or whether UEs are essentially never measured under >1 cell
at the same time in this scenario (a structural ceiling no threshold change
can fix).

Usage:
    python examples/mro_v3_sinr_gap_diagnostic.py --seeds 9101,9102,...
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
import environments.mro_zmq_env as mro_zmq_env  # noqa: E402
from environments.mro_zmq_env import MroZmqEnv  # noqa: E402

import numpy as np  # noqa: E402

# PairRoleTracker.update computes the per-UE sinr matrix internally (not
# stored on the env) via the bare module-level name _extract_ue_cell_sinr_matrix
# -- Python resolves that name from the module's own globals at call time, so
# replacing the module attribute here is visible to PairRoleTracker.update
# without touching its source. Captures the latest matrix for this script to
# read right after each env.step() call.
_LAST_SINR_MATRIX = {}
_ORIG_EXTRACT = mro_zmq_env._extract_ue_cell_sinr_matrix


def _capturing_extract(kpis):
    global _LAST_SINR_MATRIX
    matrix = _ORIG_EXTRACT(kpis)
    _LAST_SINR_MATRIX = matrix
    return matrix


mro_zmq_env._extract_ue_cell_sinr_matrix = _capturing_extract

CFG = {
    "heuristicType": [-1], "simTime": [10], "ues": [5], "RngRun": [555],
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
            # _capturing_extract's module-level monkeypatch fills this in as
            # a side effect of env._pair_tracker.update's own call to it
            # inside _reward_terms (see step() -> _reward_terms ordering).
            sinr_matrix = _LAST_SINR_MATRIX
            serving_cell = env._pair_tracker._serving_cell
            for imsi, cell_sinrs in sinr_matrix.items():
                n_cells = len(cell_sinrs)
                serving = serving_cell.get(imsi)
                if serving is None or serving not in cell_sinrs:
                    continue
                serving_sinr = cell_sinrs[serving]
                max_gap = max((sinr - serving_sinr for cell, sinr in cell_sinrs.items()
                              if cell != serving), default=None)
                rows.append({"seed": seed, "step": step_idx, "imsi": imsi,
                            "n_cells_measured": n_cells, "max_gap_db": max_gap})
            step_idx += 1
            if step_idx % 20 == 0:
                print(f"  seed {seed}: step {step_idx}, {len(rows)} UE-step samples so far",
                      flush=True)
            if terminated or truncated:
                break
    except Exception as exc:
        print(f"  seed {seed}: CRASHED ({exc!r})", flush=True)
        return []
    finally:
        env.close()
    print(f"  seed {seed}: {step_idx} steps, {len(rows)} UE-step samples", flush=True)
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
    ap.add_argument("--simtime", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--per_seed_timeout_s", type=float, default=3300)
    ap.add_argument("--total_budget_s", type=float, default=10800)
    ap.add_argument("--zmq_port", type=int, default=7200)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--results_dir", default=path.join(path.dirname(__file__), "..", "output",
                                                        "mro_v3_sinr_gap_diagnostic"))
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    deadline = t0 + args.total_budget_s

    print("=" * 78, flush=True)
    print(f"sinr-gap diagnostic: simtime={args.simtime}, {len(seeds)} seeds "
          f"({seeds[0]}..{seeds[-1]}), {args.workers} workers", flush=True)
    print("=" * 78, flush=True)

    csv_path = path.join(args.results_dir, "sinr_gap_samples.csv")
    all_rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "step", "imsi", "n_cells_measured",
                                                "max_gap_db"])
        writer.writeheader()
        for seed, rows in run_condition(seeds, args, args.workers, args.per_seed_timeout_s,
                                        deadline):
            all_rows.extend(rows)
            for r in rows:
                writer.writerow(r)
            f.flush()
            print(f"  >>> running total: {len(all_rows)} UE-step samples from "
                  f"{len(set(r['seed'] for r in all_rows))} seeds so far", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"DONE. {(time.time()-t0)/60:.1f} min elapsed. {len(all_rows)} total UE-step "
          f"samples written to {csv_path}", flush=True)

    multi_cell = [r for r in all_rows if r["n_cells_measured"] > 1]
    print(f"\nUE-steps with >1 cell measured (comparison possible): {len(multi_cell)} / "
          f"{len(all_rows)} ({100*len(multi_cell)/max(1,len(all_rows)):.1f}%)", flush=True)
    gaps = [r["max_gap_db"] for r in multi_cell if r["max_gap_db"] is not None]
    if gaps:
        import statistics
        print(f"max_gap_db distribution over those {len(gaps)} samples:", flush=True)
        print(f"    min={min(gaps):.2f}  mean={statistics.mean(gaps):.2f}  "
              f"median={statistics.median(gaps):.2f}  max={max(gaps):.2f}", flush=True)
        for thresh in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            n = sum(1 for g in gaps if g >= thresh)
            print(f"    gap >= {thresh:.1f} dB: {n} / {len(gaps)} samples "
                  f"({100*n/len(gaps):.1f}%)", flush=True)
    else:
        print("    NO multi-cell UE-step samples at all -- UEs are essentially never "
              "measured under more than one cell simultaneously in this scenario.", flush=True)
    print("=" * 78, flush=True)
