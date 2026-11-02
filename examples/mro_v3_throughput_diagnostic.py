"""Diagnostic (2026-10-26): why is v3_rule (action=[0,0], the real rule
baseline) so much slower to simulate than do_nothing (action=[-1,-1], a
verified true no-op), under otherwise identical conditions?

Prints per-step wall-clock time AND what the step actually did (handovers
this step, how many directed pairs got a nonzero margin, the role gains
applied) for both conditions, SAME seed, SAME process (sequential, not
parallel, to remove contention as a variable). Short run (a few dozen
steps) -- the point is to see WHEN and WHY the two conditions' per-step
cost diverges, not to produce a publishable number.

Usage:
    python examples/mro_v3_throughput_diagnostic.py --steps 40
"""
import argparse
import sys
import time
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import MroZmqEnv

import numpy as np

CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [7001],
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


def run(controller: str, action: np.ndarray, seed: int, steps: int, ns3_path: str,
        output_folder: str, zmq_port: int) -> list:
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    env = MroZmqEnv(ns3_path=ns3_path, scenario_configuration=cfg,
                    output_folder=output_folder, optimized=False,
                    zmq_port=zmq_port, build_ns3=False,
                    vary_rng_run_per_episode=False, purge_sim_path_on_close=True)
    rows = []
    t0 = time.time()
    obs, info = env.reset(seed=seed)
    print(f"[{controller}] reset: {time.time()-t0:.1f}s", flush=True)
    for i in range(steps):
        ts = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        dt = time.time() - ts
        terms = (info or {}).get("reward_terms") or {}
        diag = terms.get("diagnostics") or {}
        pair_margins = (info or {}).get("pair_margins_db") or {}
        n_active_pairs = sum(1 for v in pair_margins.values() if abs(v) > 1e-9)
        row = {
            "controller": controller, "step": i, "dt_s": dt,
            "handovers_this_step": diag.get("handovers_this_step"),
            "n_active_pairs": n_active_pairs,
            "role_gain_delta": diag.get("role_gain_delta_db"),
            "reward": reward,
        }
        rows.append(row)
        print(f"[{controller}] step {i:3d}: dt={dt:6.2f}s  handovers={row['handovers_this_step']}"
              f"  active_pairs={n_active_pairs}  reward={reward:.3f}"
              + (f"  sample_pairs={dict(list(pair_margins.items())[:3])}" if pair_margins else ""),
              flush=True)
        if terminated or truncated:
            break
    env.close()
    print(f"[{controller}] TOTAL: {time.time()-t0:.1f}s for {len(rows)} steps", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--zmq_port", type=int, default=6100)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    args = ap.parse_args()

    print("=" * 78, flush=True)
    print("v3_rule vs do_nothing: per-step throughput diagnostic", flush=True)
    print("=" * 78, flush=True)

    do_nothing_rows = run("do_nothing", np.array([-1.0, -1.0], dtype=np.float32),
                          args.seed, args.steps, args.ns3_path, args.output_folder, args.zmq_port)
    print()
    v3_rule_rows = run("v3_rule", np.array([0.0, 0.0], dtype=np.float32),
                       args.seed, args.steps, args.ns3_path, args.output_folder, args.zmq_port + 1)

    print("\n" + "=" * 78, flush=True)
    print("SUMMARY", flush=True)
    dn_total = sum(r["dt_s"] for r in do_nothing_rows)
    v3_total = sum(r["dt_s"] for r in v3_rule_rows)
    dn_active_steps = sum(1 for r in do_nothing_rows if r["n_active_pairs"] > 0)
    v3_active_steps = sum(1 for r in v3_rule_rows if r["n_active_pairs"] > 0)
    print(f"do_nothing: {len(do_nothing_rows)} steps, {dn_total:.1f}s total, "
          f"{dn_total/max(len(do_nothing_rows),1):.2f}s/step avg, "
          f"{dn_active_steps} steps with >=1 active pair", flush=True)
    print(f"v3_rule:    {len(v3_rule_rows)} steps, {v3_total:.1f}s total, "
          f"{v3_total/max(len(v3_rule_rows),1):.2f}s/step avg, "
          f"{v3_active_steps} steps with >=1 active pair", flush=True)
    print(f"slowdown factor: {v3_total/max(dn_total,0.01):.2f}x", flush=True)
