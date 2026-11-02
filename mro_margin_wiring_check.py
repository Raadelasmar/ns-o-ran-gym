"""Fast diagnostic: does MRO's margin now produce a clean, correctly-
directional effect on real ns-3 handover behaviour, after the (temporary,
reversible) wiring patch in lte-enb-rrc.cc's TttBasedHandover?

Short simTime (default 10s, ~100 control steps) and few seeds -- this is a
cheap, iterative diagnostic, not a publication-grade measurement. Reports
the FULL reward-term breakdown (quality/pingpong/handover_rate/outage), not
just raw handover counts, so the ping-pong/churn mechanism can actually be
seen directly.

Usage:
    python mro_margin_wiring_check.py --margin 0 --seed 2001
    python mro_margin_wiring_check.py --margin -3 --seed 2001 --simtime 10
"""
import argparse
import statistics as st
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "src"))
from environments.mro_zmq_env import CELLS, MroZmqEnv

import numpy as np

CFG = {
    "heuristicType": [-1], "simTime": [10], "ues": [5], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1],
    "controlInterval": [0.1],
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


def run(margin_db, seed, port, ns3_path, simtime, output_folder):
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    cfg["simTime"] = [simtime]
    env = MroZmqEnv(ns3_path=ns3_path, scenario_configuration=cfg,
                    output_folder=output_folder, optimized=False,
                    zmq_port=port, build_ns3=False,
                    vary_rng_run_per_episode=False, purge_sim_path_on_close=True)
    action = np.full(len(CELLS), margin_db, dtype=np.float32)
    rows = []
    try:
        obs, info = env.reset(seed=seed)
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            if "reward_terms" not in info:
                break
            rows.append(info["reward_terms"])
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows


def summarize(rows, label):
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else float("nan"), len(vals)
    q, qn = mean("quality")
    pp, ppn = mean("pingpong")
    hr, hrn = mean("handover_rate")
    og, ogn = mean("outage")
    rw, rwn = mean("reward")
    n_handovers = sum(r["diagnostics"]["handovers_this_step"] for r in rows)
    print(f"{label:>18s}: n_steps={len(rows):3d}  handovers={n_handovers:3d}  "
          f"quality={q:+.4f}({qn})  pingpong={pp:+.4f}  handover_rate={hr:+.4f}  "
          f"outage={og:+.4f}  reward={rw:+.4f}", flush=True)
    return dict(quality=q, pingpong=pp, handover_rate=hr, outage=og, reward=rw,
                n_handovers=n_handovers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--port", type=int, default=5570)
    ap.add_argument("--simtime", type=int, default=10)
    ap.add_argument("--ns3_path", default=path.expanduser("~/son_marl_project/ns-o-ran-ns3-mmwave"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "output"))
    ap.add_argument("--csv_out", default=None,
                    help="if set, write per-step rows (seed,step,reward,quality,pingpong,"
                         "handover_rate,outage) matching the established experiment-script schema")
    args = ap.parse_args()

    rows = run(args.margin, args.seed, args.port, args.ns3_path, args.simtime, args.output_folder)
    summarize(rows, f"margin={args.margin:+.1f}dB")

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "step", "reward", "quality", "pingpong",
                                              "handover_rate", "outage"])
            w.writeheader()
            for i, r in enumerate(rows):
                w.writerow({"seed": args.seed, "step": i, "reward": r["reward"],
                           "quality": r["quality"], "pingpong": r["pingpong"],
                           "handover_rate": r["handover_rate"], "outage": r["outage"]})
        print(f"wrote {len(rows)} rows to {args.csv_out}", flush=True)
