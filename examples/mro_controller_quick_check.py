"""Fast single-seed diagnostic: rule-based controller vs do-nothing, same
short simTime, same seed, weighted reward under current weights. For
rapid iteration on controller tuning -- not a substitute for a full
multi-seed validation.
"""
import argparse
import json
from os import path
import sys

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import CELLS, MroZmqEnv
from mro_rule_based_controller import RuleBasedMroController, ProportionalMroController

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

W_QUALITY, W_PINGPONG, W_CHURN, W_OUTAGE = 1.00, 9.39, 1.21, 1.0

# ALT: literature-consistent re-weighting, same terms, no retraining/resim
# needed -- the rule-based controllers trigger on raw obs (badsignal_frac,
# handover_activity), not on the reward signal, so one trajectory scores
# under both weightings. Current PINGPONG=9.39 came from a detrended-sd
# VARIANCE-equalisation procedure (mro_zmq_env.py:145-169), anchored so each
# term's CONTRIBUTION TO REWARD VARIANCE matches quality's -- a reward-
# engineering move for SAC gradient-scale stability, not a priority
# judgement. Classical/3GPP SON MRO literature treats RLF/outage reduction
# as PRIMARY and ping-pong as SECONDARY (not weighted higher), and RL papers
# that do use one combined scalar keep terms within the same order of
# magnitude (e.g. a DDPG paper weights handover-count/ping-pong/failure
# penalties all at 3; a DQN line's "effective handover rate" sums failure
# and ping-pong rates 1:1). ALT_W_PINGPONG=1.21 brings ping-pong down to
# CHURN's weight -- inside the 0.80-1.51 "natural range" the file's own
# comment already uses to justify OUTAGE's weight -- isolating this one
# term rather than changing all four at once.
ALT_W_PINGPONG = 1.21


def weighted(terms, w_pingpong=W_PINGPONG):
    r = -W_OUTAGE * terms["outage"]
    if terms.get("quality") is not None:
        r += W_QUALITY * terms["quality"]
    if terms.get("pingpong") is not None:
        r -= w_pingpong * terms["pingpong"]
    if terms.get("handover_rate") is not None:
        r -= W_CHURN * terms["handover_rate"]
    return r


def run(mode, seed, port, ns3_path, simtime, output_folder, **kwargs):
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    cfg["simTime"] = [simtime]
    env = MroZmqEnv(ns3_path=ns3_path, scenario_configuration=cfg,
                    output_folder=output_folder, optimized=False,
                    zmq_port=port, build_ns3=False,
                    vary_rng_run_per_episode=False, purge_sim_path_on_close=True)
    controller_type = kwargs.get("controller_type", "bangbang")
    if mode == "controller" and controller_type == "proportional":
        ctrl = ProportionalMroController(
            window=kwargs.get("window", 5), cadence=kwargs.get("cadence", 5),
            kp_loosen=kwargs.get("kp_loosen", 10.0), kp_tighten=kwargs.get("kp_tighten", 2.0),
            badsignal_threshold=kwargs.get("badsignal_threshold", 0.05),
            activity_threshold=kwargs.get("activity_threshold", 0.25),
            max_step=kwargs.get("max_step", 1.0),
            dead_band=kwargs.get("dead_band", 0.3), max_repeat=kwargs.get("max_repeat", 1),
        )
    elif mode == "controller":
        ctrl = RuleBasedMroController(
            window=kwargs.get("window", 5), cadence=kwargs.get("cadence", 5),
            badsignal_threshold=kwargs.get("badsignal_threshold", 0.05),
            activity_threshold=kwargs.get("activity_threshold", 0.25),
            dead_band=kwargs.get("dead_band", 0.3),
            step_db=kwargs.get("step_db", 0.5),
            max_repeat=kwargs.get("max_repeat", 1),
            multi_cell=kwargs.get("multi_cell", False),
            top_k=kwargs.get("top_k", 0),
        )
    else:
        ctrl = None
    rows = []
    try:
        obs, info = env.reset(seed=seed)
        while True:
            if mode == "controller":
                action = ctrl.act(obs)
            else:
                action = np.zeros(len(CELLS), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if "reward_terms" not in info:
                break
            rows.append(info["reward_terms"])
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["controller", "donothing"], required=True)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--port", type=int, default=5570)
    ap.add_argument("--simtime", type=int, default=10)
    ap.add_argument("--ns3_path", default=path.expanduser("~/son_marl_project/ns-o-ran-ns3-mmwave"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--cadence", type=int, default=5)
    ap.add_argument("--badsignal_threshold", type=float, default=0.05)
    ap.add_argument("--activity_threshold", type=float, default=0.25)
    ap.add_argument("--dead_band", type=float, default=0.3)
    ap.add_argument("--step_db", type=float, default=0.5)
    ap.add_argument("--max_repeat", type=int, default=1)
    ap.add_argument("--multi_cell", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--controller_type", choices=["bangbang", "proportional"], default="bangbang")
    ap.add_argument("--kp_loosen", type=float, default=10.0)
    ap.add_argument("--kp_tighten", type=float, default=2.0)
    ap.add_argument("--max_step", type=float, default=1.0)
    args = ap.parse_args()

    rows = run(args.mode, args.seed, args.port, args.ns3_path, args.simtime, args.output_folder,
               window=args.window, cadence=args.cadence,
               badsignal_threshold=args.badsignal_threshold,
               activity_threshold=args.activity_threshold, dead_band=args.dead_band,
               step_db=args.step_db, max_repeat=args.max_repeat, multi_cell=args.multi_cell,
               top_k=args.top_k, controller_type=args.controller_type,
               kp_loosen=args.kp_loosen, kp_tighten=args.kp_tighten, max_step=args.max_step)
    rewards = [weighted(r) for r in rows]
    rewards_alt = [weighted(r, w_pingpong=ALT_W_PINGPONG) for r in rows]
    n_handovers = sum(r["diagnostics"]["handovers_this_step"] for r in rows)
    print(f"mode={args.mode:<10s} seed={args.seed} n_steps={len(rows)} "
          f"handovers={n_handovers} mean_reward={np.mean(rewards):+.4f} "
          f"mean_reward_alt={np.mean(rewards_alt):+.4f}", flush=True)

    # Persist the RAW per-step terms, not just the blended scalar above --
    # the terms are what let any future re-weighting (e.g. testing a
    # different ping-pong weight) be answered by re-scoring this file
    # instead of re-running the simulation. Every run before this one in
    # today's session only printed the final weighted scalar, so none of
    # that trajectory data could be recovered afterward.
    tag = args.controller_type if args.mode == "controller" else "donothing"
    raw_path = path.join(args.output_folder, f"raw_terms_{args.mode}_{tag}_{args.seed}.jsonl")
    with open(raw_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
