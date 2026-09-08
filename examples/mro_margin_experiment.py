"""Does MRO's handover-margin action actually change ns-3's handover behaviour?

train_mro_smoke.py only proves the pipeline runs end to end with an
UNTRAINED, randomly-flailing policy -- it never checks whether the
ho_margin_db action the C++ side now consumes (LteEnbRrc::
SetHandoverMarginOffset / GetEffectiveSinrThreshold) has any real physical
effect. This script is that check: two FIXED-action arms, no RL agent, same
seed, so any difference between them is caused by the action alone.

    neutral  margin offset = 0 dB on every cell, every step
             (GetEffectiveSinrThreshold == the network's base HoSinrDifference,
             i.e. behaviour should be identical to the action not existing)
    loosen   margin offset = -3 dB on every cell, every step
             (effective threshold ~0 dB instead of the ~3 dB default --
             handovers should trigger on almost any positive SINR advantage,
             so handover/ping-pong activity should rise sharply)

Ground truth is read from ns-3's OWN UeHandoverStartStats.txt after each run,
never from the ZMQ-reported reward_terms -- the same discipline MLB's own
validation used (count churn from the simulator's own trace file, not from
the derived Python number under test). The ZMQ-side counts are also printed,
as a cross-check that the two sources agree, not as the primary evidence.

vary_rng_run_per_episode=False pins both arms to the SAME RngRun, so both see
the same UE layout and the same channel realisation -- any difference in
handover activity is attributable to the action, not to two different worlds.

Usage:
    python examples/mro_margin_experiment.py
    python examples/mro_margin_experiment.py --simtime 15 --seed 555
"""
import argparse
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import CELLS, MARGIN_LIMIT_DB, MroZmqEnv

import numpy as np

# Same locked config as train_mro_smoke.py / train_mlb_smoke.py, so cost and
# behaviour stay comparable to the numbers already on record for this CFG.
CFG = {
    "heuristicType": [-1], "simTime": [10], "ues": [3], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1],
}


def count_ns3_handovers(sim_path: str):
    """(n_lines, sample_lines) from ns-3's own UeHandoverStartStats.txt.

    Read directly from the file ns-3 wrote, not from anything this project's
    Python side derived from it -- the whole point of this script is to check
    the ZMQ-reported numbers against an independent source.
    """
    file_path = path.join(sim_path, "UeHandoverStartStats.txt")
    if not path.exists(file_path):
        return None, []
    with open(file_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return len(lines), lines[:3]


def run_arm(name: str, margin_offset_db: float, args) -> dict:
    print(f"\n=== arm '{name}': margin offset {margin_offset_db:+.1f} dB on every cell ===",
          flush=True)
    cfg = dict(CFG)
    cfg["RngRun"] = [args.seed]
    cfg["simTime"] = [args.simtime]

    env = MroZmqEnv(ns3_path=args.ns3_path, scenario_configuration=cfg,
                    output_folder=args.output_folder, optimized=False,
                    zmq_port=args.zmq_port,
                    # Both arms must see the identical world: same seed,
                    # never redrawn. See the module docstring.
                    vary_rng_run_per_episode=False,
                    purge_sim_path_on_close=False)  # need the run dir after close()

    action = np.full(len(CELLS), margin_offset_db, dtype=np.float32)
    obs, info = env.reset(seed=args.seed)
    print(f"  reset ok, rng_run={info.get('rng_run')}", flush=True)

    zmq_handovers = 0
    zmq_pingpongs = 0
    step = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        if "reward_terms" not in info:
            break  # episode ended (time limit)
        diag = info["reward_terms"]["diagnostics"]
        zmq_handovers += diag["handovers_this_step"]
        zmq_pingpongs += diag["pingpong_count"]
        step += 1
        print(f"  step {step}: handovers={diag['handovers_this_step']} "
              f"pingpong={diag['pingpong_count']} reward={reward:.4f}", flush=True)
        if terminated or truncated:
            break

    sim_path = env.sim_path
    env.close()

    n_ns3, sample = count_ns3_handovers(sim_path)
    print(f"  ZMQ-side totals:  handovers={zmq_handovers}  pingpongs={zmq_pingpongs}")
    print(f"  ns-3's own UeHandoverStartStats.txt: {n_ns3} lines")
    if sample:
        print(f"    sample: {sample}")

    return {"arm": name, "margin_offset_db": margin_offset_db, "sim_path": sim_path,
            "zmq_handovers": zmq_handovers, "zmq_pingpongs": zmq_pingpongs,
            "ns3_handover_lines": n_ns3}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--simtime", type=int, default=10)
    ap.add_argument("--zmq_port", type=int, default=5555)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    args = ap.parse_args()

    results = [
        run_arm("neutral", 0.0, args),
        run_arm("loosen", -MARGIN_LIMIT_DB, args),
    ]

    print("\n=== SUMMARY ===")
    print(f"{'arm':<10} {'margin_dB':>10} {'zmq_HO':>8} {'zmq_PP':>8} {'ns3_HO_lines':>13}")
    for r in results:
        print(f"{r['arm']:<10} {r['margin_offset_db']:>10.1f} {r['zmq_handovers']:>8} "
              f"{r['zmq_pingpongs']:>8} {r['ns3_handover_lines']:>13}")

    neutral, loosen = results
    if neutral["ns3_handover_lines"] is not None and loosen["ns3_handover_lines"] is not None:
        if loosen["ns3_handover_lines"] > neutral["ns3_handover_lines"]:
            print("\nPASS: loosening the margin increased real ns-3 handover activity, "
                  "as expected -- the action has a genuine physical effect.")
        else:
            print("\nFAIL (or inconclusive): loosening the margin did NOT increase "
                  "ns-3's own handover count. Do not trust the action wiring yet.")
    else:
        print("\nINCONCLUSIVE: could not read UeHandoverStartStats.txt for one or both arms.")
