"""Run one CIO experiment condition: a static per-cell bias, seed 555, scenario-three, 10 s.

This is the driver behind conditions C0-C4 (VERIFICATION_LOG.md V26/V27).
It reuses EnergySavingEnv (whose do_heuristic=True action path emits exactly
(cellId, value) pairs for cells 2..8) and swaps the control-file attributes to
the CIO lever before reset(). NsOranEnv only stores those attributes; the
ActionController is built in start_sim(), so the swap redirects the whole
control path. The same bias vector is (re-)written every 100 ms step;
ns-3's SetCellIndividualOffset is idempotent.

Usage:
    python examples/cio_experiment.py                 # C0: all cells 0 dB
    python examples/cio_experiment.py --cell 3 --bias -6   # C4
    python examples/cio_experiment.py --trafficModel 1 --minSpeed 2 --maxSpeed 4 \
        --handoverMode FixedTtt                           # scenario knob sweep
Conditions on record (all seed 555):
    C0 0 dB everywhere      -> output/5d523414-...  (md5 38e5f1bb, = baseline d7d1d669)
    C1 cell 2 -3 dB         -> output/c68313fb-...  (md5 ac0afa81)
    C2 cell 2 -6 dB         -> output/205da174-...  (md5 c90c50e4)
    C3 cell 2 +6 dB         -> output/7264b9cc-...  (md5 e121fed3)
    C4 cell 3 -6 dB         -> output/be7ba2b0-...  (md5 cbeca13c)
"""
import argparse
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.es_env import EnergySavingEnv

CELLS = [2, 3, 4, 5, 6, 7, 8]

CFG = {
    "heuristicType": [-1],
    "simTime": [10],
    "ues": [5],
    "RngRun": [555],
    "configuration": [0],
    "trafficModel": [3],
    "numberOfRaPreambles": [40],
    "bsOn": [5],
    "bsIdle": [0],
    "bsSleep": [0],
    "bsOff": [2],
    "reducedPmValues": [0],
    "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"],
    "indicationPeriodicity": [0.1],
    "controlFileName": ["cio_actions_for_ns3.csv"],
    "useSemaphores": [1],
    "e2nrEnabled": [1],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", type=int, choices=CELLS, help="cell to bias (omit for all-zero)")
    ap.add_argument("--bias", type=float, default=0.0, help="CIO in dB for --cell (clamped to [-6,6] by ns-3)")
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    # Scenario knobs: omit a flag to leave CFG exactly as defined above.
    # minSpeed/maxSpeed are absent from CFG by default, so omitting them lets ns-3 use its own.
    ap.add_argument("--trafficModel", type=int, choices=[0, 1, 2, 3], default=None,
                    help=f"ns-3 traffic model (default {CFG['trafficModel'][0]})")
    ap.add_argument("--minSpeed", type=float, default=None,
                    help="UE min speed in m/s (default: ns-3's own)")
    ap.add_argument("--maxSpeed", type=float, default=None,
                    help="UE max speed in m/s (default: ns-3's own)")
    ap.add_argument("--handoverMode", choices=["NoAuto", "FixedTtt", "DynamicTtt", "Threshold"],
                    default=None, help=f"handover algorithm (default {CFG['handoverMode'][0]})")
    args = ap.parse_args()

    CFG["RngRun"] = [args.seed]
    if args.trafficModel is not None:
        CFG["trafficModel"] = [args.trafficModel]
    if args.minSpeed is not None:
        CFG["minSpeed"] = [args.minSpeed]
    if args.maxSpeed is not None:
        CFG["maxSpeed"] = [args.maxSpeed]
    if args.handoverMode is not None:
        CFG["handoverMode"] = [args.handoverMode]
    action = [args.bias if c == args.cell else 0.0 for c in CELLS]
    print(f"CIO action vector (cells 2-8): {action}")

    env = EnergySavingEnv(ns3_path=args.ns3_path, scenario_configuration=CFG,
                          output_folder=args.output_folder, optimized=False,
                          do_heuristic=True)
    env.control_header = ["timestamp", "cellId", "cioDb"]
    env.log_file = "CioActions.txt"
    env.control_file = "cio_actions_for_ns3.csv"

    obs, info = env.reset()
    print(f"sim_path: {env.sim_path}", flush=True)

    steps, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if steps % 10 == 0:
            print(f"step {steps}, last_timestamp {env.last_timestamp}", flush=True)

    print(f"DONE: {steps} steps, terminated={terminated}, truncated={truncated}")
    print(f"run dir: {env.sim_path}")
    env.close()
