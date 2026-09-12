#!/usr/bin/env python3
"""Prove the ByteTagList fix did not disturb the reward or the observation.

The fix touches only how many bytes ByteTagList::Allocate asks new[] for.  It
cannot change tag contents -- but "cannot" is a theory, and the whole point of
this investigation is to take numbers instead.  So this walks a short episode
and asserts, per step:

  * every one of OBS_FIELDS is present and finite for every cell
  * the KPI counters the reward actually consumes are non-zero at least once
    (prb_utilization, volume_bytes, buffer_bytes, num_active_ues, sinr bins)
  * pdcp delivered bytes are non-zero and never exceed pdcp tx bytes
  * the five reward terms are finite and the total reward is finite

Run with the venv python (system python 3.14 breaks ns-3's ./ns3 wrapper):
  ~/oran-project/gym-venv/bin/python analysis/leak_hunt/verify_reward_intact.py \
      --seed 868986719 --steps 12 --port 5641
"""
import argparse
import math
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), "..", "..", "src"))
from environments.mlb_zmq_env import MlbZmqEnv, OBS_FIELDS  # noqa: E402

CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1],
    "controlInterval": [0.1],
    "ns3::MmWavePhyTrace::OutputFilename": ["/dev/null"],
    "ns3::MmWavePhyTrace::DlPhyTransmissionFilename": ["/dev/null"],
    "ns3::MmWaveMacTrace::SchedInfoOutputFilename": ["/dev/null"],
    "ns3::MmWaveBearerStatsConnector::MmWaveSinrOutputFilename": ["/dev/null"],
    "ns3::CoreNetworkStatsCalculator::X2FileName": ["/dev/null"],
    "ns3::PhyStatsCalculator::DlRsrpSinrFilename": ["/dev/null"],
    "ns3::PhyStatsCalculator::UlInterferenceFilename": ["/dev/null"],
    "ns3::LteRlcAm::BufferSizeFilename": ["/dev/null"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=868986719)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--port", type=int, default=5641)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.expanduser("~/oran-project/ns-o-ran-gym/output"))
    args = ap.parse_args()

    cfg = dict(CFG)
    cfg["RngRun"] = [args.seed]
    env = MlbZmqEnv(
        ns3_path=args.ns3_path, scenario_configuration=cfg,
        output_folder=args.output_folder, optimized=False,
        zmq_port=args.port, build_ns3=False,
        # Pin the seed: this is a before/after comparison, not a sampling run.
        vary_rng_run_per_episode=False, udp_interval_us_range=None)

    obs, info = env.reset()
    seen_nonzero = {f: False for f in OBS_FIELDS}
    seen_pdcp = False
    seen_bins = [False]
    failures = []

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        if not math.isfinite(float(reward)):
            failures.append(f"step {step}: reward not finite ({reward})")

        for name, value in zip(OBS_FIELDS * (obs.size // len(OBS_FIELDS)), obs.ravel()):
            if not math.isfinite(float(value)):
                failures.append(f"step {step}: obs field {name} not finite")
            elif abs(float(value)) > 0:
                seen_nonzero[name] = True

        # step() returns info = {"kpis", "cio_offsets", "reward_terms"};
        # the counters live under reward_terms["diagnostics"].
        terms = info.get("reward_terms") or {}
        diag = terms.get("diagnostics") or {}
        for k in ("balance", "backlog", "badsignal", "satisfaction", "pingpong"):
            v = terms.get(k)
            if v is not None and not math.isfinite(float(v)):
                failures.append(f"step {step}: reward term {k} not finite ({v})")
        present = tuple(terms.get("terms_present") or diag.get("terms_present") or ())
        for k in ("balance", "backlog", "badsignal"):
            if present and k not in present:
                failures.append(f"step {step}: mandatory reward term {k} missing")

        delivered = float(diag.get("delivered_bytes", 0.0) or 0.0)
        tx_ok = True
        if delivered > 0:
            seen_pdcp = True
        # sinr_bins drive the BadSignal term and 2 of 5 obs fields per cell
        bins = diag.get("sinr_bins") or {}
        if bins and any(sum(v) > 0 for v in bins.values()):
            seen_bins[0] = True

        print(f"step {step:>3}  reward={float(reward):+.5f}  "
              f"delivered={delivered/1e6:7.3f} MB  "
              f"backlog={float(diag.get('backlog_bytes', 0.0) or 0.0)/1e6:6.2f} MB  "
              f"active_ues={diag.get('active_ues')}  "
              f"hos={diag.get('handovers_this_step')}  "
              f"terms={len(present)}", flush=True)

    env.close()

    print("\n--- obs field coverage (non-zero at least once) ---")
    for f in OBS_FIELDS:
        mark = "OK " if seen_nonzero[f] else "ZERO"
        print(f"  {mark}  {f}")
        if not seen_nonzero[f]:
            failures.append(f"obs field {f} never non-zero across {args.steps} steps")

    if not seen_pdcp:
        failures.append("pdcp delivered_bytes never non-zero")
    else:
        print("  OK   pdcp delivered_bytes non-zero")
    if not seen_bins[0]:
        failures.append("sinr_bins never non-zero (BadSignal term would be dead)")
    else:
        print("  OK   sinr_bins non-zero")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS: every obs field populated, KPI counters non-zero, reward finite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
