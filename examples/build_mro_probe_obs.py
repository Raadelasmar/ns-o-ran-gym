"""Build the fixed, real-ns-3 observation batch CriticProbe measures the SAC
critic against throughout training.

CriticProbe (src/callbacks/critic_probe.py) scores the critic on the SAME
fixed batch of observations at every checkpoint, so a change in mean Q is a
change in the critic, not a change in what it was asked to score. That batch
has to come from real ns-3 -- a synthetic one would let the critic's
real-world calibration go unmeasured. MLB's own pilot uses an equivalent
file (analysis/mlb_session_2026-08-31/probe_obs.npy); this is MRO's.

Collects observations from seeded do-nothing episodes at the locked SAC
config (same CFG as every other MRO measurement this project has run), the
same low-cost, no-agent-needed pattern as mro_baseline_experiment.py --
a fixed action, real ns-3, nothing learned, just real observation vectors
recorded as they occur.

Usage:
    python examples/build_mro_probe_obs.py --n_seeds 3 --seed_start 4001
    python examples/build_mro_probe_obs.py --n_seeds 3 --workers 3   # parallel
"""
import argparse
import sys
from os import makedirs, path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import CELLS, MroZmqEnv

import numpy as np

# Identical to every other MRO experiment script's CFG this session -- see
# mro_baseline_experiment.py's docstring for why each field is set this way.
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


def collect_seed(seed: int, args, zmq_port: int) -> np.ndarray:
    """All observation vectors (reset's + every step's) from one do-nothing episode."""
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    cfg["simTime"] = [args.simtime]

    env = MroZmqEnv(ns3_path=args.ns3_path, scenario_configuration=cfg,
                    output_folder=args.output_folder, optimized=False,
                    zmq_port=zmq_port, build_ns3=False,
                    vary_rng_run_per_episode=False,
                    purge_sim_path_on_close=True)

    action = np.zeros(len(CELLS), dtype=np.float32)
    obs_rows = []
    try:
        obs, info = env.reset(seed=seed)
        obs_rows.append(obs.copy())
        print(f"  seed {seed}: reset ok, rng_run={info.get('rng_run')}", flush=True)
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            obs_rows.append(obs.copy())
            if "reward_terms" not in info:
                break
            if terminated or truncated:
                break
    except Exception as exc:
        print(f"  seed {seed}: CRASHED ({exc!r}), keeping {len(obs_rows)} rows collected so far",
              flush=True)
    finally:
        env.close()
    print(f"  seed {seed}: {len(obs_rows)} observations collected", flush=True)
    return np.stack(obs_rows) if obs_rows else np.zeros((0, len(CELLS) * 5), dtype=np.float32)


def _worker(seed, args, port, q):
    """Module-level (not nested in __main__) so spawn's pickling can find it."""
    q.put((seed, collect_seed(seed, args, port)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed_start", type=int, default=4001)
    ap.add_argument("--simtime", type=int, default=30)
    ap.add_argument("--base_port", type=int, default=5555)
    ap.add_argument("--workers", type=int, default=1,
                    help="collect this many seeds concurrently; see "
                         "mro_outage_stress_experiment.py's PARALLEL EXECUTION "
                         "docs before raising this above 1 -- build_ns3=False "
                         "is already set here so the same concurrent-rebuild "
                         "hazard that script documents does not apply, but "
                         "resource headroom still does.")
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--out", default=path.join(path.dirname(__file__), "..", "analysis",
                                               "mro_probe_obs", "probe_obs.npy"))
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    makedirs(path.dirname(args.out), exist_ok=True)

    print(f"Collecting observations from {len(seeds)} do-nothing seeds: {seeds}", flush=True)

    if args.workers <= 1:
        all_obs = [collect_seed(seed, args, args.base_port) for seed in seeds]
    else:
        # Same bounded-concurrency pattern as
        # mro_outage_stress_experiment.py's run_all_seeds, reused inline
        # since this script only ever needs ONE pass, not resumable
        # scheduling -- a worker per seed, up to the requested cap.
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        n_workers = min(args.workers, len(seeds))
        procs = []
        for i, seed in enumerate(seeds[:n_workers]):
            p = ctx.Process(target=_worker, args=(seed, args, args.base_port + i, q))
            p.start()
            procs.append(p)
        results = {}
        for _ in seeds[:n_workers]:
            seed, obs = q.get()
            results[seed] = obs
        for p in procs:
            p.join()
        # Only as many seeds as fit in one wave of workers are supported here
        # (n_seeds is small by design -- this is a one-off batch build, not
        # an ongoing experiment), so no second-wave scheduling is needed.
        all_obs = [results[s] for s in seeds[:n_workers]]

    all_obs = [o for o in all_obs if o.shape[0] > 0]
    if not all_obs:
        sys.exit("No observations collected -- every seed crashed before producing any.")

    probe = np.concatenate(all_obs, axis=0).astype(np.float32)
    np.save(args.out, probe)
    print(f"\nWrote {probe.shape[0]} observations ({probe.shape[1]} features) to {args.out}",
          flush=True)
    print(f"  mean abs value: {np.abs(probe).mean():.4f}  (sanity check -- should be well "
          f"within the observation_space bounds, not saturated at the clip edges)", flush=True)
