"""Minimal SB3 PPO smoke test for MroZmqEnv against a real ns-3 run.

Mirrors examples/train_mlb_smoke.py exactly in structure and intent: prove PPO
can construct, step and update on the live environment end to end without
crashing -- act, step, reward, policy update. It is not trying to learn
anything, and PPO here is a stand-in for whichever algorithm training
eventually uses (see docs/mro_reward_design.pdf); the point of this script is
the pipeline, not the policy.

IMPORTANT: as of this commit, MRO's action (per-cell handover-margin offset,
sent as "ho_margin_db") reaches ns-3's LteEnbNetDevice::ApplyControlPayload and
LteEnbRrc::SetHandoverMarginOffset -- that wiring compiled and linked cleanly
into the scenario-marl-zmq binary. This script proves the ROUND TRIP works
(action sent, KPIs and reward come back shaped correctly); it does not by
itself prove the margin changes handover behaviour in a measurable way -- that
needs a real A/B comparison (e.g. a large fixed offset on one cell vs none),
which is future work, not this smoke test.

It is slow for the same reason as MLB's: every model.learn() timestep is one
MroZmqEnv.step(), one scenario-marl-zmq control step over the ZeroMQ bridge.
T_control defaults to 1.0 s and simTime=10 gives an ~10-step episode, so 16
timesteps spans two ns-3 launches. Minutes, not seconds, is normal.

CFG is IDENTICAL to train_mlb_smoke.py's -- same scenario, same topology, same
three flags that must stay at their defaults (enableTraces, enableE2FileLogging,
e2cuCp/CU-CP reporting). MRO's own KPI needs are a subset of what those flags
already protect: per-UE l3_serving_sinr_db (behind e2cuCp, same as MLB's
sinr_db_mean), sinr_bins (behind enableTraces, feeds MRO's badsignal_frac
observation field the same way it feeds MLB's BadSignal). The one exception is
handovers: unlike sinr_bins, that field does NOT ride on enableTraces (it hooks
the UE RRC HandoverStart trace source directly), so it would survive even if
enableTraces were ever (wrongly) disabled -- see mro_zmq_env.py's PingPong docs.

Usage:
    python examples/train_mro_smoke.py
    python examples/train_mro_smoke.py --total_timesteps 8 --seed 555
"""
import argparse
import sys
import traceback
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import MroZmqEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Identical to train_mlb_smoke.py's CFG -- same scenario-marl-zmq topology and
# the same three do-not-touch flags. See that file's comments for the full
# source-level justification of each; not repeated here to avoid the two
# copies drifting apart in the retelling rather than the value.
CFG = {
    "heuristicType": [-1],
    "simTime": [10],
    "ues": [3],
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
    "e2nrEnabled": [1],
    "rlcAmEnabled": [1],
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--total_timesteps", type=int, default=16,
                    help="~2 rollouts at n_steps=8; enough to force one PPO update")
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    args = ap.parse_args()

    CFG["RngRun"] = [args.seed]

    env = None
    try:
        # See train_mlb_smoke.py's identical comment: traces stay on (they are
        # the KPI pipeline), so purging the run dir on close() is what keeps a
        # long run's disk usage bounded.
        env = MroZmqEnv(ns3_path=args.ns3_path, scenario_configuration=CFG,
                        output_folder=args.output_folder, optimized=False,
                        purge_sim_path_on_close=True)
        env = Monitor(env)
        print(f"obs space: {env.observation_space}")
        print(f"act space: {env.action_space}", flush=True)

        model = PPO("MlpPolicy", env, n_steps=args.n_steps,
                    batch_size=args.n_steps, verbose=1, seed=args.seed)
        print("PPO constructed", flush=True)

        model.learn(total_timesteps=args.total_timesteps)

        print("SMOKE TEST PASSED ✅")
    except Exception:
        traceback.print_exc()
        print("SMOKE TEST FAILED ❌")
        sys.exit(1)
    finally:
        if env is not None:
            env.close()
