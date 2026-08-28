"""Minimal SB3 PPO smoke test for MlbZmqEnv against a real ns-3 run.

The point is to prove PPO can construct, step and update on the live environment
end to end without crashing: act, step, reward, policy update. It is not trying
to learn anything. 16 timesteps at n_steps=8 is two tiny rollouts and the
returned policy is noise. This is the last checkpoint before real training.

It is slow because every model.learn() timestep is one MlbZmqEnv.step(), which is
one scenario-marl-zmq control step over the ZeroMQ bridge. T_control defaults to
1.0 s of simulated time and is set by the controlInterval GlobalValue. With
simTime=10 an episode is about 10 steps, so 16 timesteps spans two ns-3 launches:
SB3's VecEnv auto-resets at episode end and MlbZmqEnv.reset() relaunches the
simulator. Minutes, not seconds, is normal.

CFG matches examples/cio_zmq_experiment.py's except for rlcAmEnabled, which is 1
here and 0 there. That configuration is already proven to launch ns-3 and
complete the KPI/action round trip. The other difference is who supplies the
action: there it is env.step(None), the placeholder heuristic, here it is the
PPO policy.

Usage:
    python examples/train_mlb_smoke.py
    python examples/train_mlb_smoke.py --total_timesteps 8 --seed 555
"""
import argparse
import sys
import traceback
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mlb_zmq_env import MlbZmqEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Matches examples/cio_zmq_experiment.py's CFG except for rlcAmEnabled, which is
# 1 here and 0 there: seed 555, simTime 10, ues 3, heuristicType -1 leaves the ES
# heuristic inert, e2nrEnabled 1.
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
    # RLC AM, not UM. This used to be [0] with no justification recorded, and it
    # silently put training in a different radio regime from the one the reward
    # was validated in. Measured at seed 555 with 35 UEs and a do-nothing policy,
    # live AM reproduces the offline scenario-three baseline to 0.1% (backlog
    # 3.265 vs 3.261, satisfaction 0.391 vs 0.390), while UM differs by 48% and
    # 41%. UM also has no ARQ, so residual HARQ failures are permanent loss, a
    # ~12% floor that caps Satisfaction near 0.88 and confounds it as a delivery
    # measure.
    "rlcAmEnabled": [1],
    # Three flags deliberately left at their defaults. Each looks like a free
    # speed or disk win, and each silently destroys KPIs instead of erroring.
    #
    # enableTraces (default 1) gates MmWaveHelper::EnableTraces(), the only place
    #   RxPacketTraceUe/EnbCallback get connected (mmwave-helper.cc:3116-3141).
    #   Those callbacks are the only callers of MmWavePhyTrace::UpdateTraces(),
    #   which is the only writer of m_macVolumeUeSpecific
    #   (mmwave-phy-trace.cc:262). That same MmWavePhyTrace instance is the
    #   device's E2DuCalculator (mmwave-helper.cc:2208), so it is also the source
    #   of volume_bytes and prb_utilization. With enableTraces=0 those counters
    #   never increment: a 10-step rollout returned reward == 1.000 on every step
    #   (balance guarded on max_prb == 0, backlog 0 on volume == 0) against a
    #   traces-on run of the same seed carrying ~1.1 MB/step. Traces are the KPI
    #   pipeline, not a logging luxury. Use purge_sim_path_on_close for the disk
    #   cost instead.
    #
    # enableE2FileLogging (default 1) is not a logging switch. It selects offline
    #   files instead of connecting to a live RIC. At 0,
    #   lte-enb-net-device.cc:1153 calls e2term->RegisterKpmCallbackToE2Sm(),
    #   which needs a real E2 termination; with none, ns-3 died mid-episode at
    #   step 5 of 10 with "Simulation exited with an error".
    #
    # e2cuCp / CU-CP reporting (default on): the callback filling m_l3sinrMap is
    #   registered only under `if (m_sendCuCp)` (mmwave-enb-net-device.cc:347),
    #   and that map is the only source of the per-UE l3_serving_sinr_db behind
    #   the observation's 7 sinr_db_mean fields. Disabling it pins all 7 to
    #   -40 dB.
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
        # close() runs on every reset, so each finished episode's run dir is
        # deleted instead of accumulating. Traces have to stay on (they are the
        # KPI pipeline, see the CFG note), so ns-3 writes ~4-7 MB per step, on
        # the order of 500 GB over a 100k-step run. Nothing in the training loop
        # reads those files back, so purging is what makes long runs possible.
        env = MlbZmqEnv(ns3_path=args.ns3_path, scenario_configuration=CFG,
                        output_folder=args.output_folder, optimized=False,
                        purge_sim_path_on_close=True)
        # Monitor logs per-episode return and length, which is how we can tell
        # the act/step/reward loop actually turned rather than short-circuiting.
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
