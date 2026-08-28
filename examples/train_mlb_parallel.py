"""Parallel PPO training for MlbZmqEnv: N ns-3 simulations at once.

One control step costs about 200 s solo and 250-300 s at 5 workers, and a
single-worker run pays that on every PPO timestep. scenario-marl-zmq.cc used to
hardcode tcp://localhost:5555, which made parallelism impossible; it now takes a
zmqPort, so N workers can run side by side and the effective cost falls to about
50 s per timestep at 5 workers.

Measured at 2 workers: 1.86x throughput, with results byte-identical to
sequential, so parallelism does not perturb the simulation. That figure was
measured at two workers under RLC UM. At 5 workers under RLC AM the per-step cost
is higher, so do not extrapolate 1.86x without measuring it.

Two things that will bite you if you write this yourself. The first is a build
race: NsOranEnv.setup_sim() calls configure_and_build_ns3() on every env
construction, so under SubprocVecEnv that is N simultaneous ns-3 builds
contending for one CMake cache, one .lock-ns3_* file and one binary. We build
once here in the parent and construct workers with build_ns3=False. The second is
port collision: every worker needs its own ZMQ port and its own ns-3 instance
dialling that port, which base_port + rank gives. Check nothing else is on those
ports first, because a stray run silently steals the socket.

Prefer fewer, longer episodes. Cold start is about 190 s solo but about 730 s
under N-way contention (the ns-3 build check plus 35 UEs attaching, all workers
at once), so short episodes pay that toll repeatedly.

Usage:
    python examples/train_mlb_parallel.py --n_envs 5 --total_timesteps 1000
    python examples/train_mlb_parallel.py --n_envs 5 --ues 3 --simtime 10   # faster pilot
"""
import argparse
import socket
import sys
import time
from os import makedirs, path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from callbacks.step_logger import StepLogger
from environments.mlb_zmq_env import MlbZmqEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# The locked training configuration.
#
# rlcAmEnabled=1 is RLC AM, not UM, and it is load-bearing. It used to be [0]
# with no justification recorded, which silently put training in a different
# radio regime from the one the reward was validated in. At seed 555 with 35 UEs
# and a do-nothing policy, live AM reproduces the offline scenario-three baseline
# to 0.1% (backlog 3.265 vs 3.261, satisfaction 0.391 vs 0.390), while UM differs
# by 48% and 41%. UM also has no ARQ, so residual HARQ failures are permanent
# loss, a ~12% floor that caps Satisfaction near 0.88 and confounds it as a
# delivery measure.
#
# For the three flags that silently zero the KPIs rather than erroring
# (enableTraces, enableE2FileLogging, CU-CP reporting), see the CFG notes in
# train_mlb_smoke.py and docs/mlb_training_fixes.md.
CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1],
}


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def make_env(rank: int, args, cfg: dict, monitor_dir: str):
    def _init():
        env = MlbZmqEnv(
            ns3_path=args.ns3_path, scenario_configuration=dict(cfg),
            output_folder=args.output_folder, optimized=False,
            zmq_port=args.base_port + rank,
            # built once in the parent, see the module docstring
            build_ns3=False,
            # Each episode is a new world. Roughly 1 seed in 10 kills ns-3 at
            # start-up, and reset() re-draws rather than losing the run.
            vary_rng_run_per_episode=True,
            # Per-episode offered load. None leaves it fixed at the scenario's
            # own value.
            udp_interval_us_range=args.udp_interval_range,
            # ns-3 writes ~250 MB per run dir; without this a long run fills the disk
            purge_sim_path_on_close=True,
        )
        # Monitor gets a filename so its per-episode totals land on disk as an
        # independent record: StepLogger's per-step rewards must sum, per
        # episode, to Monitor's `r`. That cross-check is the one the offline
        # reconstruction could never pass, which is why both are kept.
        return Monitor(env, filename=path.join(monitor_dir, f"monitor_rank{rank}"))
    return _init


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n_envs", type=int, default=5)
    ap.add_argument("--total_timesteps", type=int, default=1000)
    ap.add_argument("--n_steps", type=int, default=16, help="rollout length PER ENV")
    ap.add_argument("--base_port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--ues", type=int, default=None)
    ap.add_argument("--simtime", type=int, default=None)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--save_dir", default=path.join(path.dirname(__file__), "..", "output", "ppo_runs"))
    ap.add_argument("--udp_interval_range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="randomise udpFullBufferIntervalUs per episode over "
                         "[LO, HI] us inclusive (e.g. 450 550). Larger = LESS "
                         "offered DL load. Omitted = fixed at the scenario's value, "
                         "i.e. unchanged behaviour.")
    ap.add_argument("--log_dir", default=None,
                    help="where the per-step reward-term/action CSV and the Monitor "
                         "files go (default: <save_dir>/logs_<timestamp>)")
    args = ap.parse_args()

    if args.udp_interval_range is not None:
        args.udp_interval_range = tuple(args.udp_interval_range)
        print(f"per-episode offered load: udpFullBufferIntervalUs ~ U"
              f"{args.udp_interval_range} us (inclusive)", flush=True)

    log_dir = args.log_dir or path.join(args.save_dir,
                                        "logs_" + time.strftime("%Y%m%d_%H%M%S"))
    makedirs(log_dir, exist_ok=True)

    cfg = dict(CFG)
    if args.ues is not None:
        cfg["ues"] = [args.ues]
    if args.simtime is not None:
        cfg["simTime"] = [args.simtime]

    busy = [p for p in range(args.base_port, args.base_port + args.n_envs) if not port_is_free(p)]
    if busy:
        sys.exit(f"ports already in use: {busy} -- another run is holding them. "
                 f"Use --base_port to move out of the way.")

    print("building ns-3 ONCE in the parent (workers will skip it) ...", flush=True)
    t0 = time.time()
    probe = MlbZmqEnv(ns3_path=args.ns3_path, scenario_configuration=dict(cfg),
                      output_folder=args.output_folder, optimized=False,
                      zmq_port=args.base_port, build_ns3=True)
    print(f"  build/lookup done in {time.time()-t0:.1f}s -> {probe.script_executable}", flush=True)
    del probe

    venv = SubprocVecEnv([make_env(i, args, cfg, log_dir) for i in range(args.n_envs)],
                         start_method="spawn")
    print(f"{args.n_envs} workers up on ports "
          f"{args.base_port}..{args.base_port + args.n_envs - 1}", flush=True)

    # batch_size must divide n_steps * n_envs
    batch = args.n_steps * args.n_envs
    model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=batch,
                verbose=1, seed=args.seed, tensorboard_log=None)
    # CheckpointCallback counts callback calls, not timesteps, and one call is
    # one vec-step across all workers. So this fires every n_steps * 4 vec-steps,
    # which is n_steps * 4 * n_envs timesteps.
    ckpt = CheckpointCallback(save_freq=max(args.n_steps * 4, 1), save_path=args.save_dir,
                              name_prefix="mlb_ppo")
    # Records info["reward_terms"] and info["cio_offsets"] at the source. See
    # src/callbacks/step_logger.py for why this is not Monitor(info_keywords=).
    steplog = StepLogger(path.join(log_dir, "reward_terms_steps.csv"), verbose=1)
    print(f"per-step reward terms + actions -> {log_dir}", flush=True)
    try:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=CallbackList([ckpt, steplog]), progress_bar=False)
        model.save(path.join(args.save_dir, "mlb_ppo_final"))
        print("TRAINING RUN COMPLETE")
    finally:
        venv.close()
