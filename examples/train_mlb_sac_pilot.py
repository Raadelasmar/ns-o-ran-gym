"""Short SAC pilot on real ns-3: does the critic climb?

Not a performance run. Pass criteria in AGENT_BUILD_LOG.md Step 12. The PPO pilot
baseline it is read against is Step 11a-11c: critic 6.212 mean V (13-15% of a
42-49 target), 130 gradient updates in the whole 16.5 h run.

learning_starts is 5000, not the production 25000. SB3 does no gradient updates
before it, so a short pilot at 25000 would collect for ~21 h and see a flat
critic. A pass here does not validate 25000.

gamma is 0.9948838031 = 0.95 ** 0.1, what arm C ran, not Step 10j-6's
0.99 ** 0.1: 19.6 s of horizon against 99.5 s over a 30 s episode (Step 11d).
Passed explicitly because SB3's unset default of 0.99 is a 10 s horizon at T=0.1.

Usage:
    python examples/train_mlb_sac_pilot.py                    # the pilot as specified
    python examples/train_mlb_sac_pilot.py --total_timesteps 100 --n_envs 1   # smoke
"""
# Must precede the numpy/torch imports. Step 10j-5a: unpinned, concurrent runs
# put ~102 threads on 12 cores and one advanced <5,000 steps in three hours.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                  # noqa: E402
import json                                                      # noqa: E402
import shutil                                                    # noqa: E402
import socket                                                    # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
from os import makedirs, path                                    # noqa: E402

import torch                                                     # noqa: E402
torch.set_num_threads(1)

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from callbacks.critic_probe import CriticProbe                   # noqa: E402
from callbacks.step_logger import StepLogger                     # noqa: E402
from environments.mlb_zmq_env import MlbZmqEnv                   # noqa: E402

from stable_baselines3 import SAC                                # noqa: E402
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,   # noqa: E402
                                                CheckpointCallback)
from stable_baselines3.common.monitor import Monitor             # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv       # noqa: E402

# The PPO pilot's config, unchanged except for controlInterval. enableTraces,
# enableE2FileLogging and CU-CP reporting fail silently rather than erroring, so
# they are deliberately absent (docs/mlb_training_fixes.md). rlcAmEnabled=1 is AM:
# UM has no ARQ, a ~12% loss floor that caps Satisfaction near 0.88. ues=5 is per
# cell, so 7 cells x 5 = the 35 UEs every measurement is quoted at.
CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1],
    # Fix 2. Not control_period_s: that is the 0.1 s E2 indication window the
    # byte counters cover, a different quantity that shares the value.
    "controlInterval": [0.1],
}

# Arm C (C_sac_gs16), the only T=0.1 arm that converted samples into learning.
# analysis/mlb_session_2026-08-30/runs/C_sac_gs16_T0.1_rs1_s1.json.
ARM_C = dict(
    gamma=0.9948838031,      # see module docstring; not 10j-6's table value
    learning_rate=3e-4,
    batch_size=256,
    train_freq=1,            # 1 vec-step -> n_envs transitions, then gradient_steps
    gradient_steps=16,       # gs=64 scored below doing nothing
    buffer_size=200_000,     # nothing is evicted at these budgets
    ent_coef="auto_0.1",
)

# ns-3 writes ~180 MB per 30 s episode; this leaves ~3 episode-boundaries of slack.
MIN_FREE_GB = 3.0

# From analysis/mlb_session_2026-08-31/memprobe. Ceiling is our own anon; the
# floor keeps systemd-oomd below its 50% pressure trigger.
MEM_ANON_CEILING_GB = 6.0
MIN_AVAIL_GB = 2.5


class DiskGuard(BaseCallback):
    """Stops training if free space falls below MIN_FREE_GB.

    Catches a purge (MlbZmqEnv.close -> _purge_sim_path) that has silently stopped
    working, which would otherwise surface as a full filesystem hours in.
    """

    def __init__(self, path_to_watch: str, min_free_gb: float = MIN_FREE_GB, verbose: int = 0):
        super().__init__(verbose)
        self.path_to_watch = path_to_watch
        self.min_free_gb = min_free_gb
        self.tripped = False

    def _on_step(self) -> bool:
        free_gb = shutil.disk_usage(self.path_to_watch).free / 1e9
        if free_gb < self.min_free_gb:
            self.tripped = True
            print(f"\n!! DiskGuard: only {free_gb:.2f} GB free at "
                  f"{self.path_to_watch} (floor {self.min_free_gb} GB). "
                  f"STOPPING at {self.num_timesteps} timesteps.\n", flush=True)
            return False
        return True


class MemGuard(BaseCallback):
    """Stops training before systemd-oomd notices, and checkpoints on the way out.

    Step 13: oomd shot gnome-shell (238 MB) rather than the 9.1 GB pilot, because
    it picks by reclaim activity within the pressured slice. Waiting for the kill
    is useless: it lands elsewhere, as SIGKILL, with nothing saved.

    Watches our own cgroup's memory.stat anon (not memory.current, which counts
    reclaimable ns-3 trace page cache) and system-wide MemAvailable, which catches
    pressure that is somebody else's fault.
    """

    def __init__(self, anon_ceiling_gb: float, min_avail_gb: float,
                 checkpoint_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.anon_ceiling_gb = anon_ceiling_gb
        self.min_avail_gb = min_avail_gb
        # Explicit, not model.logger.dir: SB3 defaults that to a /tmp/SB3-* temp
        # dir, so the emergency checkpoint used to land outside the run's own
        # directories with only a log line saying where.
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_written = ""
        self.tripped = False
        self.reason = ""
        self.peak_anon_gb = 0.0
        self.cg = self._own_cgroup()

    @staticmethod
    def _own_cgroup():
        """Resolve this process's cgroup v2 directory, or None if unavailable."""
        try:
            for line in open("/proc/self/cgroup", encoding="utf8"):
                # cgroup v2 has exactly one line, "0::<path>"
                if line.startswith("0::"):
                    d = path.join("/sys/fs/cgroup", line.strip()[3:].lstrip("/"))
                    return d if path.isdir(d) else None
        except OSError:
            pass
        return None

    def _anon_gb(self) -> float:
        if not self.cg:
            return 0.0
        try:
            with open(path.join(self.cg, "memory.stat"), encoding="utf8") as fh:
                for line in fh:
                    if line.startswith("anon "):
                        return int(line.split()[1]) / 1e9
        except (OSError, ValueError):
            pass
        return 0.0

    @staticmethod
    def _avail_gb() -> float:
        try:
            with open("/proc/meminfo", encoding="utf8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1e6
        except (OSError, ValueError):
            pass
        return float("inf")

    def _stop(self, reason: str) -> bool:
        self.tripped = True
        self.reason = reason
        print(f"\n!! MemGuard: {reason}. CHECKPOINTING then STOPPING at "
              f"{self.num_timesteps} timesteps.\n", flush=True)
        # Save here rather than trusting the finally block: the run may be killed
        # without warning at this point.
        try:
            save_dir = self.checkpoint_dir
            self.model.save(path.join(save_dir, "memguard_model"))
            self.model.save_replay_buffer(path.join(save_dir, "memguard_replay"))
            self.checkpoint_written = path.join(save_dir, "memguard_model")
            print(f"   MemGuard checkpoint written to {save_dir}", flush=True)
        except Exception as exc:                       # noqa: BLE001 - never mask the stop
            print(f"   MemGuard checkpoint FAILED: {exc}", flush=True)
        return False

    def _on_step(self) -> bool:
        anon = self._anon_gb()
        self.peak_anon_gb = max(self.peak_anon_gb, anon)
        if self.cg and anon > self.anon_ceiling_gb:
            return self._stop(f"cgroup anon {anon:.2f} GB > ceiling "
                              f"{self.anon_ceiling_gb:.2f} GB")
        avail = self._avail_gb()
        if avail < self.min_avail_gb:
            return self._stop(f"system MemAvailable {avail:.2f} GB < floor "
                              f"{self.min_avail_gb:.2f} GB")
        return True


class ReplayBufferSaver(BaseCallback):
    """Periodically writes the replay buffer to one atomically-replaced file.

    Step 13's run lost 4,000 samples because CheckpointCallback saves weights but
    not the buffer, and SAC is off-policy: the buffer is the run. Written to a
    temp file and os.replace()d, so a kill mid-write keeps the previous copy.
    """

    def __init__(self, save_path: str, every: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.every = max(int(every), 1)
        self.saves = 0
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        tmp = self.save_path + ".tmp"
        try:
            self.model.save_replay_buffer(tmp)
            src = tmp if path.exists(tmp) else tmp + ".pkl"
            os.replace(src, self.save_path + ".pkl")
            self.saves += 1
            if self.verbose:
                sz = path.getsize(self.save_path + ".pkl") / 1e6
                print(f"   replay buffer saved ({sz:.0f} MB, "
                      f"{self.num_timesteps} timesteps)", flush=True)
        except Exception as exc:                       # noqa: BLE001 - a failed save must not kill the run
            print(f"   !! replay buffer save failed: {exc}", flush=True)
        return True


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
            # Built once in the parent: N workers calling configure_and_build_ns3()
            # concurrently race one CMake cache, lock file and binary.
            build_ns3=False,
            # ~1 seed in 10 kills ns-3 at start-up; reset() re-draws.
            vary_rng_run_per_episode=True,
            # Not set: the PPO pilot predates per-episode load randomisation
            # (Step 7f), so leaving it off keeps the two comparable.
            udp_interval_us_range=None,
            # ~180 MB per episode per worker; tested in tests/test_purge_sim_path.py.
            purge_sim_path_on_close=True,
        )
        return Monitor(env, filename=path.join(monitor_dir, f"monitor_rank{rank}"))
    return _init


def assert_actuals(model, venv, args, learn_budget):
    """Read the live model back and refuse to train if it does not match intent.

    On a resume, BaseAlgorithm.load does __dict__.update(data)
    (base_class.py:738), so the checkpoint's pickled hyperparameters can override
    the command line silently. Two did:

      Defect A  gradient_steps came back as the previous run's value, giving 5.00
                updates/sample against arm C's 3.20 while the banner printed 3.00.
      Defect B  learn(total_timesteps=N, reset_num_timesteps=False) treats N as a
                remainder, not a target (base_class.py:416).

    Both are fixed at the source; this is the belt to that pair of braces. It also
    pins replay_buffer.n_envs, which load_replay_buffer() never reconciles
    (off_policy_algorithm.py:241-257) and which fails minutes later inside add().
    """
    rb = model.replay_buffer
    rows = [
        # name,                          intended,                              actual
        ("gradient_steps",               args.gradient_steps,                   model.gradient_steps),
        ("learning_starts",              args.learning_starts,                  model.learning_starts),
        ("gamma",                        ARM_C["gamma"],                        model.gamma),
        ("learning_rate",                ARM_C["learning_rate"],                model.learning_rate),
        ("batch_size",                   ARM_C["batch_size"],                   model.batch_size),
        ("buffer_size",                  ARM_C["buffer_size"],                  model.buffer_size),
        ("ent_coef",                     ARM_C["ent_coef"],                     model.ent_coef),
        ("train_freq",                   (ARM_C["train_freq"], "step"),
         (model.train_freq.frequency, model.train_freq.unit.value)),
        ("n_envs",                       args.n_envs,                           model.n_envs),
        ("venv.num_envs",                args.n_envs,                           venv.num_envs),
        # The crash-3 check: everything above can be right and this still wrong.
        ("replay_buffer.n_envs",         args.n_envs,                           rb.n_envs),
        ("replay_buffer.buffer_size",    ARM_C["buffer_size"] // args.n_envs,   rb.buffer_size),
        ("rb.optimize_memory_usage",     False,                                 rb.optimize_memory_usage),
        ("rb.handle_timeout_termination", True,                                 rb.handle_timeout_termination),
        ("rb.observation_space",         model.observation_space,               rb.observation_space),
        ("rb.action_space",              model.action_space,                    rb.action_space),
        ("total_timesteps (absolute)",   args.total_timesteps,
         model.num_timesteps + learn_budget),
    ]

    print("-" * 78, flush=True)
    print("ACTUALS READ BACK OFF THE LIVE MODEL (not what was asked for, what it is)",
          flush=True)
    print("-" * 78, flush=True)
    bad = []
    for name, want, got in rows:
        same = want == got
        if not same:
            bad.append((name, want, got))
        w, g = str(want), str(got)
        if len(w) > 24:
            w, g = w[:21] + "...", g[:21] + "..."
        print(f"  {'ok ' if same else '!! '} {name:32s} intent {w:<26s}"
              + ("" if same else f" ACTUAL {g}"), flush=True)

    # train_freq=1 fires gradient_steps updates per vec-step regardless of worker
    # count, so n_envs and gradient_steps are one decision.
    ratio = model.gradient_steps / model.n_envs
    arm_c = ARM_C["gradient_steps"] / 5          # gs=16 at the 5 workers arm C ran
    dev = ratio / arm_c - 1
    print(f"      {'updates/sample':32s} {ratio:.2f}  (arm C {arm_c:.2f}, {dev:+.0%})"
          + ("  <-- !! DIFFERENT ARM, not a shorter arm C" if abs(dev) > 0.10
             else "  within 10%, comparable"), flush=True)

    held = rb.size() * rb.n_envs
    print(f"      {'transitions in buffer':32s} {held}", flush=True)
    print(f"      {'model.num_timesteps':32s} {model.num_timesteps}", flush=True)
    if held != model.num_timesteps:
        # Not an error: zip and buffer are written by different callbacks, so the
        # counter can lag. Quote the run's sample count from the buffer.
        print(f"      {'^ offset':32s} {held - model.num_timesteps:+d} "
              f"(buffer vs counter, quote the buffer)", flush=True)
    print(f"      {'samples still to collect':32s} {learn_budget}", flush=True)
    print("-" * 78, flush=True)

    if bad:
        print("\nREFUSING TO TRAIN. The live model does not match intent:", flush=True)
        for name, want, got in bad:
            print(f"  {name}: intent {want!r}, actual {got!r}", flush=True)
        sys.exit(f"{len(bad)} hyperparameter mismatch(es); nothing has been trained.")
    return rows, ratio


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n_envs", type=int, default=5)
    ap.add_argument("--total_timesteps", type=int, default=8000)
    ap.add_argument("--learning_starts", type=int, default=5000)
    ap.add_argument("--probe_every", type=int, default=250)
    ap.add_argument("--base_port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--ues", type=int, default=None)
    ap.add_argument("--simtime", type=int, default=None)
    ap.add_argument("--min_free_gb", type=float, default=MIN_FREE_GB)
    # Ceiling is our own anon (the incompressible part); the floor is system
    # MemAvailable, which protects the machine from a hog that is not us.
    ap.add_argument("--mem_anon_ceiling_gb", type=float, default=MEM_ANON_CEILING_GB)
    ap.add_argument("--min_avail_gb", type=float, default=MIN_AVAIL_GB)
    # updates/sample = gradient_steps / n_envs, so changing n_envs alone silently
    # moves the ratio off arm C's 3.20. Not a free knob: gs=64 scored below
    # doing nothing.
    ap.add_argument("--gradient_steps", type=int, default=ARM_C["gradient_steps"])
    ap.add_argument("--rb_save_every", type=int, default=500,
                    help="save the replay buffer every N aggregate timesteps")
    ap.add_argument("--resume_from", default=None,
                    help="path prefix of a saved model to resume; its replay "
                         "buffer is loaded from <prefix>_replay.pkl if present")
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--save_dir", default=path.join(path.dirname(__file__), "..", "output", "sac_pilot"))
    ap.add_argument("--probe_obs", default=path.join(
        path.dirname(__file__), "..", "analysis", "mlb_session_2026-08-31", "probe_obs.npy"))
    ap.add_argument("--log_dir", default=None)
    args = ap.parse_args()

    log_dir = args.log_dir or path.join(args.save_dir, "logs_" + time.strftime("%Y%m%d_%H%M%S"))
    makedirs(log_dir, exist_ok=True)
    makedirs(args.save_dir, exist_ok=True)

    # MemGuard's emergency checkpoint lands in save_dir, so prove it is writable
    # now rather than finding out during an out-of-memory stop.
    _probe = path.join(args.save_dir, ".write_test")
    try:
        with open(_probe, "w") as fh:
            fh.write("ok")
        os.remove(_probe)
    except OSError as exc:
        sys.exit(f"save_dir is not writable: {args.save_dir} ({exc}). "
                 f"MemGuard could not checkpoint on an out-of-memory stop.")

    cfg = dict(CFG)
    if args.ues is not None:
        cfg["ues"] = [args.ues]
    if args.simtime is not None:
        cfg["simTime"] = [args.simtime]

    if not path.exists(args.probe_obs):
        sys.exit(f"probe batch missing: {args.probe_obs}\n"
                 f"build it with: python analysis/mlb_session_2026-08-31/build_probe_obs.py")

    busy = [p for p in range(args.base_port, args.base_port + args.n_envs) if not port_is_free(p)]
    if busy:
        sys.exit(f"ports already in use: {busy}. Another run is holding them. "
                 f"Use --base_port to move out of the way.")

    free_gb = shutil.disk_usage(args.output_folder).free / 1e9
    if free_gb < args.min_free_gb:
        sys.exit(f"only {free_gb:.2f} GB free at {args.output_folder}, "
                 f"floor is {args.min_free_gb} GB. Refusing to start.")

    hypers = dict(ARM_C, learning_starts=args.learning_starts,
                  gradient_steps=args.gradient_steps)
    updates_per_sample = args.gradient_steps / args.n_envs
    ARM_C_RATIO = ARM_C["gradient_steps"] / 5     # gs=16 at the 5 workers arm C ran
    expected_updates = int(max(args.total_timesteps - args.learning_starts, 0) * updates_per_sample)

    print("=" * 78, flush=True)
    print("SAC PILOT ON REAL ns-3: does the critic climb?", flush=True)
    print("=" * 78, flush=True)
    print(f"  samples          {args.total_timesteps} ABSOLUTE target across "
          f"{args.n_envs} workers = {args.total_timesteps // args.n_envs}/worker", flush=True)
    if args.resume_from:
        print(f"  ** RESUMING **   the numbers below are INTENT. What the model "
              f"actually loaded is printed after it is built, and any mismatch "
              f"aborts the run before training.", flush=True)
    print(f"  simulated time   {args.total_timesteps * cfg['controlInterval'][0]:.0f} s aggregate, "
          f"{args.total_timesteps * cfg['controlInterval'][0] / args.n_envs:.0f} s/worker", flush=True)
    print(f"  T_control        {cfg['controlInterval'][0]} s", flush=True)
    print(f"  learning_starts  {args.learning_starts}  (production is 25000, see docstring)", flush=True)
    print(f"  gradient budget  {updates_per_sample:.2f} updates/sample "
          f"-> ~{expected_updates} updates (PPO pilot: 130 total)", flush=True)
    print(f"  vs arm C         {ARM_C_RATIO:.2f} updates/sample "
          f"({100*(updates_per_sample/ARM_C_RATIO-1):+.0f}%)"
          + ("  <-- !! DIFFERENT ARM, not a shorter arm C" 
             if abs(updates_per_sample/ARM_C_RATIO-1) > 0.10 else "  (within 10%, comparable)"),
          flush=True)
    print(f"  gamma            {ARM_C['gamma']}  "
          f"(horizon {1/(1-ARM_C['gamma']):.1f} steps = "
          f"{1/(1-ARM_C['gamma'])*cfg['controlInterval'][0]:.1f} s)", flush=True)
    print(f"  seed             {args.seed}   ports {args.base_port}..{args.base_port+args.n_envs-1}", flush=True)
    print(f"  free disk        {free_gb:.2f} GB (floor {args.min_free_gb} GB)", flush=True)
    print(f"  mem guard        anon ceiling {args.mem_anon_ceiling_gb} GB, "
          f"MemAvailable floor {args.min_avail_gb} GB", flush=True)
    print(f"  replay buffer    saved every {args.rb_save_every} timesteps", flush=True)
    print(f"  logs             {log_dir}", flush=True)
    print("=" * 78, flush=True)

    with open(path.join(log_dir, "run_config.json"), "w") as fh:
        json.dump({"argv": sys.argv, "cfg": cfg, "hypers": hypers,
                   "n_envs": args.n_envs, "seed": args.seed,
                   "total_timesteps": args.total_timesteps,
                   "updates_per_sample": updates_per_sample,
                   "expected_gradient_updates": expected_updates,
                   "probe_obs": path.abspath(args.probe_obs),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "ppo_pilot_baseline": "AGENT_BUILD_LOG.md Step 11a-11c"},
                  fh, indent=2)

    print("building ns-3 ONCE in the parent (workers skip it) ...", flush=True)
    t0 = time.time()
    probe_env = MlbZmqEnv(ns3_path=args.ns3_path, scenario_configuration=dict(cfg),
                          output_folder=args.output_folder, optimized=False,
                          zmq_port=args.base_port, build_ns3=True)
    print(f"  build/lookup done in {time.time()-t0:.1f}s -> {probe_env.script_executable}", flush=True)
    del probe_env

    venv = SubprocVecEnv([make_env(i, args, cfg, log_dir) for i in range(args.n_envs)],
                         start_method="spawn")
    print(f"{args.n_envs} workers up on ports "
          f"{args.base_port}..{args.base_port + args.n_envs - 1}", flush=True)

    if args.resume_from:
        # Off-policy resume: the buffer is the run. reset_num_timesteps=False
        # below keeps the step counter, so learning_starts still means what it did.
        print(f"resuming from {args.resume_from} ...", flush=True)
        # Defect A fix. load applies __dict__.update(data) then update(kwargs)
        # (base_class.py:738-739), so passing hypers here is what stops the
        # checkpoint's pickled values from winning.
        model = SAC.load(args.resume_from, env=venv, device="cpu", **hypers)
        rb = args.resume_from + "_replay.pkl"
        if not path.exists(rb):
            # Step 15e: this used to warn and continue with an empty buffer.
            sys.exit(f"no replay buffer at {rb}. An off-policy resume without "
                     f"its buffer would discard every collected transition. "
                     f"Refusing. (The buffer file must be named "
                     f"'<prefix>_replay.pkl' next to '<prefix>.zip'.)")
        model.load_replay_buffer(rb)
        print(f"  replay buffer restored: {model.replay_buffer.size()} rows x "
              f"{model.replay_buffer.n_envs} lanes = "
              f"{model.replay_buffer.size() * model.replay_buffer.n_envs} transitions",
              flush=True)
        print(f"  resumed at {model.num_timesteps} timesteps", flush=True)
        # Defect B fix. _setup_learn treats total_timesteps as a remainder when
        # reset_num_timesteps=False (base_class.py:416); convert to absolute.
        learn_budget = args.total_timesteps - model.num_timesteps
        if learn_budget <= 0:
            sys.exit(f"--total_timesteps {args.total_timesteps} is already reached: "
                     f"the checkpoint is at {model.num_timesteps} timesteps. "
                     f"Raise --total_timesteps or there is nothing to do.")
    else:
        model = SAC("MlpPolicy", venv, seed=args.seed, verbose=1, device="cpu", **hypers)
        learn_budget = args.total_timesteps

    # save_freq counts vec-steps, so this is every 1000 aggregate timesteps.
    ckpt = CheckpointCallback(save_freq=max(1000 // args.n_envs, 1),
                              save_path=args.save_dir, name_prefix="mlb_sac")
    steplog = StepLogger(path.join(log_dir, "reward_terms_steps.csv"), verbose=1)
    critic = CriticProbe(path.join(log_dir, "critic_probe.csv"), args.probe_obs,
                         every=args.probe_every, output_folder=args.output_folder,
                         verbose=1)
    guard = DiskGuard(args.output_folder, args.min_free_gb, verbose=1)
    memguard = MemGuard(args.mem_anon_ceiling_gb, args.min_avail_gb,
                        checkpoint_dir=args.save_dir, verbose=1)
    rbsaver = ReplayBufferSaver(path.join(args.save_dir, "mlb_sac_replay_live"),
                                args.rb_save_every, verbose=1)
    if memguard.cg is None:
        print("  !! MemGuard: own cgroup not resolvable, the anon ceiling is "
              "INACTIVE, only the MemAvailable floor will fire.", flush=True)
    else:
        print(f"  MemGuard cgroup  {memguard.cg}", flush=True)

    # After the buffer is loaded: replay_buffer.n_envs is only wrong once the
    # pickle is in place.
    actual_rows, updates_per_sample_actual = assert_actuals(model, venv, args, learn_budget)
    with open(path.join(log_dir, "run_actuals.json"), "w") as fh:
        json.dump({"checked": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "resumed_from": args.resume_from,
                   "learn_budget": int(learn_budget),
                   "total_timesteps_absolute": int(args.total_timesteps),
                   "num_timesteps_at_start": int(model.num_timesteps),
                   "transitions_in_buffer": int(model.replay_buffer.size()
                                                * model.replay_buffer.n_envs),
                   "updates_per_sample": updates_per_sample_actual,
                   "arm_c_updates_per_sample": ARM_C["gradient_steps"] / 5,
                   "fields": [{"name": n, "intent": str(w), "actual": str(g)}
                              for n, w, g in actual_rows]}, fh, indent=2)

    status = "COMPLETE"
    try:
        model.learn(total_timesteps=learn_budget,
                    reset_num_timesteps=not bool(args.resume_from),
                    callback=CallbackList([ckpt, steplog, critic, guard,
                                           memguard, rbsaver]),
                    log_interval=1, progress_bar=False)
        # A callback returning False makes learn() return normally, so a guard trip
        # would otherwise be recorded as COMPLETE.
        if memguard.tripped:
            status = f"STOPPED_EARLY_MEM: {memguard.reason}"
        elif guard.tripped:
            status = "STOPPED_EARLY_DISK"
        elif model.num_timesteps < args.total_timesteps:
            status = f"STOPPED_EARLY: {model.num_timesteps}/{args.total_timesteps}"
    except BaseException as exc:                     # noqa: BLE001 - record then re-raise
        status = f"FAILED: {type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            model.save(path.join(args.save_dir, "mlb_sac_final"))
            # Holds every transition of the run, which the post-hoc
            # explained-variance check needs: SB3's SAC logs no EV of its own.
            model.save_replay_buffer(path.join(args.save_dir, "mlb_sac_replay"))
        finally:
            venv.close()
            with open(path.join(log_dir, "run_status.json"), "w") as fh:
                json.dump({"status": status,
                           "n_envs": args.n_envs,
                           "gradient_steps": int(model.gradient_steps),
                           "updates_per_sample": model.gradient_steps / args.n_envs,
                           "total_timesteps_absolute": int(args.total_timesteps),
                           "transitions_in_buffer": int(model.replay_buffer.size()
                                                        * model.replay_buffer.n_envs),
                           "disk_guard_tripped": guard.tripped,
                           "mem_guard_tripped": memguard.tripped,
                           "mem_guard_reason": memguard.reason,
                           "peak_cgroup_anon_gb": round(memguard.peak_anon_gb, 3),
                           "memguard_checkpoint_dir": memguard.checkpoint_dir,
                           "memguard_checkpoint_written": memguard.checkpoint_written,
                           "replay_buffer_saves": rbsaver.saves,
                           "resumed_from": args.resume_from,
                           "timesteps_done": int(model.num_timesteps),
                           "gradient_updates": int(getattr(model, "_n_updates", 0)),
                           "probe_rows": critic.rows_written,
                           "step_rows": steplog.rows_written,
                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)
    print(f"SAC PILOT {status}", flush=True)
