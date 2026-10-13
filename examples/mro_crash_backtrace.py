"""Reproduce one mid-run crash under gdb to get a real backtrace.

The post-sim-baseline-merge do-nothing baseline (2026-10-09) crashed on all
5/5 seeds with empty stdout/stderr -- "no diagnosable cause from the logs
alone", the same silent-exit signature documented since the original round-2
baseline, except now 100% of seeds hit it (was 40%) and the worst case lost
88% of its episode (was 40%). That regression coincides exactly with this
session's merge of Ra'ed's sim-baseline branch, so the new code is the prime
suspect, not a pre-existing flaky seed.

One concrete, testable hypothesis going in: LteEnbRrc::CheckCioHelperMatchesArgmax
contains a hard NS_ABORT_MSG_IF(candidate > best, ...) -- a real abort(), not
a log line -- gated on `m_handoverHysteresisDb > 0.0 || m_tttFromBiasedSinr`.
MRO's CFG sets handoverHysteresisDb=0 but tttFromBiasedSinr=1, so that `||`
makes the check run on every TttBasedHandover call regardless. If
GetCioBiasedSinrDb ever disagrees with the inline CIO-biased argmax that
picked a cell, ns-3 aborts -- which would look exactly like "exited with an
error, empty stdout/stderr" from the Python side if the abort message is
written to a buffered stream the subprocess pipe never flushes before dying.
Not assumed true -- this script exists to check it against a real backtrace,
not to confirm it by assertion.

Does NOT modify MroZmqEnv: subclasses it and overrides only start_sim(), to
wrap the real launch command in `gdb -batch -ex run -ex bt -ex quit` instead
of a bare subprocess. Everything else (the ZeroMQ server, the do-nothing
action loop) is the real, already-validated code path, so the simulation
gets real traffic and should crash at the same natural point gdb or no gdb.

Usage:
    python examples/mro_crash_backtrace.py --seed 2001
"""
import argparse
import os
import subprocess
import sys
import time
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import CELLS, MroZmqEnv

import numpy as np

CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [555],
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


class _DirectCaptureMroZmqEnv(MroZmqEnv):
    """Identical to MroZmqEnv.start_sim(), except stdout/stderr are
    redirected straight to files on disk via a shell wrapper, bypassing
    this project's own PIPE + selector-based capture entirely.

    Why: two real crashes (the original overnight run and a reproduction
    via this same script) both showed completely empty stdout/stderr
    through the normal capture path, despite ns-3's own NS_ABORT_MSG_IF
    documenting that its default std::terminate handler flushes streams
    before aborting -- so a real message is expected to exist. Before
    concluding the message is genuinely never written, rule out the
    possibility that THIS project's own pipe-reading (read_streams(), the
    selector-based capture in ns_env.py) is dropping output written very
    close to process death -- a plausible, separate bug in its own right.
    CONFIRMED: this is exactly what was happening -- direct capture
    surfaced the real NS_ASSERT message the PIPE path was losing.

    GDB_WRAP = True additionally wraps the real binary in a gdb batch
    session for a full backtrace. Combined with direct capture because the
    first gdb attempt (no direct capture) was misread: gdb -batch's OWN
    exit code is independent of whether the INFERIOR crashed -- plain
    `quit` after a signal-stopped process still returns 0 from gdb itself.
    self.sim_process IS the gdb process here, so ns3_exit_code in that
    earlier run was gdb's exit code, not the inferior's, and gdb's own
    backtrace text would have gone through the same lossy PIPE capture
    this class works around. Both fixed at once here.
    """

    GDB_WRAP = False

    def start_sim(self):
        if self.is_open:
            raise ValueError('The environment is open and a new start_sim has been called.')
        self.is_open = True

        import selectors
        import uuid

        parameters = self.scenario_configuration
        self.sim_result = {'params': dict(parameters), 'meta': {}}
        sim_uuid = str(uuid.uuid4())
        self.sim_result['meta']['id'] = sim_uuid
        self.sim_path = os.path.join(self.output_folder, sim_uuid)
        os.makedirs(self.sim_path)

        from ns_o_ran_gym.bridge.zmq_database import ZmqStateDatabase
        self.zmq_db = ZmqStateDatabase(port=self.zmq_port, history_maxlen=self.history_maxlen)
        self.zmq_db.start()
        self._pending_kpis = None
        self._handover_history = []
        self._last_ue_sinr = {}
        self._last_action = None

        self.direct_stdout_path = os.path.join(self.sim_path, "direct_stdout.txt")
        self.direct_stderr_path = os.path.join(self.sim_path, "direct_stderr.txt")
        real_command = [self.script_executable] + [f'--{k}={v}' for k, v in parameters.items()]
        if self.GDB_WRAP:
            real_command = ["gdb", "-batch",
                            "-ex", "run",
                            "-ex", "thread apply all bt full",
                            "-ex", "quit"] + ["--args"] + real_command
        quoted = " ".join(f"'{c}'" for c in real_command)
        shell_cmd = f"{quoted} > '{self.direct_stdout_path}' 2> '{self.direct_stderr_path}'"
        command = ["bash", "-c", shell_cmd]
        print(f"  [direct-capture{'+gdb' if self.GDB_WRAP else ''} launch] {shell_cmd}", flush=True)
        self.sim_result['meta']['start_time'] = time.time()
        self.sim_process = subprocess.Popen(command, cwd=self.sim_path, env=self.environment,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.selector = selectors.DefaultSelector()
        self._set_nonblocking(self.sim_process.stdout)
        self._set_nonblocking(self.sim_process.stderr)
        self.selector.register(self.sim_process.stdout, selectors.EVENT_READ)
        self.selector.register(self.sim_process.stderr, selectors.EVENT_READ)


class _GdbDirectCaptureMroZmqEnv(_DirectCaptureMroZmqEnv):
    """_DirectCaptureMroZmqEnv with the gdb batch wrapper turned on -- see
    that class's docstring for why both fixes (direct file capture AND gdb)
    are needed together, not either alone. Supersedes the first attempt at
    a gdb-wrapped env, which wrapped gdb around a bare subprocess (no
    direct capture) and misread gdb's OWN exit code as the inferior's.
    """
    GDB_WRAP = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2001,
                    help="2001 crashed earliest (36/299 steps) in the triggering "
                         "run, so it should reproduce fastest under gdb too")
    ap.add_argument("--simtime", type=int, default=30)
    ap.add_argument("--zmq_port", type=int, default=5555)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--plain", action="store_true",
                    help="skip the gdb wrapper entirely and run the real, "
                         "unmodified MroZmqEnv -- use this when gdb itself "
                         "perturbs the crash (ptrace overhead changing "
                         "timing-sensitive behaviour). Still prints the real "
                         "ns3_exit_code on the way out, which is enough to "
                         "tell a signal (segfault/abort) from a deliberate "
                         "nonzero exit() even without a backtrace.")
    ap.add_argument("--direct", action="store_true",
                    help="redirect stdout/stderr straight to files on disk "
                         "via a shell wrapper, bypassing this project's own "
                         "PIPE-based capture -- use this to rule out a "
                         "harness-side capture bug if --plain shows an "
                         "empty stderr despite a signal that should print "
                         "a message (e.g. SIGABRT). CONFIRMED needed: the "
                         "normal capture path genuinely loses this output.")
    ap.add_argument("--gdb", action="store_true",
                    help="wrap the binary in a gdb batch session for a "
                         "full backtrace. Only meaningful together with "
                         "--direct: gdb's own stdout (including the "
                         "backtrace) goes through the same lossy PIPE path "
                         "otherwise, AND gdb -batch's own exit code is "
                         "independent of whether the inferior crashed, so "
                         "ns3_exit_code is meaningless here -- read the "
                         "captured files, not the printed exit code.")
    args = ap.parse_args()

    cfg = dict(CFG)
    cfg["RngRun"] = [args.seed]
    cfg["simTime"] = [args.simtime]

    if args.gdb and not args.direct:
        sys.exit("--gdb requires --direct (gdb's own output needs the direct-capture "
                 "file redirection too, or it is lost the same way the inferior's was).")

    env_cls = (_GdbDirectCaptureMroZmqEnv if (args.direct and args.gdb)
              else _DirectCaptureMroZmqEnv if args.direct
              else MroZmqEnv if args.plain
              else MroZmqEnv)
    env = env_cls(ns3_path=args.ns3_path, scenario_configuration=cfg,
                  output_folder=args.output_folder, optimized=False,
                  zmq_port=args.zmq_port, build_ns3=False,
                  vary_rng_run_per_episode=False,
                  purge_sim_path_on_close=False)  # keep the sim_path for inspection either way

    action = np.zeros(len(CELLS), dtype=np.float32)
    step_idx = 0
    try:
        obs, info = env.reset(seed=args.seed)
        print(f"reset ok, rng_run={info.get('rng_run')}, sim_path={env.sim_path}", flush=True)
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            if "reward_terms" not in info:
                exit_code = info.get('ns3_exit_code')
                crashed = info.get('crashed')
                # A negative returncode from Python's subprocess means the
                # process was killed by signal -N (SIGSEGV=11, SIGABRT=6,
                # SIGBUS=7, ...), not a deliberate exit() call -- the single
                # most useful fact available without a backtrace.
                signal_note = (f" (killed by signal {-exit_code}"
                              + {11: " -- SIGSEGV, a real memory-safety bug",
                                 6: " -- SIGABRT, likely an NS_ABORT_MSG_IF "
                                    "assertion or similar deliberate abort()",
                                 7: " -- SIGBUS, a misaligned/invalid memory access"}
                                .get(-exit_code, "") + ")"
                              if isinstance(exit_code, int) and exit_code < 0 else "")
                gdb_caveat = (" (meaningless under --gdb -- that's gdb's own exit "
                             "code, not the inferior's; read the captured files below)"
                             if args.gdb else "")
                print(f"\nEpisode ended at step {step_idx} -- "
                      f"ns3_exit_code={exit_code}, crashed={crashed}{signal_note}{gdb_caveat}",
                      flush=True)
                break
            step_idx += 1
            if step_idx % 20 == 0:
                print(f"  step {step_idx} ok", flush=True)
            if terminated or truncated:
                break
    except Exception as exc:
        print(f"\nPython-side exception at step {step_idx}: {exc!r}", flush=True)
    finally:
        sim_path = getattr(env, "sim_path", None)
        env.close()

    if args.direct and sim_path:
        for name in ("direct_stdout.txt", "direct_stderr.txt"):
            fp = path.join(sim_path, name)
            try:
                with open(fp) as fh:
                    content = fh.read()
            except OSError as exc:
                content = f"<could not read: {exc}>"
            print(f"\n=== {name} ({len(content)} bytes) ===")
            print(content if content else "(empty)")

    print(f"\nFull gdb-wrapped process output is in: {env.sim_path} "
          f"(purge_sim_path_on_close=False, so the directory survives)", flush=True)
