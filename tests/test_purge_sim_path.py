"""The episode-boundary disk purge: MlbZmqEnv._purge_sim_path and close().

ns-3 writes ~180 MB per 30 s episode and a run crosses a boundary every 300 steps
per worker, so a purge that silently stops working fills the disk hours in.
test_mlb_zmq_env.py's stub overrides close() and so never reaches
_purge_sim_path; this one stubs only the ns-3 process and the ZMQ socket, and
runs the real close().

Run:  python tests/test_purge_sim_path.py
"""
import os
import sys
import shutil
import tempfile
import uuid
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mlb_zmq_env import MlbZmqEnv          # noqa: E402

CFG = {"ues": [3], "simTime": [10], "RngRun": [555]}
_FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f": {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


class _FakeProc:
    """Stands in for the ns-3 Popen. close() kills and reaps it."""
    def __init__(self):
        self.killed = self.waited = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        return 0


class _FakeZmqDb:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _PurgeStubEnv(MlbZmqEnv):
    """Only setup_sim/start_sim are stubbed; close() is the real one."""
    def setup_sim(self):
        self.environment = {}
        self.script_executable = "/bin/true"

    def start_sim(self):
        # Mirror the real start_sim: a uuid4 dir directly inside output_folder.
        self.sim_path = path.join(self.output_folder, str(uuid.uuid4()))
        os.makedirs(self.sim_path, exist_ok=True)
        with open(path.join(self.sim_path, "du-cell-2.txt"), "w") as fh:
            fh.write("x" * 4096)
        self.is_open = True
        self.sim_process = _FakeProc()
        self.zmq_db = _FakeZmqDb()


def make_env(tmp, purge):
    return _PurgeStubEnv(ns3_path="/unused", scenario_configuration=dict(CFG),
                         output_folder=tmp, optimized=False,
                         purge_sim_path_on_close=purge)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        env = make_env(tmp, purge=True)
        env.start_sim()
        p = env.sim_path
        check("test_run_dir_exists_before_close", path.isdir(p))
        env.close()
        check("test_close_purges_the_run_dir", not path.exists(p), f"{p} survived")
        check("test_close_reaped_the_child", env.sim_process.killed and env.sim_process.waited)
        check("test_close_closed_the_socket", env.zmq_db is None)

    # Mutation proof: with purging off the dir must survive, or the check above
    # could be satisfied by the TemporaryDirectory teardown instead.
    with tempfile.TemporaryDirectory() as tmp:
        env = make_env(tmp, purge=False)
        env.start_sim()
        p = env.sim_path
        env.close()
        check("test_purge_off_leaves_the_run_dir", path.isdir(p),
              f"{p} was deleted with purging DISABLED")

    # A training run never calls close() explicitly, reset() does it once per
    # episode. Crossing episodes must not accumulate directories.
    with tempfile.TemporaryDirectory() as tmp:
        env = make_env(tmp, purge=True)
        seen = []
        for _ in range(5):
            env.close()          # what reset() does before start_sim()
            env.start_sim()
            seen.append(env.sim_path)
        live = [d for d in os.listdir(tmp)]
        check("test_episode_boundaries_do_not_accumulate_dirs",
              len(live) == 1 and live[0] == path.basename(seen[-1]),
              f"{len(live)} dirs left after 5 episodes: {live}")
        check("test_all_but_current_were_purged",
              all(not path.exists(d) for d in seen[:-1]))
        env.close()

    # the guards: each refuses, and leaves the directory alone
    with tempfile.TemporaryDirectory() as tmp:
        # (a) a name that is not a uuid4
        env = make_env(tmp, purge=True)
        env.start_sim()
        bad = path.join(tmp, "not-a-uuid")
        os.makedirs(bad, exist_ok=True)
        env.sim_path = bad
        env._purge_sim_path()
        check("test_guard_refuses_non_uuid_name", path.isdir(bad))
        shutil.rmtree(bad, ignore_errors=True)

        # (b) a uuid dir that is NOT directly inside output_folder
        outside = tempfile.mkdtemp()
        nested = path.join(outside, str(uuid.uuid4()))
        os.makedirs(nested, exist_ok=True)
        env.sim_path = nested
        env._purge_sim_path()
        check("test_guard_refuses_dir_outside_output_folder", path.isdir(nested))
        shutil.rmtree(outside, ignore_errors=True)

        # (c) a symlink pointing at a real uuid dir inside output_folder
        real = path.join(tmp, str(uuid.uuid4()))
        os.makedirs(real, exist_ok=True)
        link = path.join(tmp, str(uuid.uuid4()) + "-link")
        os.symlink(real, link)
        env.sim_path = link
        env._purge_sim_path()
        check("test_guard_refuses_symlink", path.isdir(real) and path.islink(link))
        os.unlink(link)
        shutil.rmtree(real, ignore_errors=True)

        # (d) sim_path unset
        env.sim_path = None
        try:
            env._purge_sim_path()
            check("test_guard_tolerates_sim_path_none", True)
        except Exception as exc:
            check("test_guard_tolerates_sim_path_none", False, repr(exc))

    with tempfile.TemporaryDirectory() as tmp:
        env = make_env(tmp, purge=True)
        env.start_sim()
        env.close()
        try:
            env.close()
            check("test_double_close_is_safe", True)
        except Exception as exc:
            check("test_double_close_is_safe", False, repr(exc))

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {_FAILURES}")
        sys.exit(1)
    print("all purge tests passed")


if __name__ == "__main__":
    main()
