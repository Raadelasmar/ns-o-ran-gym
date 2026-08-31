"""Gym-API and reward/observation checks for MlbZmqEnv, WITHOUT ns-3 or ZeroMQ.

MlbZmqEnv normally builds and launches ns-3 (NsOranEnv.setup_sim) and binds a
real ZMQ REP socket (bridge/zmq_database.py). Neither is available in a unit
test, so _StubMlbZmqEnv replaces exactly two seams -- setup_sim() and the
start_sim()/recv/close plumbing -- with a deterministic synthetic KPI source in
the payload shape scenario-marl-zmq.cc actually sends (see tests/test_server.py).
Everything under test (spaces, _get_obs, _cio_from_action, _reward_terms, the
step()/reset() contract) is the real code path.

Run:  pytest tests/test_mlb_zmq_env.py      or      python tests/test_mlb_zmq_env.py
"""
import sys
import warnings
from os import path

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mlb_zmq_env import (BUFFER_SCALE_BYTES, CELLS, CIO_LIMIT_DB, KPI_WINDOW_S,
                                      OBS_FIELDS, REWARD_FORMULA_VERSION, SINR_FLOOR_DB,
                                      SINR_SCALE_DB, VOLUME_SCALE_BYTES, MlbZmqEnv, obs_index)

CFG = {"ues": [3], "simTime": [10], "RngRun": [555]}

EPISODE_STEPS = 6

# check_env's own warnings that are expected here and not defects:
#  - the action space is CIO in dB, deliberately +/-6 rather than +/-1 (an SB3
#    user normalises with a wrapper; the env must speak the physical unit).
CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "For Box action spaces, we recommend using a symmetric and normalized space "
        "(range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/"
        "guide/rl_tips.html for more information.",
        "Casting input x to numpy array.",
    ]
]


class _FakeZmqDb:
    """Stand-in for ZmqStateDatabase: records what step() sends, no socket."""

    def __init__(self):
        self.kpi_history = []
        self.sent = []

    def send_control_actions(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


def _snapshot(step: int) -> dict:
    """One synthetic KPI snapshot, deterministic in `step`.

    Exercises the awkward cases on purpose:
      - step 2 drops cell 8 entirely            -> zero-filled obs block
      - cell 7 always has zero UEs and no SINR  -> SINR floor, not 0 dB
    """
    cells = {}
    for i, cell in enumerate(CELLS):
        if step == 2 and cell == 8:
            continue
        n_ues = 0 if cell == 7 else (i + step) % 4
        cells[str(cell)] = {
            "prb_utilization": min(0.05 * i + 0.1 * (step % 3) + 0.2, 1.0),
            "buffer_bytes": 1000.0 * i + 500.0 * step,
            "volume_bytes": 20000.0 * (i + 1) + 1000.0 * step,
            "num_active_ues": n_ues,
            "ues": {str(1000 + 10 * cell + u): {"l3_serving_sinr_db": 5.0 + i - u}
                    for u in range(n_ues)},
        }
    return {"timestamp": float(step), "cells": cells}


class _StubMlbZmqEnv(MlbZmqEnv):
    def setup_sim(self):
        # Bypass configure_and_build_ns3() + the build-status lookup.
        self.environment = {}
        self.script_executable = "/bin/true"

    def start_sim(self):
        self.is_open = True
        self.sim_path = "/tmp/stub-mlb-zmq"
        self.zmq_db = _FakeZmqDb()
        self._pending_kpis = None
        self._step_count = 0

    def _recv_next_kpis(self):
        if self._step_count >= EPISODE_STEPS:
            return None
        kpis = _snapshot(self._step_count)
        self.zmq_db.kpi_history.append(kpis)
        self._step_count += 1
        return kpis

    def is_simulation_over(self):
        return self._step_count >= EPISODE_STEPS

    def close(self):
        if self.is_open:
            if self.zmq_db is not None:
                self.zmq_db.close()
                self.zmq_db = None
            self.is_open = False


def make_env() -> _StubMlbZmqEnv:
    return _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                          output_folder="/tmp", optimized=False)


def test_check_env():
    """The Gymnasium API contract, via the official checker."""
    env = make_env()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    env.close()

    unexpected = [str(w.message) for w in caught
                  if str(w.message) not in CHECK_ENV_IGNORE_WARNINGS]
    if unexpected:
        raise gym.error.Error(f"Unexpected warnings: {unexpected}")


def test_obs_matches_space():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray) and obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape == (len(CELLS) * len(OBS_FIELDS),)
    assert env.observation_space.contains(obs)

    # Values land where the layout says they do, DIVIDED BY THEIR FIXED SCALE
    # (see the observation-scaling block in mlb_zmq_env: the raw fields span
    # seven orders of magnitude and MlpPolicy does not normalise).
    snap = _snapshot(0)["cells"]
    assert obs[obs_index(2, "prb_utilization")] == np.float32(snap["2"]["prb_utilization"])
    assert obs[obs_index(3, "num_active_ues")] == np.float32(
        snap["3"]["num_active_ues"] / env.total_ues)
    assert obs[obs_index(2, "buffer_bytes")] == np.float32(
        snap["2"]["buffer_bytes"] / BUFFER_SCALE_BYTES)
    assert obs[obs_index(2, "volume_bytes")] == np.float32(
        snap["2"]["volume_bytes"] / VOLUME_SCALE_BYTES)
    # Cell 7 has no UEs -> no SINR report -> floor, not a healthy-looking 0 dB.
    assert obs[obs_index(7, "sinr_db_mean")] == np.float32(SINR_FLOOR_DB / SINR_SCALE_DB)

    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


def test_absent_cell_is_zero_filled():
    """An absent cell reads zero on the four load fields and FLOOR on SINR.

    The SINR slot deliberately does NOT read 0: 0 dB is a perfectly good link,
    so zero-filling it made a cell that is not reporting at all look healthier
    than one reporting a weak signal. Absent and present-but-silent now agree.
    """
    env = make_env()
    env.reset(seed=0)
    env.step(env.action_space.sample())          # consumes snapshot 1
    obs, *_ = env.step(env.action_space.sample())  # snapshot 2: cell 8 missing
    for field in ("prb_utilization", "num_active_ues", "buffer_bytes", "volume_bytes"):
        assert obs[obs_index(8, field)] == 0.0, field
    assert obs[obs_index(8, "sinr_db_mean")] == np.float32(SINR_FLOOR_DB / SINR_SCALE_DB)
    env.close()


def test_action_is_consumed_and_clipped():
    env = make_env()
    env.reset(seed=0)
    action = np.linspace(-20.0, 20.0, len(CELLS), dtype=np.float32)   # out of range on purpose
    _, _, _, _, info = env.step(action)

    sent = env.zmq_db.sent[-1]["cells"]
    assert set(sent) == {str(c) for c in CELLS}
    expected = np.clip(action, -CIO_LIMIT_DB, CIO_LIMIT_DB)
    for i, cell in enumerate(CELLS):
        assert sent[str(cell)]["cio_offset"] == float(expected[i])
    assert info["cio_offsets"] == sent
    env.close()


def test_action_none_falls_back_to_heuristic():
    """examples/cio_zmq_experiment.py's path: env.step(None) still round-trips."""
    env = make_env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(None)

    sent = env.zmq_db.sent[-1]["cells"]
    snap = _snapshot(0)["cells"]
    for cell_id, cell_kpis in snap.items():
        expected = max(-CIO_LIMIT_DB, min(CIO_LIMIT_DB,
                                          -12.0 * (cell_kpis["prb_utilization"] - 0.5)))
        assert sent[cell_id]["cio_offset"] == expected
    assert np.isfinite(reward)
    env.close()


def test_reward_terms():
    env = make_env()
    env.reset(seed=0)
    kpis = _snapshot(1)
    terms = env._reward_terms(kpis)

    cells = kpis["cells"]
    buf = sum(c["buffer_bytes"] for c in cells.values())
    vol = sum(c["volume_bytes"] for c in cells.values())
    # control_period_s is the 0.1 s E2 indication window volume_bytes actually
    # accumulates over, NOT the 1.0 s T_control step period -- see KPI_WINDOW_S.
    assert env.control_period_s == KPI_WINDOW_S
    assert terms["backlog"] == buf / (vol / env.control_period_s)
    assert 0.0 < terms["balance"] <= 1.0
    assert terms["reward"] == env.w_balance * terms["balance"] - env.w_backlog * terms["backlog"]
    assert terms["diagnostics"]["median_mcs_gate_applied"] is False
    # Every weight applied to the reward must also be reported. w_pingpong was
    # applied but missing here, so anything reconstructing the reward from this
    # dict came out short by exactly w_pingpong * pingpong.
    assert set(terms["weights"].keys()) == {"balance", "backlog", "badsignal",
                                            "satisfaction", "pingpong"}, terms["weights"]

    # Idle network: Balance is neutralised, Backlog charges nothing.
    idle = {"timestamp": 0.0, "cells": {str(c): {"prb_utilization": 0.0, "buffer_bytes": 0.0,
                                                 "volume_bytes": 0.0, "num_active_ues": 0,
                                                 "ues": {}} for c in CELLS}}
    idle_terms = env._reward_terms(idle)
    assert idle_terms["balance"] == 1.0 and idle_terms["backlog"] == 0.0
    assert idle_terms["diagnostics"]["balance_guarded"] is True
    assert np.isfinite(idle_terms["reward"])
    env.close()


def test_backlog_drain_rate_comes_from_the_full_control_step():
    """FIX 13: Backlog's denominator must be the PDCP full-step delivery rate.

    It used to be volume_bytes / 0.1 s -- the E2 indication volume, which covers
    only a 10 % SAMPLE of the 1.0 s control step. MEASURED on the AM
    discrimination runs: that sampled rate correlates r = +0.257 with real
    full-step delivery, so Backlog was being divided by noise. Satisfaction
    already used the PDCP accumulator; the two terms described different seconds
    of the same network.

    Both halves are pinned here: PDCP present -> PDCP rate wins; PDCP absent ->
    fall back to the old E2 volume rather than crashing on an older ns-3 build.
    """
    env = make_env()
    env.reset(seed=0)

    # --- PDCP present: the 1.0 s accumulator must be what divides.
    kpis = _snapshot(1)
    window_s = 1.0
    per_imsi = {"1": 4_000_000.0, "2": 6_000_000.0}     # 10 MB over the full step
    for cell in kpis["cells"].values():
        cell["pdcp_delivered_bytes"] = {}
        cell["pdcp_window_s"] = window_s
    # Bank all of it on one cell; the helper sums across cells either way.
    first = next(iter(kpis["cells"].values()))
    first["pdcp_delivered_bytes"] = per_imsi

    terms = env._reward_terms(kpis)
    buf = sum(c["buffer_bytes"] for c in kpis["cells"].values())
    expected_rate = sum(per_imsi.values()) / window_s
    assert terms["diagnostics"]["backlog_rate_source"] == "pdcp"
    assert terms["backlog"] == buf / expected_rate

    # It must NOT be the old E2 answer -- otherwise the fix is inert.
    vol = sum(c["volume_bytes"] for c in kpis["cells"].values())
    assert terms["backlog"] != buf / (vol / env.control_period_s)

    # --- PDCP absent: degrade to the E2 volume, do not crash.
    legacy = _snapshot(1)
    legacy_terms = env._reward_terms(legacy)
    lbuf = sum(c["buffer_bytes"] for c in legacy["cells"].values())
    lvol = sum(c["volume_bytes"] for c in legacy["cells"].values())
    assert legacy_terms["diagnostics"]["backlog_rate_source"] == "e2_volume"
    assert legacy_terms["backlog"] == lbuf / (lvol / env.control_period_s)
    env.close()


def test_backlog_and_satisfaction_read_one_parse():
    """The two delivery-driven terms must never disagree about the same payload.

    Backlog (drain rate) and Satisfaction (delivered vs demanded) both come from
    _pdcp_totals(). If one of them were re-parsing the payload separately, a
    change to the dedup rule could silently apply to one term and not the other.
    """
    env = make_env()
    env.reset(seed=0)
    kpis = _snapshot(1)
    imsi = env.udp_imsis[0] if env.udp_imsis else 1
    for cell in kpis["cells"].values():
        cell["pdcp_delivered_bytes"] = {}
        cell["pdcp_window_s"] = 1.0
    # Same UE banked on TWO cells: the documented rule is SUM, not max.
    cs = list(kpis["cells"].values())
    cs[0]["pdcp_delivered_bytes"] = {str(imsi): 1_000_000.0}
    cs[1]["pdcp_delivered_bytes"] = {str(imsi): 3_000_000.0}

    total, window_s, present, per_imsi = env._pdcp_totals(
        [kpis["cells"].get(str(c), {}) for c in CELLS])
    assert present and window_s == 1.0
    assert per_imsi[imsi] == 4_000_000.0        # summed across cells
    assert total == 4_000_000.0

    terms = env._reward_terms(kpis)
    buf = sum(c["buffer_bytes"] for c in kpis["cells"].values())
    assert terms["backlog"] == buf / (total / window_s)
    env.close()


def test_time_limit_is_truncation_not_termination():
    """simTime running out is a TIME LIMIT, so truncated=True / terminated=False.

    Both flags used to be set. SB3 computes
        info["TimeLimit.truncated"] = truncated and not terminated
    so with both True the flag came out False and SB3 never bootstrapped the
    terminal value -- the critic was trained toward 0 at every episode end.
    """
    env = make_env()
    env.reset(seed=0)
    for _ in range(EPISODE_STEPS + 5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            assert truncated and not terminated, (terminated, truncated)
            break
    else:
        raise AssertionError("episode never ended")
    env.close()


def test_reset_retries_past_a_crashing_seed():
    """A seed that kills ns-3 at start-up must be re-drawn, not fatal.

    MEASURED: ~1 seed in 10 makes ns-3 exit before its first control step
    (seed 1003 did it twice, on two different ports). Since FIX 4 draws a fresh
    seed every episode, a long training run is GUARANTEED to hit one, and
    without a retry the entire run dies on a single unlucky draw. This path
    fires ~10% of the time in reality, so it needs a deterministic test.
    """
    class _CrashyEnv(_StubMlbZmqEnv):
        crashes_left = 2
        seeds_tried = []

        def start_sim(self):
            super().start_sim()
            self.seeds_tried.append(self.scenario_configuration["RngRun"])
            if self.crashes_left > 0:
                self.crashes_left -= 1
                self._step_count = EPISODE_STEPS   # -> _recv_next_kpis returns None

    env = _CrashyEnv(ns3_path="/unused", scenario_configuration=CFG,
                     output_folder="/tmp", optimized=False)
    obs, info = env.reset(seed=0)
    assert not info.get("reset_failed"), info
    assert info["reset_attempts"] == 3, info          # 2 crashes then success
    assert len(set(env.seeds_tried)) == 3, env.seeds_tried   # a NEW seed each try
    assert env.observation_space.contains(obs)
    assert env.reset_retry_count == 2, env.reset_retry_count
    env.close()


def test_reset_does_not_retry_when_seed_is_pinned():
    """With vary_rng_run_per_episode=False a retry would replay the SAME failing
    simulation, so the failure must surface instead of spinning."""
    class _AlwaysCrash(_StubMlbZmqEnv):
        def start_sim(self):
            super().start_sim()
            self._step_count = EPISODE_STEPS

    env = _AlwaysCrash(ns3_path="/unused", scenario_configuration=CFG,
                       output_folder="/tmp", optimized=False,
                       vary_rng_run_per_episode=False)
    obs, info = env.reset(seed=0)
    assert info.get("reset_failed") is True, info
    assert info["reset_attempts"] == 1, info      # exactly one, no spinning
    env.close()


def test_env_tells_ns3_the_port_it_binds():
    """The port Python BINDS and the port ns-3 DIALS must be the same one.

    They used to be independent: zmq_port bound the socket, while the scenario's
    zmqPort GlobalValue decided what ns-3 connected to. Setting only the first
    produced a SILENT deadlock -- both sides alive at ~0% CPU, no error, no log
    line -- the most expensive kind of failure to diagnose.
    """
    env = _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                         output_folder="/tmp", optimized=False, zmq_port=5599)
    assert env.zmq_port == 5599
    assert env.scenario_configuration["zmqPort"] == 5599, env.scenario_configuration
    env.close()


def test_pingpong_detection():
    """The PingPong detector, including the case a single payload cannot see.

    Y = 0.8 s but a control step spans 1.0 s, so an A->B / B->A pair can STRADDLE
    two payloads. Counting inside one payload would miss exactly those, which is
    the same window bug that corrupted Field 2's first implementation.
    """
    env = make_env()
    env.reset(seed=0)

    def terms(handovers, t):
        snap = _snapshot(1)
        snap["handovers"] = [{"t": ht, "imsi": i, "src": s_, "dst": d}
                             for ht, i, s_, d in handovers]
        snap["timestamp"] = t
        return env._reward_terms(snap)

    # 1. a complete ping-pong inside ONE payload: UE 7 goes 2->3 then 3->2
    d = terms([(1.10, 7, 2, 3), (1.50, 7, 3, 2)], 2.0)["diagnostics"]
    assert d["pingpong_count"] == 1, d

    # 2. STRADDLE: outbound in one payload, return in the next, gap < Y
    env._handover_history = []
    terms([(2.90, 9, 4, 5)], 3.0)
    d = terms([(3.40, 9, 5, 4)], 4.0)["diagnostics"]
    assert d["pingpong_count"] == 1, ("straddling pair missed", d)

    # 3. NOT a ping-pong: onward move A->B then B->C
    env._handover_history = []
    d = terms([(1.10, 11, 2, 3), (1.40, 11, 3, 6)], 2.0)["diagnostics"]
    assert d["pingpong_count"] == 0, d

    # 4. a return, but LATER than Y -> not a ping-pong
    env._handover_history = []
    terms([(1.00, 13, 2, 3)], 2.0)
    d = terms([(2.50, 13, 3, 2)], 3.0)["diagnostics"]
    assert d["pingpong_count"] == 0, ("return outside Y was counted", d)

    # 5. field present but empty -> a REAL zero, not "unavailable"
    env._handover_history = []
    t = terms([], 5.0)
    assert t["pingpong"] == 0.0 and not t["diagnostics"]["pingpong_unavailable"]

    # 6. field ABSENT -> None, flagged, and DROPPED from the reward sum
    snap = _snapshot(1)
    t = env._reward_terms(snap)
    assert t["pingpong"] is None
    assert t["diagnostics"]["pingpong_unavailable"] is True
    assert "pingpong" not in t["diagnostics"]["terms_present"]
    env.close()


def test_backlog_is_a_level_not_a_rate():
    """Backlog must be INVARIANT to the reported window -- the exact opposite of
    PingPong, and for the same reason.

    Backlog is seconds of drain time: a queued LEVEL divided by a delivery RATE.
    Report the same physical flow over a 10x shorter window and both the bytes
    and the window shrink 10x, so the rate -- and therefore the term -- must not
    move at all. PingPong, a count per step, must scale 10x. Those two tests are
    a matched pair and neither is meaningful without the other.

    WHY THIS EXISTS (Step 10i). Step 10h-9 reported Backlog as +32.6%
    T_control-dependent on an arm whose physics was bit-identical at both
    control periods. It was not: `am_neutral` was recorded BEFORE FIX 13 (E2
    volume / KPI_WINDOW_S) and the T=0.1 arm after it (PDCP / reported window),
    so a pure formula change was read as a physical effect. With one formula on
    both sides the same arm comes out at +0.20%. The term was correct all along;
    the COMPARISON was not. This test pins the property so the question never
    has to be re-litigated from run artefacts again.
    """
    env = make_env()
    env.reset(seed=0)

    # One physical flow, reported two ways. Same queue, same bytes-per-second,
    # only the accounting window differs.
    BUFFER_BYTES = 4_000_000.0
    RATE_BPS = 10_000_000.0

    def terms(window_s):
        snap = _snapshot(1)
        for i, (cell, ck) in enumerate(snap["cells"].items()):
            # All the queue on one cell; Backlog sums over cells anyway.
            ck["buffer_bytes"] = BUFFER_BYTES if i == 0 else 0.0
            ck["pdcp_delivered_bytes"] = ({"1": RATE_BPS * window_s}
                                          if i == 0 else {})
            ck["pdcp_window_s"] = window_s
        return env._reward_terms(snap)

    ref = terms(1.0)          # the T_control = 1.0 reference
    short = terms(0.1)        # the same flow over a 10x shorter window

    d_ref, d_short = ref["diagnostics"], short["diagnostics"]

    # NON-VACUITY. A zero term satisfies any invariance claim, and a term that
    # fell back to the E2-volume branch would not be testing FIX 13 at all.
    assert ref["backlog"] > 0.0, ref["backlog"]
    assert d_ref["backlog_rate_source"] == "pdcp", d_ref["backlog_rate_source"]
    assert d_short["backlog_rate_source"] == "pdcp", d_short["backlog_rate_source"]
    assert d_ref["pdcp_window_s"] == 1.0 and d_short["pdcp_window_s"] == 0.1

    # The queued LEVEL is the same physical queue in both.
    assert d_ref["backlog_bytes"] == d_short["backlog_bytes"] == BUFFER_BYTES

    # The property under test, EXACTLY. bytes/window is (r*w)/w, which for
    # IEEE-754 returns r exactly for these values, so this is not an
    # approximate claim.
    assert short["backlog"] == ref["backlog"], (
        "Backlog is not window-invariant: the same flow reported over a 10x "
        "shorter window changed the drain time",
        ref["backlog"], short["backlog"])

    # And it is the drain time it claims to be: level / rate, in seconds.
    assert abs(ref["backlog"] - BUFFER_BYTES / RATE_BPS) < 1e-12, ref["backlog"]
    assert d_ref["delivery_rate_bytes_per_s"] == d_short["delivery_rate_bytes_per_s"]

    # The reward must carry the same invariance, weight included.
    assert abs((ref["reward"] - short["reward"])) < 1e-12, (
        ref["reward"], short["reward"])

    # The stamp that makes a cross-formula comparison self-detecting.
    assert d_ref["reward_formula_version"] == REWARD_FORMULA_VERSION
    assert REWARD_FORMULA_VERSION >= 13


def test_backlog_and_pingpong_are_a_matched_pair():
    """The two window behaviours in one place: one scales, the other must not.

    Guards against a future edit 'fixing' the wrong one -- normalising Backlog
    by the window (it already is) or de-normalising PingPong.
    """
    env = make_env()
    env.reset(seed=0)

    def terms(window_s):
        snap = _snapshot(1)
        snap["timestamp"] = 2.0
        snap["handover_window_s"] = window_s
        snap["handovers"] = [{"t": 1.92, "imsi": 7, "src": 2, "dst": 3},
                             {"t": 1.98, "imsi": 7, "src": 3, "dst": 2}]
        for i, (cell, ck) in enumerate(snap["cells"].items()):
            ck["buffer_bytes"] = 4_000_000.0 if i == 0 else 0.0
            ck["pdcp_delivered_bytes"] = ({"1": 1e7 * window_s} if i == 0 else {})
            ck["pdcp_window_s"] = window_s
        env._handover_history = []
        return env._reward_terms(snap)

    ref, short = terms(1.0), terms(0.1)
    assert ref["diagnostics"]["pingpong_count"] == 1, "vacuous: no ping-pong detected"
    assert ref["backlog"] > 0.0 and ref["pingpong"] > 0.0

    # A LEVEL: unchanged.  A RATE: exactly 10x.
    assert short["backlog"] == ref["backlog"], ("Backlog moved with the window",
                                                ref["backlog"], short["backlog"])
    assert short["pingpong"] == 10.0 * ref["pingpong"], ("PingPong did not scale",
                                                         ref["pingpong"], short["pingpong"])


def test_pingpong_is_a_rate_not_a_per_step_count():
    """PingPong must scale with handover_window_s, not with the step count.

    PingPong is the ONLY one of the five reward terms whose magnitude depends on
    T_control: the other four are levels or ratios (Jain index, kbps/kbps, a bin
    fraction, seconds of drain time). While it was `pingpong_count/active_ues`,
    a COUNT PER STEP, shortening the control period deflated it while the other
    four stayed put -- which REVERSES the recorded neutral-vs-flipflop ordering
    of the Step 6o validation (reward gap +0.2213 -> -0.0242), scoring the
    flip-flop controller better than do-nothing. See PINGPONG_REF_S.

    So this asserts the property, not the implementation: the SAME ping-pong
    count reported over a 10x shorter window must yield EXACTLY 10x the term.
    Exact equality is deliberate -- at window == PINGPONG_REF_S the factor is
    1.0/1.0, an IEEE-754 exact multiply, which is what lets every weight tuned
    at T_control = 1.0 and the whole Step 6o PASS carry over unchanged.
    """
    env = make_env()
    env.reset(seed=0)

    # UE 7 goes 2->3 then 3->2, 60 ms apart: one ping-pong (gap < Y = 0.8 s),
    # and close enough together to sit inside a 0.1 s window as well as a 1.0 s
    # one, so the DETECTED COUNT is held fixed and only the reported window
    # varies. That isolates the normalisation from the detector.
    HANDOVERS = [{"t": 1.92, "imsi": 7, "src": 2, "dst": 3},
                 {"t": 1.98, "imsi": 7, "src": 3, "dst": 2}]

    def terms(window_s):
        snap = _snapshot(1)
        snap["timestamp"] = 2.0
        snap["handovers"] = [dict(h) for h in HANDOVERS]
        if window_s is not None:
            snap["handover_window_s"] = window_s
        env._handover_history = []          # identical detector state each call
        return env._reward_terms(snap)

    ref = terms(1.0)
    short = terms(0.1)
    d_ref, d_short = ref["diagnostics"], short["diagnostics"]

    # Non-vacuity: a zero term would satisfy any ratio. Guard it explicitly.
    assert d_ref["pingpong_count"] == 1, d_ref
    assert ref["pingpong"] > 0.0, ref["pingpong"]

    # The count must be IDENTICAL, so the 10x below comes from the window and
    # not from the detector seeing something different.
    assert d_short["pingpong_count"] == d_ref["pingpong_count"], (d_ref, d_short)

    # The property under test, exactly.
    assert short["pingpong"] == 10.0 * ref["pingpong"], (
        "PingPong is not rate-normalised: a 10x shorter window must give a 10x "
        "larger term", ref["pingpong"], short["pingpong"])
    assert d_ref["pingpong_rate_scale"] == 1.0
    assert d_short["pingpong_rate_scale"] == 10.0
    assert d_ref["pingpong_window_s"] == 1.0
    assert d_short["pingpong_window_s"] == 0.1

    # At the reference window the term is bit-identical to the old raw count
    # form. This is the anchor that keeps the tuned weights valid.
    assert ref["pingpong"] == d_ref["pingpong_count"] / d_ref["active_ues"]
    assert d_ref["pingpong_ref_s"] == 1.0

    # Both fallbacks must reproduce the reference EXACTLY rather than dividing
    # by zero or reaching for control_period_s (the 0.1 s E2 KPI window, a
    # different quantity, which would inflate the term 10x). See KPI_WINDOW_S.
    for label, w in (("field absent", None), ("non-positive", 0.0)):
        fb = terms(w)
        assert fb["pingpong"] == ref["pingpong"], (label, fb["pingpong"])
        assert fb["diagnostics"]["pingpong_rate_scale"] == 1.0, label

    # And it must reach the reward with the configured weight.
    assert ref["reward"] - short["reward"] == (
        env.w_pingpong * (short["pingpong"] - ref["pingpong"]))
    env.close()


def test_rng_run_varies_per_episode():
    """Successive episodes must get DIFFERENT ns-3 seeds, or training sees one world.

    Gymnasium semantics matter here: reset(seed=X) RE-seeds, so it must stay
    reproducible, while a bare reset() draws the next seed in the stream. SB3
    seeds once and then calls reset() with no seed for every later episode, so
    the bare-reset path is the one training actually takes.
    """
    env = make_env()
    env.reset(seed=0)
    seq = [env.scenario_configuration["RngRun"]]
    for _ in range(2):
        env.reset()                      # how SB3 resets between episodes
        seq.append(env.scenario_configuration["RngRun"])
    env.close()
    assert len(set(seq)) == 3, seq       # three episodes, three worlds

    # Same starting seed reproduces the same episode sequence.
    env2 = make_env()
    env2.reset(seed=0)
    again = [env2.scenario_configuration["RngRun"]]
    for _ in range(2):
        env2.reset()
        again.append(env2.scenario_configuration["RngRun"])
    env2.close()
    assert seq == again, (seq, again)


def test_episode_runs_to_completion():
    env = make_env()
    env.reset(seed=0)
    steps, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        steps += 1
        assert steps <= EPISODE_STEPS + 1
    env.close()


# --- per-episode offered-load randomisation (FIX: udp_ue_rate_kbps recompute) ---
# The Satisfaction denominator is n_udp * udp_ue_rate_kbps, and udp_ue_rate_kbps
# is derived from udpFullBufferIntervalUs. Randomising the interval WITHOUT
# recomputing the rate scores every episode against a different episode's demand:
# Satisfaction stays in range, still moves with the network, and is silently
# wrong. These tests exist mainly to make that impossible.
UDP_SDU_BITS_PER_MS = 1310.0 * 8.0 * 1000.0     # 1310 B SDU -> kbit/s at 1 us


def _expected_rate(interval_us):
    return UDP_SDU_BITS_PER_MS / interval_us


def _command_line(env):
    """The argv start_sim() would build, without launching anything."""
    return [f'--{k}={v}' for k, v in env.scenario_configuration.items()]


def test_udp_interval_randomised_and_denominator_recomputed():
    """THE CENTREPIECE. Interval is an int inside the range, and the Satisfaction
    denominator equals 1310*8*1000/interval EXACTLY, every episode."""
    env = _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                         output_folder="/tmp", optimized=False,
                         udp_interval_us_range=(450, 550))
    seen = set()
    for _ in range(25):
        env.reset(seed=None)
        interval = env.scenario_configuration['udpFullBufferIntervalUs']
        assert isinstance(interval, int), f"not an int: {type(interval)}"
        assert 450 <= interval <= 550, interval
        seen.add(interval)
        assert env.udp_ue_rate_kbps == _expected_rate(interval), (
            f"denominator not recomputed: interval {interval} -> "
            f"{env.udp_ue_rate_kbps} != {_expected_rate(interval)}")
    env.close()
    assert len(seen) > 1, "the interval never actually varied"


def test_udp_interval_reported_in_reset_info():
    """reset() hands back the drawn interval and the rate it implies, so the
    load an episode was scored against is on record next to its reward."""
    env = _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                         output_folder="/tmp", optimized=False,
                         udp_interval_us_range=(450, 550))
    _obs, info = env.reset(seed=7)
    assert 450 <= info["udp_interval_us"] <= 550
    assert info["udp_interval_us"] == env.scenario_configuration['udpFullBufferIntervalUs']
    assert info["udp_ue_rate_kbps"] == _expected_rate(info["udp_interval_us"])
    env.close()


def test_udp_interval_command_line_is_scalar_not_list():
    """ns_env.py:61 stores {k: v[0]} and start_sim renders f'--{param}={value}',
    so a list would emit --udpFullBufferIntervalUs=[480] and ns-3's
    UintegerValue parser would reject it. Same trap RngRun documents."""
    env = _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                         output_folder="/tmp", optimized=False,
                         udp_interval_us_range=(480, 480))
    env.reset(seed=1)
    args = [a for a in _command_line(env) if a.startswith("--udpFullBufferIntervalUs")]
    assert args == ["--udpFullBufferIntervalUs=480"], args
    assert "[" not in args[0] and "]" not in args[0]
    env.close()


def test_udp_interval_none_is_byte_identical_to_before():
    """The default must change NOTHING: no key added to the command line, and
    the rate stays at the scenario's own value."""
    plain = _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                           output_folder="/tmp", optimized=False)
    plain.reset(seed=3)
    cmd = _command_line(plain)
    assert not any("udpFullBufferIntervalUs" in a for a in cmd), cmd
    assert plain.udp_ue_rate_kbps == _expected_rate(500.0)   # ns-3's own default
    assert plain.scenario_configuration.get('udpFullBufferIntervalUs') is None
    rate_before = plain.udp_ue_rate_kbps
    for _ in range(5):
        plain.reset(seed=None)
        assert plain.udp_ue_rate_kbps == rate_before, "rate drifted with the knob off"
    plain.close()

    # And a config that PINS the interval keeps honouring it, untouched.
    pinned = _StubMlbZmqEnv(ns3_path="/unused",
                            scenario_configuration=dict(CFG, udpFullBufferIntervalUs=[750]),
                            output_folder="/tmp", optimized=False)
    pinned.reset(seed=3)
    assert pinned.scenario_configuration['udpFullBufferIntervalUs'] == 750
    assert pinned.udp_ue_rate_kbps == _expected_rate(750)
    pinned.close()


def test_udp_interval_is_seed_reproducible_and_paired_with_rng_run():
    """A given reset(seed=) must replay the same (RngRun, interval) SEQUENCE --
    otherwise a run cannot be reproduced."""
    def sequence():
        env = _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                             output_folder="/tmp", optimized=False,
                             udp_interval_us_range=(450, 550))
        out = []
        for i in range(4):
            _obs, info = env.reset(seed=99 if i == 0 else None)
            out.append((info["rng_run"], info["udp_interval_us"]))
        env.close()
        return out
    a, b = sequence(), sequence()
    assert a == b, f"not reproducible:\n  {a}\n  {b}"
    assert len({x[1] for x in a}) > 1, "interval constant across episodes"


def test_udp_interval_range_validated():
    for bad in [(0, 100), (-5, 5), (600, 400)]:
        try:
            _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                           output_folder="/tmp", optimized=False,
                           udp_interval_us_range=bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid range {bad}")


def test_satisfaction_scales_with_the_drawn_interval():
    """The whole point, at the arithmetic level: hold DELIVERED BYTES fixed and
    vary only the interval. In the capacity-limited regime delivery is roughly
    flat while demand goes as 1/interval, so Satisfaction must scale LINEARLY
    with the interval. If the denominator were not recomputed it would instead
    be CONSTANT -- which is exactly the silent failure being guarded against."""
    got = {}
    for interval in (450, 500, 550, 950):
        env = _StubMlbZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                             output_folder="/tmp", optimized=False,
                             udp_interval_us_range=(interval, interval))
        env.reset(seed=5)
        # One synthetic snapshot with a FIXED delivered volume.
        delivered = {str(i): 2.0e5 for i in env.udp_imsis}
        kpis = _snapshot(0)
        for cell_kpis in kpis["cells"].values():
            cell_kpis["pdcp_window_s"] = 1.0
            cell_kpis["pdcp_delivered_bytes"] = {}
        first = next(iter(kpis["cells"].values()))
        first["pdcp_delivered_bytes"] = delivered
        got[interval] = env._reward_terms(kpis)["satisfaction"]
        env.close()

    assert len(set(got.values())) == len(got), (
        f"Satisfaction did NOT move with the interval -- the denominator is "
        f"stale: {got}")
    # Linear in the interval: sat(I) / I is the same constant for every I.
    ratios = [got[i] / i for i in got]
    assert max(ratios) - min(ratios) < 1e-9, f"not proportional to interval: {got}"
    assert got[950] > got[550] > got[500] > got[450], got


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
