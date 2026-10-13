"""Gym-API and reward/observation checks for MroZmqEnv, WITHOUT ns-3 or ZeroMQ.

Same stubbing strategy as tests/test_mlb_zmq_env.py: _StubMroZmqEnv replaces
setup_sim()/start_sim()/_recv_next_kpis()/close() with a deterministic
synthetic KPI source in the payload shape scenario-marl-zmq.cc actually sends.
Everything under test (spaces, _get_obs, _margin_from_action, _reward_terms,
the step()/reset() contract) is the real code path.

Run:  pytest tests/test_mro_zmq_env.py      or      python tests/test_mro_zmq_env.py
"""
import sys
import warnings
from os import path

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import (CELLS, HANDOVER_RATE_REF_S, HANDOVER_WINDOW_FLOOR_S,
                                      MARGIN_LIMIT_DB, OBS_FIELDS, OUTAGE_SINR_DB, PINGPONG_Y_S,
                                      SINR_FLOOR_DB, SINR_SCALE_DB, MroZmqEnv, obs_index)

CFG = {"ues": [3], "simTime": [10], "RngRun": [555]}

EPISODE_STEPS = 6

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


def _snapshot(step: int, handovers=None) -> dict:
    """One synthetic KPI snapshot, deterministic in `step`.

    Exercises the same awkward cases MLB's stub does:
      - step 2 drops cell 8 entirely            -> zero-filled obs block
      - cell 7 always has zero UEs and no SINR  -> SINR floor, not 0 dB
    Every UE gets a stable IMSI (1000 + 10*cell + u) so tests can track one
    across steps for HandoverQuality. `handovers` defaults to an empty list
    (field present, genuinely no events) rather than absent (field missing).
    """
    cells = {}
    for i, cell in enumerate(CELLS):
        if step == 2 and cell == 8:
            continue
        n_ues = 0 if cell == 7 else (i + step) % 4
        cells[str(cell)] = {
            "num_active_ues": n_ues,
            "sinr_bins": [1 if cell == 3 else 0, 0, 0, 0, 0, 0, 9],
            "ues": {str(1000 + 10 * cell + u): {"l3_serving_sinr_db": 5.0 + i - u}
                    for u in range(n_ues)},
        }
    return {"timestamp": float(step), "cells": cells,
            "handovers": [] if handovers is None else handovers,
            "handover_window_s": 1.0}


class _StubMroZmqEnv(MroZmqEnv):
    def __init__(self, *a, snapshots=None, **kw):
        # snapshots: optional list[dict] overriding _snapshot() for a scripted
        # sequence (used by the handover/quality/ping-pong tests below).
        self._snapshots = snapshots
        super().__init__(*a, **kw)

    def setup_sim(self):
        self.environment = {}
        self.script_executable = "/bin/true"

    def start_sim(self):
        self.is_open = True
        self.sim_path = "/tmp/stub-mro-zmq"
        self.zmq_db = _FakeZmqDb()
        self._pending_kpis = None
        self._step_count = 0
        # This override bypasses MroZmqEnv.start_sim() entirely, so it must
        # repeat the same episode-boundary resets that method does -- without
        # this, state from a PRIOR episode (e.g. a stray call from
        # check_env's own env_step_passive_checker) leaks into the next one.
        # Caught by test_check_env's determinism check: two reset(seed=X) +
        # step(action) sequences disagreed because a leftover self._last_action
        # from an earlier, unrelated step() was still set when the "first"
        # sequence's reward was computed.
        self._handover_history = []
        self._last_ue_sinr = {}
        self._last_action = None
        # Real start_sim() always sets this (meta.exitcode is read by
        # step()'s truncation branch); default to a clean exit, matching
        # what a stub-driven "the episode ran fine" test should mean. A test
        # exercising the crash path overwrites meta.exitcode directly before
        # its final step() call.
        self.sim_result = {'params': {}, 'meta': {'exitcode': 0}}

    def _recv_next_kpis(self):
        if self._step_count >= EPISODE_STEPS:
            return None
        if self._snapshots is not None:
            if self._step_count >= len(self._snapshots):
                return None
            kpis = self._snapshots[self._step_count]
        else:
            kpis = _snapshot(self._step_count)
        self.zmq_db.kpi_history.append(kpis)
        self._step_count += 1
        return kpis

    def is_simulation_over(self):
        limit = len(self._snapshots) if self._snapshots is not None else EPISODE_STEPS
        return self._step_count >= limit

    def close(self):
        if self.is_open:
            if self.zmq_db is not None:
                self.zmq_db.close()
                self.zmq_db = None
            self.is_open = False


def make_env(**kw) -> _StubMroZmqEnv:
    return _StubMroZmqEnv(ns3_path="/unused", scenario_configuration=CFG,
                          output_folder="/tmp", optimized=False, **kw)


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

    # Cell 7 has zero UEs and no SINR report -> floor, not 0 dB.
    idx = obs_index(7, "sinr_db_mean")
    assert obs[idx] == np.float32(SINR_FLOOR_DB / SINR_SCALE_DB)

    # last_margin_db is 0.0 for every cell before the first step().
    for cell in CELLS:
        assert obs[obs_index(cell, "last_margin_db")] == 0.0
    env.close()


def test_absent_cell_is_zero_filled():
    env = make_env()
    env.reset(seed=0)
    # step 0 -> snapshot(1) is returned by the first step(); drive to step 2,
    # where cell 8 is dropped entirely from the payload.
    action = np.zeros(len(CELLS), dtype=np.float32)
    env.step(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["kpis"]["timestamp"] == 2.0
    block = slice(obs_index(8, "sinr_db_mean"), obs_index(8, "sinr_db_mean") + len(OBS_FIELDS))
    # Every field except SINR is a real zero for an absent cell; SINR floors.
    assert obs[obs_index(8, "sinr_db_mean")] == np.float32(SINR_FLOOR_DB / SINR_SCALE_DB)
    assert obs[obs_index(8, "num_active_ues")] == 0.0
    env.close()


def test_margin_from_action_clips_and_maps_cells():
    env = make_env()
    env.reset(seed=0)
    # Deliberately out-of-range on cell 2 (index 0) to exercise clipping.
    action = np.array([MARGIN_LIMIT_DB + 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, -MARGIN_LIMIT_DB - 5.0],
                      dtype=np.float32)
    env.step(action)
    sent = env.zmq_db.sent[-1]
    assert sent["cells"]["2"]["ho_margin_db"] == MARGIN_LIMIT_DB
    assert sent["cells"]["8"]["ho_margin_db"] == -MARGIN_LIMIT_DB
    assert "cio_offset" not in sent["cells"]["2"]  # this env speaks ho_margin_db only
    env.close()


def test_reward_terms_present_when_handovers_available():
    env = make_env()
    env.reset(seed=0)
    action = np.zeros(len(CELLS), dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    terms = info["reward_terms"]
    assert set(terms["diagnostics"]["terms_present"]) >= {"outage", "smoothness"}
    assert isinstance(terms["reward"], float)
    env.close()


def test_handovers_unavailable_drops_three_terms_not_scores_zero():
    """A payload with no 'handovers' key at all (older ns-3 build) must drop
    PingPong, HandoverRate and HandoverQuality from the sum entirely -- never
    silently score them 0."""
    snap = _snapshot(0)
    del snap["handovers"]
    # Two copies: reset() consumes the first, step() consumes the second and
    # computes the reward under test -- a single-entry list would be fully
    # consumed by reset() alone, ending the episode before step() runs.
    env = make_env(snapshots=[snap, dict(snap)])
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(len(CELLS), dtype=np.float32))
    terms = info["reward_terms"]
    assert terms["pingpong"] is None
    assert terms["handover_rate"] is None
    assert terms["quality"] is None
    present = terms["diagnostics"]["terms_present"]
    assert "pingpong" not in present
    assert "handover_rate" not in present
    assert "quality" not in present
    env.close()


def test_outage_exposure_counts_ues_below_threshold():
    # Two UEs on cell 2: one clearly in outage, one clearly healthy.
    cells = {str(c): {"num_active_ues": 0, "sinr_bins": [0] * 7, "ues": {}} for c in CELLS}
    cells["2"] = {
        "num_active_ues": 2,
        "sinr_bins": [0] * 7,
        "ues": {"1": {"l3_serving_sinr_db": OUTAGE_SINR_DB - 1.0},
                "2": {"l3_serving_sinr_db": OUTAGE_SINR_DB + 10.0}},
    }
    snap = {"timestamp": 0.0, "cells": cells, "handovers": [], "handover_window_s": 1.0}
    env = make_env(snapshots=[snap, dict(snap)])
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(len(CELLS), dtype=np.float32))
    assert info["reward_terms"]["outage"] == 0.5  # 1 of 2 UEs in outage
    env.close()


def test_outage_exposure_empty_network_is_zero_not_dropped():
    cells = {str(c): {"num_active_ues": 0, "sinr_bins": [0] * 7, "ues": {}} for c in CELLS}
    snap = {"timestamp": 0.0, "cells": cells, "handovers": [], "handover_window_s": 1.0}
    env = make_env(snapshots=[snap, dict(snap)])
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(len(CELLS), dtype=np.float32))
    assert info["reward_terms"]["outage"] == 0.0
    assert "outage" in info["reward_terms"]["diagnostics"]["terms_present"]
    env.close()


def test_action_smoothness_zero_on_first_step_then_tracks_delta():
    env = make_env()
    env.reset(seed=0)
    a1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info1 = env.step(a1)
    assert info1["reward_terms"]["smoothness"] == 0.0  # no prior action yet

    a2 = np.array([1.0 + 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # clips to +3
    _, _, _, _, info2 = env.step(a2)
    # sent a1=1.0, sent a2=clip(4.0)=3.0 -> squared delta on cell 2 is 4.0, mean/7
    expected = float(np.mean((np.array([3.0, 0, 0, 0, 0, 0, 0]) - np.array([1.0, 0, 0, 0, 0, 0, 0])) ** 2))
    assert abs(info2["reward_terms"]["smoothness"] - expected) < 1e-6
    env.close()


def test_handover_quality_classifies_good_and_bad_moves():
    """UE 100: src cell 2 (SINR -10) -> dst cell 3 (SINR +5): a GOOD move.
    UE 200: src cell 3 (SINR +10) -> dst cell 4 (SINR -5): a BAD move.
    UE 300 has a handover but never reappears: unmeasurable, dropped."""
    snap0 = {
        "timestamp": 0.0,
        "cells": {
            "2": {"num_active_ues": 1, "sinr_bins": [0] * 7, "ues": {"100": {"l3_serving_sinr_db": -10.0}}},
            "3": {"num_active_ues": 2, "sinr_bins": [0] * 7, "ues": {"200": {"l3_serving_sinr_db": 10.0},
                                                                      "300": {"l3_serving_sinr_db": 0.0}}},
            **{str(c): {"num_active_ues": 0, "sinr_bins": [0] * 7, "ues": {}} for c in CELLS if c not in (2, 3)},
        },
        "handovers": [],
        "handover_window_s": 1.0,
    }
    snap1 = {
        "timestamp": 1.0,
        "cells": {
            "3": {"num_active_ues": 1, "sinr_bins": [0] * 7, "ues": {"100": {"l3_serving_sinr_db": 5.0}}},
            "4": {"num_active_ues": 1, "sinr_bins": [0] * 7, "ues": {"200": {"l3_serving_sinr_db": -5.0}}},
            **{str(c): {"num_active_ues": 0, "sinr_bins": [0] * 7, "ues": {}} for c in CELLS if c not in (3, 4)},
        },
        # 300's handover is real but 300 never reappears in any later snapshot.
        "handovers": [{"t": 0.5, "imsi": 100, "src": 2, "dst": 3},
                      {"t": 0.6, "imsi": 200, "src": 3, "dst": 4},
                      {"t": 0.7, "imsi": 300, "src": 3, "dst": 5}],
        "handover_window_s": 1.0,
    }
    env = make_env(snapshots=[snap0, snap1])
    env.reset(seed=0)  # consumes snap0
    _, _, _, _, info = env.step(np.zeros(len(CELLS), dtype=np.float32))  # consumes snap1
    terms = info["reward_terms"]
    assert terms["diagnostics"]["quality_measured"] == 2  # 300 dropped
    assert terms["diagnostics"]["quality_good"] == 1      # only UE 100's move helped
    assert terms["quality"] == 0.5
    env.close()


def test_pingpong_detects_a_return_leg():
    snap0 = _snapshot(0, handovers=[])
    snap1 = _snapshot(1, handovers=[{"t": 0.3, "imsi": 5, "src": 2, "dst": 3}])
    snap2 = _snapshot(2, handovers=[{"t": 0.7, "imsi": 5, "src": 3, "dst": 2}])  # return, within Y=0.8s
    env = make_env(snapshots=[snap0, snap1, snap2])
    env.reset(seed=0)  # consumes snap0
    env.step(np.zeros(len(CELLS), dtype=np.float32))  # consumes snap1
    _, _, _, _, info = env.step(np.zeros(len(CELLS), dtype=np.float32))  # consumes snap2
    assert info["reward_terms"]["diagnostics"]["pingpong_count"] == 1
    assert info["reward_terms"]["pingpong"] is not None and info["reward_terms"]["pingpong"] > 0.0
    env.close()


def test_pingpong_and_handover_rate_are_rates_not_per_step_counts():
    """PingPong AND HandoverRate must scale with handover_window_s, not with
    the raw event count -- the same property test c576d6c added for MLB's own
    PingPong, extended to cover MRO's copy of it plus the new HandoverRate
    term, which has the identical count-per-step shape and was never checked
    for this before docs/mro_training_fixes.md flagged it.

    Asserts the property, not the implementation: the SAME detected handovers
    reported over a 10x shorter window must yield EXACTLY 10x both terms.
    Exact equality is deliberate -- at window == HANDOVER_RATE_REF_S the
    factor is 1.0/1.0, an IEEE-754 exact multiply, which is what lets every
    weight measured at the reference window (docs/mro_training_fixes.md's
    round-1 baseline) carry over unchanged.
    """
    env = make_env()
    env.reset(seed=0)

    # Two legs for the SAME UE, 60 ms apart -- close enough to sit inside a
    # 0.1 s window as well as a 1.0 s one, so the DETECTED COUNT is held fixed
    # and only the reported window varies. This isolates the normalisation
    # from the ping-pong detector's own pairing logic, and gives
    # HandoverRate a nonzero n_handovers=2 in the same fixture.
    HANDOVERS = [{"t": 1.92, "imsi": 7, "src": 2, "dst": 3},
                 {"t": 1.98, "imsi": 7, "src": 3, "dst": 2}]

    def terms(window_s):
        snap = _snapshot(1)
        snap["timestamp"] = 2.0
        snap["handovers"] = [dict(h) for h in HANDOVERS]
        snap["handover_window_s"] = window_s
        env._handover_history = []  # each call is an independent measurement
        return env._reward_terms(snap, np.zeros(len(CELLS), dtype=np.float32))

    ref = terms(HANDOVER_RATE_REF_S)
    short = terms(HANDOVER_RATE_REF_S / 10.0)

    # Non-vacuity guard: both reference terms must be genuinely nonzero, or a
    # 10x check would pass trivially on 0.0 == 0.0.
    assert ref["pingpong"] is not None and ref["pingpong"] > 0.0
    assert ref["handover_rate"] is not None and ref["handover_rate"] > 0.0
    assert ref["diagnostics"]["pingpong_count"] == 1
    assert ref["diagnostics"]["handovers_this_step"] == 2

    assert short["pingpong"] == 10.0 * ref["pingpong"]
    assert short["handover_rate"] == 10.0 * ref["handover_rate"]
    assert short["diagnostics"]["handover_rate_scale"] == 10.0
    assert ref["diagnostics"]["handover_rate_scale"] == 1.0

    # Fallbacks: field absent, and field <= 0.0, both reproduce the
    # reference-window behaviour (scale == 1.0) rather than dividing by zero.
    snap_absent = _snapshot(1)
    snap_absent["timestamp"] = 2.0
    del snap_absent["handovers"]
    env._handover_history = []
    absent = env._reward_terms(snap_absent, np.zeros(len(CELLS), dtype=np.float32))
    assert absent["diagnostics"]["handover_rate_scale"] == 1.0
    assert absent["pingpong"] is None and absent["handover_rate"] is None

    env.close()


def test_handover_rate_scale_is_capped_for_a_pathologically_small_window():
    """A reported handover_window_s well below any real control period must
    not blow PingPong/HandoverRate up unboundedly.

    Prompted by the reward-weight-methodology boundedness audit (see
    HANDOVER_WINDOW_FLOOR_S): the pre-existing guard only covered
    handover_window_s <= 0.0 (falls back to the 1.0x reference scale), not a
    small POSITIVE window, which is the actually dangerous case -- e.g. a
    window of 0.001 s would have inflated both terms 1000x with no cap at
    all. This asserts the floor, not just "some large-ish number": scale
    must be exactly HANDOVER_RATE_REF_S / HANDOVER_WINDOW_FLOOR_S (100.0 at
    current constants), and must NOT keep growing as the reported window
    shrinks further below the floor.
    """
    env = make_env()
    env.reset(seed=0)

    def terms(window_s):
        snap = _snapshot(1)
        snap["timestamp"] = 2.0
        snap["handovers"] = [{"t": 1.92, "imsi": 7, "src": 2, "dst": 3},
                             {"t": 1.98, "imsi": 7, "src": 3, "dst": 2}]
        snap["handover_window_s"] = window_s
        env._handover_history = []
        return env._reward_terms(snap, np.zeros(len(CELLS), dtype=np.float32))

    at_floor = terms(HANDOVER_WINDOW_FLOOR_S)
    below_floor = terms(HANDOVER_WINDOW_FLOOR_S / 10.0)  # e.g. a 0.001 s window
    expected_cap = HANDOVER_RATE_REF_S / HANDOVER_WINDOW_FLOOR_S

    assert at_floor["diagnostics"]["handover_rate_scale"] == expected_cap
    # The dangerous case: reporting an even smaller window must NOT scale
    # the term any further -- both must hit the same cap, not 10x apart.
    assert below_floor["diagnostics"]["handover_rate_scale"] == expected_cap
    assert below_floor["pingpong"] == at_floor["pingpong"]
    assert below_floor["handover_rate"] == at_floor["handover_rate"]
    env.close()


def _drive_to_truncation(env, exit_code: int):
    """reset() + enough step()s to reach the truncation branch, with the
    stub's sim_result.meta.exitcode set to `exit_code` right before the
    final step() call that triggers it.
    """
    env.reset(seed=0)
    for _ in range(EPISODE_STEPS - 1):
        env.step(np.zeros(len(CELLS), dtype=np.float32))
    env.sim_result['meta']['exitcode'] = exit_code
    return env.step(np.zeros(len(CELLS), dtype=np.float32))


def test_truncation_surfaces_a_clean_exit_as_not_crashed():
    """A clean simTime finish (ns-3 exitcode 0) must report crashed=False
    and the real exit code, not silently drop both."""
    env = make_env()
    obs, reward, terminated, truncated, info = _drive_to_truncation(env, exit_code=0)
    assert truncated and not terminated
    assert info["ns3_exit_code"] == 0
    assert info["crashed"] is False
    env.close()


def test_truncation_flags_a_nonzero_exit_as_crashed():
    """The bug this guards against: a clean finish and a crash used to be
    indistinguishable from step()'s return value alone, both landing on the
    exact same truncation branch. A nonzero ns-3 exit code, even on an
    otherwise-complete episode, must now be visible in info."""
    env = make_env()
    obs, reward, terminated, truncated, info = _drive_to_truncation(env, exit_code=1)
    assert truncated and not terminated
    assert info["ns3_exit_code"] == 1
    assert info["crashed"] is True
    env.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
