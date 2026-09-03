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
from environments.mro_zmq_env import (CELLS, MARGIN_LIMIT_DB, OBS_FIELDS, OUTAGE_SINR_DB,
                                      PINGPONG_Y_S, SINR_FLOOR_DB, SINR_SCALE_DB,
                                      MroZmqEnv, obs_index)

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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
