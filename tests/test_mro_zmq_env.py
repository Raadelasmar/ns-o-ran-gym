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
from environments.mro_zmq_env import (BASE_CHURN_GAIN_DB, BASE_TOO_LATE_GAIN_DB, CELLS,
                                      CHURN_RATE_THRESHOLD, HANDOVER_RATE_REF_S,
                                      HANDOVER_WINDOW_FLOOR_S, MARGIN_LIMIT_DB, OBS_FIELDS,
                                      OUTAGE_SINR_DB, PAIR_RATE_LIMIT_DB_PER_STEP,
                                      PAIR_RATE_WINDOW_STEPS, PINGPONG_Y_S, ROLE_DELTA_MAX_DB,
                                      SINR_FLOOR_DB, SINR_SCALE_DB, TOO_LATE_RATE_THRESHOLD,
                                      TOO_LATE_SINR_MARGIN_DB, MroZmqEnv, PairMarginState,
                                      PairRoleTracker, _extract_ue_cell_sinr_matrix,
                                      _pairwise_margin_targets, obs_index)

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

    # last_pair_margin_db is 0.0 for every cell before the first step().
    for cell in CELLS:
        assert obs[obs_index(cell, "last_pair_margin_db")] == 0.0
    env.close()


def test_absent_cell_is_zero_filled():
    env = make_env()
    env.reset(seed=0)
    # step 0 -> snapshot(1) is returned by the first step(); drive to step 2,
    # where cell 8 is dropped entirely from the payload.
    action = np.zeros(2, dtype=np.float32)
    env.step(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["kpis"]["timestamp"] == 2.0
    block = slice(obs_index(8, "sinr_db_mean"), obs_index(8, "sinr_db_mean") + len(OBS_FIELDS))
    # Every field except SINR is a real zero for an absent cell; SINR floors.
    assert obs[obs_index(8, "sinr_db_mean")] == np.float32(SINR_FLOOR_DB / SINR_SCALE_DB)
    assert obs[obs_index(8, "num_active_ues")] == 0.0
    env.close()


def test_action_delta_is_clipped_to_role_delta_max():
    env = make_env()
    env.reset(seed=0)
    # Deliberately out-of-range on both dims to exercise the action_space clip.
    action = np.array([ROLE_DELTA_MAX_DB + 5.0, -ROLE_DELTA_MAX_DB - 5.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    applied = info["reward_terms"]["diagnostics"]["role_gain_delta_db"]
    assert applied["too_late"] == ROLE_DELTA_MAX_DB
    assert applied["churn"] == -ROLE_DELTA_MAX_DB
    env.close()


def test_pairwise_payload_schema_and_sign():
    """With a known too-late pair and a known churn pair injected directly
    into env._pair_rates (bypassing PairRoleTracker's own detection, which is
    tested separately), one step() must send
    {"cells": {src: {"ho_margin_pairs": {dst: dB}}}} -- never the old
    "ho_margin_db" key -- with too-late LOWERING its one direction and churn
    RAISING both directions of its pair.
    """
    env = make_env()
    env.reset(seed=0)
    env._pair_rates = ({(2, 3): 1.0}, {(4, 5): 1.0})  # both fully saturated rates
    action = np.zeros(2, dtype=np.float32)  # pure base-gain rule
    env.step(action)
    sent = env.zmq_db.sent[-1]

    # too-late(2 -> 3): only that direction, negative.
    assert sent["cells"]["2"]["ho_margin_pairs"]["3"] < 0.0
    assert "ho_margin_db" not in sent["cells"]["2"]
    assert "3" not in sent["cells"].get("3", {}).get("ho_margin_pairs", {})

    # churn(4, 5): both directions, positive.
    assert sent["cells"]["4"]["ho_margin_pairs"]["5"] > 0.0
    assert sent["cells"]["5"]["ho_margin_pairs"]["4"] > 0.0
    env.close()


def test_pair_margin_state_rate_limits_then_clamps():
    """Pure unit test, no env: a target far outside the bound must be
    approached at exactly PAIR_RATE_LIMIT_DB_PER_STEP per step and never
    exceed +-MARGIN_LIMIT_DB even after many steps."""
    state = PairMarginState()
    key = (2, 3)
    seen = []
    for _ in range(20):
        offsets = state.step({key: 100.0})
        seen.append(offsets[key])
    # Monotonically rising at exactly the rate limit until the clamp bites.
    for i in range(1, len(seen)):
        assert seen[i] - seen[i - 1] <= PAIR_RATE_LIMIT_DB_PER_STEP + 1e-9
    assert seen[-1] == MARGIN_LIMIT_DB

    # Retargeting to 0.0 must descend at the same rate, not jump.
    for _ in range(3):
        offsets = state.step({})
    assert offsets[key] == MARGIN_LIMIT_DB - 3 * PAIR_RATE_LIMIT_DB_PER_STEP

    # Once it reaches exactly 0.0, the pair is pruned from state but the
    # final 0.0 must still have been sent at least once.
    steps_to_zero = int(round((MARGIN_LIMIT_DB - 3 * PAIR_RATE_LIMIT_DB_PER_STEP)
                              / PAIR_RATE_LIMIT_DB_PER_STEP))
    final_offsets = {}
    for _ in range(steps_to_zero):
        final_offsets = state.step({})
    assert final_offsets.get(key) == 0.0
    assert key not in state._offsets


def test_reward_terms_present_when_handovers_available():
    env = make_env()
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    terms = info["reward_terms"]
    # smoothness deliberately NOT in terms_present -- it's logged (still in
    # terms["smoothness"]) but not summed into reward, since
    # REWARD_FORMULA_VERSION 2 dropped it from the formula entirely.
    assert set(terms["diagnostics"]["terms_present"]) >= {"outage"}
    assert "smoothness" not in terms["diagnostics"]["terms_present"]
    assert isinstance(terms["reward"], float)
    assert isinstance(terms["smoothness"], float)
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
    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
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
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["reward_terms"]["outage"] == 0.5  # 1 of 2 UEs in outage
    env.close()


def test_outage_exposure_empty_network_is_zero_not_dropped():
    cells = {str(c): {"num_active_ues": 0, "sinr_bins": [0] * 7, "ues": {}} for c in CELLS}
    snap = {"timestamp": 0.0, "cells": cells, "handovers": [], "handover_window_s": 1.0}
    env = make_env(snapshots=[snap, dict(snap)])
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["reward_terms"]["outage"] == 0.0
    assert "outage" in info["reward_terms"]["diagnostics"]["terms_present"]
    env.close()


def test_action_smoothness_zero_on_first_step_then_tracks_delta():
    """v3: smoothness compares the 2-dim role-gain DELTA step to step (the
    agent's own action, clipped to +-ROLE_DELTA_MAX_DB), not an applied
    per-cell margin -- see MroZmqEnv._reward_terms."""
    env = make_env()
    env.reset(seed=0)
    a1 = np.array([1.0, 0.0], dtype=np.float32)
    _, _, _, _, info1 = env.step(a1)
    assert info1["reward_terms"]["smoothness"] == 0.0  # no prior action yet

    a2 = np.array([1.0 + ROLE_DELTA_MAX_DB, 0.0], dtype=np.float32)  # clips to ROLE_DELTA_MAX_DB
    _, _, _, _, info2 = env.step(a2)
    expected = float(np.mean((np.array([ROLE_DELTA_MAX_DB, 0.0])
                              - np.array([1.0, 0.0])) ** 2))
    assert abs(info2["reward_terms"]["smoothness"] - expected) < 1e-6
    env.close()


def test_action_smoothness_is_logged_but_not_rewarded():
    """REWARD_FORMULA_VERSION 2: ActionSmoothness is diagnostic-only. Two envs
    fed the IDENTICAL kpi sequence (so quality/pingpong/handover_rate/outage
    match exactly) but wildly different action deltas at step 2 -- a jerky
    swing to the ROLE_DELTA_MAX_DB bound vs. holding completely still -- must
    produce the SAME reward, even though "smoothness" itself differs hugely
    between them. Mutation check: reintroducing `- self.w_smooth * smoothness`
    into the sum makes this fail (reward would differ by the smoothness
    term's weight).
    """
    snaps = [_snapshot(0), _snapshot(1), _snapshot(1)]  # same snap repeats for step 2
    env_still = make_env(snapshots=[dict(s) for s in snaps])
    env_jerky = make_env(snapshots=[dict(s) for s in snaps])

    a1 = np.zeros(2, dtype=np.float32)
    env_still.reset(seed=0)
    env_jerky.reset(seed=0)
    env_still.step(a1)
    env_jerky.step(a1)

    a2_still = np.zeros(2, dtype=np.float32)
    a2_jerky = np.array([ROLE_DELTA_MAX_DB, -ROLE_DELTA_MAX_DB], dtype=np.float32)
    _, r_still, _, _, info_still = env_still.step(a2_still)
    _, r_jerky, _, _, info_jerky = env_jerky.step(a2_jerky)

    assert info_still["reward_terms"]["smoothness"] == 0.0
    assert info_jerky["reward_terms"]["smoothness"] > 1.0  # genuinely large delta
    assert r_still == r_jerky  # reward must not see the difference
    env_still.close()
    env_jerky.close()


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
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))  # consumes snap1
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
    env.step(np.zeros(2, dtype=np.float32))  # consumes snap1
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))  # consumes snap2
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
        return env._reward_terms(snap, np.zeros(2, dtype=np.float32))

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
    absent = env._reward_terms(snap_absent, np.zeros(2, dtype=np.float32))
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
        return env._reward_terms(snap, np.zeros(2, dtype=np.float32))

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


def _two_cell_sinr_snapshot(cell_a_sinr, cell_b_sinr=None, imsi=100, cell_a=2, cell_b=3):
    """A minimal synthetic payload reporting one UE's SINR under cell_a (and
    optionally also under cell_b) -- for PairRoleTracker's too-late tests,
    which need control over exactly which cells report a given imsi."""
    cells = {str(c): {"num_active_ues": 0, "sinr_bins": [0] * 7, "ues": {}} for c in CELLS}
    cells[str(cell_a)] = {"num_active_ues": 1, "sinr_bins": [0] * 7,
                          "ues": {str(imsi): {"l3_serving_sinr_db": cell_a_sinr}}}
    if cell_b_sinr is not None:
        cells[str(cell_b)] = {"num_active_ues": 1, "sinr_bins": [0] * 7,
                              "ues": {str(imsi): {"l3_serving_sinr_db": cell_b_sinr}}}
    return {"cells": cells}


def test_extract_ue_cell_sinr_matrix_keeps_every_candidate_cell():
    """Unlike _extract_ue_sinr (which collapses to one value per imsi), the
    pairwise matrix must keep BOTH cells' readings for the same UE -- this is
    the real ns-3 payload shape (see MroZmqEnv's class docstring), and the
    too-late diagnostic depends on seeing both at once."""
    kpis = _two_cell_sinr_snapshot(-2.0, 5.0)
    matrix = _extract_ue_cell_sinr_matrix(kpis)
    assert matrix[100] == {2: -2.0, 3: 5.0}


def test_pair_role_tracker_too_late_fires_after_sustained_advantage():
    tracker = PairRoleTracker()
    history: list = []

    # Step 1: only cell 2 reports -- establishes serving cell 2 unambiguously
    # via the single-cell-report inference, no qualifying sample yet.
    too_late, _ = tracker.update(_two_cell_sinr_snapshot(0.0, None), [], history)
    assert too_late[(2, 3)] == 0.0

    # PAIR_RATE_WINDOW_STEPS more steps: cell 3 now clearly better.
    for _ in range(PAIR_RATE_WINDOW_STEPS):
        too_late, _ = tracker.update(
            _two_cell_sinr_snapshot(0.0, TOO_LATE_SINR_MARGIN_DB), [], history)

    assert too_late[(2, 3)] == 1.0            # every window sample qualifies
    assert too_late[(3, 2)] == 0.0            # directional -- the reverse never fires
    assert too_late[(2, 4)] == 0.0            # an untouched pair stays at 0
    assert sum(too_late.values()) == 1.0      # nothing else lit up


def test_pair_role_tracker_too_late_does_not_fire_below_margin():
    """A candidate that's only marginally better (under TOO_LATE_SINR_MARGIN_DB)
    must never qualify, however long it persists."""
    tracker = PairRoleTracker()
    history: list = []
    tracker.update(_two_cell_sinr_snapshot(0.0, None), [], history)
    for _ in range(PAIR_RATE_WINDOW_STEPS):
        too_late, _ = tracker.update(
            _two_cell_sinr_snapshot(0.0, TOO_LATE_SINR_MARGIN_DB - 0.1), [], history)
    assert too_late[(2, 3)] == 0.0


def test_pair_role_tracker_churn_fires_on_return_leg():
    """Mirrors test_pingpong_detects_a_return_leg's scenario, but checked
    through PairRoleTracker's own pair-level bucketing instead of the
    reward's global pingpong_count."""
    tracker = PairRoleTracker()
    history: list = []

    too_late, churn = tracker.update({"cells": {}}, [], history)
    assert churn[(2, 3)] == 0.0

    # Pair-wise-role tracker expects already-PARSED (t, imsi, src, dst)
    # tuples, matching _parse_handovers's output -- the real format
    # _reward_terms passes it, not raw payload dicts.
    leg_out = [(0.3, 5, 2, 3)]
    history.extend(leg_out)  # mirrors _reward_terms's extend-before-call ordering
    _, churn = tracker.update({"cells": {}}, leg_out, history)
    assert churn[(2, 3)] == 0.0  # no return leg yet, must not fire early

    leg_back = [(0.7, 5, 3, 2)]  # within PINGPONG_Y_S=0.8
    history.extend(leg_back)
    _, churn = tracker.update({"cells": {}}, leg_back, history)
    assert churn[(2, 3)] > 0.0
    assert churn[(4, 5)] == 0.0  # an untouched pair stays at 0


def test_pairwise_margin_targets_combines_both_roles_additively():
    """The core of the literature-grounded conflict-resolution rule: both
    roles contribute a signed, rate-excess-weighted term to the SAME
    directed offset rather than one overriding the other. Uses DIFFERENT
    gains for the two roles specifically so an override bug (one role's
    contribution silently discarded) would change the numeric result, not
    just its sign.
    """
    too_late_rate = {(2, 3): 1.0}   # only this one directed pair fires
    churn_rate = {(2, 3): 1.0}      # the SAME pair, pair-level
    too_late_gain, churn_gain = 2.0, 1.0
    targets = _pairwise_margin_targets(too_late_rate, churn_rate, too_late_gain, churn_gain)

    excess = 1.0 - TOO_LATE_RATE_THRESHOLD  # == 1.0 - CHURN_RATE_THRESHOLD here
    expected_2_to_3 = churn_gain * excess - too_late_gain * excess  # both contribute
    expected_3_to_2 = churn_gain * excess                            # churn only
    assert abs(targets[(2, 3)] - expected_2_to_3) < 1e-9
    assert abs(targets[(3, 2)] - expected_3_to_2) < 1e-9


def test_pairwise_margin_targets_zero_delta_reproduces_base_gains():
    """Pins the warm-start property the whole v3 design depends on: with the
    agent's action_delta == 0 (so the effective gain passed in here is
    exactly BASE_TOO_LATE_GAIN_DB / BASE_CHURN_GAIN_DB), the computed target
    is a PURE function of the fixed base gains and the measured rates --
    nothing related to a trained policy. See MroZmqEnv's class docstring on
    action_space for why this must hold exactly.
    """
    too_late_rate = {(2, 3): 1.0}
    churn_rate = {(4, 5): 1.0}
    targets = _pairwise_margin_targets(too_late_rate, churn_rate,
                                       BASE_TOO_LATE_GAIN_DB, BASE_CHURN_GAIN_DB)
    expected_too_late = -BASE_TOO_LATE_GAIN_DB * (1.0 - TOO_LATE_RATE_THRESHOLD)
    expected_churn = BASE_CHURN_GAIN_DB * (1.0 - CHURN_RATE_THRESHOLD)
    assert abs(targets[(2, 3)] - expected_too_late) < 1e-9
    assert abs(targets[(4, 5)] - expected_churn) < 1e-9
    assert abs(targets[(5, 4)] - expected_churn) < 1e-9
    assert (3, 2) not in targets


def test_last_pair_margin_db_obs_reflects_touching_pairs():
    """A cell touched by an applied pair offset (as either serving or
    candidate) must report a nonzero last_pair_margin_db; an untouched cell
    must report exactly 0.0."""
    env = make_env()
    env.reset(seed=0)
    env._pair_rates = ({}, {(2, 3): 1.0})  # churn only, both directions
    # One step only: _get_obs below reads self._pair_margin_state as updated
    # by THIS step's _pairwise_payload_from_action, before _reward_terms
    # (called later in the same step()) overwrites self._pair_rates with a
    # fresh (here: all-zero) reading that would start decaying it back down.
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert obs[obs_index(2, "last_pair_margin_db")] > 0.0
    assert obs[obs_index(3, "last_pair_margin_db")] > 0.0
    assert obs[obs_index(4, "last_pair_margin_db")] == 0.0
    env.close()


def _drive_to_truncation(env, exit_code: int):
    """reset() + enough step()s to reach the truncation branch, with the
    stub's sim_result.meta.exitcode set to `exit_code` right before the
    final step() call that triggers it.
    """
    env.reset(seed=0)
    for _ in range(EPISODE_STEPS - 1):
        env.step(np.zeros(2, dtype=np.float32))
    env.sim_result['meta']['exitcode'] = exit_code
    return env.step(np.zeros(2, dtype=np.float32))


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
