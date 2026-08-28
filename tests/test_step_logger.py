"""Schema and bookkeeping checks for StepLogger, WITHOUT SB3, ns-3 or ZeroMQ.

StepLogger only ever READS `self.locals` and `self.training_env`, so it can be
driven with a synthetic rollout: a fake vec env supplying num_envs/get_attr, and
hand-built `locals` dicts in exactly the shape
OnPolicyAlgorithm.collect_rollouts hands to callback.update_locals(). That keeps
the test honest about the two shapes that actually bite --

  * the TERMINAL step, where MlbZmqEnv's time-limit path returns reward 0.0 and
    NO "reward_terms" key, and
  * a term legitimately DROPPED from the sum (satisfaction/pingpong None),
    which must stay distinguishable from a term that scored 0.0

-- without needing a simulator to produce them.

Run:  pytest tests/test_step_logger.py     or     python tests/test_step_logger.py
"""
import csv
import json
import sys
import tempfile
from os import path

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from callbacks.step_logger import FIELDS, StepLogger

CELLS = [2, 3, 4, 5, 6, 7, 8]
N_ENVS = 2


class _FakeModel:
    """BaseCallback reads training_env off model.get_env() and num_timesteps off
    the model, and both are read-only properties on the callback -- so the seam
    for a test is a fake MODEL, not a fake attribute."""

    def __init__(self, venv):
        self._venv = venv
        self.num_timesteps = 0

    def get_env(self):
        return self._venv


class _FakeVecEnv:
    """Just the surface StepLogger touches: num_envs, get_attr, action_space."""

    def __init__(self, n_envs=N_ENVS):
        self.num_envs = n_envs
        self._rng = [1001 + i for i in range(n_envs)]
        self.action_space = type("S", (), {"high": np.full(len(CELLS), 6.0)})()

    def get_attr(self, name, indices=None):
        if name == "scenario_configuration":
            return [{"RngRun": r, "ues": 5} for r in self._rng]
        if name.startswith("w_"):
            return [{"w_balance": 1.0, "w_backlog": 0.1, "w_badsignal": 1.0,
                     "w_satisfaction": 1.1, "w_pingpong": 1.5}[name]] * self.num_envs
        if name in ("control_period_s", "udp_ue_rate_kbps", "n_udp", "total_ues"):
            return [{"control_period_s": 0.1, "udp_ue_rate_kbps": 20960.0,
                     "n_udp": 9, "total_ues": 35}[name]] * self.num_envs
        raise AttributeError(name)

    def new_episode(self):
        self._rng = [r + 500 for r in self._rng]


def _cells_diag(key_type=int):
    """The per-cell blocks of reward_terms["diagnostics"], keyed as the env keys
    them (int cell ids). key_type=str simulates a JSON round trip."""
    k = (lambda c: key_type(c)) if key_type is not int else (lambda c: c)
    return {
        "ci": {k(c): 0.05 + 0.01 * i for i, c in enumerate(CELLS)},
        "prb_utilization": {k(c): 0.30 + 0.05 * i for i, c in enumerate(CELLS)},
        "n_ues": {k(c): float(3 + i) for i, c in enumerate(CELLS)},
        "sinr_bins": {k(c): [2.0 + i, 10.0, 20.0, 40.0, 30.0, 15.0, 5.0]
                      for i, c in enumerate(CELLS)},
    }


def _kpis(drop_cell=None):
    """A payload carrying the per-cell PDCP dicts StepLogger reads out of it,
    plus a BULK field that must never reach the CSV."""
    cells = {}
    for i, c in enumerate(CELLS):
        if c == drop_cell:
            continue
        cells[str(c)] = {"pdcp_delivered_bytes": {str(1 + 4 * i): 1.0e5 + 1.0e3 * i,
                                                  str(2 + 4 * i): 3.0e4 + 1.0e2 * i},
                         "bulk_payload_sentinel": "…" * 20}
    return {"cells": cells, "bulk_payload_sentinel": "…" * 20}


def _terms(reward, *, satisfaction=0.5, pingpong=0.1, key_type=int):
    """A reward_terms dict in MlbZmqEnv._reward_terms' real shape.

    The scalar aggregates are DERIVED from the per-cell blocks, exactly as
    _reward_terms derives them. Hardcoding them instead let the fixture claim
    active_ues=35 while the per-cell n_ues summed to 42 -- and the consistency
    assertion below caught it, which is the assertion's whole point.
    """
    present = (["balance", "backlog", "badsignal"]
               + (["satisfaction"] if satisfaction is not None else [])
               + (["pingpong"] if pingpong is not None else []))
    cells = _cells_diag(key_type)
    n_ues_vals = list(cells["n_ues"].values())
    prb_vals = list(cells["prb_utilization"].values())
    bin_rows = list(cells["sinr_bins"].values())
    return {
        "reward": reward, "balance": 0.6, "backlog": 3.2, "badsignal": 0.0,
        "satisfaction": satisfaction, "pingpong": pingpong,
        "weights": {"balance": 1.0, "backlog": 0.1, "badsignal": 1.0, "satisfaction": 1.1},
        "diagnostics": {
            "active_ues": float(sum(n_ues_vals)),
            "max_prb_utilization": float(max(prb_vals)),
            "backlog_bytes": 4.2e6,
            "delivered_bytes": 1.3e6, "delivery_rate_bytes_per_s": 1.3e7,
            "delivered_udp_bytes": 9.1e5, "satisfaction_demand_kbps": 188640.0,
            "udp_ue_rate_kbps": 20960.0, "n_udp": 9, "pingpong_count": 3,
            "handovers_this_step": 12, "handover_history_len": 21,
            "badsignal_bad_count": float(sum(b[0] for b in bin_rows)),
            "badsignal_total_tx": float(sum(sum(b) for b in bin_rows)),
            "pingpong_y_s": 0.8, "control_period_s": 0.1, "pdcp_window_s": 1.0,
            "balance_guarded": False, "backlog_denominator_zero": False,
            "backlog_rate_source": "pdcp", "badsignal_denominator_zero": False,
            "satisfaction_unavailable": satisfaction is None,
            "pdcp_field_present": True, "pingpong_unavailable": pingpong is None,
            "median_mcs_gate_applied": False,
            "delivered_by_imsi": {1: 1.0e5, 5: 2.0e5},
            "udp_imsis": [1 + 4 * i for i in range(len(CELLS))],
            "terms_present": tuple(present),
            **cells,
        },
    }


def _offsets(vals):
    return {str(c): {"cio_offset": float(v)} for c, v in zip(CELLS, vals)}


def _drive(cb, venv, rollout):
    """Feed StepLogger a rollout: [(infos, rewards, dones, raw, clipped), ...].

    Mirrors OnPolicyAlgorithm.collect_rollouts: bump the model's timestep
    counter by num_envs, publish `locals`, then call the PUBLIC on_step() so the
    callback's own num_timesteps bookkeeping is exercised rather than bypassed.
    """
    cb.on_training_start({}, {})
    for infos, rewards, dones, raw, clipped in rollout:
        cb.update_locals({"infos": infos, "rewards": np.array(rewards, dtype=np.float32),
                          "dones": np.array(dones, dtype=bool),
                          "actions": np.array(raw, dtype=np.float32),
                          "clipped_actions": np.array(clipped, dtype=np.float32)})
        cb.model.num_timesteps += venv.num_envs
        cb.on_step()
        if any(dones):
            venv.new_episode()
    cb.on_training_end()


def _make(tmpdir):
    venv = _FakeVecEnv()
    cb = StepLogger(path.join(tmpdir, "steps.csv"))
    cb.init_callback(_FakeModel(venv))
    return venv, cb


def _rollout(n_steps, done_at=None):
    """n_steps vec steps; the step at index done_at ends the episode."""
    out = []
    for k in range(n_steps):
        done = (k == done_at)
        raw = [[(-1) ** i * (k + 0.5 + j) for j in range(len(CELLS))] for i in range(N_ENVS)]
        clipped = [[float(np.clip(v, -6.0, 6.0)) for v in row] for row in raw]
        if done:
            # Time-limit end: MlbZmqEnv returns reward 0.0 and NO reward_terms;
            # Monitor then attaches info["episode"].
            infos = [{"kpis": _kpis(), "cio_offsets": _offsets(clipped[i]),
                      "episode": {"r": 9.5 + i, "l": done_at + 1}}
                     for i in range(N_ENVS)]
            rewards = [0.0] * N_ENVS
        else:
            infos = [{"kpis": _kpis(), "cio_offsets": _offsets(clipped[i]),
                      "reward_terms": _terms(0.4 + 0.1 * k + i)} for i in range(N_ENVS)]
            rewards = [0.4 + 0.1 * k + i for i in range(N_ENVS)]
        out.append((infos, rewards, [done] * N_ENVS, raw, clipped))
    return out


def _read(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


# -- the tests -------------------------------------------------------------
def test_schema_and_row_count():
    """Header is exactly FIELDS, and there is one row per (vec step, env)."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        n_steps = 5
        _drive(cb, venv, _rollout(n_steps))
        rows = _read(cb.csv_path)

        with open(cb.csv_path) as fh:
            header = next(csv.reader(fh))
        assert header == FIELDS, "header drifted from FIELDS"
        assert len(rows) == n_steps * N_ENVS == cb.rows_written
        for r in rows:
            assert set(r) == set(FIELDS), "row has columns outside the schema"


def test_no_gaps_in_env_step_index():
    """(env_rank, vec_step) covers the grid exactly once -- no gaps, no repeats."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        n_steps = 6
        _drive(cb, venv, _rollout(n_steps))
        rows = _read(cb.csv_path)

        seen = {(int(r["env_rank"]), int(r["vec_step"])) for r in rows}
        assert seen == {(e, s) for e in range(N_ENVS) for s in range(n_steps)}
        # num_timesteps advances by num_envs per vec step
        for r in rows:
            assert int(r["num_timesteps"]) == (int(r["vec_step"]) + 1) * N_ENVS


def test_terms_actions_and_offsets_recorded():
    """All five terms, the raw action, the clipped action and the applied CIO
    offsets land in the row -- and the raw action is NOT overwritten by the
    clipped one (the whole point of logging both)."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        _drive(cb, venv, _rollout(4))
        rows = [r for r in _read(cb.csv_path) if r["phase"] == "step"]
        assert rows

        for r in rows:
            for t in ("balance", "backlog", "badsignal", "satisfaction", "pingpong"):
                assert r[t] != "", f"{t} missing on a normal step"
            assert r["terms_present"] == "balance|backlog|badsignal|satisfaction|pingpong"
            assert r["pingpong_unavailable"] == "0"
            for c in CELLS:
                assert r[f"action_raw_cell{c}"] != ""
                assert r[f"action_clipped_cell{c}"] != ""
                assert r[f"cio_offset_cell{c}"] != ""

        # Somewhere in the rollout the raw action exceeds +/-6 dB and is clipped;
        # if the two columns were the same value we would never see it.
        clipped_somewhere = any(
            abs(float(r[f"action_raw_cell{c}"]) - float(r[f"action_clipped_cell{c}"])) > 1e-9
            for r in rows for c in CELLS)
        assert clipped_somewhere, "clipping never observed -- the two columns are redundant"
        # The applied offset always equals the clipped action.
        for r in rows:
            for c in CELLS:
                assert abs(float(r[f"cio_offset_cell{c}"])
                           - float(r[f"action_clipped_cell{c}"])) < 1e-6


def test_terminal_step_shape():
    """The episode-ending step has phase=terminal, empty terms, reward 0.0 and
    Monitor's episode totals -- recorded as a real shape, not dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        _drive(cb, venv, _rollout(4, done_at=2))
        rows = _read(cb.csv_path)

        term = [r for r in rows if r["phase"] == "terminal"]
        assert len(term) == N_ENVS
        for r in term:
            assert r["done"] == "1"
            assert float(r["reward"]) == 0.0
            assert r["balance"] == "" and r["satisfaction"] == ""
            assert r["monitor_ep_r"] != "" and r["monitor_ep_l"] != ""
            assert r[f"cio_offset_cell{CELLS[0]}"] != ""   # the action still went out


def test_episode_bookkeeping_and_rng_refresh():
    """episode_step counts inside the episode, the terminal step still belongs
    to it, and the ns-3 seed is re-read after the reset."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        _drive(cb, venv, _rollout(5, done_at=2))
        rows = [r for r in _read(cb.csv_path) if int(r["env_rank"]) == 0]

        assert [int(r["episode_step"]) for r in rows] == [0, 1, 2, 0, 1]
        assert [int(r["episode_index"]) for r in rows] == [0, 0, 0, 1, 1]
        assert rows[2]["phase"] == "terminal", "the done step must close episode 0"

        rng = [r["rng_run"] for r in rows]
        assert rng[0] == rng[1] == rng[2], "seed changed mid-episode"
        assert rng[3] == rng[4] and rng[3] != rng[0], "seed not re-read after reset"


def test_dropped_term_distinguishable_from_zero():
    """satisfaction=None (dropped from the sum) must not be written as 0.0."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        infos = [{"kpis": _kpis(), "cio_offsets": _offsets([0.0] * len(CELLS)),
                  "reward_terms": _terms(0.3, satisfaction=None, pingpong=0.0)}
                 for _ in range(N_ENVS)]
        zeros = [[0.0] * len(CELLS)] * N_ENVS
        _drive(cb, venv, [(infos, [0.3] * N_ENVS, [False] * N_ENVS, zeros, zeros)])
        r = _read(cb.csv_path)[0]

        assert r["satisfaction"] == "", "a DROPPED term was written as a value"
        assert r["satisfaction_unavailable"] == "1"
        assert float(r["pingpong"]) == 0.0, "a term that really scored 0.0 was blanked"
        assert r["pingpong_unavailable"] == "0"
        assert r["terms_present"] == "balance|backlog|badsignal|pingpong"


def test_meta_sidecar():
    """Run-level constants land in the sidecar rather than in every row."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        _drive(cb, venv, _rollout(2))
        meta = json.load(open(cb.meta_path))

        assert meta["n_envs"] == N_ENVS
        assert meta["fields"] == FIELDS
        assert meta["w_pingpong"] == 1.5 and meta["w_backlog"] == 0.1
        assert meta["scenario_configuration"]["ues"] == 5


def test_per_cell_columns_populate_and_reconstruct_the_aggregates():
    """Every per-cell block is filled, varies BETWEEN cells, and sums back to
    the aggregate the reward actually used. A block that silently collapsed to
    one value per step would pass a mere "is it populated" check."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        _drive(cb, venv, _rollout(3))
        rows = [r for r in _read(cb.csv_path) if r["phase"] == "step"]
        assert rows

        def block(r, fmt):
            return [float(r[fmt.format(c=c)]) for c in CELLS]

        for r in rows:
            for f in ("prb_utilization", "n_ues", "ci",
                      "delivered_bytes", "delivered_udp_bytes"):
                v = block(r, f + "_cell{c}")
                assert all(x == x for x in v), f"{f} has a gap"
                assert len(set(v)) > 1, f"{f} is the same on every cell"
            bins = [block(r, "sinr_" + b + "_cell{c}")
                    for b in ("le_m6", "le_0", "le_6", "le_12", "le_18", "le_24", "gt_24")]
            assert len(set(bins[0])) > 1, "sinr bin 0 identical on every cell"

            # The reward's own aggregates must be reproducible from the blocks.
            assert abs(sum(block(r, "n_ues_cell{c}")) - float(r["active_ues"])) < 1e-6
            assert abs(max(block(r, "prb_utilization_cell{c}"))
                       - float(r["max_prb_utilization"])) < 1e-6
            assert abs(sum(bins[0]) - float(r["badsignal_bad_count"])) < 1e-6
            assert abs(sum(sum(b) for b in bins) - float(r["badsignal_total_tx"])) < 1e-6
            # UDP delivery is a strict subset of total, per cell.
            tot = block(r, "delivered_bytes_cell{c}")
            udp = block(r, "delivered_udp_bytes_cell{c}")
            assert all(u <= t + 1e-6 for u, t in zip(udp, tot))
            assert any(t - u > 1.0 for u, t in zip(udp, tot)), "udp == total everywhere"


def test_per_cell_absent_cell_is_empty_not_zero():
    """A cell missing from the payload has NO delivery figure. Writing 0.0 would
    claim it delivered nothing, which is the three-zeros mistake."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        gone = CELLS[-1]
        infos = [{"kpis": _kpis(drop_cell=gone), "cio_offsets": _offsets([0.0] * len(CELLS)),
                  "reward_terms": _terms(0.3)} for _ in range(N_ENVS)]
        zeros = [[0.0] * len(CELLS)] * N_ENVS
        _drive(cb, venv, [(infos, [0.3] * N_ENVS, [False] * N_ENVS, zeros, zeros)])
        r = _read(cb.csv_path)[0]

        assert r[f"delivered_bytes_cell{gone}"] == ""
        assert r[f"delivered_udp_bytes_cell{gone}"] == ""
        assert float(r[f"delivered_bytes_cell{CELLS[0]}"]) > 0
        # diagnostics still covers every cell, so those blocks stay filled.
        assert r[f"prb_utilization_cell{gone}"] != ""


def test_per_cell_accepts_str_or_int_diagnostic_keys():
    """diagnostics is keyed by int cell id in-process; a JSON round trip would
    make it str. Both must read."""
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        infos = [{"kpis": _kpis(), "cio_offsets": _offsets([0.0] * len(CELLS)),
                  "reward_terms": _terms(0.3, key_type=str)} for _ in range(N_ENVS)]
        zeros = [[0.0] * len(CELLS)] * N_ENVS
        _drive(cb, venv, [(infos, [0.3] * N_ENVS, [False] * N_ENVS, zeros, zeros)])
        r = _read(cb.csv_path)[0]
        for c in CELLS:
            assert r[f"prb_utilization_cell{c}"] != "", f"str key lost cell {c}"
            assert r[f"ci_cell{c}"] != ""


def test_bulk_kpi_payload_never_logged():
    """The per-cell PDCP dicts are read OUT of info["kpis"], but the payload
    itself must not land in the CSV."""
    assert not any("kpi" in f for f in FIELDS)
    with tempfile.TemporaryDirectory() as tmp:
        venv, cb = _make(tmp)
        _drive(cb, venv, _rollout(2))
        text = open(cb.csv_path).read()
        assert "…" not in text, "bulk payload leaked into the CSV"
        assert "bulk_payload_sentinel" not in text


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            bad += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
