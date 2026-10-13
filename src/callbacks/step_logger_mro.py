"""Per-step, in-process logging of MRO's reward terms and actions.

Mirrors callbacks/step_logger.py's design exactly (same reasons: reconstruct-
after-the-fact from ns-3 KPI files breaks when the reward formula changes and
structurally misses the last step of every episode, since
MroZmqEnv(purge_sim_path_on_close=True) deletes the run dir at every episode
boundary -- see that module's docstring for the full rationale, identical
here), but NOT a subclass or an import of StepLogger: MRO's reward_terms
shape is its own (five terms, mostly network-wide scalars, not MLB's
elaborate per-cell KPI breakdown), and the two envs' info dicts use different
keys (`margin_offsets` vs `cio_offsets`). Kept as an independent copy for the
same file-independence reason the rest of this project's MRO-side modules
already follow.

Also logs `ns3_exit_code` / `crashed` on terminal rows -- MroZmqEnv.step()
only started surfacing these recently (previously a clean simTime finish and
a mid-episode crash were indistinguishable from the return value alone, both
landing on the same truncation branch). Logging them here means a real
training run's data can finally show, after the fact, whether a given
episode's end was a legitimate truncation or a crash that happened to land
on a complete step count.
"""
import csv
import json
import time
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Same 7-cell mmWave topology as MroZmqEnv.CELLS.
_CELLS = [2, 3, 4, 5, 6, 7, 8]

# Scalar terms of the reward, in the order they appear in the sum.
_TERMS = ("quality", "pingpong", "handover_rate", "outage", "smoothness")

# Diagnostic scalars from reward_terms["diagnostics"]. margin_action_db is
# per-cell and emitted separately, below.
_DIAG_FLAGS = ("handovers_unavailable",)
_DIAG_NUMS = ("active_ues", "n_ues_reporting_sinr", "handovers_this_step",
              "handover_history_len", "handover_window_s", "handover_rate_scale",
              "handover_rate_ref_s", "reward_formula_version", "pingpong_count",
              "pingpong_y_s", "quality_good", "quality_measured", "outage_sinr_db")

FIELDS = (["wall_iso", "vec_step", "num_timesteps", "env_rank", "episode_index",
           "episode_step", "phase", "rng_run", "reward", "done"]
          + list(_TERMS)
          + ["terms_present"]
          + list(_DIAG_FLAGS)
          + list(_DIAG_NUMS)
          + ["ns3_exit_code", "crashed"]
          + [f"action_raw_cell{c}" for c in _CELLS]
          + [f"action_clipped_cell{c}" for c in _CELLS]
          + [f"margin_action_db_cell{c}" for c in _CELLS]
          + ["monitor_ep_r", "monitor_ep_l"])


def _f(x):
    """CSV cell for a value that may be None/np scalar/bool/tuple."""
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (tuple, list)):
        return "|".join(str(v) for v in x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


class MroStepLogger(BaseCallback):
    """Writes one CSV row per (vec step, env) with MRO's reward terms and actions.

    Parameters
    ----------
    csv_path : str
        Row-per-step output. Flushed on every row, same reasoning as
        StepLogger: a run killed hours in must not lose its tail.
    meta_path : str, optional
        Sidecar JSON with the run-level constants (weights, margin limit,
        scenario config). Defaults to csv_path with a "_meta.json" suffix.
    """

    def __init__(self, csv_path: str, meta_path: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.meta_path = meta_path or (csv_path.rsplit(".", 1)[0] + "_meta.json")
        self.rows_written = 0
        self._fh = None
        self._w = None
        self._vec_step = 0
        self._episode_index = None
        self._episode_step = None
        self._rng_runs = None
        self._refresh_rng = True

    # -- lifecycle ---------------------------------------------------------
    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._episode_index = [0] * n
        self._episode_step = [0] * n
        self._rng_runs = [None] * n
        self._refresh_rng = True

        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=FIELDS, extrasaction="raise")
        self._w.writeheader()
        self._fh.flush()
        self._write_meta()
        if self.verbose:
            print(f"MroStepLogger -> {self.csv_path} ({len(FIELDS)} columns, {n} envs)",
                  flush=True)

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def _write_meta(self) -> None:
        """Run-level constants, read off the live envs rather than re-declared."""
        meta = {"n_envs": self.training_env.num_envs, "cells": _CELLS, "fields": FIELDS}
        for attr in ("w_quality", "w_pingpong", "w_churn", "w_outage", "w_smooth",
                     "total_ues", "scenario_configuration"):
            try:
                meta[attr] = self._sanitise(self.training_env.get_attr(attr)[0])
            except Exception as exc:                  # attribute absent on a stub env
                meta[attr] = f"<unavailable: {type(exc).__name__}>"
        try:
            meta["action_space_high"] = [float(v) for v in self.training_env.action_space.high]
        except Exception:
            pass
        with open(self.meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

    @staticmethod
    def _sanitise(v):
        if isinstance(v, dict):
            return {str(k): MroStepLogger._sanitise(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [MroStepLogger._sanitise(x) for x in v]
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            return v.item()
        return v

    # -- per-step ----------------------------------------------------------
    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        raw = np.asarray(self.locals.get("actions", []), dtype=float)
        clipped = np.asarray(self.locals.get("clipped_actions", raw), dtype=float)
        raw = raw.reshape(len(infos), -1) if raw.size else np.zeros((len(infos), 0))
        clipped = (clipped.reshape(len(infos), -1) if clipped.size
                   else np.zeros((len(infos), 0)))

        if self._refresh_rng:
            self._read_rng_runs()
            self._refresh_rng = False

        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for i, info in enumerate(infos):
            self._w.writerow(self._row(stamp, i, info, rewards, dones, raw, clipped))
            self.rows_written += 1

        for i, d in enumerate(dones):
            self._episode_step[i] += 1
            if d:
                self._episode_index[i] += 1
                self._episode_step[i] = 0
                self._refresh_rng = True

        self._fh.flush()
        self._vec_step += 1
        return True

    def _row(self, stamp, i, info, rewards, dones, raw, clipped) -> dict:
        terms = (info or {}).get("reward_terms")
        offsets = (info or {}).get("margin_offsets") or {}
        diag = (terms or {}).get("diagnostics") or {}
        ep = (info or {}).get("episode") or {}

        if terms is not None:
            phase = "step"
        elif "margin_offsets" in (info or {}):
            # Time-limit end (clean finish OR a crash -- see ns3_exit_code/
            # crashed below, the whole point of logging them here).
            phase = "terminal"
        else:
            phase = "void"

        row = {f: "" for f in FIELDS}
        row.update({
            "wall_iso": stamp,
            "vec_step": self._vec_step,
            "num_timesteps": self.num_timesteps,
            "env_rank": i,
            "episode_index": self._episode_index[i],
            "episode_step": self._episode_step[i],
            "phase": phase,
            "rng_run": _f(self._rng_runs[i]),
            "reward": _f(rewards[i]) if i < rewards.size else "",
            "done": _f(bool(dones[i])) if i < dones.size else "",
            "monitor_ep_r": _f(ep.get("r")),
            "monitor_ep_l": _f(ep.get("l")),
            "ns3_exit_code": _f((info or {}).get("ns3_exit_code")),
            "crashed": _f((info or {}).get("crashed")),
        })
        for t in _TERMS:
            row[t] = _f((terms or {}).get(t))
        row["terms_present"] = _f(diag.get("terms_present"))
        for k in _DIAG_FLAGS:
            row[k] = _f(diag.get(k))
        for k in _DIAG_NUMS:
            row[k] = _f(diag.get(k))

        margin_db = diag.get("margin_action_db") or {}
        for j, c in enumerate(_CELLS):
            row[f"action_raw_cell{c}"] = _f(raw[i][j]) if j < raw.shape[1] else ""
            row[f"action_clipped_cell{c}"] = (_f(clipped[i][j]) if j < clipped.shape[1] else "")
            # margin_action_db is keyed by int cell id (dict(zip(CELLS, ...))
            # in mro_zmq_env.py), not str -- unlike offsets, which comes
            # through the ZMQ/JSON payload and is str-keyed.
            if c in margin_db:
                row[f"margin_action_db_cell{c}"] = _f(margin_db[c])
            else:
                cell = offsets.get(str(c))
                row[f"margin_action_db_cell{c}"] = _f(cell.get("ho_margin_db")) if cell else ""
        return row

    def _read_rng_runs(self) -> None:
        """The per-episode ns-3 seed, re-read after every reset.

        reset() redraws RngRun (vary_rng_run_per_episode); reset()'s own info
        dict carries it, but a reset info never reaches _on_step, so it is
        read off the envs instead -- identical approach to StepLogger's own
        _read_rng_runs (MRO has no equivalent of MLB's udpFullBufferIntervalUs
        randomisation knob, so that half is simply absent here).
        """
        try:
            cfgs = self.training_env.get_attr("scenario_configuration")
        except Exception:
            return
        for i, cfg in enumerate(cfgs):
            if i >= len(self._rng_runs):
                break
            cfg = cfg or {}
            try:
                self._rng_runs[i] = int(cfg["RngRun"])
            except (KeyError, TypeError, ValueError):
                self._rng_runs[i] = None
