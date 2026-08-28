"""Per-step, in-process logging of the MLB reward terms and the actions.

The reward terms and the CIO offsets used to be reconstructed after the run from
the ns-3 KPI files, by a separate out-of-process script. That broke twice: the
script drifted out of sync with the env when the Backlog denominator changed at
FIX 13, and a wrong ping-pong column mapping produced a whole CSV that had to be
thrown away. Reconstruction also cannot be made complete, because
MlbZmqEnv(purge_sim_path_on_close=True) deletes the run dir at every episode
boundary, so an out-of-process tailer structurally misses the last step of every
episode. Measured on the 2026-08-22 pilot: 31% of steps missing overall, and
k=29 missing from 25 out of 25 episodes.

This records the numbers at the source instead. MlbZmqEnv.step() already puts
them in `info`, so the training loop can just write them down.

Not Monitor(info_keywords=...): Monitor copies info_keywords into ep_info inside
its `if terminated or truncated` branch, so it yields one row per episode
carrying the last step's values, not the per-step series this needs. Keep Monitor
as well, since it gives an independent per-episode reward total to check this
file against, but the per-step record has to come from a callback reading
`self.locals`.

Everything is logged per cell because the action is per cell. Aggregates cannot
answer the only question an MLB agent poses: I biased cell 5 down, did cell 3
gain the UEs? The reward is a scalar and max_prb/active_ues are scalars, so a run
logged in aggregate can show the reward rising without ever showing which cell
moved. So every per-cell input to every term is logged per cell: prb_utilization,
n_ues, ci (the Balance numerator), delivered bytes (total and UDP-only, the
Satisfaction numerator) and the 7 SINR bins (the BadSignal numerator).

info["kpis"] as a whole is deliberately not logged. It already crosses the
SubprocVecEnv pipe every step, so logging it would cost no extra IPC, but it
would balloon the CSV by orders of magnitude for data the terms already
summarise. Two specific fields are read out of it, the per-cell PDCP delivery
dicts, because _pdcp_totals merges them across cells before reaching
diagnostics, so the per-cell split exists nowhere else. Everything else comes
from reward_terms["diagnostics"], which is what the reward actually consumed
rather than a second reading of the payload.

Two SB3 details this relies on. First, OnPolicyAlgorithm.collect_rollouts calls
callback.update_locals(locals()) after env.step(), so `self.locals` holds both
`actions` (the raw policy output) and `clipped_actions` (clipped to the action
space). Logging both makes the clipping visible. MlbZmqEnv._cio_from_action then
clips again to +/-CIO_LIMIT_DB, and info["cio_offsets"] is what ns-3 was actually
sent. Second, on the step that ends an episode the VecEnv auto-resets and
MlbZmqEnv's time-limit path returns reward 0.0 with no "reward_terms" key. That
is a real, expected shape rather than missing data, so it is recorded as
phase="terminal" rather than dropped or written as zeros.
"""
import csv
import json
import time
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# One column per cell, in the env's fixed CELLS order.
_CELLS = [2, 3, 4, 5, 6, 7, 8]

# Scalar terms of the reward, in the order they appear in the sum.
_TERMS = ("balance", "backlog", "badsignal", "satisfaction", "pingpong")

# The 7 L1M.RS-SINR bins, named by their dB edge rather than by the offline
# analysis' "bin34..bin127" labels. Same order as _sinr_bins() documents:
#   [0] <= -6 dB  [1] <= 0  [2] <= 6  [3] <= 12  [4] <= 18  [5] <= 24  [6] > 24
# Index 0 is BadSignal's numerator, so `sinr_le_m6_cell5 / sum(sinr_*_cell5)` is
# cell 5's own BadSignal.
_SINR_BINS = ("le_m6", "le_0", "le_6", "le_12", "le_18", "le_24", "gt_24")

# Per-cell blocks. ci, prb_utilization and n_ues come from
# reward_terms["diagnostics"]; the two delivery columns come from
# info["kpis"]["cells"][c]["pdcp_delivered_bytes"] (see the module docstring).
# The SINR bins are diagnostics-sourced too but are emitted separately, via
# _SINR_BINS.
_PER_CELL = ("prb_utilization", "n_ues", "ci", "delivered_bytes", "delivered_udp_bytes")

# Flags and counters from reward_terms["diagnostics"] that say how a term was
# reached. A term that scored 0.0 and a term that was dropped from the sum must
# stay distinguishable: the env keeps those three zeros distinct, and so does
# this file.
_DIAG_FLAGS = ("balance_guarded", "backlog_denominator_zero", "backlog_rate_source",
               "badsignal_denominator_zero", "satisfaction_unavailable",
               "pdcp_field_present", "pingpong_unavailable", "median_mcs_gate_applied")

# Diagnostic scalars: the physical numerators and denominators behind the terms.
# These are what a reward-hacking check correlates the reward against, so keeping
# them in the same row makes the CSV self-sufficient.
_DIAG_NUMS = ("active_ues", "max_prb_utilization", "backlog_bytes", "delivered_bytes",
              "delivery_rate_bytes_per_s", "delivered_udp_bytes", "delivered_by_imsi_total",
              "satisfaction_demand_kbps", "udp_ue_rate_kbps", "n_udp",
              "pingpong_count", "handovers_this_step", "handover_history_len",
              "badsignal_bad_count", "badsignal_total_tx", "pingpong_y_s",
              "control_period_s", "pdcp_window_s")

FIELDS = (["wall_iso", "vec_step", "num_timesteps", "env_rank", "episode_index",
           "episode_step", "phase", "rng_run", "udp_interval_us", "reward", "done"]
          + list(_TERMS)
          + ["terms_present"]
          + list(_DIAG_FLAGS)
          + list(_DIAG_NUMS)
          + [f"{f}_cell{c}" for f in _PER_CELL for c in _CELLS]
          + [f"sinr_{b}_cell{c}" for b in _SINR_BINS for c in _CELLS]
          + [f"action_raw_cell{c}" for c in _CELLS]
          + [f"action_clipped_cell{c}" for c in _CELLS]
          + [f"cio_offset_cell{c}" for c in _CELLS]
          + ["monitor_ep_r", "monitor_ep_l"])


def _cell_key(d, cell):
    """diagnostics dicts are keyed by int cell id, but a JSON round trip would
    make them str. Accept either rather than depending on which side built them."""
    if not isinstance(d, dict):
        return None
    if cell in d:
        return d[cell]
    return d.get(str(cell))


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


class StepLogger(BaseCallback):
    """Writes one CSV row per (vec step, env) with the reward terms and actions.

    Parameters
    ----------
    csv_path : str
        Row-per-step output. Flushed on every row: at ~57 s/timestep the flush
        is free, and a run killed after 16 h must not lose its tail.
    meta_path : str, optional
        Sidecar JSON with the run-level constants (weights, CIO limit, scenario
        config). Defaults to csv_path with a "_meta.json" suffix. These are
        constant for the run, so they belong here rather than in every row.
    """

    def __init__(self, csv_path: str, meta_path: Optional[str] = None, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.meta_path = meta_path or (csv_path.rsplit(".", 1)[0] + "_meta.json")
        self.rows_written = 0
        self._fh = None
        self._w = None
        self._vec_step = 0
        self._episode_index = None    # per env: episodes completed so far
        self._episode_step = None     # per env: steps taken in the current episode
        self._rng_runs = None         # per env: ns-3 seed of the current episode
        self._udp_intervals = None    # per env: offered-load knob this episode
        self._refresh_rng = True      # re-read both after every reset

    # -- lifecycle ---------------------------------------------------------
    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._episode_index = [0] * n
        self._episode_step = [0] * n
        self._rng_runs = [None] * n
        self._udp_intervals = [None] * n
        self._refresh_rng = True

        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=FIELDS, extrasaction="raise")
        self._w.writeheader()
        self._fh.flush()
        self._write_meta()
        if self.verbose:
            print(f"StepLogger -> {self.csv_path} ({len(FIELDS)} columns, {n} envs)",
                  flush=True)

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def _write_meta(self) -> None:
        """Run-level constants, read off the live envs rather than re-declared."""
        meta = {"n_envs": self.training_env.num_envs, "cells": _CELLS, "fields": FIELDS}
        for attr in ("w_balance", "w_backlog", "w_badsignal", "w_satisfaction",
                     "w_pingpong", "control_period_s", "udp_ue_rate_kbps", "n_udp",
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
            return {str(k): StepLogger._sanitise(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [StepLogger._sanitise(x) for x in v]
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

        # Episode bookkeeping AFTER the rows, so the terminal step is still
        # numbered inside the episode that produced it.
        for i, d in enumerate(dones):
            self._episode_step[i] += 1
            if d:
                self._episode_index[i] += 1
                self._episode_step[i] = 0
                self._refresh_rng = True      # the env reset, so a new ns-3 seed

        self._fh.flush()
        self._vec_step += 1
        return True

    def _row(self, stamp, i, info, rewards, dones, raw, clipped) -> dict:
        terms = (info or {}).get("reward_terms")
        offsets = (info or {}).get("cio_offsets") or {}
        diag = (terms or {}).get("diagnostics") or {}
        ep = (info or {}).get("episode") or {}

        if terms is not None:
            phase = "step"
        elif "cio_offsets" in (info or {}):
            # Time-limit end: MlbZmqEnv returns reward 0.0 and no reward_terms.
            phase = "terminal"
        else:
            # step() called with no pending KPI snapshot, so info is empty.
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
            "udp_interval_us": _f(self._udp_intervals[i]),
            "reward": _f(rewards[i]) if i < rewards.size else "",
            "done": _f(bool(dones[i])) if i < dones.size else "",
            "monitor_ep_r": _f(ep.get("r")),
            "monitor_ep_l": _f(ep.get("l")),
        })
        for t in _TERMS:
            row[t] = _f((terms or {}).get(t))
        row["terms_present"] = _f(diag.get("terms_present"))
        for k in _DIAG_FLAGS:
            row[k] = _f(diag.get(k))
        for k in _DIAG_NUMS:
            if k == "delivered_by_imsi_total":
                by_imsi = diag.get("delivered_by_imsi") or {}
                row[k] = _f(float(sum(by_imsi.values())) if by_imsi else None)
            else:
                row[k] = _f(diag.get(k))
        # --- per-cell KPI blocks -------------------------------------------
        kpi_cells = ((info or {}).get("kpis") or {}).get("cells") or {}
        udp_imsis = set(diag.get("udp_imsis") or ())
        for c in _CELLS:
            for f in ("prb_utilization", "n_ues", "ci"):
                row[f"{f}_cell{c}"] = _f(_cell_key(diag.get(f), c))
            bins = _cell_key(diag.get("sinr_bins"), c) or []
            for b, name in enumerate(_SINR_BINS):
                row[f"sinr_{name}_cell{c}"] = _f(bins[b]) if b < len(bins) else ""
            # A cell absent from the payload leaves these empty; a cell that
            # reported and delivered nothing writes 0.0. Same three-zeros
            # discipline the terms use.
            cell_kpis = kpi_cells.get(str(c), kpi_cells.get(c))
            drained = (cell_kpis or {}).get("pdcp_delivered_bytes")
            if drained is None:
                row[f"delivered_bytes_cell{c}"] = ""
                row[f"delivered_udp_bytes_cell{c}"] = ""
            else:
                tot = udp = 0.0
                for imsi_str, val in drained.items():
                    try:
                        imsi, val = int(imsi_str), float(val)
                    except (TypeError, ValueError):
                        continue
                    tot += val
                    if imsi in udp_imsis:
                        udp += val
                row[f"delivered_bytes_cell{c}"] = _f(tot)
                row[f"delivered_udp_bytes_cell{c}"] = _f(udp)

        for j, c in enumerate(_CELLS):
            row[f"action_raw_cell{c}"] = _f(raw[i][j]) if j < raw.shape[1] else ""
            row[f"action_clipped_cell{c}"] = (_f(clipped[i][j]) if j < clipped.shape[1] else "")
            cell = offsets.get(str(c))
            row[f"cio_offset_cell{c}"] = _f(cell.get("cio_offset")) if cell else ""
        return row

    def _read_rng_runs(self) -> None:
        """The per-episode scenario knobs, re-read after every reset.

        reset() redraws RngRun (vary_rng_run_per_episode) and, when a range is
        configured, udpFullBufferIntervalUs, so neither can be cached once at
        start-up. reset()'s own info dict carries both, but a reset info never
        reaches _on_step, so they are read off the envs instead.

        udpFullBufferIntervalUs is absent from the config when randomisation is
        off, which must read as "not randomised" (empty) rather than as a value.
        """
        try:
            cfgs = self.training_env.get_attr("scenario_configuration")
        except Exception:
            return
        for i, cfg in enumerate(cfgs):
            if i >= len(self._rng_runs):
                break
            cfg = cfg or {}
            for key, store in (("RngRun", self._rng_runs),
                               ("udpFullBufferIntervalUs", self._udp_intervals)):
                try:
                    store[i] = int(cfg[key])
                except (KeyError, TypeError, ValueError):
                    store[i] = None
