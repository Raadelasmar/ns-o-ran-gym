"""Observation builder for the MLB RL agent (read-only).

Given a completed run's database.db and a single timestamp, returns that step's
observation as a 7x5 float array.

Rows: mmWave cells 2..8, always in that fixed order (load_metric.NR_CELLS).
Cols: 0 prb_pct, 1 n_ues, 2 buffer_bytes, 3 sinr_p10, 4 median_mcs.

Cols 0-3 are taken from load_metric.build_load_view and are NOT re-queried here.
Col 4 is computed in this module from the du CARR.PDSCHMCSDist bins, which
build_load_view does not read.

Verified facts this module relies on (checked against run b3efb1ee, 2026-08-06):
  - The du cell-level columns are repeated identically on every UE row of a cell
    at a timestamp (0 disagreeing groups DB-wide), so the MCS bins are taken
    once per (cellid, timestamp) with .first() -- never summed. Summing would
    multiply every bin count by the cell's UE count.
  - The du cell-id column is nrcellid; gnb_cu_cp uses cellid. build_load_view
    aliases its own query, so only the direct MCS query below needs nrcellid.
  - A cell that served no UEs at a timestamp is ABSENT from du entirely, not
    present as a zero row (16 of 99 timestamps in b3efb1ee, all cell 4). Absent
    cells are therefore emitted here as an all-zero row rather than dropped.

Values are returned RAW and un-normalized, as specified, with one exception: a
final guard replaces every non-finite entry, so np.isfinite(obs).all() always
holds. Col 3 would otherwise carry -inf for a present cell whose worst serving
UE is in outage (the datalake stores -Inf as a REAL, and the collision rule in
datalake.py deliberately keeps it); such entries become -40.0 dB, and non-finite
entries in any other column become 0.0. Finite values are never altered.
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

from load_metric import NR_CELLS, build_load_view

# Cell-level MCS histogram columns, bin 1 (MCS 0-4) .. bin 6 (MCS 25-29).
MCS_BIN_COLS = [f"carrpdschmcsdistbin{i}" for i in range(1, 7)]

# Column order of the returned observation matrix.
OBS_COLUMNS = ["prb_pct", "n_ues", "buffer_bytes", "sinr_p10", "median_mcs"]


def _median_mcs_bin(counts: np.ndarray) -> float:
    """Weighted-median bin index (1-6) of an MCS histogram; 0 if the bins are empty.

    `counts` holds the number of transmissions falling in each of the 6 MCS
    ranges. Returns the lowest bin index whose cumulative count reaches half of
    the total, i.e. the bin containing the median transmission.
    """
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    cumulative = np.cumsum(counts.astype(float))
    return float(np.searchsorted(cumulative, total / 2.0, side="left") + 1)


def median_mcs_at(db_path: str, timestamp: int) -> pd.Series:
    """Median MCS bin per mmWave cell at one timestamp, indexed by cellid.

    Queries du directly (build_load_view does not read the MCS bins). Cells
    absent from du at this timestamp are simply missing from the result; the
    caller reindexes and fills them with 0.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        du = pd.read_sql_query(
            "SELECT timestamp, ueimsicomplete, nrcellid AS cellid, "
            + ", ".join(MCS_BIN_COLS)
            + " FROM du WHERE timestamp = ? AND nrcellid BETWEEN 2 AND 8",
            con,
            params=(int(timestamp),),
        )
    finally:
        con.close()

    if du.empty:
        return pd.Series(dtype=float, name="median_mcs")

    # Same guard as load_metric.build_load_view: these are cell-level columns
    # repeated across the cell's UE rows, so every group must agree before we
    # collapse it with .first().
    nunique = du.groupby(["cellid", "timestamp"])[MCS_BIN_COLS].nunique(dropna=False)
    disagreeing = nunique[(nunique > 1).any(axis=1)]
    if not disagreeing.empty:
        raise ValueError(
            "cell-level du MCS bins disagree within (cellid, timestamp) groups "
            f"({len(disagreeing)} groups):\n{disagreeing}"
        )

    cell_bins = du.groupby(["cellid", "timestamp"], as_index=False)[MCS_BIN_COLS].first()
    medians = cell_bins[MCS_BIN_COLS].fillna(0.0).to_numpy(dtype=float)
    return pd.Series(
        [_median_mcs_bin(row) for row in medians],
        index=cell_bins["cellid"].to_numpy(),
        name="median_mcs",
    )


def get_observation(db_path: str, timestamp: int,
                    view: pd.DataFrame | None = None) -> np.ndarray:
    """Return the 7x5 observation for `timestamp`.

    Rows follow NR_CELLS ([2..8]) in fixed order; a cell absent from the data at
    this timestamp is emitted as an all-zero row.

    `view` lets a caller build_load_view() once and reuse it across timestamps;
    when omitted it is rebuilt on every call (build_load_view scans the whole
    run and prints its own diagnostics, so pass it in for stepwise use).
    """
    if view is None:
        view = build_load_view(db_path)

    step = view[view["timestamp"] == timestamp]
    if step.empty:
        raise ValueError(f"timestamp {timestamp} not present in {db_path}")

    step = step.set_index("cellid").reindex(NR_CELLS)
    mcs = median_mcs_at(db_path, timestamp).reindex(NR_CELLS).fillna(0.0)

    obs = np.zeros((len(NR_CELLS), len(OBS_COLUMNS)), dtype=float)
    obs[:, 0] = step["prb_pct"].to_numpy(dtype=float)
    obs[:, 1] = step["n_ues"].to_numpy(dtype=float)
    obs[:, 2] = step["buffer_bytes"].to_numpy(dtype=float)
    obs[:, 3] = step["sinr_p10"].to_numpy(dtype=float)
    obs[:, 4] = mcs.to_numpy(dtype=float)

    # build_load_view zeroes prb/n_ues/vol/buffer for idle cells but leaves the
    # SINR columns NaN; rule is that an absent cell contributes an all-zero row.
    absent = ~step["present"].fillna(False).to_numpy(dtype=bool)
    obs[absent, :] = 0.0

    # Infinity/NaN guard: no non-finite value may leave this function. Column 3
    # (sinr_p10) floors at -40 dB, a physically-reasonable dead-signal level;
    # every other column floors at 0. Finite entries are left untouched -- this
    # is a guard, not normalization.
    nonfinite = ~np.isfinite(obs)
    if nonfinite.any():
        floors = np.zeros(len(OBS_COLUMNS), dtype=float)
        floors[OBS_COLUMNS.index("sinr_p10")] = -40.0
        obs[nonfinite] = np.broadcast_to(floors, obs.shape)[nonfinite]
    return obs


def first_timestamp(db_path: str) -> int:
    """Earliest du timestamp in the database."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return int(con.execute("SELECT MIN(timestamp) FROM du").fetchone()[0])
    finally:
        con.close()


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else None
    if db is None:
        from load_metric import DEFAULT_DB
        db = DEFAULT_DB

    ts = first_timestamp(db)
    observation = get_observation(db, ts)

    print(f"\ndb        : {db}")
    print(f"timestamp : {ts} (first in du)")
    print(f"shape     : {observation.shape}\n")
    header = "cell  " + "".join(f"{name:>16}" for name in OBS_COLUMNS)
    print(header)
    print("-" * len(header))
    for cell, row in zip(NR_CELLS, observation):
        print(f"{cell:<6}" + "".join(f"{v:>16.2f}" for v in row))
