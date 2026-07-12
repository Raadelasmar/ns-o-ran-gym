"""Standalone read-only load-metric analysis over a completed run's database.db.

Builds one row per (cellid, timestamp) for NR cells 2-8 (build_load_view),
computes a compound load metric in [0,1] (compute_load), and — when run as a
script — validates whether the compound metric orders cells any differently
than raw PRB utilization alone.

Facts this module relies on (verified against run ee4dc79c, 2026-07-12):
  - du cell-level columns (dlprbusage, qosflowpdcppduvolumedl_filter,
    drbbuffersizeqos) are repeated identically on every UE row of a cell at a
    timestamp; they must be taken once, never summed. Asserted here.
  - gnb_cu_cp.numactiveues is likewise cell-repeated (= ueMap.size() at report
    time, mmwave-enb-net-device.cc:1087).
  - A cell absent from du at a timestamp genuinely served zero UEs (raw
    du-cell-*.txt files contain no rows there; dedup discarded nothing in the
    audited run), so absence is encoded as an idle row, not dropped.
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

NR_CELLS = list(range(2, 9))
TOTAL_UES = 21  # 3 UEs/gNB * 7 gNBs in scenario-three

CELL_LEVEL_DU_COLS = ["dlprbusage", "qosflowpdcppduvolumedl_filter", "drbbuffersizeqos"]

DEFAULT_WEIGHTS = {"prb": 0.4, "ues": 0.2, "vol": 0.2, "qual": 0.2}

DEFAULT_DB = ("/home/raadelasmar/oran-project/ns-o-ran-gym/output/"
              "ee4dc79c-69d5-47e5-8f2c-95ffa5a7568c/database.db")


def build_load_view(db_path: str) -> pd.DataFrame:
    """One row per (cellid, timestamp), NR cells 2-8, full timestamp grid.

    Columns: cellid, timestamp, prb_pct, n_ues, vol_bytes, sinr_min, sinr_p10,
    sinr_mean, buffer_bytes, present. Idle cells (no du rows) get zeros for the
    volume/count columns, NaN SINR, present=False.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    du = pd.read_sql_query(
        "SELECT timestamp, ueimsicomplete, nrcellid AS cellid, "
        "dlprbusage, qosflowpdcppduvolumedl_filter, drbbuffersizeqos "
        "FROM du WHERE nrcellid BETWEEN 2 AND 8", con)
    cp = pd.read_sql_query(
        "SELECT timestamp, ueimsicomplete, cellid, numactiveues, l3_serving_sinr "
        "FROM gnb_cu_cp WHERE cellid BETWEEN 2 AND 8", con)
    con.close()

    # Cell-level du columns must be identical across a cell's UE rows: take
    # first, never sum. Raise if any group disagrees.
    nunique = du.groupby(["cellid", "timestamp"])[CELL_LEVEL_DU_COLS].nunique(dropna=False)
    disagreeing = nunique[(nunique > 1).any(axis=1)]
    if not disagreeing.empty:
        raise ValueError(
            "cell-level du columns disagree within (cellid, timestamp) groups "
            f"({len(disagreeing)} groups):\n{disagreeing}")

    cell_du = du.groupby(["cellid", "timestamp"], as_index=False).agg(
        prb_pct=("dlprbusage", "first"),
        vol_bytes=("qosflowpdcppduvolumedl_filter", "first"),
        buffer_bytes=("drbbuffersizeqos", "first"),
        n_du_rows=("ueimsicomplete", "size"),
    )

    # numactiveues is cell-repeated too; verify before taking first.
    nu = cp.groupby(["cellid", "timestamp"])["numactiveues"].nunique(dropna=False)
    if (nu > 1).any():
        raise ValueError(
            f"gnb_cu_cp.numactiveues disagrees within {(nu > 1).sum()} (cellid, timestamp) groups")

    cell_cp = cp.groupby(["cellid", "timestamp"], as_index=False).agg(
        n_ues=("numactiveues", "first"),
        n_cp_rows=("ueimsicomplete", "size"),
        sinr_min=("l3_serving_sinr", "min"),
        sinr_p10=("l3_serving_sinr", lambda s: s.quantile(0.10)),
        sinr_mean=("l3_serving_sinr", "mean"),
    )

    # Cross-check numactiveues against the surviving per-UE row count.
    mismatch = cell_cp[cell_cp["n_ues"] != cell_cp["n_cp_rows"]]
    if mismatch.empty:
        print("cross-check OK: numactiveues == per-cell gnb_cu_cp row count for "
              f"all {len(cell_cp)} (cell, timestamp) groups")
    else:
        print(f"cross-check MISMATCH in {len(mismatch)} (cell, timestamp) groups "
              "(numactiveues vs surviving row count) — keeping numactiveues, "
              "mismatching groups follow:")
        print(mismatch.to_string(index=False))

    # Full grid: cells 2-8 x every timestamp present anywhere in the run.
    all_ts = sorted(set(du["timestamp"]).union(cp["timestamp"]))
    grid = pd.MultiIndex.from_product([NR_CELLS, all_ts], names=["cellid", "timestamp"])
    view = (pd.DataFrame(index=grid).reset_index()
            .merge(cell_du, on=["cellid", "timestamp"], how="left")
            .merge(cell_cp, on=["cellid", "timestamp"], how="left"))

    view["present"] = view["n_du_rows"].notna()

    # A cell in one table but not the other would make 'present' ambiguous;
    # surface it rather than paper over it (0 such groups in the audited run).
    cp_only = view["present"].ne(view["n_cp_rows"].notna())
    if cp_only.any():
        print(f"WARNING: {cp_only.sum()} (cell, timestamp) groups present in only "
              "one of du/gnb_cu_cp; idle-row rule below uses du presence")

    idle = ~view["present"]
    view.loc[idle, ["prb_pct", "n_ues", "vol_bytes", "buffer_bytes"]] = 0.0
    view.loc[idle, ["sinr_min", "sinr_p10", "sinr_mean"]] = np.nan

    n_idle = int(idle.sum())
    per_cell_idle = view.loc[idle, "cellid"].value_counts().sort_index().to_dict()
    print(f"present=False rows: {n_idle} (per cell: {per_cell_idle}) — "
          f"expected 52 with {{3: 2, 4: 12, 6: 38}} for run ee4dc79c")

    return view[["cellid", "timestamp", "prb_pct", "n_ues", "vol_bytes",
                 "sinr_min", "sinr_p10", "sinr_mean", "buffer_bytes", "present"]]


def compute_load(view: pd.DataFrame, weights: dict | None = None,
                 sinr_agg: str = "min") -> pd.DataFrame:
    """Add component columns c_prb/c_ues/c_vol/c_qual and a 'load' column in [0,1]."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    assert set(w) == set(DEFAULT_WEIGHTS), f"weights must have keys {sorted(DEFAULT_WEIGHTS)}"
    assert abs(sum(w.values()) - 1.0) < 1e-9, f"weights must sum to 1, got {sum(w.values())}"
    assert sinr_agg in ("min", "p10", "mean"), sinr_agg

    df = view.copy()
    df["c_prb"] = df["prb_pct"] / 100.0
    df["c_ues"] = df["n_ues"] / TOTAL_UES

    vol_p95 = df.loc[df["present"], "vol_bytes"].quantile(0.95)
    print(f"VOL_P95 (95th pct of vol_bytes over present rows) = {vol_p95:.1f} bytes/period")
    df["c_vol"] = (df["vol_bytes"] / vol_p95).clip(0.0, 1.0)

    sinr = df[f"sinr_{sinr_agg}"]
    df["c_qual"] = ((30.0 - sinr) / 35.0).clip(0.0, 1.0)
    df.loc[df["n_ues"] == 0, "c_qual"] = 0.0

    for c in ("c_prb", "c_ues", "c_vol", "c_qual"):
        col = df[c]
        assert col.notna().all(), f"{c} has NaN"
        assert col.between(0.0, 1.0).all(), f"{c} outside [0,1]"

    df["load"] = (w["prb"] * df["c_prb"] + w["ues"] * df["c_ues"]
                  + w["vol"] * df["c_vol"] + w["qual"] * df["c_qual"])
    assert df["load"].between(0.0, 1.0).all(), "load outside [0,1]"
    return df


# --------------------------------------------------------------------------
# Part 3 — validation
# --------------------------------------------------------------------------

COMPONENTS = ["c_prb", "c_ues", "c_vol", "c_qual"]


def _busiest_per_ts(df: pd.DataFrame, col: str) -> pd.Series:
    """cellid with the max of `col` at each timestamp (first cell on ties)."""
    return df.loc[df.groupby("timestamp")[col].idxmax()].set_index("timestamp")["cellid"]


def _per_ts_spearman(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.Series:
    """Spearman correlation of the 7-cell load ranking at each timestamp."""
    a = df_a.pivot(index="timestamp", columns="cellid", values="load")
    b = df_b.pivot(index="timestamp", columns="cellid", values="load")
    return pd.Series({ts: a.loc[ts].corr(b.loc[ts], method="spearman") for ts in a.index})


def validate(db_path: str) -> None:
    view = build_load_view(db_path)
    df = compute_load(view)  # default weights, sinr_agg='min'

    print("\n=== per-cell stats: load and components (default weights, sinr_agg=min) ===")
    stats = df.groupby("cellid")[["load"] + COMPONENTS].agg(["mean", "std", "min", "max"])
    print(stats.round(3).to_string())

    print("\n=== falsification test: busiest-by-load vs busiest-by-prb_pct ===")
    by_load = _busiest_per_ts(df, "load")
    by_prb = _busiest_per_ts(df, "prb_pct")
    disagree = by_load[by_load != by_prb]
    prb_wide = df.pivot(index="timestamp", columns="cellid", values="prb_pct")
    n_prb_ties = int((prb_wide.eq(prb_wide.max(axis=1), axis=0).sum(axis=1) > 1)
                     .loc[disagree.index].sum()) if len(disagree) else 0
    print(f"timestamps: {by_load.size}; disagreements: {len(disagree)} "
          f"({len(disagree) / by_load.size:.0%}); of those, {n_prb_ties} occur where "
          "prb_pct max is tied across cells (argmax then picks the first tied cell)")
    if disagree.empty:
        print("They NEVER disagree: on this run the compound metric selects the same "
              "busiest cell as raw PRB utilization at every timestamp — the extra "
              "components change nothing that PRB alone doesn't.")
    else:
        print("\nfirst 5 disagreements (components for both cells):")
        for ts in disagree.index[:5]:
            rows = df[(df["timestamp"] == ts) & (df["cellid"].isin([by_load[ts], by_prb[ts]]))]
            cols = ["cellid", "prb_pct", "n_ues", "vol_bytes", "load"] + COMPONENTS
            print(f"\n  ts {ts}: busiest by load = cell {by_load[ts]}, "
                  f"by prb_pct = cell {by_prb[ts]}")
            print(rows[cols].round(3).to_string(index=False))

    print("\n=== weight sensitivity (sinr_agg=min) ===")
    weightings = {
        "prb_only": {"prb": 1.0, "ues": 0.0, "vol": 0.0, "qual": 0.0},
        "default": DEFAULT_WEIGHTS,
        "ue_qual_heavy": {"prb": 0.2, "ues": 0.3, "vol": 0.1, "qual": 0.4},
    }
    loads = {name: compute_load(view, weights=w) for name, w in weightings.items()}
    names = list(weightings)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            rho = _per_ts_spearman(loads[na], loads[nb])
            ba, bb = _busiest_per_ts(loads[na], "load"), _busiest_per_ts(loads[nb], "load")
            print(f"{na} vs {nb}: per-ts Spearman mean={rho.mean():.3f} "
                  f"min={rho.min():.3f}; busiest cell differs at "
                  f"{(ba != bb).sum()}/{ba.size} timestamps")

    print("\n=== sinr_agg sensitivity (default weights) ===")
    aggs = {agg: compute_load(view, sinr_agg=agg) for agg in ("min", "p10", "mean")}
    busiest = {agg: _busiest_per_ts(d, "load") for agg, d in aggs.items()}
    for i, aa in enumerate(list(aggs)):
        for ab in list(aggs)[i + 1:]:
            n_diff = int((busiest[aa] != busiest[ab]).sum())
            print(f"sinr_agg={aa} vs {ab}: busiest cell differs at "
                  f"{n_diff}/{busiest[aa].size} timestamps")


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB)
