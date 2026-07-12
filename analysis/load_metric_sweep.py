"""Throwaway multi-seed sweep of analysis/load_metric.py over the Phase-0
baseline runs (seed -> run dir taken from baseline_sweep/run_index.csv).

Read-only: only reads database.db files and prints. Seeds whose database does
not cover the full 99-timestep episode are excluded loudly, not averaged in.
"""

import csv
import io
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from load_metric import build_load_view, compute_load, COMPONENTS

RUN_INDEX = "/home/raadelasmar/oran-project/baseline_sweep/run_index.csv"
OUTPUT_ROOT = "/home/raadelasmar/oran-project/ns-o-ran-gym/output"
FULL_EPISODE_TS = 99

SCRATCH = os.environ.get(
    "SWEEP_SCRATCH",
    os.path.join(tempfile.gettempdir(), "load_metric_sweep_dbs"))


def coerced_copy(db_path: str, seed: int) -> str:
    """Copy the db to scratch and coerce TEXT leftovers in gnb_cu_cp.l3_serving_sinr.

    Baseline dbs predate the V13 ingestion fix: '-inf' outage markers were stored
    as TEXT in the REAL column. Python float() parses '-inf' correctly; SQLite's
    CAST would yield 0.0, so the coercion must happen here. The original file is
    never opened for writing.
    """
    os.makedirs(SCRATCH, exist_ok=True)
    dst = os.path.join(SCRATCH, f"seed_{seed}.db")
    shutil.copyfile(db_path, dst)
    con = sqlite3.connect(dst)
    cur = con.cursor()
    rows = cur.execute("SELECT rowid, l3_serving_sinr FROM gnb_cu_cp "
                       "WHERE typeof(l3_serving_sinr) = 'text'").fetchall()
    for rowid, val in rows:
        cur.execute("UPDATE gnb_cu_cp SET l3_serving_sinr = ? WHERE rowid = ?",
                    (float(val), rowid))
    con.commit()
    con.close()
    if rows:
        vals = sorted({v for _, v in rows})
        print(f"seed {seed}: coerced {len(rows)} TEXT l3_serving_sinr values "
              f"{vals[:4]}{'...' if len(vals) > 4 else ''} in scratch copy")
    return dst

def load_seed_views():
    views = {}
    with open(RUN_INDEX) as fh:
        for row in csv.DictReader(fh):
            seed, run_dir = int(row["seed"]), row["run_dir"]
            db = os.path.join(OUTPUT_ROOT, run_dir, "database.db")
            if not os.path.isfile(db):
                print(f"seed {seed}: NO database.db ({run_dir}) — excluded")
                continue
            db = coerced_copy(db, seed)
            buf = io.StringIO()
            with redirect_stdout(buf):
                view = build_load_view(db)
            n_ts = view["timestamp"].nunique()
            checks = "; ".join(l for l in buf.getvalue().splitlines()
                               if "MISMATCH" in l or "WARNING" in l)
            if checks:
                print(f"seed {seed}: {checks}")
            if n_ts < FULL_EPISODE_TS:
                print(f"seed {seed}: TRUNCATED database ({n_ts}/{FULL_EPISODE_TS} "
                      f"timestamps, {run_dir}) — excluded from episode stats")
                continue
            idle = (~view["present"]).sum()
            print(f"seed {seed}: OK ({n_ts} ts, present=False rows: {idle})")
            views[seed] = view
    return views


def busiest_series(df, col):
    return df.loc[df.groupby("timestamp")[col].idxmax()].set_index("timestamp")["cellid"].sort_index()


def churn(series):
    return int((series.values[1:] != series.values[:-1]).sum())


def main():
    print("=== seed inventory ===")
    views = load_seed_views()
    print(f"\nproceeding with {len(views)} seeds: {sorted(views)}")

    dfs = {}
    for seed, view in views.items():
        buf = io.StringIO()
        with redirect_stdout(buf):
            dfs[seed] = compute_load(view)  # default weights, sinr_agg='min'
        p95_line = [l for l in buf.getvalue().splitlines() if "VOL_P95" in l][0]
        print(f"seed {seed}: {p95_line}")

    # ---- 1. hotspot tracking
    print("\n=== 1. hotspot tracking: busiest cell by mean load vs by mean prb_pct ===")
    print("seed | busiest-by-load (mean) | busiest-by-PRB (mean) | agree?")
    hot_load, hot_prb = {}, {}
    for seed, df in dfs.items():
        ml = df.groupby("cellid")["load"].mean()
        mp = df.groupby("cellid")["prb_pct"].mean()
        hot_load[seed], hot_prb[seed] = int(ml.idxmax()), int(mp.idxmax())
        print(f"{seed} | {hot_load[seed]} ({ml.max():.3f}) | {hot_prb[seed]} ({mp.max():.1f}) "
              f"| {'YES' if hot_load[seed] == hot_prb[seed] else 'NO'}")
    print(f"busiest-by-load values across seeds: {sorted(set(hot_load.values()))}")
    print(f"busiest-by-PRB  values across seeds: {sorted(set(hot_prb.values()))}")

    # ---- 2. temporal variation
    print("\n=== 2. temporal variation of load ===")
    for seed, df in dfs.items():
        stats = df.groupby("cellid")["load"].agg(["mean", "std", "min", "max"]).round(3)
        line = "  ".join(f"c{c}:{r['mean']:.2f}±{r['std']:.2f}[{r['min']:.2f},{r['max']:.2f}]"
                         for c, r in stats.iterrows())
        busiest = hot_load[seed]
        sd_busiest = df[df["cellid"] == busiest]["load"].std()
        sd_across = df.groupby("timestamp")["load"].std().mean()
        print(f"seed {seed}: {line}")
        print(f"          sd over time of busiest cell (c{busiest}) load = {sd_busiest:.3f}; "
              f"mean over time of across-cell sd = {sd_across:.3f}")

    # ---- 3. rank churn
    print("\n=== 3. rank churn: busiest-cell identity changes per episode (98 transitions) ===")
    print("seed | churn by load | churn by prb_pct")
    for seed, df in dfs.items():
        print(f"{seed} | {churn(busiest_series(df, 'load'))} | {churn(busiest_series(df, 'prb_pct'))}")

    # ---- 4. component dominance
    print("\n=== 4. component dominance (pooled across all seeds and cells) ===")
    pooled = pd.concat(dfs.values(), keys=dfs.keys(), names=["seed"]).reset_index(level=0)
    for comp in COMPONENTS:
        pear = pooled["load"].corr(pooled[comp])
        spear = pooled["load"].corr(pooled[comp], method="spearman")
        flag = "  <-- >0.95" if max(pear, spear) > 0.95 else ""
        print(f"{comp}: Pearson={pear:.3f}  Spearman={spear:.3f}{flag}")
    print(f"(pooled rows: {len(pooled)} = seeds x cells x timestamps)")


if __name__ == "__main__":
    main()
