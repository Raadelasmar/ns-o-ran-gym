"""Eyeball inspection of the compound load score, 7 complete baseline seeds.

Two weightings:
  A (current):  prb .40, ues .20, vol .20, qual .20
  B (proposed): prb .15, ues .40, vol .10, qual .35

No statistics beyond counting. Prints full per-cell context so a human can
judge whether the flagged cell is one they would offload.
Read-only: works on the sweep's coerced scratch copies; load_metric.py untouched.
"""

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from load_metric import build_load_view, compute_load
from load_metric_sweep import coerced_copy, RUN_INDEX, OUTPUT_ROOT

SEEDS = [555, 556, 560, 561, 562, 563, 564]
W_A = {"prb": 0.40, "ues": 0.20, "vol": 0.20, "qual": 0.20}
W_B = {"prb": 0.15, "ues": 0.40, "vol": 0.10, "qual": 0.35}

SHOW = ["prb_pct", "n_ues", "vol_bytes", "sinr_min", "sinr_mean", "buffer_bytes"]


def load_all() -> pd.DataFrame:
    import csv
    with open(RUN_INDEX) as fh:
        dirs = {int(r["seed"]): r["run_dir"] for r in csv.DictReader(fh)}
    frames = []
    for seed in SEEDS:
        db = coerced_copy(os.path.join(OUTPUT_ROOT, dirs[seed], "database.db"), seed)
        with redirect_stdout(io.StringIO()):
            view = build_load_view(db)
            a = compute_load(view, weights=W_A)
            b = compute_load(view, weights=W_B)
        a = a.rename(columns={"load": "score_A"})
        a["score_B"] = b["load"]
        a["seed"] = seed
        frames.append(a)
    df = pd.concat(frames, ignore_index=True)
    df["rank_A"] = df.groupby(["seed", "timestamp"])["score_A"].rank(
        ascending=False, method="min").astype(int)
    df["rank_B"] = df.groupby(["seed", "timestamp"])["score_B"].rank(
        ascending=False, method="min").astype(int)
    return df


def fmt_cell(r, mark="") -> str:
    sinr_min = "  nan" if pd.isna(r["sinr_min"]) else f"{r['sinr_min']:5.1f}"
    sinr_mean = "  nan" if pd.isna(r["sinr_mean"]) else f"{r['sinr_mean']:5.1f}"
    return (f"    c{int(r['cellid'])}{mark:<3} prb={r['prb_pct']:5.1f}  ues={int(r['n_ues']):2d}  "
            f"vol={r['vol_bytes']:9.0f}  sinr_min={sinr_min}  sinr_mean={sinr_mean}  "
            f"buf={r['buffer_bytes']:9.0f}  A={r['score_A']:.3f}(r{r['rank_A']})  "
            f"B={r['score_B']:.3f}(r{r['rank_B']})")


def print_case(df, seed, ts, marks: dict):
    snap = df[(df["seed"] == seed) & (df["timestamp"] == ts)].sort_values("cellid")
    for _, r in snap.iterrows():
        print(fmt_cell(r, marks.get(int(r["cellid"]), "")))


def top15(df, col, other_rank):
    top = df.nlargest(15, col)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"\n#{i}: seed {r['seed']} ts {r['timestamp']} cell {int(r['cellid'])}  "
              f"{col}={r[col]:.3f}  rank under {'A' if other_rank == 'rank_A' else 'B'}: "
              f"{int(r[other_rank])}")
        print_case(df, r["seed"], r["timestamp"], {int(r["cellid"]): "<--"})


def main():
    df = load_all()
    n_ts = df.groupby("seed")["timestamp"].nunique().sum()
    print(f"rows: {len(df)}, seed-timesteps: {n_ts}")

    print("\n================ Part 1: 15 highest under B ================")
    top15(df, "score_B", "rank_A")

    print("\n================ Part 2: 15 highest under A ================")
    top15(df, "score_A", "rank_B")

    print("\n================ Part 3: A vs B disagreement on busiest cell ================")
    busiest = df.loc[df.groupby(["seed", "timestamp"])["score_A"].idxmax(),
                     ["seed", "timestamp", "cellid"]].rename(columns={"cellid": "cell_A"})
    bB = df.loc[df.groupby(["seed", "timestamp"])["score_B"].idxmax(),
                ["seed", "timestamp", "cellid"]].rename(columns={"cellid": "cell_B"})
    busiest = busiest.merge(bB, on=["seed", "timestamp"])
    dis = busiest[busiest["cell_A"] != busiest["cell_B"]].reset_index(drop=True)
    print(f"disagreeing seed-timesteps: {len(dis)}/{len(busiest)}")
    picks = dis.iloc[np.linspace(0, len(dis) - 1, min(10, len(dis))).astype(int)]
    for _, d in picks.iterrows():
        seed, ts, ca, cb = d["seed"], d["timestamp"], int(d["cell_A"]), int(d["cell_B"])
        snap = df[(df["seed"] == seed) & (df["timestamp"] == ts)]
        others = snap[~snap["cellid"].isin([ca, cb])].copy()
        # obvious offload target: fewest users, then best (highest) sinr_min
        others["sinr_key"] = others["sinr_min"].fillna(np.inf)
        tgt = int(others.sort_values(["n_ues", "sinr_key"],
                                     ascending=[True, False]).iloc[0]["cellid"])
        print(f"\nseed {seed} ts {ts}: A picks c{ca}, B picks c{cb}, "
              f"offload target c{tgt} (fewest users, best sinr_min)")
        print_case(df, seed, ts, {ca: "<A", cb: "<B", tgt: "<T"})

    print("\n================ Part 4: summary counts ================")
    for name, col in (("A", "score_A"), ("B", "score_B")):
        worst_sinr = most_ues = n_sinr = 0
        for (seed, ts), snap in df.groupby(["seed", "timestamp"]):
            top = snap.loc[snap[col].idxmax()]
            big = snap[snap["n_ues"] >= 3]
            if not big.empty and big["sinr_min"].notna().any():
                n_sinr += 1
                worst = big.loc[big["sinr_min"].idxmin()]
                if top["cellid"] == worst["cellid"]:
                    worst_sinr += 1
            if top["n_ues"] == snap["n_ues"].max():
                most_ues += 1
        n_all = len(busiest)
        print(f"{name}: busiest == worst-sinr_min cell (among cells with >=3 UEs): "
              f"{worst_sinr}/{n_sinr} ({worst_sinr / n_sinr:.0%});  "
              f"busiest == most-UEs cell: {most_ues}/{n_all} ({most_ues / n_all:.0%})")


if __name__ == "__main__":
    main()
