"""Throwaway diagnostic for the compound load metric, over the 7 complete
baseline seeds (555, 556, 560, 561, 562, 563, 564).

A: component collinearity (pooled Pearson/Spearman matrices).
B: spectral_eff = vol_bytes / max(RRU.PrbUsedDl, 1) as a candidate term.
C: predictive test — AUC of each predictor at t against a degradation label
   at t+D (D = 1, 3, 5 timesteps of 100 ms).
D: weight grid-search over {c_prb, c_ues, c_qual, spectral_eff}, reported only.

Read-only: originals untouched; works on the sweep's coerced scratch copies.
Does not modify load_metric.py.
"""

import io
import itertools
import os
import sqlite3
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from load_metric import build_load_view, compute_load
from load_metric_sweep import coerced_copy, RUN_INDEX, OUTPUT_ROOT

SEEDS = [555, 556, 560, 561, 562, 563, 564]
COMPONENTS = ["c_prb", "c_ues", "c_vol", "c_qual"]
DELTAS = [1, 3, 5]


def run_dirs():
    import csv
    with open(RUN_INDEX) as fh:
        return {int(r["seed"]): r["run_dir"] for r in csv.DictReader(fh)}


def load_seed(seed: int, run_dir: str) -> pd.DataFrame:
    db = coerced_copy(os.path.join(OUTPUT_ROOT, run_dir, "database.db"), seed)
    with redirect_stdout(io.StringIO()):
        view = build_load_view(db)
        df = compute_load(view)  # default weights, sinr_agg='min'

    # Extra du columns not exposed by build_load_view: cell-level RRU.PrbUsedDl
    # (assert cell-repeated) and per-cell mean of per-UE DRB.UEThpDl.UEID.
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    extra = pd.read_sql_query(
        "SELECT timestamp, nrcellid AS cellid, "
        "MIN(rruprbuseddl) AS rru_min, MAX(rruprbuseddl) AS rru, "
        "AVG(drbuethpdlueid) AS thp_mean "
        "FROM du WHERE nrcellid BETWEEN 2 AND 8 GROUP BY timestamp, nrcellid", con)
    con.close()
    if not np.allclose(extra["rru_min"], extra["rru"]):
        raise ValueError(f"seed {seed}: rruprbuseddl not cell-repeated")
    df = df.merge(extra.drop(columns="rru_min"), on=["cellid", "timestamp"], how="left")
    df.loc[~df["present"], "rru"] = 0.0  # idle: no PRBs used, thp stays NaN

    # Part B term. p95 normalization over present rows of this run, clipped
    # like c_vol for symmetry.
    se_raw = df["vol_bytes"] / df["rru"].clip(lower=1.0)
    se_p95 = se_raw[df["present"]].quantile(0.95)
    df["spectral_eff"] = (se_raw / se_p95).clip(0.0, 1.0)

    # Per-run label thresholds.
    df["buf_p90"] = df.loc[df["present"], "buffer_bytes"].quantile(0.90)
    df["thp_p10"] = df.loc[df["present"], "thp_mean"].quantile(0.10)
    df["seed"] = seed
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Degradation label at t+D per sub-condition, aligned onto the row at t."""
    ts_sorted = np.sort(df["timestamp"].unique())
    pos = {ts: i for i, ts in enumerate(ts_sorted)}
    df = df.sort_values(["cellid", "timestamp"]).reset_index(drop=True)
    df["tpos"] = df["timestamp"].map(pos)

    cond_sinr = df["sinr_min"] < 0.0                       # NaN -> False
    cond_buf = df["buffer_bytes"] > df["buf_p90"]
    cond_thp = df["thp_mean"] < df["thp_p10"]              # NaN -> False
    df["cond_sinr"], df["cond_buf"], df["cond_thp"] = cond_sinr, cond_buf, cond_thp
    df["degraded"] = cond_sinr | cond_buf | cond_thp

    for d in DELTAS:
        for c in ("cond_sinr", "cond_buf", "cond_thp", "degraded"):
            df[f"{c}_t{d}"] = df.groupby("cellid")[c].shift(-d)
    return df


def auc(scores: pd.Series, labels: pd.Series) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    s, y = scores.to_numpy(float), labels.to_numpy(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    dirs = run_dirs()
    frames = [add_labels(load_seed(s, dirs[s])) for s in SEEDS]
    pooled = pd.concat(frames, ignore_index=True)
    print(f"pooled rows: {len(pooled)} ({len(SEEDS)} seeds x 7 cells x 99 ts)")

    # ---------------- Part A
    print("\n=== A. collinearity over pooled rows ===")
    comps = pooled[COMPONENTS]
    for method in ("pearson", "spearman"):
        print(f"\n{method.capitalize()} matrix:")
        print(comps.corr(method=method).round(3).to_string())
    for method in ("pearson", "spearman"):
        m = comps.corr(method=method)
        hi = [(a, b, m.loc[a, b]) for i, a in enumerate(COMPONENTS)
              for b in COMPONENTS[i + 1:] if m.loc[a, b] > 0.85]
        print(f"{method} pairs > 0.85: "
              + (", ".join(f"({a},{b})={v:.3f}" for a, b, v in hi) if hi else "none"))
    print(f"corr(c_prb, c_vol): Pearson={comps['c_prb'].corr(comps['c_vol']):.3f} "
          f"Spearman={comps['c_prb'].corr(comps['c_vol'], method='spearman'):.3f}")

    # ---------------- Part B
    print("\n=== B. spectral_eff = vol_bytes / max(RRU.PrbUsedDl, 1), p95-normalized ===")
    for other in ("c_prb", "c_qual", "c_vol", "c_ues"):
        p = pooled["spectral_eff"].corr(pooled[other])
        s = pooled["spectral_eff"].corr(pooled[other], method="spearman")
        print(f"corr(spectral_eff, {other}): Pearson={p:.3f} Spearman={s:.3f}")

    # ---------------- Part C
    print("\n=== C. predictive test ===")
    eligible = pooled[pooled["present"] & (pooled["n_ues"] >= 1)]
    print("current-time base rates on eligible rows "
          f"(n={len(eligible)}): sinr_min<0: {eligible['cond_sinr'].mean():.3f}, "
          f"buffer>p90: {eligible['cond_buf'].mean():.3f}, "
          f"thp_mean<p10: {eligible['cond_thp'].mean():.3f}, "
          f"combined: {eligible['degraded'].mean():.3f}")

    predictors = ["c_prb", "c_ues", "c_vol", "c_qual", "spectral_eff", "load"]
    print("\nlabel base rates at t+D (eligible rows with a valid t+D):")
    rows = {}
    for d in DELTAS:
        sub = eligible.dropna(subset=[f"degraded_t{d}"])
        rows[d] = sub
        print(f"  D={d}: n={len(sub)}  sinr:{sub[f'cond_sinr_t{d}'].mean():.3f}  "
              f"buf:{sub[f'cond_buf_t{d}'].mean():.3f}  "
              f"thp:{sub[f'cond_thp_t{d}'].mean():.3f}  "
              f"combined:{sub[f'degraded_t{d}'].mean():.3f}")

    print("\nAUC (predictor at t -> degraded at t+D):")
    print(f"{'predictor':>14} | " + " | ".join(f"D={d}" for d in DELTAS))
    for p in predictors:
        vals = [auc(rows[d][p], rows[d][f"degraded_t{d}"].astype(bool)) for d in DELTAS]
        print(f"{p:>14} | " + " | ".join(f"{v:.3f}" for v in vals))

    # ---------------- Part D
    print("\n=== D. weight grid-search over {c_prb, c_ues, c_qual, spectral_eff} "
          "(c_vol dropped: see Part A), step 0.1, sum=1; scored at D=3 ===")
    sub3 = rows[3]
    y3 = sub3["degraded_t3"].astype(bool)
    feats = ["c_prb", "c_ues", "c_qual", "spectral_eff"]
    X = sub3[feats]
    check_against = feats + ["c_vol"]
    results = []
    for w in itertools.product(range(11), repeat=3):
        if sum(w) > 10:
            continue
        weights = np.array([w[0], w[1], w[2], 10 - sum(w)]) / 10.0
        score = X.to_numpy() @ weights
        score_s = pd.Series(score, index=sub3.index)
        a = auc(score_s, y3)
        corr_prb = score_s.corr(sub3["c_prb"])
        max_corr = max(abs(score_s.corr(sub3[c])) for c in check_against)
        results.append((a, tuple(weights), corr_prb, max_corr))
    results.sort(reverse=True)

    print("top 5 by AUC(D=3):  weights (prb, ues, qual, spec_eff) | AUC | corr with c_prb")
    for a, w, cp, _ in results[:5]:
        print(f"  {w} | {a:.3f} | {cp:.3f}")
    diverse = [r for r in results if r[3] < 0.90]
    if diverse:
        a, w, cp, mc = diverse[0]
        print(f"best with max|corr(score, any single component incl c_vol)| < 0.90: "
              f"weights {w} AUC={a:.3f} corr_prb={cp:.3f} max_corr={mc:.3f}")
    else:
        print("no weighting in the grid keeps max|corr| < 0.90 with every single component")


if __name__ == "__main__":
    main()
