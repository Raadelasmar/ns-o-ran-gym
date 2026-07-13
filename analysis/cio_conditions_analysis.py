"""Cross-condition analysis of the CIO experiments (V26/V27, VERIFICATION_LOG.md).

Per-cell episode means from load_metric.build_load_view/compute_load, plus
handovers (CellIdStatsHandover.txt: time imsi rnti targetCellId, per
mmwave-bearer-stats-connector.cc PrintUeEndHandover), RLF rows
(gnb_cu_cp.l3_serving_sinr < -5, the codebase RLF definition incl. -inf),
total delivered volume and total DL backlog.

Load is reported under both weightings:
  W_A: compute_load's DEFAULT_WEIGHTS (prb .40/ues .20/vol .20/qual .20) — the
       pre-decision defaults the V26/V27 reports used;
  W_B: the chosen Phase-1 weights (prb .15/ues .40/vol .10/qual .35).
Cell 3 ranks top in C0 under both, so the V27 target choice is weighting-robust.

Usage: python analysis/cio_conditions_analysis.py [label=run_dir ...]
Defaults to the five seed-555 conditions on record.
"""
import sys
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from load_metric import build_load_view, compute_load

OUT = "/home/raadelasmar/oran-project/ns-o-ran-gym/output"
DEFAULT_CONDS = {
    "C0 (0dB)":        "5d523414-fa58-4eb5-bccc-3206530e99de",
    "C1 (-3dB c2)":    "c68313fb-2a35-49aa-97a5-6fa62c3b7980",
    "C2 (-6dB c2)":    "205da174-4f41-47c7-afb6-ea7ba39307fb",
    "C3 (+6dB c2)":    "7264b9cc-eaa6-40bd-869a-dc712f9dd18b",
    "C4 (-6dB c3)":    "be7ba2b0-f5ab-410f-9f08-ad058bd8da93",
}
W_B = {"prb": 0.15, "ues": 0.40, "vol": 0.10, "qual": 0.35}


def analyze(conds: dict[str, str]) -> None:
    per_cell, headline = {}, {}
    for cond, run in conds.items():
        db = f"{OUT}/{run}/database.db"
        print(f"\n########## {cond}  ({run}) ##########")
        view = build_load_view(db)
        df = compute_load(view)                    # W_A defaults
        df_b = compute_load(view, weights=W_B)

        tbl = df.groupby("cellid")[["n_ues", "prb_pct", "vol_bytes", "buffer_bytes"]].mean()
        fin = df[np.isfinite(df["sinr_min"])]      # -inf outage rows counted via RLF instead
        tbl["sinr_min"] = fin.groupby("cellid")["sinr_min"].mean()
        tbl["load_A"] = df.groupby("cellid")["load"].mean()
        tbl["load_B"] = df_b.groupby("cellid")["load"].mean()
        per_cell[cond] = tbl

        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cp = pd.read_sql_query(
            "SELECT cellid, l3_serving_sinr FROM gnb_cu_cp WHERE cellid BETWEEN 2 AND 8", con)
        con.close()
        ho = pd.read_csv(f"{OUT}/{run}/CellIdStatsHandover.txt", sep=r"\s+",
                         names=["t", "imsi", "rnti", "dst"]).sort_values("t")
        ho["src"] = ho.groupby("imsi")["dst"].shift()  # previous target = source

        headline[cond] = {
            "handovers": len(ho),
            "rlf_rows": int((cp["l3_serving_sinr"] < -5).sum()),
            "total_vol_MB": df["vol_bytes"].sum() / 1e6,
            "total_buffer_MBticks": df["buffer_bytes"].sum() / 1e6,
            "mean_ue_sinr": float(cp.loc[np.isfinite(cp["l3_serving_sinr"]),
                                         "l3_serving_sinr"].mean()),
            "dst_counts": ho["dst"].value_counts().sort_index().to_dict(),
        }

    print("\n\n============ PER-CELL EPISODE MEANS ============")
    for cond, tbl in per_cell.items():
        print(f"\n--- {cond} ---")
        print(tbl.round(3).to_string())

    for cell in (2, 3):
        print(f"\n============ CELL {cell} ACROSS CONDITIONS ============")
        print(pd.DataFrame({c: t.loc[cell] for c, t in per_cell.items()}).T.round(3).to_string())

    print("\n============ n_ues delta vs first condition ============")
    nues = pd.DataFrame({c: t["n_ues"] for c, t in per_cell.items()}).T
    print((nues - nues.iloc[0]).round(3).to_string())

    print("\n============ HEADLINE ============")
    hl = pd.DataFrame(headline).T
    print(hl.drop(columns="dst_counts").round(3).to_string())
    for c in conds:
        print(f"  {c}: HO dst counts {headline[c]['dst_counts']}")


if __name__ == "__main__":
    conds = (dict(a.split("=", 1) for a in sys.argv[1:])
             if len(sys.argv) > 1 else DEFAULT_CONDS)
    analyze(conds)
