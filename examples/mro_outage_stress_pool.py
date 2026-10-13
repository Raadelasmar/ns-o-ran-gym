"""Pool mro_outage_stress_experiment.py's batch CSVs into one combined
statistics table.

Each batch writes its own CSV (the script opens --csv_out in overwrite
mode, so two batches sharing one file would destroy the first), so pooling
the final weight measurement across batches needs a small separate read-and-
recombine step. Same detrended_sd method as the experiment script itself --
this does not re-derive anything, it just widens the pool the existing
formula runs over.

Usage:
    python examples/mro_outage_stress_pool.py output/mro_outage_stress_terms_batch1.csv output/mro_outage_stress_terms_batch2.csv
"""
import argparse
import csv

import numpy as np

TERMS = ("quality", "pingpong", "handover_rate", "outage")


def detrended_sd(rows: list, term: str) -> tuple:
    """Identical method to mro_outage_stress_experiment.py's own detrended_sd."""
    by_seed = {}
    for r in rows:
        if r[term] in (None, ""):
            continue
        by_seed.setdefault(r["seed"], []).append((int(r["step"]), float(r[term])))

    raw_values, residuals = [], []
    for seed, pairs in by_seed.items():
        if len(pairs) < 2:
            continue
        steps = np.array([p[0] for p in pairs], dtype=float)
        values = np.array([p[1] for p in pairs], dtype=float)
        raw_values.extend(values.tolist())
        slope, intercept = np.polyfit(steps, values, 1)
        residuals.extend((values - (slope * steps + intercept)).tolist())

    raw_values = np.array(raw_values)
    residuals = np.array(residuals)
    return (float(raw_values.std()) if raw_values.size else float("nan"),
            float(residuals.std()) if residuals.size else float("nan"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="+", help="batch CSVs to pool together")
    ap.add_argument("--anchor_sd", type=float, default=0.273,
                    help="HandoverSignalQuality's round-2 detrended sd -- see "
                         "docs/mro_training_fixes.md")
    args = ap.parse_args()

    all_rows = []
    seeds_seen = set()
    for f in args.csvs:
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        batch_seeds = sorted(set(r["seed"] for r in rows))
        print(f"{f}: {len(rows)} rows, seeds {batch_seeds}")
        overlap = seeds_seen & set(batch_seeds)
        if overlap:
            raise SystemExit(f"Seed(s) {overlap} appear in more than one file -- "
                             f"pooling would double-count them. Refusing.")
        seeds_seen.update(batch_seeds)
        all_rows.extend(rows)

    print(f"\nPooled {len(all_rows)} rows across {len(args.csvs)} files, "
          f"{len(seeds_seen)} distinct seeds: {sorted(seeds_seen)}")

    print("\n=== per-term statistics (pooled across ALL batches, sticky +3 dB arm) ===")
    print(f"{'term':<16} {'n':>6} {'mean':>10} {'raw_sd':>10} {'detrended_sd':>13}")
    stats = {}
    for term in TERMS:
        values = [float(r[term]) for r in all_rows if r[term] not in (None, "")]
        raw_sd, det_sd = detrended_sd(all_rows, term)
        mean = float(np.mean(values)) if values else float("nan")
        stats[term] = {"mean": mean, "raw_sd": raw_sd, "detrended_sd": det_sd, "n": len(values)}
        print(f"{term:<16} {len(values):>6} {mean:>10.4f} {raw_sd:>10.4f} {det_sd:>13.4f}")

    outage_values = [float(r["outage"]) for r in all_rows if r["outage"] not in (None, "")]
    n_fired = sum(1 for v in outage_values if v > 0.0)
    print(f"\noutage fired on {n_fired}/{len(outage_values)} steps "
          f"({100*n_fired/len(outage_values):.3f}%) under the sticky arm, pooled")
    print("(round-2 do-nothing baseline: fired on 0.01% of UE-steps)")

    print("\n=== derived outage weight (anchored to round-2's HandoverSignalQuality sd) ===")
    det_sd = stats["outage"]["detrended_sd"]
    w = (args.anchor_sd / det_sd) if det_sd and det_sd > 0 else float("nan")
    print(f"  w_outage = anchor_sd({args.anchor_sd}) / detrended_sd({det_sd:.4f}) = {w:.4f}")
