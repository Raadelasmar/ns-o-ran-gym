"""Trajectory-ranking sanity check: does the candidate final MRO weight set
actually score the sticky (too-restrictive) margin arm worse than doing
nothing?

Variance-matching (what derived every weight below) only checks that terms
have comparable spread -- it never checks that the resulting weighted
reward ranks good/neutral/bad behaviour correctly, which is the specific
gap 20260915_Dani_RewardWeightMethodologyReport.pdf Section 5.3 flagged and
recommended closing before trusting the weights. This script closes it: it
applies ONE candidate weight set to two REAL datasets already on record at
the correct locked config (ues=5, simTime=30, controlInterval=0.1) --

    do-nothing   : output/mro_baseline_terms_round2.csv      (all-zero action)
    sticky +3 dB : output/mro_outage_stress_terms_batch{1,2}.csv (pooled)

-- and checks the sign: a too-restrictive margin is a real MRO failure mode
(lingering on a degrading cell), so it must score WORSE than doing nothing,
not better and not indistinguishably close.

ActionSmoothness is NOT part of this check -- both datasets used a FIXED
action every step (all-zero, or constant +3 dB), so smoothness is exactly
0.0 in both regardless of its weight, contributing nothing to the
comparison either way.

Usage:
    python examples/mro_weight_ranking_check.py \\
        --baseline output/mro_baseline_terms_round2.csv \\
        --sticky output/mro_outage_stress_terms_batch1.csv output/mro_outage_stress_terms_batch2.csv \\
        --w_quality 1.00 --w_pingpong 1.51 --w_churn 0.80 --w_outage 1.0
"""
import argparse
import csv

import numpy as np


def load_rows(paths: list) -> list:
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def weighted_reward(rows: list, w_quality, w_pingpong, w_churn, w_outage) -> np.ndarray:
    """Per-step reward under the given weights, mirroring MroZmqEnv._reward_terms'
    own formula exactly (quality/pingpong/handover_rate dropped, not zeroed,
    when the source CSV recorded them as empty -- same "missing != measured
    zero" discipline the live env uses).
    """
    out = []
    for r in rows:
        reward = -w_outage * float(r["outage"])
        if r["quality"] not in (None, ""):
            reward += w_quality * float(r["quality"])
        if r["pingpong"] not in (None, ""):
            reward -= w_pingpong * float(r["pingpong"])
        if r["handover_rate"] not in (None, ""):
            reward -= w_churn * float(r["handover_rate"])
        out.append(reward)
    return np.array(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, help="do-nothing CSV (round-2 baseline)")
    ap.add_argument("--sticky", nargs="+", required=True, help="sticky-arm CSV(s) to pool")
    ap.add_argument("--w_quality", type=float, default=1.00)
    ap.add_argument("--w_pingpong", type=float, default=1.51)
    ap.add_argument("--w_churn", type=float, default=0.80)
    ap.add_argument("--w_outage", type=float, default=1.0)
    args = ap.parse_args()

    baseline_rows = load_rows([args.baseline])
    sticky_rows = load_rows(args.sticky)

    baseline_reward = weighted_reward(baseline_rows, args.w_quality, args.w_pingpong,
                                      args.w_churn, args.w_outage)
    sticky_reward = weighted_reward(sticky_rows, args.w_quality, args.w_pingpong,
                                    args.w_churn, args.w_outage)

    print(f"Weights under test: w_quality={args.w_quality} w_pingpong={args.w_pingpong} "
          f"w_churn={args.w_churn} w_outage={args.w_outage}")
    print(f"(ActionSmoothness excluded -- both arms use a fixed action, so it is "
          f"exactly 0.0 in both regardless of its weight)\n")

    print(f"{'condition':<20} {'n':>6} {'mean reward':>12} {'sd':>10}")
    print(f"{'do-nothing':<20} {len(baseline_reward):>6} {baseline_reward.mean():>12.4f} "
          f"{baseline_reward.std():>10.4f}")
    print(f"{'sticky +3 dB':<20} {len(sticky_reward):>6} {sticky_reward.mean():>12.4f} "
          f"{sticky_reward.std():>10.4f}")

    gap = baseline_reward.mean() - sticky_reward.mean()
    print(f"\ngap (do-nothing minus sticky) = {gap:+.4f}")
    if gap > 0:
        print("PASS: the sticky (too-restrictive-margin) arm scores WORSE than doing "
              "nothing under these weights, as a real MRO failure mode should.")
    else:
        print("FAIL: the sticky arm does not score worse than do-nothing under these "
              "weights -- do not trust this weight set before investigating why.")
