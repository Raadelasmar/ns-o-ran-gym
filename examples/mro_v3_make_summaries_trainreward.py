"""Rebuilds summary_*.json for all four MRO v3 final-eval controllers using
the TRAINING reward (REWARD_FORMULA_VERSION=2: quality - 9.39*pingpong -
1.21*handover_rate - 1.00*outage) as the per-seed score, instead of the
separate evaluation function used by mro_v3_overnight_eval.summarize().

Per-seed score = mean of the "reward" column over all steps in that seed's
episode (the column mro_zmq_env.py's env.step() actually returned at each
step -- verified byte-for-byte against a direct recomputation from the
reward weights in an earlier session).

v2c reuses its own historical raw per-step CSVs (output/mro_v2c_final_eval_
seed*.csv, seeds 2001-2005 only -- its mechanism cannot be re-run on the
current ns-3 build), which also carry a "reward" column computed under the
same formula.

Usage:
    python examples/mro_v3_make_summaries_trainreward.py
"""
import csv
import json
from os import path

import numpy as np

EVAL_DIR = path.join(path.dirname(__file__), "..", "output", "mro_v3_final_eval")
ROOT = path.join(path.dirname(__file__), "..")


def mean_reward_per_seed(csv_path, controller=None):
    by_seed = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if controller and row.get("controller") != controller:
                continue
            r = row.get("reward")
            if r in (None, ""):
                continue
            by_seed.setdefault(int(row["seed"]), []).append(float(r))
    return {s: float(np.mean(v)) for s, v in by_seed.items()}


def write_summary(name, per_seed):
    seeds = sorted(per_seed)
    vals = np.array([per_seed[s] for s in seeds])
    summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "per_seed_scores": per_seed,
        "mean_score": float(vals.mean()),
        "sd_score": float(vals.std()),
    }
    out = path.join(EVAL_DIR, f"summary_{name}_trainreward.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}: mean={vals.mean():.4f} sd={vals.std():.4f} n={len(seeds)}")


if __name__ == "__main__":
    write_summary("do_nothing", mean_reward_per_seed(
        path.join(EVAL_DIR, "raw_steps_do_nothing.csv")))
    write_summary("v3_rule", mean_reward_per_seed(
        path.join(EVAL_DIR, "raw_steps_v3_rule.csv")))
    write_summary("trained_policy", mean_reward_per_seed(
        path.join(EVAL_DIR, "raw_steps_trained_policy.csv")))

    v2c = {}
    for seed in (2001, 2002, 2003, 2004, 2005):
        d = mean_reward_per_seed(path.join(ROOT, "output", f"mro_v2c_final_eval_seed{seed}.csv"))
        v2c[seed] = d[seed]
    write_summary("v2c_verified", v2c)
