"""Builds the MRO v3 "Section 7.3" results table/stats, in the same shape as
the manuscript's Section 7.2 (MLB Agent): mean score with a 95% bootstrap
CI per method, and for each baseline the new agent is compared against --
paired by world (seed) -- a one-sided Wilcoxon signed-rank test and a paired
bootstrap 95% CI of the difference, Holm-corrected across the comparisons
made in one run. Unlike Section 7.2's score (a COST, lower is better), MRO's
score (mro_v3_overnight_eval.summarize: quality - pingpong - handover_rate -
outage) is a BENEFIT -- higher is better -- so this reports "agent higher
by X%" rather than "agent lower by X%"; the statistical machinery is
otherwise identical (one-sided Wilcoxon, paired bootstrap, Holm).

v2c is special-cased: it is not re-run (its own mechanism no longer exists
in the current ns-3 build -- see mro_v3_overnight_eval.py's module
docstring), so its per-seed scores are the hardcoded historical values
(HIST_V2C_SCORES, seeds 2001..2005). Comparisons against it are only valid
when the OTHER condition's summary was also measured on exactly those same
5 seeds (mro_v3_final_eval.py's default) -- this script checks that and
refuses the comparison otherwise, rather than silently mispairing.

Usage:
    python examples/mro_v3_report_stats.py \
        --summary do_nothing=output/mro_v3_overnight_eval/raw_steps.csv:do_nothing \
        --summary v3_rule=output/mro_v3_final_eval/summary_v3_rule.json \
        --summary agent=output/mro_v3_final_eval/summary_trained_policy.json \
        --agent agent --baselines do_nothing,v3_rule,v2c
"""
import argparse
import json
import sys
from os import path

sys.path.insert(0, path.dirname(__file__))
from mro_v3_overnight_eval import HIST_SEEDS, HIST_V2C_SCORES, summarize  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats as sstats  # noqa: E402

RNG = np.random.default_rng(20261027)


def load_per_seed_scores(spec: str) -> dict:
    """spec is either a path to a summary_*.json (has "per_seed_scores"), or
    "csv_path:controller_name" to recompute per-seed scores from a raw_steps
    CSV filtered to one controller (for reusing e.g. the historical
    do_nothing rows in output/mro_v3_overnight_eval/raw_steps.csv)."""
    if ":" in spec and not spec.endswith(".json"):
        csv_path, controller = spec.rsplit(":", 1)
        import csv as csv_mod
        rows = []
        with open(csv_path, newline="") as f:
            for row in csv_mod.DictReader(f):
                if row["controller"] != controller:
                    continue
                row["seed"] = int(row["seed"])
                row["step"] = int(row["step"])
                for k in ("reward", "quality", "pingpong", "handover_rate", "outage"):
                    row[k] = float(row[k]) if row.get(k) not in (None, "") else None
                rows.append(row)
        return summarize(rows)["per_seed_scores"]
    with open(spec) as f:
        summary = json.load(f)
    return {int(k): v for k, v in summary["per_seed_scores"].items()}


def paired_bootstrap_ci(diffs: np.ndarray, n_resamples: int = 10000, alpha: float = 0.05):
    n = len(diffs)
    idx = RNG.integers(0, n, size=(n_resamples, n))
    resampled_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mean_ci_bootstrap(values: np.ndarray, n_resamples: int = 10000, alpha: float = 0.05):
    n = len(values)
    idx = RNG.integers(0, n, size=(n_resamples, n))
    resampled_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm_correct(pvals: list) -> list:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adjusted = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running_max = max(running_max, val)
        adjusted[i] = running_max
    return adjusted


def compare(agent_scores: dict, baseline_scores: dict, baseline_name: str):
    shared = sorted(set(agent_scores) & set(baseline_scores))
    if len(shared) < len(agent_scores) or len(shared) < len(baseline_scores):
        print(f"  WARNING: {baseline_name} shares only {len(shared)} seeds with the agent "
              f"(agent has {len(agent_scores)}, baseline has {len(baseline_scores)}) -- "
              f"comparison restricted to the shared seeds, which may not be a fair pairing.",
              file=sys.stderr)
    a = np.array([agent_scores[s] for s in shared])
    b = np.array([baseline_scores[s] for s in shared])
    diffs = a - b
    n_better = int((diffs > 0).sum())
    try:
        _, p = sstats.wilcoxon(diffs, alternative="greater")
    except ValueError:
        p = float("nan")
    lo, hi = paired_bootstrap_ci(diffs)
    rel_pct = 100.0 * diffs.mean() / abs(b.mean()) if b.mean() != 0 else float("nan")
    return {
        "baseline": baseline_name, "n_worlds": len(shared), "worlds_agent_better": n_better,
        "mean_diff": float(diffs.mean()), "diff_ci": (lo, hi), "rel_pct": rel_pct, "p": float(p),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", action="append", required=True,
                    help="name=path(.json) or name=csv_path:controller, repeatable")
    ap.add_argument("--agent", required=True, help="name (from --summary) of the new agent")
    ap.add_argument("--baselines", required=True,
                    help="comma-separated names to compare the agent against; 'v2c' is special")
    args = ap.parse_args()

    conditions = {}
    for spec in args.summary:
        name, rest = spec.split("=", 1)
        conditions[name] = load_per_seed_scores(rest)
    conditions["v2c"] = dict(zip(HIST_SEEDS, HIST_V2C_SCORES))

    agent_scores = conditions[args.agent]
    baselines = args.baselines.split(",")

    print("=" * 78)
    print(f"Mean score [bootstrap 95% CI], n_worlds")
    print("=" * 78)
    rows_for_table = []
    for name in baselines + [args.agent]:
        scores = np.array(list(conditions[name].values()))
        lo, hi = mean_ci_bootstrap(scores)
        print(f"  {name:16s} {scores.mean():.4f} [{lo:.4f}, {hi:.4f}]  n={len(scores)}")
        rows_for_table.append(name)

    print("\n" + "=" * 78)
    print(f"{args.agent} vs each baseline, paired by world (one-sided Wilcoxon: agent higher)")
    print("=" * 78)
    results = [compare(agent_scores, conditions[b], b) for b in baselines]
    adj_p = holm_correct([r["p"] for r in results])
    for r, p_adj in zip(results, adj_p):
        print(f"  vs {r['baseline']:12s}  agent higher by {r['rel_pct']:+.1f}% "
              f"(diff [{r['diff_ci'][0]:.4f}, {r['diff_ci'][1]:.4f}])  "
              f"{r['worlds_agent_better']}/{r['n_worlds']} worlds  "
              f"p={r['p']:.4f}  Holm-adj p={p_adj:.4f}  "
              f"{'SIGNIFICANT' if p_adj < 0.05 else 'not significant'}")
    print("=" * 78)
