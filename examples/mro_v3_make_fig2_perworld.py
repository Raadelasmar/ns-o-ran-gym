"""Builds fig_mro_pair_offsets.pdf as a per-world (per-seed) substitute for
the true per-relation-class forest plot (abandoned: a short probe run
showed trigger events are sparse and concentrated later in the episode, so
a short/cheap probe can't collect them -- see chat). Uses data already on
disk from the real final-eval runs (output/mro_v3_final_eval/raw_steps_
v3_rule.csv and raw_steps_trained_policy.csv, full ~290-step episodes,
8 seeds each): per seed (world), mean |pairwise margin correction| with a
bootstrap 95% CI over that world's own steps, v3 rule vs agent -- same
dot+whisker forest-plot visual style as Ra'ed's Figure 7.2, grouped by
world instead of relation class.

Usage:
    python examples/mro_v3_make_fig2_perworld.py
"""
import csv
from os import path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_DIR = path.join(path.dirname(__file__), "..", "output", "mro_v3_final_eval")
SEEDS = (2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008)

GRAY = "#A6A6A6"
BLUE = "#1A56DB"
RNG = np.random.default_rng(20261029)


def load_margins_per_seed(tag):
    by_seed = {}
    with open(path.join(EVAL_DIR, f"raw_steps_{tag}.csv"), newline="") as f:
        for row in csv.DictReader(f):
            seed = int(row["seed"])
            v = row.get("mean_abs_pair_margin_db")
            if v in (None, ""):
                continue
            by_seed.setdefault(seed, []).append(float(v))
    return by_seed


def boot_ci(vals):
    vals = np.asarray(vals)
    n = len(vals)
    idx = RNG.integers(0, n, size=(10000, n))
    rs = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


if __name__ == "__main__":
    rule = load_margins_per_seed("v3_rule")
    agent = load_margins_per_seed("trained_policy")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(SEEDS))[::-1]

    rule_stats, agent_stats = {}, {}
    for seed in SEEDS:
        rule_stats[seed] = boot_ci(rule[seed])
        agent_stats[seed] = boot_ci(agent[seed])

    for stats, color, dy in ((rule_stats, GRAY, +0.14), (agent_stats, BLUE, -0.14)):
        for yi, seed in zip(y, SEEDS):
            m, lo, hi = stats[seed]
            ax.errorbar(m, yi + dy, xerr=[[m - lo], [hi - m]], fmt="o",
                        color=color, ecolor=color, elinewidth=1.2, capsize=3,
                        markersize=5)

    for yi, seed in zip(y, SEEDS):
        rm = rule_stats[seed][0]
        am = agent_stats[seed][0]
        delta = am - rm
        if abs(delta) > 0.0015:
            ax.annotate(f"{delta:+.4f} dB", (max(rm, am) + 0.0008, yi),
                        va="center", fontsize=8, color=BLUE, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"world {s}" for s in SEEDS], fontsize=9.5)
    ax.set_xlabel("mean |pairwise handover margin correction| in force, dB", fontsize=9.5)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Line2D([0], [0], marker="o", color=GRAY, linestyle="",
                          markersize=6, label="TMRO-PAIR"),
              plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="",
                          markersize=6, label="MRO agent")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    out = path.join(EVAL_DIR, "fig_mro_pair_offsets.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)

    rv_all = np.array([rule_stats[s][0] for s in SEEDS])
    av_all = np.array([agent_stats[s][0] for s in SEEDS])
    print("rule  mean-of-world-means:", rv_all.mean())
    print("agent mean-of-world-means:", av_all.mean())
    print("worlds where agent > rule:", int((av_all > rv_all).sum()), "/", len(SEEDS))
    for seed in SEEDS:
        print(f"  world {seed}: rule={rule_stats[seed][0]:.5f} agent={agent_stats[seed][0]:.5f} "
              f"delta={agent_stats[seed][0]-rule_stats[seed][0]:+.5f}")
