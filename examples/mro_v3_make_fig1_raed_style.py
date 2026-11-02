"""Rebuilds fig_mro_mean_ci.pdf to match Ra'ed's own res_overview.pdf
EXACTLY in style (see plot_results.py): same rcParams, figsize proportions,
bar style (white-edged bars, left-aligned titles, in-bar value labels,
light grid, errorbar drawn separately from the bars) and colors -- using
real MRO v3 final-eval data, training-reward formula (see
mro_v3_make_summaries_trainreward.py).

Panel choice (MRO has no direct analogue of MLB's "neighbour congestion" /
"95th-pct delay"): (a) score -- same role as 7.1(a); (b) handover rate --
the network-churn KPI, playing the role 7.1(b)'s congestion term plays
(a side effect of handover activity); (c) ping-pong rate -- the QoS/KPI
term the training reward penalises most heavily (9.39x), playing the role
7.1(c)'s tail-delay term plays (worst-case / most policy-sensitive KPI).
Outage was considered for (c) but is ~flat across all three controllers
(0.05-0.06%) and not discriminating; ping-pong is the far more informative
choice here.

NOTE ON DIRECTION: unlike Fig 7.1 where all three panels are cost-type
(lower-better), MRO's score is benefit-type (higher-better) while handover
rate and ping-pong rate are cost-type (lower-better). The methods are
ordered by (a)'s ranking (worst-to-best, top-to-bottom) throughout, which
empirically also orders panels (b)/(c) worst-to-best, so the visual
pattern still reads the same way as Fig 7.1 despite the underlying metric
directions differing.

Usage:
    python examples/mro_v3_make_fig1_raed_style.py
"""
from os import path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = path.join(path.dirname(__file__), "..", "output", "mro_v3_final_eval")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555", "axes.linewidth": 0.6,
    "xtick.color": "#333", "ytick.color": "#333", "pdf.fonttype": 42,
})
ACC = "#2a6fdb"
GRY = "#b8bcc4"
INK = "#222"

# order: worst -> best (top -> bottom), per the training-reward score
ORDER = ["do_nothing", "v3_rule", "agent"]
DISPLAY = {
    "do_nothing": "Z (no control)",
    "v3_rule": "TMRO-PAIR",
    "agent": "MRO agent",
}

# (mean, lo, hi) bootstrap 95% CI, from mro_v3_report_stats.py /
# the diagnostic computation in this session (training-reward formula).
SCORE = {
    "do_nothing": (0.1535, 0.1377, 0.1693),
    "v3_rule":    (0.1575, 0.1519, 0.1645),
    "agent":      (0.1626, 0.1549, 0.1706),
}
HANDOVER_PCT = {  # x100 from fig1_bc_panel_data.json
    "do_nothing": (11.886, 11.026, 12.740),
    "v3_rule":    (11.896, 11.015, 12.838),
    "agent":      (11.652, 10.631, 12.682),
}
PINGPONG_PCT = {
    "do_nothing": (0.196, 0.074, 0.333),
    "v3_rule":    (0.129, 0.065, 0.191),
    "agent":      (0.096, 0.045, 0.161),
}

PANELS = [
    ("(a) score", SCORE, "{:.3f}", "training-reward score"),
    ("(b) handover rate", HANDOVER_PCT, "{:.2f}", "handover rate (%)"),
    ("(c) ping-pong rate", PINGPONG_PCT, "{:.2f}", "ping-pong rate (%)"),
]

names = [DISPLAY[k] for k in ORDER]
cols = [GRY] * (len(ORDER) - 1) + [ACC]
y = list(range(len(ORDER)))

fig, axs = plt.subplots(1, 3, figsize=(6.3, 1.2), sharey=True, gridspec_kw={"wspace": 0.12})

for ax, (title, data, fmt, xlabel) in zip(axs, PANELS):
    v = [data[k][0] for k in ORDER]
    lo = [data[k][1] for k in ORDER]
    hi = [data[k][2] for k in ORDER]
    ax.barh(y, v, color=cols, height=0.62, edgecolor="white", linewidth=1)
    ax.errorbar(v, y, xerr=[[a - b for a, b in zip(v, lo)], [b - a for a, b in zip(v, hi)]],
                fmt="none", ecolor="#444", elinewidth=0.8, capsize=2)
    for yi, val in zip(y, v):
        ax.text(val * 0.03, yi, fmt.format(val), va="center", ha="left", fontsize=7.5,
                 color="white" if yi == len(ORDER) - 1 else INK,
                 fontweight="bold" if yi == len(ORDER) - 1 else "normal")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)

axs[0].set_yticks(y)
axs[0].set_yticklabels(names)
axs[0].get_yticklabels()[-1].set_fontweight("bold")
axs[0].get_yticklabels()[-1].set_color(ACC)
axs[0].invert_yaxis()

out = path.join(OUT_DIR, "fig_mro_mean_ci.pdf")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
