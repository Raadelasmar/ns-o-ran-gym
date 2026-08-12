"""Generic vertically-stacked, multi-axis time-series plotting.

Built for experiment driver scripts (examples/cio_zmq_experiment.py and
similar) that want one figure per data source (KPIs, actions, reward, ...)
with one row of axes per entity (cell, agent, ...), sharing an x-axis. Not
specific to any KPI set or use case: everything is driven by the metric names
present in the `panels` argument.

Design:
  - One Figure, one column of Axes (via plt.subplots(n, 1, sharex=True)).
  - Within a panel (row), metrics whose values fall in comparable ranges
    share one y-axis; metrics on a different scale get their own y-axis
    (Axes.twinx()), offset so multiple right-hand axes don't overlap. This
    keeps a panel with e.g. a [0,1] fraction, a byte count and a UE count
    from squashing the fraction to a flat line, without spawning a separate
    axis per metric when several already share a comparable range.
  - The same metric name always gets the same color on every panel in the
    figure (and across figures, if you pass a shared `colors` map). Each
    panel gets its own legend, since panels can have different metric sets
    and a single figure-wide legend can't be relied on to label every line.
"""
import csv
import math
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def assign_colors(metric_names: Sequence[str]) -> dict:
    """One consistent color per metric name. tab10 up to 10 metrics, tab20 beyond."""
    names = list(dict.fromkeys(metric_names))  # de-dup, preserve order
    cmap = plt.get_cmap("tab10") if len(names) <= 10 else plt.get_cmap("tab20")
    return {name: cmap(i % cmap.N) for i, name in enumerate(names)}


def _axis_bucket(vmin: float, vmax: float) -> str:
    """Group metrics with comparable value ranges so they can share a y-axis.

    Metrics that stay within roughly [0, 1] (fractions, normalized scores)
    are bucketed together regardless of their exact bounds. Everything else
    is bucketed by the order of magnitude of its largest absolute value, so
    two similarly-sized byte counts share an axis but a byte count and a
    small integer count don't.
    """
    if vmax <= 1.5 and vmin >= -0.5:
        return "unit"
    amax = max(abs(vmin), abs(vmax), 1e-9)
    return f"mag{math.floor(math.log10(amax))}"


def plot_panels(fig, axs, x: Sequence[float], panels: Sequence[Mapping[str, Sequence[float]]],
                 titles: Optional[Sequence[str]] = None, xlabel: str = "step",
                 colors: Optional[dict] = None) -> dict:
    """Plot one vertically-stacked, shared-x-axis column of panels.

    Args:
        fig, axs: the return of plt.subplots(len(panels), 1, sharex=True,
            squeeze=False) (or any array-like of that many Axes).
        x: shared x-axis values, one per data point.
        panels: one dict per panel/row, mapping metric name -> sequence of
            values (same length as x, NaN for missing points). Panels may
            have different metric sets -- a panel with a single metric still
            works, ending up with a single y-axis.
        titles: optional per-panel titles (e.g. cell ids); defaults to
            "Panel {i}".
        xlabel: label drawn under the bottom panel only.
        colors: optional explicit {metric_name: color} map. Computed from
            every metric name across all panels if not given -- pass the
            map this function returns to reuse the same colors on another
            figure (e.g. so a KPI figure and an action figure agree).

    Returns:
        The {metric_name: color} map actually used.
    """
    axs = np.atleast_1d(np.asarray(axs)).reshape(-1)
    if len(axs) != len(panels):
        raise ValueError(f"{len(axs)} axes for {len(panels)} panels")

    if colors is None:
        colors = assign_colors([name for panel in panels for name in panel])

    for i, (ax, panel) in enumerate(zip(axs, panels)):
        # Group this panel's metrics by value range so comparable-scale
        # metrics share a y-axis instead of each spawning its own.
        buckets: dict = {}
        for name, values in panel.items():
            values = np.asarray(values, dtype=float)
            finite = values[np.isfinite(values)]
            vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            buckets.setdefault(_axis_bucket(vmin, vmax), []).append(name)

        panel_handles, panel_labels = [], []
        for b, names in enumerate(buckets.values()):
            bucket_ax = ax if b == 0 else ax.twinx()
            if b > 1:
                # Extra right-hand spines would otherwise all stack at the
                # same position and overlap.
                bucket_ax.spines["right"].set_position(("axes", 1.0 + 0.12 * (b - 1)))

            for name in names:
                values = np.asarray(panel[name], dtype=float)
                (line,) = bucket_ax.plot(x, values, color=colors[name], label=name)
                panel_handles.append(line)
                panel_labels.append(name)
            bucket_ax.set_ylabel(" / ".join(names), fontsize=8)
            bucket_ax.tick_params(axis="y", labelsize=7)

        ax.set_title(titles[i] if titles else f"Panel {i}", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        # Drawn on the primary axis so it stacks above every twinx() axis in
        # this panel; handles/labels are collected explicitly since lines on
        # a twinned axis aren't picked up by ax.legend() on its own.
        ax.legend(panel_handles, panel_labels, loc="upper right",
                  fontsize=7, framealpha=0.6, ncol=min(len(panel_labels), 3) or 1)

    axs[-1].set_xlabel(xlabel)
    fig.tight_layout()
    return colors


def save_panels_csv(csv_path: str, x: Sequence[float], panels: Sequence[Mapping[str, Sequence[float]]],
                     panel_names: Optional[Sequence[str]] = None, xlabel: str = "x") -> None:
    """Save exactly the data plot_panels() would draw, in long format, so the
    same figure can be redrawn later (load_panels_csv() + plot_panels())
    without re-running whatever produced this data.

    One row per (x, panel, metric) point: [xlabel, panel, metric, value].
    """
    panel_names = list(panel_names) if panel_names is not None else [f"Panel {i}" for i in range(len(panels))]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([xlabel, "panel", "metric", "value"])
        for panel_name, panel in zip(panel_names, panels):
            for metric, values in panel.items():
                for xi, v in zip(x, values):
                    writer.writerow([xi, panel_name, metric, v])


def load_panels_csv(csv_path: str):
    """Inverse of save_panels_csv(). Returns (x, panels, panel_names, xlabel),
    ready to pass straight into plot_panels(fig, axs, x, panels,
    titles=panel_names, xlabel=xlabel).

    All panels/metrics in one file are assumed to share one x sequence, same
    as the single `x` argument save_panels_csv()/plot_panels() take per call.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        xlabel = reader.fieldnames[0]
        panel_order = []
        data: dict = {}  # panel -> metric -> list[(x, value)]
        for row in reader:
            p, m = row["panel"], row["metric"]
            if p not in data:
                data[p] = {}
                panel_order.append(p)
            xi = float(row[xlabel])
            v = row["value"]
            v = float(v) if v not in ("", "nan") else float("nan")
            data[p].setdefault(m, []).append((xi, v))

    x = None
    panels = []
    for p in panel_order:
        panel = {}
        for m, pairs in data[p].items():
            xs, vs = zip(*pairs)
            panel[m] = list(vs)
            if x is None:
                x = list(xs)
        panels.append(panel)
    return x, panels, panel_order, xlabel
