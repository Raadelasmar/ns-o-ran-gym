"""Run scenario-marl-zmq end to end, seed 555, 10 s, delivered purely over the
ZeroMQ bridge (bridge/zmq_database.py) -- no CSV/semaphore ActionController
path, unlike examples/cio_experiment.py's scenario-three run.

Uses environments.mlb_zmq_env.MlbZmqEnv, which binds the ZMQ REP socket,
launches scenario-marl-zmq, and on each step() replies to that step's KPI
snapshot with a CIO offset per cell computed live from that snapshot
(scratch/scenario-marl-zmq.cc's MarlControlStep/StepSync), then blocks for
the next snapshot -- the same recv/compute/send loop tests/test_server.py
exercises standalone, wrapped as a Gym env. There's no static per-cell bias to
configure here: the CIO offsets come out of the iterative KPI/action loop
itself, not a value picked ahead of time. controlFileName is deliberately
left unset so ns-3's legacy semaphore-gated control file path never
activates.

At the end of the run, saves three figures into the run's output directory
(analysis/timeseries_plots.py's plot_panels, one column of axes per cell/
panel, shared x-axis, metrics grouped onto shared y-axes by value range), plus
the underlying data as a CSV next to each figure (save_panels_csv), so the
same figure can be redrawn later (load_panels_csv + plot_panels) without
re-running the simulation:
  kpis.png/.csv    -- every KPI the ZMQ bridge sent, one row per cell
  actions.png/.csv -- the CIO offset actually sent back each step, one row per cell
  reward.png/.csv  -- the scalar reward each step (single row)

Usage:
    python examples/cio_zmq_experiment.py
"""
import argparse
import sys
from datetime import datetime
from os import path

import matplotlib
matplotlib.use("Agg")  # this script only saves PNGs, never plt.show()
import matplotlib.pyplot as plt

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
sys.path.insert(0, path.join(path.dirname(__file__), "..", "analysis"))
from environments.mlb_zmq_env import MlbZmqEnv
from timeseries_plots import plot_panels, save_panels_csv

CELLS = [2, 3, 4, 5, 6, 7, 8]

CFG = {
    "heuristicType": [-1],
    "simTime": [10],
    "ues": [3],
    "RngRun": [555],
    "configuration": [0],
    "trafficModel": [3],
    "numberOfRaPreambles": [40],
    "bsOn": [5],
    "bsIdle": [0],
    "bsSleep": [0],
    "bsOff": [2],
    "reducedPmValues": [0],
    "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"],
    "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1],
    "rlcAmEnabled": [0],
}


def _cell_kpis(kpis: dict, cell: int) -> dict:
    return (kpis or {}).get("cells", {}).get(str(cell), {})


def _mean_sinr(cell_kpis: dict) -> float:
    sinrs = [u.get("l3_serving_sinr_db") for u in cell_kpis.get("ues", {}).values()
             if u.get("l3_serving_sinr_db") is not None]
    return sum(sinrs) / len(sinrs) if sinrs else float("nan")


def build_kpi_panels(kpi_log: list) -> list:
    """One panel per cell, one series per KPI the ZMQ bridge sends (see
    tests/test_server.py's logged fields): prb_utilization, buffer_bytes,
    volume_bytes, num_active_ues, and the per-cell mean of per-UE
    l3_serving_sinr_db (UE-level in the raw payload, averaged here since a
    panel needs one series per metric, not one per UE).
    """
    panels = []
    for cell in CELLS:
        prb, buf, vol, nues, sinr = [], [], [], [], []
        for kpis in kpi_log:
            ck = _cell_kpis(kpis, cell)
            prb.append(ck.get("prb_utilization", float("nan")))
            buf.append(ck.get("buffer_bytes", float("nan")))
            vol.append(ck.get("volume_bytes", float("nan")))
            nues.append(ck.get("num_active_ues", float("nan")))
            sinr.append(_mean_sinr(ck))
        panels.append({
            "prb_utilization": prb,
            "buffer_bytes": buf,
            "volume_bytes": vol,
            "num_active_ues": nues,
            "sinr_db_mean": sinr,
        })
    return panels


def build_action_panels(action_log: list) -> list:
    """One panel per cell, the CIO offset actually sent that step."""
    panels = []
    for cell in CELLS:
        cio = [actions.get(str(cell), {}).get("cio_offset", float("nan")) for actions in action_log]
        panels.append({"cio_offset": cio})
    return panels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    args = ap.parse_args()

    CFG["RngRun"] = [args.seed]

    env = MlbZmqEnv(ns3_path=args.ns3_path, scenario_configuration=CFG,
                     output_folder=args.output_folder, optimized=False)

    obs, info = env.reset()
    print(f"sim_path: {env.sim_path}", flush=True)

    kpi_log = [info.get("kpis")]
    action_log = []
    reward_log = []

    steps, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        # action is unused by MlbZmqEnv.step() -- the CIO reply is computed
        # live from the pending KPI snapshot, not from a caller-supplied action.
        # THE AGENT (AND EVEN LATER THE COORDINATOR/AGGREGATOR) WILL CALL LIKE:
        # ... = env.step(action)
        obs, reward, terminated, truncated, info = env.step(None)
        steps += 1
        kpi_log.append(info.get("kpis"))
        action_log.append(info.get("cio_offsets", {}))
        reward_log.append(reward)
        print(f"step {steps}, reward {reward:.4f}", flush=True)

    print(f"DONE: {steps} steps, terminated={terminated}, truncated={truncated}")
    print(f"run dir: {env.sim_path}")
    env.close()

    # x-axis: sim timestamp when each KPI snapshot arrived. action_log[i]/
    # reward_log[i] were computed from kpi_log[i] and produced kpi_log[i+1],
    # so they're plotted at kpi_log[i+1]'s timestamp (x_kpi[1:]).
    x_kpi = [(kpis or {}).get("timestamp", i) for i, kpis in enumerate(kpi_log)]
    x_step = x_kpi[1:]

    # One timestamp shared by all three filenames, so they're grouped
    # together (and don't clobber a previous run's plots in the same dir).
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cell_titles = [f"Cell {c}" for c in CELLS]

    kpi_panels = build_kpi_panels(kpi_log)
    fig, axs = plt.subplots(len(CELLS), 1, sharex=True, squeeze=False, figsize=(10, 2.2 * len(CELLS)))
    plot_panels(fig, axs, x_kpi, kpi_panels, titles=cell_titles, xlabel="timestamp (s)")
    kpi_fig_path = path.join(env.sim_path, f"{run_ts}_kpis.png")
    fig.savefig(kpi_fig_path, dpi=150)
    print(f"saved {kpi_fig_path}")
    kpi_csv_path = path.join(env.sim_path, f"{run_ts}_kpis.csv")
    save_panels_csv(kpi_csv_path, x_kpi, kpi_panels, panel_names=cell_titles, xlabel="timestamp (s)")
    print(f"saved {kpi_csv_path}")

    action_panels = build_action_panels(action_log)
    fig, axs = plt.subplots(len(CELLS), 1, sharex=True, squeeze=False, figsize=(10, 2.2 * len(CELLS)))
    plot_panels(fig, axs, x_step, action_panels, titles=cell_titles, xlabel="timestamp (s)")
    action_fig_path = path.join(env.sim_path, f"{run_ts}_actions.png")
    fig.savefig(action_fig_path, dpi=150)
    print(f"saved {action_fig_path}")
    action_csv_path = path.join(env.sim_path, f"{run_ts}_actions.csv")
    save_panels_csv(action_csv_path, x_step, action_panels, panel_names=cell_titles, xlabel="timestamp (s)")
    print(f"saved {action_csv_path}")

    reward_panels = [{"reward": reward_log}]
    fig, axs = plt.subplots(1, 1, sharex=True, squeeze=False, figsize=(10, 2.5))
    plot_panels(fig, axs, x_step, reward_panels, titles=["Reward"], xlabel="timestamp (s)")
    reward_fig_path = path.join(env.sim_path, f"{run_ts}_reward.png")
    fig.savefig(reward_fig_path, dpi=150)
    print(f"saved {reward_fig_path}")
    reward_csv_path = path.join(env.sim_path, f"{run_ts}_reward.csv")
    save_panels_csv(reward_csv_path, x_step, reward_panels, panel_names=["Reward"], xlabel="timestamp (s)")
    print(f"saved {reward_csv_path}")
