"""Overnight validation run (2026-10-26): does v3's pure rule-based baseline
(RuleBasedMroControllerV3 -- a constant zero action, i.e. BASE_TOO_LATE_GAIN_DB
/ BASE_CHURN_GAIN_DB with no agent correction) beat v2c, the best traditional
MRO controller measured under the old per-cell design?

v2c CANNOT be re-run live tonight: its 7-dim per-cell action and the ns-3
mechanism it drove (LteEnbRrc::m_hysteresisMarginOffset keyed by target cell
only) no longer exist -- v3 replaced both with a pairwise (serving, target)
mechanism (see mro_investigation_log.md / mro_zmq_env.py's module docstring).
So this script measures TWO conditions fresh tonight, under the CURRENT
(v3) ns-3 build and env, both reusing mro_baseline_experiment.py's exact
locked CFG and parallel-worker pattern:

  do_nothing: action=[-1,-1] every step. Under v3's gain = clip(BASE + delta,
      0, MARGIN_LIMIT_DB), delta=-1 with BASE_*_GAIN_DB=1.0 clips the
      effective gain to exactly 0 for both roles -- a TRUE no-op, verified
      against _pairwise_margin_targets's own zero-gain behaviour (gain=0
      makes every target 0.0 regardless of measured rates).
  v3_rule:    action=[0,0] every step -- the rule's fixed base gains, no
      agent correction. This IS RuleBasedMroControllerV3's own behaviour.

Run on the SAME seeds as the historical v2c measurement (2001-2005) for
do_nothing specifically -- this gives a live, same-night, same-build
reproducibility check against the historical do-nothing numbers already on
record (examples/mro_baseline_experiment.py's own prior output, reused in
mro_three_way_eval.pdf), which is what lets v3_rule's fresh numbers be
compared against the HISTORICAL v2c reference in the same frame, without
needing to resurrect the old ns-3 binary.

Built-in overnight safety (unattended, must finish well before a hard
deadline): a per-seed subprocess timeout, and an overall wall-clock deadline
after which no NEW seed is dispatched (whatever has already finished is kept;
anything still in flight at the deadline is cancelled). Both conditions'
results, plus the comparison against the hardcoded historical v2c/do-nothing
numbers, are written to a single results file even if the run is cut short.

Usage:
    python examples/mro_v3_overnight_eval.py --ns3_path ... --output_folder ...
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import MroZmqEnv

import numpy as np

# Identical to mro_baseline_experiment.py's own locked CFG -- same topology,
# traffic, and the exact config the historical v2c/do-nothing numbers below
# were themselves measured under.
CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1],
    "controlInterval": [0.1],
    "channelConditionUpdatePeriodMs": [0], "handoverSinrFilterTauMs": [400],
    "handoverHysteresisDb": [0], "tttFromBiasedSinr": [1],
    "controlPhaseOffsetS": [0.001],
    "ns3::MmWavePhyTrace::OutputFilename": ["/dev/null"],
    "ns3::MmWavePhyTrace::DlPhyTransmissionFilename": ["/dev/null"],
    "ns3::MmWaveMacTrace::SchedInfoOutputFilename": ["/dev/null"],
    "ns3::MmWaveBearerStatsConnector::MmWaveSinrOutputFilename": ["/dev/null"],
    "ns3::CoreNetworkStatsCalculator::X2FileName": ["/dev/null"],
    "ns3::PhyStatsCalculator::DlRsrpSinrFilename": ["/dev/null"],
    "ns3::PhyStatsCalculator::UlInterferenceFilename": ["/dev/null"],
    "ns3::LteRlcAm::BufferSizeFilename": ["/dev/null"],
}

TERMS = ("quality", "pingpong", "handover_rate", "outage")

# Evaluation-function weights -- 3GPP TR 36.902 / Kwong et al. 2024 ESWA
# 246:122871 equal-weight-on-penalty-terms convention, identical to every
# prior controller-ranking comparison this session (mro_three_way_eval.pdf).
# NOT the training reward's weights (9.39x on PingPong) -- see that PDF's
# Section 1 for the full rationale on why ranking uses this, not the other.
SCORE_W_QUALITY, SCORE_W_PINGPONG, SCORE_W_HANDOVER, SCORE_W_OUTAGE = 1.00, 1.21, 1.21, 1.00

# Historical reference, read verbatim off this session's own prior delivered
# artifacts (make_fig_3way.py / mro_three_way_eval.pdf, 2026-10-25), NOT
# remeasured here -- v2c's own mechanism no longer exists to re-run. Per-seed
# scores for seeds [2001..2005] in that exact order.
HIST_SEEDS = (2001, 2002, 2003, 2004, 2005)
HIST_DO_NOTHING_SCORES = (0.1994, 0.1425, 0.1613, 0.1720, 0.1737)
HIST_V2C_SCORES = (0.1896, 0.1880, 0.1668, 0.1730, 0.1251)


def action_for(controller: str) -> np.ndarray:
    if controller == "do_nothing":
        # gain = clip(BASE_*_GAIN_DB + delta, 0, MARGIN_LIMIT_DB); delta=-1
        # with BASE_*_GAIN_DB=1.0 clips both effective gains to exactly 0.
        return np.array([-1.0, -1.0], dtype=np.float32)
    if controller == "v3_rule":
        return np.array([0.0, 0.0], dtype=np.float32)
    raise ValueError(f"unknown controller {controller!r}")


def run_seed(seed: int, controller: str, args, zmq_port: int) -> list:
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    cfg["simTime"] = [args.simtime]

    env = MroZmqEnv(ns3_path=args.ns3_path, scenario_configuration=cfg,
                    output_folder=args.output_folder, optimized=False,
                    zmq_port=zmq_port, build_ns3=False,
                    vary_rng_run_per_episode=False,
                    purge_sim_path_on_close=True)

    action = action_for(controller)
    rows = []
    try:
        obs, info = env.reset(seed=seed)
        print(f"  [{controller}] seed {seed}: reset ok, rng_run={info.get('rng_run')}",
              flush=True)
        step_idx = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(action)
            if "reward_terms" not in info:
                break
            terms = info["reward_terms"]
            row = {"controller": controller, "seed": seed, "step": step_idx, "reward": reward}
            for t in TERMS:
                row[t] = terms[t]
            rows.append(row)
            step_idx += 1
            if terminated or truncated:
                break
    except Exception as exc:
        print(f"  [{controller}] seed {seed}: CRASHED ({exc!r}), discarding this seed's rows",
              flush=True)
        return []
    finally:
        env.close()
    print(f"  [{controller}] seed {seed}: {len(rows)} steps collected", flush=True)
    return rows


def _worker_entry(seed, controller, args, zmq_port, result_queue) -> None:
    rows = run_seed(seed, controller, args, zmq_port)
    result_queue.put((zmq_port, seed, rows))


def run_condition(controller: str, seeds: list, args, workers: int,
                  per_seed_timeout_s: float, deadline_unix_s: float):
    """Yields (seed, rows) as each seed finishes. Adapted from
    mro_baseline_experiment.py's run_all_seeds, with two overnight-safety
    additions: a per-seed subprocess timeout (a stuck worker is killed, not
    left to hang the whole run), and a hard wall-clock deadline (no NEW seed
    is dispatched once past it -- whatever is already running is still given
    up to per_seed_timeout_s to finish, then the condition ends regardless of
    how many seeds in `seeds` were actually reached).
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    n_workers = min(workers, len(seeds))
    pending = list(seeds)
    running: dict = {}   # port -> (process, seed, start_time)

    def launch(port: int) -> None:
        if not pending or time.time() >= deadline_unix_s:
            return
        seed = pending.pop(0)
        p = ctx.Process(target=_worker_entry, args=(seed, controller, args, port, result_queue))
        p.start()
        running[port] = (p, seed, time.time())
        print(f"  [{controller}] seed {seed}: launched on port {port} (pid {p.pid})", flush=True)

    for port in (args.zmq_port + i for i in range(n_workers)):
        launch(port)

    while running:
        try:
            port, seed, rows = result_queue.get(timeout=1.0)
        except queue.Empty:
            now = time.time()
            timed_out = [p_ for p_, (proc, sd, started) in running.items()
                        if now - started > per_seed_timeout_s]
            for p_ in timed_out:
                proc, dead_seed, _ = running.pop(p_)
                print(f"  [{controller}] seed {dead_seed}: TIMED OUT after "
                      f"{per_seed_timeout_s:.0f}s, interrupting (pid {proc.pid})", flush=True)
                # SIGINT, not terminate()/SIGTERM: run_seed()'s own `finally:
                # env.close()` only runs if the worker gets a chance to
                # unwind via a caught exception, which Python's default
                # SIGINT handling (KeyboardInterrupt) provides and SIGTERM
                # does not. Confirmed by this script's own dry run:
                # proc.terminate() left every killed worker's ns-3 child
                # process (env.close()'s sim_process) running as an orphan,
                # un-reaped, still burning CPU minutes later. Same fix
                # mro_baseline_experiment.py's own handle_sigint already
                # uses for exactly this reason.
                os.kill(proc.pid, signal.SIGINT)
                proc.join(timeout=15)
                if proc.is_alive():
                    print(f"  [{controller}] seed {dead_seed}: did not exit after SIGINT, "
                          f"SIGKILL (pid {proc.pid}) -- its ns-3 child may be orphaned", flush=True)
                    proc.kill()
                    proc.join(timeout=5)
                yield dead_seed, []
                launch(p_)
            dead = [p_ for p_, (proc, sd, started) in running.items()
                   if p_ not in timed_out and not proc.is_alive()]
            for p_ in dead:
                proc, dead_seed, _ = running.pop(p_)
                print(f"  [{controller}] seed {dead_seed}: worker exited without reporting "
                      f"(exit code {proc.exitcode})", flush=True)
                yield dead_seed, []
                launch(p_)
            if now >= deadline_unix_s and not pending:
                continue  # let currently-running seeds finish or time out
            continue
        running.pop(port, None)
        yield seed, rows
        launch(port)

    skipped = len(pending)
    if skipped:
        print(f"  [{controller}] DEADLINE HIT: {skipped} seed(s) never dispatched: {pending}",
              flush=True)


def summarize(rows: list, score_w=(SCORE_W_QUALITY, SCORE_W_PINGPONG,
                                   SCORE_W_HANDOVER, SCORE_W_OUTAGE)) -> dict:
    """Per-seed mean of each term (over steps where it was measured, not
    dropped), per-seed evaluation-function score, then mean+sd across seeds
    -- identical method to mro_three_way_eval.pdf's own scoring."""
    by_seed: dict = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    per_seed_scores = {}
    per_seed_terms = {}
    for seed, seed_rows in by_seed.items():
        means = {}
        for t in TERMS:
            vals = [r[t] for r in seed_rows if r[t] is not None]
            means[t] = float(np.mean(vals)) if vals else None
        per_seed_terms[seed] = means
        q, pp, hr, out = (means[t] if means[t] is not None else 0.0
                         for t in ("quality", "pingpong", "handover_rate", "outage"))
        w_q, w_pp, w_hr, w_out = score_w
        per_seed_scores[seed] = w_q * q - w_pp * pp - w_hr * hr - w_out * out
    scores = list(per_seed_scores.values())
    return {
        "n_seeds": len(scores),
        "seeds": sorted(per_seed_scores),
        "per_seed_scores": per_seed_scores,
        "per_seed_terms": per_seed_terms,
        "mean_score": float(np.mean(scores)) if scores else None,
        "sd_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--simtime", type=int, default=30)
    ap.add_argument("--zmq_port", type=int, default=5700)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--results_dir", default=path.join(path.dirname(__file__), "..", "output",
                                                        "mro_v3_overnight_eval"))
    ap.add_argument("--do_nothing_workers", type=int, default=5)
    ap.add_argument("--v3_rule_workers", type=int, default=8)
    ap.add_argument("--v3_rule_seeds", type=int, default=30,
                    help="how many fresh seeds (6001, 6002, ...) to run v3_rule over")
    ap.add_argument("--per_seed_timeout_s", type=float, default=1500,
                    help="kill a single seed's worker if it runs longer than this (25 min default)")
    ap.add_argument("--do_nothing_budget_s", type=float, default=2700,
                    help="max wall-clock time to spend dispatching do_nothing seeds (45 min default)")
    ap.add_argument("--total_budget_s", type=float, default=13500,
                    help="max wall-clock time for the whole script, both conditions combined "
                         "(3h45m default) -- a hard overnight safety net")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    t_start = time.time()
    overall_deadline = t_start + args.total_budget_s
    do_nothing_deadline = min(t_start + args.do_nothing_budget_s, overall_deadline)

    print("=" * 78, flush=True)
    print("MRO v3 OVERNIGHT EVAL: does the v3 rule-based baseline beat v2c?", flush=True)
    print(f"  started           {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  hard deadline     {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(overall_deadline))} "
          f"({args.total_budget_s/3600:.2f} h budget)", flush=True)
    print(f"  do_nothing budget {args.do_nothing_budget_s/60:.0f} min, seeds {list(HIST_SEEDS)}", flush=True)
    print(f"  v3_rule seeds     {args.v3_rule_seeds} (6001..{6000+args.v3_rule_seeds})", flush=True)
    print("=" * 78, flush=True)

    all_rows = {"do_nothing": [], "v3_rule": []}
    raw_csv_path = path.join(args.results_dir, "raw_steps.csv")
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "seed", "step", "reward"] + list(TERMS))
        writer.writeheader()

        print(f"\n--- condition: do_nothing ({len(HIST_SEEDS)} seeds, same as the historical "
              f"v2c/do-nothing measurement) ---", flush=True)
        for seed, rows in run_condition("do_nothing", list(HIST_SEEDS), args,
                                        args.do_nothing_workers, args.per_seed_timeout_s,
                                        do_nothing_deadline):
            all_rows["do_nothing"].extend(rows)
            for r in rows:
                writer.writerow(r)
            f.flush()

        v3_seeds = list(range(6001, 6001 + args.v3_rule_seeds))
        print(f"\n--- condition: v3_rule ({len(v3_seeds)} seeds) ---", flush=True)
        for seed, rows in run_condition("v3_rule", v3_seeds, args,
                                        args.v3_rule_workers, args.per_seed_timeout_s,
                                        overall_deadline):
            all_rows["v3_rule"].extend(rows)
            for r in rows:
                writer.writerow(r)
            f.flush()

    summary_do_nothing = summarize(all_rows["do_nothing"])
    summary_v3_rule = summarize(all_rows["v3_rule"])

    hist_do_nothing_mean = float(np.mean(HIST_DO_NOTHING_SCORES))
    hist_v2c_mean = float(np.mean(HIST_V2C_SCORES))
    hist_v2c_sd = float(np.std(HIST_V2C_SCORES, ddof=1))

    lines = []
    lines.append("=" * 78)
    lines.append("MRO v3 OVERNIGHT EVAL -- RESULTS")
    lines.append(f"finished {time.strftime('%Y-%m-%dT%H:%M:%S')}, "
                 f"{(time.time()-t_start)/60:.1f} min elapsed")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"do_nothing (tonight, n={summary_do_nothing['n_seeds']}/{len(HIST_SEEDS)} seeds, "
                 f"same seeds as history): mean score "
                 f"{summary_do_nothing['mean_score']:.4f}" if summary_do_nothing['mean_score'] is not None
                 else "do_nothing: NO DATA (every seed crashed or timed out)")
    if summary_do_nothing['sd_score'] is not None:
        lines[-1] += f" (sd {summary_do_nothing['sd_score']:.4f})"
    lines.append(f"do_nothing (HISTORICAL, 2026-10-21/25, same 5 seeds): mean score "
                 f"{hist_do_nothing_mean:.4f}")
    lines.append("  -> reproducibility check: tonight's do_nothing vs the historical do_nothing "
                 "number, same build family, same seeds, same CFG.")
    lines.append("")
    lines.append(f"v3_rule (tonight, n={summary_v3_rule['n_seeds']}/{args.v3_rule_seeds} seeds): "
                 f"mean score " + (f"{summary_v3_rule['mean_score']:.4f}"
                                   if summary_v3_rule['mean_score'] is not None else "NO DATA"))
    if summary_v3_rule['sd_score'] is not None:
        lines[-1] += f" (sd {summary_v3_rule['sd_score']:.4f})"
    lines.append(f"v2c (HISTORICAL, 2026-10-21, 5 seeds {HIST_SEEDS}): mean score "
                 f"{hist_v2c_mean:.4f} (sd {hist_v2c_sd:.4f})")
    if summary_v3_rule['mean_score'] is not None:
        gap = summary_v3_rule['mean_score'] - hist_v2c_mean
        verdict = ("v3_rule BEATS historical v2c" if gap > 0 else
                  "v3_rule is BELOW historical v2c")
        lines.append(f"  -> {verdict} by {gap:+.4f} "
                     f"(v2c's own sd was {hist_v2c_sd:.4f} -- read the gap against that)")
    lines.append("")
    lines.append("Per-seed scores:")
    lines.append(f"  do_nothing tonight : " + json.dumps(summary_do_nothing['per_seed_scores']))
    lines.append(f"  do_nothing history : " + json.dumps(dict(zip(HIST_SEEDS, HIST_DO_NOTHING_SCORES))))
    lines.append(f"  v2c history        : " + json.dumps(dict(zip(HIST_SEEDS, HIST_V2C_SCORES))))
    lines.append(f"  v3_rule tonight    : " + json.dumps(summary_v3_rule['per_seed_scores']))
    lines.append("")
    lines.append(f"Raw per-step CSV: {raw_csv_path}")
    lines.append("=" * 78)

    summary_text = "\n".join(lines)
    print("\n" + summary_text, flush=True)

    with open(path.join(args.results_dir, "SUMMARY.txt"), "w") as fh:
        fh.write(summary_text + "\n")
    with open(path.join(args.results_dir, "summary.json"), "w") as fh:
        json.dump({
            "do_nothing_tonight": summary_do_nothing,
            "v3_rule_tonight": summary_v3_rule,
            "historical_do_nothing_mean": hist_do_nothing_mean,
            "historical_v2c_mean": hist_v2c_mean,
            "historical_v2c_sd": hist_v2c_sd,
            "historical_v2c_per_seed": dict(zip(HIST_SEEDS, HIST_V2C_SCORES)),
            "historical_do_nothing_per_seed": dict(zip(HIST_SEEDS, HIST_DO_NOTHING_SCORES)),
            "elapsed_minutes": (time.time() - t_start) / 60,
        }, fh, indent=2)

    print(f"\nWrote {args.results_dir}/SUMMARY.txt and summary.json", flush=True)
