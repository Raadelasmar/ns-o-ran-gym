"""Final MRO v3 evaluation (2026-10-27), post threshold-fix
(TOO_LATE_RATE_THRESHOLD / CHURN_RATE_THRESHOLD: 0.3 -> 0.0, see
src/environments/mro_zmq_env.py's module-level comment for the empirical
grounding). Runs ONE controller -- do_nothing, v3_rule, or a trained SAC
policy -- on a given seed set and simTime, writing per-step rows and a
mro_v3_overnight_eval.summarize()-compatible summary (mean/sd score,
per-seed scores for paired statistics).

Default seeds/simtime (2001..2005, simTime=30) match
mro_v3_overnight_eval.HIST_SEEDS/HIST_V2C_SCORES exactly, so do_nothing,
v3_rule and a trained policy run with these defaults are directly
PAIRED-by-world with the historical v2c reference for a Wilcoxon/bootstrap
comparison (examples/mro_v3_report_stats.py) -- v2c's own mechanism cannot
be re-run (see mro_v3_overnight_eval.py's module docstring), so this is the
only way to compare against it.

do_nothing does NOT need to be re-run after the threshold fix: its action
(delta=-1) clips both effective gains to exactly 0 regardless of
TOO_LATE_RATE_THRESHOLD/CHURN_RATE_THRESHOLD (see _pairwise_margin_targets:
delta = gain * excess = 0 * excess = 0 for any excess). The historical
output/mro_v3_overnight_eval/raw_steps.csv do_nothing rows (seeds
2001..2005, simTime=30) remain valid and should be reused rather than
re-run.

Usage:
    python examples/mro_v3_final_eval.py --controller v3_rule
    python examples/mro_v3_final_eval.py --controller trained_policy \
        --model output/mro_v3_pilot_v2/mro_sac_final.zip
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from os import path

sys.path.insert(0, path.dirname(__file__))
from mro_v3_overnight_eval import TERMS, HIST_SEEDS, summarize  # noqa: E402

WORKER_SCRIPT = path.join(path.dirname(__file__), "mro_v3_single_seed_worker.py")


def _read_csv_rows(csv_path: str) -> list:
    if not path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row["seed"] = int(row["seed"])
            row["step"] = int(row["step"])
            for k in ("reward",) + TERMS:
                row[k] = float(row[k]) if row.get(k) not in (None, "") else None
            rows.append(row)
    return rows


def run_condition(controller: str, model_path, seeds: list, args, workers: int,
                  per_seed_timeout_s: float, deadline_unix_s: float):
    """Launches one subprocess.Popen of mro_v3_single_seed_worker.py per seed
    -- NOT multiprocessing.Process, which hangs forever on the first
    post-reset env.step() for reasons never isolated (see that script's own
    module docstring). Each worker writes its own CSV; this orchestrator
    polls, enforces the per-seed timeout, and reads results back off disk."""
    n_workers = min(workers, len(seeds))
    pending = list(seeds)
    running: dict = {}  # port -> (Popen, seed, started_ts, csv_path)
    tmp_dir = path.join(args.results_dir, f"_tmp_{controller}")
    os.makedirs(tmp_dir, exist_ok=True)

    def launch(port: int) -> None:
        if not pending or time.time() >= deadline_unix_s:
            return
        seed = pending.pop(0)
        csv_path = path.join(tmp_dir, f"seed_{seed}.csv")
        cmd = [sys.executable, WORKER_SCRIPT,
              "--controller", controller, "--seed", str(seed),
              "--simtime", str(args.simtime), "--zmq_port", str(port),
              "--output_csv", csv_path, "--ns3_path", args.ns3_path,
              "--output_folder", args.output_folder]
        if controller == "trained_policy":
            cmd += ["--model", model_path]
        # No stdout=PIPE here: an unread pipe fills after enough per-step
        # prints and then blocks the WORKER's own print() call -- looks
        # exactly like an env.step() hang from the orchestrator's side.
        # Let the child inherit our actual stdout/stderr fds instead.
        proc = subprocess.Popen(cmd)
        running[port] = (proc, seed, time.time(), csv_path)
        print(f"  [{controller}] seed {seed}: launched on port {port} (pid {proc.pid})",
              flush=True)

    for port in (args.zmq_port + i for i in range(n_workers)):
        launch(port)

    while running:
        time.sleep(1.0)
        now = time.time()
        finished_ports = []
        for port, (proc, seed, started, csv_path) in list(running.items()):
            ret = proc.poll()
            if ret is not None:
                finished_ports.append(port)
                if ret != 0:
                    print(f"  [{controller}] seed {seed}: worker exited with code {ret}",
                          flush=True)
                rows = _read_csv_rows(csv_path)
                running.pop(port)
                yield seed, rows
                launch(port)
            elif now - started > per_seed_timeout_s:
                print(f"  [{controller}] seed {seed}: TIMED OUT after "
                      f"{per_seed_timeout_s:.0f}s, terminating (pid {proc.pid})", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                rows = _read_csv_rows(csv_path)
                running.pop(port)
                yield seed, rows
                launch(port)

    skipped = len(pending)
    if skipped:
        print(f"  [{controller}] DEADLINE HIT: {skipped} seed(s) never dispatched: {pending}",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--controller", required=True,
                    choices=["do_nothing", "v3_rule", "trained_policy"])
    ap.add_argument("--model", default=None, help="required if --controller trained_policy")
    ap.add_argument("--seeds", default=",".join(str(s) for s in HIST_SEEDS),
                    help="default: the historical v2c seeds (2001..2005) for direct pairing")
    ap.add_argument("--simtime", type=int, default=30,
                    help="default 30: matches the historical v2c/do_nothing measurement")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--per_seed_timeout_s", type=float, default=10800)
    ap.add_argument("--total_budget_s", type=float, default=18000)
    ap.add_argument("--zmq_port", type=int, default=7400)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--results_dir", default=path.join(path.dirname(__file__), "..", "output",
                                                        "mro_v3_final_eval"))
    args = ap.parse_args()

    if args.controller == "trained_policy" and not args.model:
        raise SystemExit("--model is required for --controller trained_policy")

    os.makedirs(args.results_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    deadline = t0 + args.total_budget_s

    print("=" * 78, flush=True)
    print(f"final eval: controller={args.controller} simtime={args.simtime} "
          f"{len(seeds)} seeds ({seeds}), {args.workers} workers", flush=True)
    if args.model:
        print(f"  model: {args.model}", flush=True)
    print("=" * 78, flush=True)

    tag = args.controller
    csv_path = path.join(args.results_dir, f"raw_steps_{tag}.csv")
    all_rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "seed", "step", "reward",
                                                "action_too_late_delta", "action_churn_delta",
                                                "mean_abs_pair_margin_db"] + list(TERMS))
        writer.writeheader()
        for seed, rows in run_condition(args.controller, args.model, seeds, args, args.workers,
                                        args.per_seed_timeout_s, deadline):
            all_rows.extend(rows)
            for r in rows:
                writer.writerow(r)
            f.flush()
            print(f"  >>> running total: {len(all_rows)} rows from "
                  f"{len(set(r['seed'] for r in all_rows))} seeds so far", flush=True)

    summary = summarize(all_rows)
    print("\n" + "=" * 78, flush=True)
    print(f"{tag} -- DONE. {(time.time()-t0)/60:.1f} min elapsed", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    summary_path = path.join(args.results_dir, f"summary_{tag}.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {csv_path} and {summary_path}", flush=True)
