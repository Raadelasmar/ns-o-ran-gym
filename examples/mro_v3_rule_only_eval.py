"""Third attempt (2026-10-26, ~04:15): v3_rule ALONE, reusing
mro_v3_overnight_eval.py's own functions. do_nothing already succeeded (5/5
seeds, real data, see output/mro_v3_overnight_eval/raw_steps.csv) and is not
re-run here. v3_rule's own corrections appear to meaningfully slow ns-3
(every wave-1 seed either crashed or hit the 50-min timeout at simtime=10
under 14-way parallel load) -- this attempt uses a shorter simtime and a
longer per-seed timeout, calibrated off seed 6001's real partial progress
(43/99 steps before an unrelated crash) from the previous attempt.
"""
import sys
import time
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
sys.path.insert(0, path.dirname(__file__))
from mro_v3_overnight_eval import run_condition, summarize  # noqa: E402

import csv  # noqa: E402
import json  # noqa: E402
import argparse  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["v3_rule", "do_nothing"], default="v3_rule")
    ap.add_argument("--simtime", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=14)
    ap.add_argument("--seed_start", type=int, default=6101)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--per_seed_timeout_s", type=float, default=3300)
    ap.add_argument("--total_budget_s", type=float, default=1800)
    ap.add_argument("--zmq_port", type=int, default=5980)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--results_dir", default=path.join(path.dirname(__file__), "..", "output",
                                                        "mro_v3_overnight_eval"))
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + args.total_budget_s
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    import os
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 78, flush=True)
    print(f"{args.controller}: simtime={args.simtime}, {len(seeds)} seeds "
          f"({seeds[0]}..{seeds[-1]}), {args.workers} workers", flush=True)
    print(f"  started  {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  deadline {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(deadline))}", flush=True)
    print("=" * 78, flush=True)

    csv_path = path.join(args.results_dir, f"raw_steps_{args.controller}.csv")
    all_rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "seed", "step", "reward",
                                               "quality", "pingpong", "handover_rate", "outage"])
        writer.writeheader()
        for seed, rows in run_condition(args.controller, seeds, args, args.workers,
                                        args.per_seed_timeout_s, deadline):
            all_rows.extend(rows)
            for r in rows:
                writer.writerow(r)
            f.flush()
            print(f"  >>> running total: {len(all_rows)} rows from "
                  f"{len(set(r['seed'] for r in all_rows))} seeds so far", flush=True)

    summary = summarize(all_rows)
    print("\n" + "=" * 78, flush=True)
    print(f"{args.controller} -- DONE. {(time.time()-t0)/60:.1f} min elapsed", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    summary_path = path.join(args.results_dir, f"{args.controller}_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {csv_path} and {summary_path}", flush=True)
