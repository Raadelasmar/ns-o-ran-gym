"""Find the real parallelism ceiling on this machine for v3 (2026-10-26).

Sweeps candidate worker counts, each running that many ns-3 processes in
parallel for a SHORT episode (simtime=2 -> 20 control steps at
controlInterval=0.1), one wave, do_nothing action (a true no-op, so this
measures pure simulation throughput, not controller-specific behaviour --
already shown in mro_v3_throughput_diagnostic.py to track v3_rule closely
under no contention).

For each candidate N: launch N seeds on N workers (one wave), with a
per-seed timeout generous enough to let a genuine completion land but not so
long the whole sweep takes forever if a level is broken. Records: how many
of the N seeds completed within the window, and the WALL-CLOCK time for the
whole batch (not sum of individual seed times -- they run in parallel, so
this is what actually matters for training throughput:
aggregate_steps / wall_clock_time).

Usage:
    python examples/mro_nenvs_sweep.py --candidates 5,6,7,8,9,10,12,14
"""
import argparse
import subprocess
import sys
import time
from os import path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="5,6,7,8,9,10,12,14")
    ap.add_argument("--simtime", type=int, default=2)
    ap.add_argument("--per_seed_timeout_s", type=float, default=600)
    ap.add_argument("--zmq_port_base", type=int, default=6400)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    ap.add_argument("--results_dir", default=path.join(path.dirname(__file__), "..", "output",
                                                        "mro_nenvs_sweep"))
    args = ap.parse_args()

    candidates = [int(x) for x in args.candidates.split(",")]
    script = path.join(path.dirname(__file__), "mro_v3_rule_only_eval.py")
    venv_py = path.join(path.dirname(__file__), "..", ".venv", "bin", "python3")

    results = []
    print("=" * 78, flush=True)
    print(f"n_envs sweep: candidates={candidates}, simtime={args.simtime} "
          f"({args.simtime*10} steps/seed)", flush=True)
    print("=" * 78, flush=True)

    for i, n in enumerate(candidates):
        results_dir = path.join(args.results_dir, f"n{n}")
        port_base = args.zmq_port_base + i * 100  # wide spacing, no port reuse across rounds
        t0 = time.time()
        print(f"\n--- n_envs={n}: starting {time.strftime('%H:%M:%S')} ---", flush=True)
        proc = subprocess.run(
            [venv_py, script, "--controller", "do_nothing", "--simtime", str(args.simtime),
             "--seeds", str(n), "--seed_start", str(10000 + i * 1000), "--workers", str(n),
             "--per_seed_timeout_s", str(args.per_seed_timeout_s),
             "--total_budget_s", "90", "--zmq_port", str(port_base),
             "--ns3_path", args.ns3_path, "--output_folder", args.output_folder,
             "--results_dir", results_dir],
            capture_output=True, text=True,
            timeout=args.per_seed_timeout_s + 120,
        )
        wall_s = time.time() - t0
        csv_path = path.join(results_dir, "raw_steps_do_nothing.csv")
        completed_seeds = set()
        try:
            with open(csv_path) as f:
                next(f)  # header
                for line in f:
                    completed_seeds.add(line.split(",")[1])
        except FileNotFoundError:
            pass
        n_completed = len(completed_seeds)
        total_steps = sum(1 for _ in open(csv_path)) - 1 if n_completed else 0
        throughput = total_steps / wall_s if wall_s > 0 else 0.0
        result = {
            "n_envs": n, "wall_s": wall_s, "completed": n_completed, "of": n,
            "total_steps": total_steps, "aggregate_steps_per_s": throughput,
            "returncode": proc.returncode,
        }
        results.append(result)
        print(f"  n_envs={n}: {n_completed}/{n} seeds completed, {total_steps} total steps, "
              f"{wall_s:.0f}s wall, {throughput:.3f} aggregate steps/s "
              f"(returncode={proc.returncode})", flush=True)
        if proc.returncode != 0:
            print(f"  stderr tail: {proc.stderr[-500:]}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("SWEEP SUMMARY", flush=True)
    print(f"{'n_envs':>8} {'completed':>12} {'wall_s':>8} {'steps/s':>10}", flush=True)
    for r in results:
        print(f"{r['n_envs']:>8} {r['completed']:>5}/{r['of']:<5} {r['wall_s']:>8.0f} "
              f"{r['aggregate_steps_per_s']:>10.3f}", flush=True)
    print("=" * 78, flush=True)
