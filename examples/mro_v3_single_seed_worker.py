import argparse
import csv
import sys
import time
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
sys.path.insert(0, path.dirname(__file__))
from environments.mro_zmq_env import MroZmqEnv

import numpy as np
from mro_v3_overnight_eval import CFG, TERMS  # ACTUALLY USED, unlike the earlier ablation


def run(controller, action, seed, steps, ns3_path, output_folder, zmq_port, csv_writer, csv_file):
    # csv_writer.writerow() + csv_file.flush() happen after EVERY step (not
    # just once at the end) so that a timeout-triggered proc.terminate() from
    # the orchestrator (mro_v3_final_eval.py) leaves whatever steps already
    # ran on disk, instead of losing the whole episode's progress.
    cfg = dict(CFG)
    cfg["RngRun"] = [seed]
    env = MroZmqEnv(ns3_path=ns3_path, scenario_configuration=cfg,
                    output_folder=output_folder, optimized=False,
                    zmq_port=zmq_port, build_ns3=False,
                    vary_rng_run_per_episode=False, purge_sim_path_on_close=True)
    rows = []
    try:
        t0 = time.time()
        obs, info = env.reset(seed=seed)
        print(f"[{controller}] reset: {time.time()-t0:.1f}s", flush=True)
        i = 0
        while True:
            ts = time.time()
            step_action = action(obs) if callable(action) else action
            obs, reward, terminated, truncated, info = env.step(step_action)
            dt = time.time() - ts
            if "reward_terms" not in info:
                break
            terms = info["reward_terms"]
            pair_margins = info.get("pair_margins_db") or {}
            mean_abs_margin = (sum(abs(v) for v in pair_margins.values()) / len(pair_margins)
                               if pair_margins else 0.0)
            act = np.asarray(step_action, dtype=np.float32).reshape(-1)
            row = {"controller": controller, "seed": seed, "step": i, "reward": reward,
                   "action_too_late_delta": float(act[0]), "action_churn_delta": float(act[1]),
                   "mean_abs_pair_margin_db": mean_abs_margin}
            for t in TERMS:
                row[t] = terms[t]
            print(f"[{controller}] step {i:3d}: dt={dt:6.2f}s reward={reward}", flush=True)
            rows.append(row)
            csv_writer.writerow(row)
            csv_file.flush()
            i += 1
            if terminated or truncated:
                break
    except Exception as exc:
        print(f"CRASHED: {exc!r}", flush=True)
        raise
    finally:
        env.close()
    print(f"[{controller}] TOTAL: {time.time()-t0:.1f}s for {len(rows)} steps", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True,
                    choices=["do_nothing", "v3_rule", "trained_policy"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--simtime", type=int, default=30)
    ap.add_argument("--zmq_port", type=int, required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder", default=path.join(path.dirname(__file__), "..", "output"))
    args = ap.parse_args()

    if args.controller == "do_nothing":
        action = np.array([-1.0, -1.0], dtype=np.float32)
    elif args.controller == "v3_rule":
        action = np.array([0.0, 0.0], dtype=np.float32)
    else:
        if not args.model:
            raise SystemExit("--model is required for --controller trained_policy")
        from stable_baselines3 import SAC
        _model = SAC.load(args.model, device="cpu")

        def action(obs):
            a, _ = _model.predict(obs, deterministic=True)
            return a
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "seed", "step", "reward",
                                                "action_too_late_delta", "action_churn_delta",
                                                "mean_abs_pair_margin_db"] + list(TERMS))
        writer.writeheader()
        f.flush()
        rows = run(args.controller, action, args.seed, None, args.ns3_path,
                  args.output_folder, args.zmq_port, writer, f)
    print(f"  [{args.controller}] seed {args.seed}: wrote {len(rows)} rows to "
          f"{args.output_csv}", flush=True)
