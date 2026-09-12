#!/usr/bin/env python3
"""Stage A: reproduce the 17.7 GB ns-3 balloon in ONE worker, no SB3.

Runs a single MlbZmqEnv with a PINNED RngRun and random actions, sampling the
ns-3 child's RSS from /proc/<pid>/status every step. No trainer, no
SubprocVecEnv, no replay buffer -- so anything that grows here is ns-3's.

Why the seed can be pinned exactly: MlbZmqEnv.reset() draws RngRun from
self.np_random only when vary_rng_run_per_episode is True. With it False the
env launches ns-3 with scenario_configuration['RngRun'] verbatim, so passing a
seed observed in the killed run's rng_run column replays that worker's episode
0 world bit-for-bit. udp_interval_us_range is None in the SAC pilot, so no other
per-episode randomisation exists to diverge on.

Sampling uses /proc/<pid>/status (RssAnon is the exact quantity the kernel OOM
dump reported as anon-rss) because it is O(1) counter reads. /proc/<pid>/smaps
walks every VMA and gets slow at multi-GB RSS, so it is only dumped at the
--dump-at thresholds and at exit -- that dump is what names the growing region.

SAFETY: --abort-at-gb kills ns-3 and exits cleanly BEFORE the cgroup OOM killer
fires, so the CSV and the smaps dumps survive. Run inside a capped scope anyway:

  systemd-run --user --scope -p MemoryMax=20G -p MemoryHigh=infinity \
      ~/oran-project/gym-venv/bin/python \
      analysis/mlb_session_2026-09-08/memprofile_stage_a.py --seed 868986719
"""
import argparse
import csv
import os
import sys
import time
from os import path

sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), "..", "..", "src"))
from environments.mlb_zmq_env import MlbZmqEnv  # noqa: E402

# Identical to examples/train_mlb_sac_pilot.py CFG, except RngRun, which is
# overridden per run. Kept as a literal copy rather than an import so this
# script stays runnable if the trainer is edited mid-investigation.
CFG = {
    "heuristicType": [-1], "simTime": [30], "ues": [5], "RngRun": [555],
    "configuration": [0], "trafficModel": [3], "numberOfRaPreambles": [40],
    "bsOn": [5], "bsIdle": [0], "bsSleep": [0], "bsOff": [2],
    "reducedPmValues": [0], "outageThreshold": [-5.0],
    "handoverMode": ["DynamicTtt"], "indicationPeriodicity": [0.1],
    "e2nrEnabled": [1], "rlcAmEnabled": [1],
    "controlInterval": [0.1],
    "ns3::MmWavePhyTrace::OutputFilename": ["/dev/null"],
    "ns3::MmWavePhyTrace::DlPhyTransmissionFilename": ["/dev/null"],
    "ns3::MmWaveMacTrace::SchedInfoOutputFilename": ["/dev/null"],
    "ns3::MmWaveBearerStatsConnector::MmWaveSinrOutputFilename": ["/dev/null"],
    "ns3::CoreNetworkStatsCalculator::X2FileName": ["/dev/null"],
    "ns3::PhyStatsCalculator::DlRsrpSinrFilename": ["/dev/null"],
    "ns3::PhyStatsCalculator::UlInterferenceFilename": ["/dev/null"],
    "ns3::LteRlcAm::BufferSizeFilename": ["/dev/null"],
}

STATUS_FIELDS = ("VmSize", "VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM", "VmData")


def read_status(pid):
    """{field: kB} from /proc/<pid>/status. Empty dict once the process is gone."""
    out = {}
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in STATUS_FIELDS:
                    out[key] = int(rest.split()[0])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    return out


def dump_smaps(pid, dest, tag):
    """Full /proc/<pid>/smaps plus a sorted top-region summary.

    The summary is the payload: it says whether the growth is one enormous
    anonymous mapping (a single huge container/vector), many medium ones (an
    mmap-per-object pattern), or [heap] brk growth (ordinary malloc churn).
    """
    try:
        with open(f"/proc/{pid}/smaps") as fh:
            raw = fh.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    with open(dest, "w") as fh:
        fh.write(raw)

    regions, cur = [], None
    for line in raw.splitlines():
        head = line.split()
        if not head:
            continue
        if "-" in head[0] and not line.startswith(" "):
            cur = {"addr": head[0], "perms": head[1],
                   "name": head[5] if len(head) > 5 else "[anon]", "rss": 0}
            regions.append(cur)
        elif cur is not None and line.startswith("Rss:"):
            cur["rss"] = int(line.split()[1])

    regions.sort(key=lambda r: -r["rss"])
    by_name = {}
    for r in regions:
        by_name[r["name"]] = by_name.get(r["name"], 0) + r["rss"]

    summary = [f"--- smaps summary @ {tag} (pid {pid}), {len(regions)} regions ---",
               "  top 15 single regions by RSS:"]
    for r in regions[:15]:
        summary.append(f"    {r['rss']/1048576:9.3f} GB  {r['perms']}  "
                       f"{r['addr']}  {r['name']}")
    summary.append("  top 10 by mapping name:")
    for name, rss in sorted(by_name.items(), key=lambda kv: -kv[1])[:10]:
        summary.append(f"    {rss/1048576:9.3f} GB  {name}")
    text = "\n".join(summary)
    with open(dest + ".summary", "w") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True,
                    help="RngRun to pin, e.g. one of the killed run's rng_run values")
    ap.add_argument("--steps", type=int, default=210)
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--ns3_path", default=path.expanduser("~/oran-project/ns-3-mmwave-oran"))
    ap.add_argument("--output_folder",
                    default=path.join(path.dirname(path.abspath(__file__)), "runs"))
    ap.add_argument("--out_dir",
                    default=path.join(path.dirname(path.abspath(__file__)), "memprofile"))
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--dump_at", default="4,8,12",
                    help="comma-separated RssAnon GB thresholds to dump smaps at")
    ap.add_argument("--abort_at_gb", type=float, default=15.0,
                    help="kill ns-3 and exit cleanly at this RssAnon, before the OOM killer")
    ap.add_argument("--preload", default=None,
                    help="absolute path to mallocinfo_sampler.so; injected into the "
                         "ns-3 child's LD_PRELOAD (Stage B1)")
    ap.add_argument("--mallocinfo_interval", type=int, default=60)
    ap.add_argument("--preload-env", action="append", default=None,
                    metavar="KEY=VALUE",
                    help="extra env var for the preloaded .so; repeatable")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)
    thresholds = sorted(float(t) for t in args.dump_at.split(",") if t.strip())
    csv_path = path.join(args.out_dir, f"rss_seed{args.seed}.csv")

    cfg = dict(CFG)
    cfg["RngRun"] = [args.seed]

    env = MlbZmqEnv(
        ns3_path=args.ns3_path, scenario_configuration=cfg,
        output_folder=args.output_folder, optimized=False,
        zmq_port=args.port,
        build_ns3=False,
        # THE point of this script: replay one exact seed. Also disables the
        # reset retry loop, so a bad seed surfaces instead of being swapped out.
        vary_rng_run_per_episode=False,
        udp_interval_us_range=None,
        # Keep the run dir: 083550e's trace redirect already cut it to ~5 MB,
        # and stdout/stderr in it are evidence if ns-3 dies.
        purge_sim_path_on_close=False,
    )

    # NsOranEnv.setup_sim() builds self.environment as ONLY LD_LIBRARY_PATH +
    # DYLD_LIBRARY_PATH, and start_sim passes it to Popen(env=...) -- which
    # REPLACES the environment rather than extending it. So an LD_PRELOAD
    # exported in the parent shell never reaches ns-3; it has to be injected
    # here, after __init__ has built the dict and before reset() launches.
    if args.preload:
        preload = path.abspath(args.preload)
        if not path.exists(preload):
            sys.exit(f"--preload {preload} does not exist; build it with the gcc line "
                     f"in mallocinfo_sampler.c")
        mi_dir = path.join(args.out_dir, f"mallocinfo_seed{args.seed}")
        os.makedirs(mi_dir, exist_ok=True)
        env.environment["LD_PRELOAD"] = preload
        env.environment["MALLOCINFO_DIR"] = mi_dir
        env.environment["MALLOCINFO_INTERVAL"] = str(args.mallocinfo_interval)
        # Generic passthrough so a different .so (e.g. sizehist.so) can be
        # configured without editing this script again: --preload-env K=V
        for kv in args.preload_env or []:
            k, _, v = kv.partition("=")
            env.environment[k] = v
        print(f"malloc_info       every {args.mallocinfo_interval}s -> {mi_dir}", flush=True)

    print(f"seed(RngRun)      {args.seed}", flush=True)
    print(f"binary            {env.script_executable}", flush=True)
    print(f"csv               {csv_path}", flush=True)
    print(f"smaps dumps at    {thresholds} GB RssAnon, abort at {args.abort_at_gb} GB",
          flush=True)

    t0 = time.time()
    obs, info = env.reset()
    if info.get("reset_failed"):
        sys.exit(f"ns-3 died at start-up on seed {args.seed}; nothing to profile.")
    pid = env.sim_process.pid
    print(f"ns-3 pid          {pid}   reset in {time.time()-t0:.1f}s", flush=True)

    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["step", "wall_s", "step_ms", "sim_timestamp",
                     "vmsize_kb", "vmrss_kb", "rss_anon_kb", "rss_file_kb",
                     "rss_shmem_kb", "vmhwm_kb", "vmdata_kb",
                     "py_rss_kb", "reward", "backlog_bytes_sum"])

    self_pid = os.getpid()
    fired = set()
    reason = "completed"
    try:
        for step in range(1, args.steps + 1):
            ts = time.time()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_ms = (time.time() - ts) * 1000.0

            st = read_status(pid)
            if not st:
                reason = f"ns-3 process gone at step {step}"
                break
            py = read_status(self_pid)
            kpis = info.get("kpis") or {}
            backlog = sum(float(c.get("buffer_bytes", 0.0) or 0.0)
                          for c in (kpis.get("cells") or {}).values())

            writer.writerow([step, round(time.time() - t0, 2), round(step_ms, 1),
                             kpis.get("timestamp"),
                             st.get("VmSize"), st.get("VmRSS"), st.get("RssAnon"),
                             st.get("RssFile"), st.get("RssShmem"), st.get("VmHWM"),
                             st.get("VmData"), py.get("VmRSS"),
                             round(float(reward), 4), int(backlog)])
            fh.flush()

            anon_gb = st.get("RssAnon", 0) / 1048576.0
            if step % args.print_every == 0 or step == 1:
                print(f"step {step:4d}  t={time.time()-t0:6.1f}s  "
                      f"anon={anon_gb:7.3f} GB  rss={st.get('VmRSS',0)/1048576:7.3f} GB  "
                      f"vsz={st.get('VmSize',0)/1048576:7.3f} GB  "
                      f"py={py.get('VmRSS',0)/1024:6.0f} MB  "
                      f"step={step_ms:6.0f} ms  backlog={backlog/1e6:6.1f} MB", flush=True)

            for th in thresholds:
                if anon_gb >= th and th not in fired:
                    fired.add(th)
                    dest = path.join(args.out_dir, f"smaps_seed{args.seed}_{th:g}GB_step{step}")
                    text = dump_smaps(pid, dest, f"{th:g} GB / step {step}")
                    print(f"\n[dumped smaps -> {dest}]\n{text}\n", flush=True)

            if anon_gb >= args.abort_at_gb:
                dest = path.join(args.out_dir, f"smaps_seed{args.seed}_ABORT_step{step}")
                text = dump_smaps(pid, dest, f"ABORT {anon_gb:.2f} GB / step {step}")
                print(f"\n[ABORT threshold hit -> {dest}]\n{text}\n", flush=True)
                reason = f"aborted at {anon_gb:.2f} GB, step {step}"
                break

            if terminated or truncated:
                reason = f"episode ended at step {step} (term={terminated} trunc={truncated})"
                break
    finally:
        fh.close()
        final = read_status(pid)
        if final:
            dest = path.join(args.out_dir, f"smaps_seed{args.seed}_final")
            dump_smaps(pid, dest, "final")
        env.close()

    print(f"\nfinished: {reason}", flush=True)
    if final:
        print(f"peak VmHWM        {final.get('VmHWM',0)/1048576:.3f} GB", flush=True)
        print(f"final RssAnon     {final.get('RssAnon',0)/1048576:.3f} GB", flush=True)
    print(f"csv               {csv_path}", flush=True)


if __name__ == "__main__":
    main()
