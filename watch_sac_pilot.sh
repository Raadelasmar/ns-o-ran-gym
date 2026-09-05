#!/usr/bin/env bash
# One-shot status for the running SAC pilot. Read-only, safe to run any time.
#
#   ./watch_sac_pilot.sh          one snapshot
#   watch -n 60 ./watch_sac_pilot.sh   refresh every minute
UNIT=sacpilot
GYM="${GYM:-$HOME/oran-project/ns-o-ran-gym}"
LOGDIR=$(cat "$GYM/output/sac_pilot/LATEST_RUN" 2>/dev/null)

hr() { printf '%s\n' "------------------------------------------------------------------------------"; }

echo "=============================================================================="
echo " SAC PILOT STATUS   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

# 1. alive?
STATE=$(systemctl --user show "$UNIT.service" -p ActiveState --value 2>/dev/null)
SUB=$(systemctl --user show "$UNIT.service" -p SubState --value 2>/dev/null)
MAINPID=$(systemctl --user show "$UNIT.service" -p MainPID --value 2>/dev/null)
echo " unit        ${STATE:-not-loaded} / ${SUB:-} (MainPID ${MAINPID:-0})"
if [ "${STATE:-}" != "active" ]; then
    echo " !! NOT RUNNING. Exit info:"
    systemctl --user show "$UNIT.service" -p Result -p ExecMainStatus -p ExecMainCode 2>/dev/null | sed 's/^/    /'
    echo "    run status file:"
    [ -n "$LOGDIR" ] && cat "$LOGDIR/run_status.json" 2>/dev/null | sed 's/^/    /'
fi
SINCE=$(systemctl --user show "$UNIT.service" -p ActiveEnterTimestamp --value 2>/dev/null)
[ -n "$SINCE" ] && echo " started     $SINCE"

# 2. memory (what killed the Step 13 run)
hr
U=$(id -u)
CG=/sys/fs/cgroup/user.slice/user-$U.slice/user@$U.service/app.slice/$UNIT.service
if [ -d "$CG" ]; then
    g() { awk -v n="$1" '$1==n {printf "%.2f", $2/1e9}' "$CG/memory.stat"; }
    b() { awk '{printf "%.2f", $1/1e9}' "$CG/$1" 2>/dev/null; }
    printf " cgroup mem  current %s GB | anon %s GB (the one that matters) | cache %s GB | swap %s GB | peak %s GB\n" \
        "$(b memory.current)" "$(g anon)" "$(g file)" "$(b memory.swap.current)" "$(b memory.peak)"
    printf " pressure    %s\n" "$(awk '/^full/ {print $2, $3, $4}' "$CG/memory.pressure" 2>/dev/null)"
    printf " ns-3 procs  "
    NW=0
    for p in $(cat "$CG/cgroup.procs" 2>/dev/null); do
        if grep -qa 'scenario-marl-zmq' "/proc/$p/cmdline" 2>/dev/null; then
            A=$(awk '/^RssAnon/ {printf "%d", $2/1024}' "/proc/$p/status" 2>/dev/null)
            printf "%s " "${A}M"; NW=$((NW+1))
        fi
    done
    echo "(${NW} workers)"
else
    echo " cgroup mem  (cgroup not present, unit not running)"
fi
awk '/MemTotal|MemAvailable|SwapFree/ {printf " %-12s %.2f GB\n", $1, $2/1e6}' /proc/meminfo

# 3. progress
hr
if [ -n "$LOGDIR" ] && [ -f "$LOGDIR/critic_probe.csv" ]; then
    python3 - "$LOGDIR" <<'PY'
import csv, json, os, sys, time
d = sys.argv[1]
try:
    total = json.load(open(os.path.join(d, "run_config.json")))["total_timesteps"]
except Exception:
    total = None
rows = list(csv.DictReader(open(os.path.join(d, "critic_probe.csv"))))
if not rows:
    print(" progress    no probe rows yet (first row lands at the first probe interval)")
else:
    r = rows[-1]
    t, el = int(r["num_timesteps"]), float(r["elapsed_s"])
    sph = float(r["samples_per_hour"]) or 1.0
    pct = f"{100*t/total:.1f}%" if total else "?"
    eta = f"{(total-t)/sph:.1f} h" if total and t < total else "--"
    print(f" progress    {t}{'/'+str(total) if total else ''} timesteps ({pct}) "
          f"| {el/3600:.2f} h elapsed | {sph:.0f} samples/h | ETA {eta}")
    print(f" learning    n_updates={r['n_updates']}  Q={r['q_mean']} (sd {r['q_sd']})  "
          f"{r['q_pct_of_target_recent']}% of target  drift {r['action_drift_db']} dB  "
          f"alpha {r['ent_coef']}")
    print(f" last probe  {r['wall_iso']}  ({(time.time()-time.mktime(time.strptime(r['wall_iso'],'%Y-%m-%dT%H:%M:%S')))/60:.0f} min ago)")
PY
else
    echo " progress    no critic_probe.csv yet at ${LOGDIR:-<no LATEST_RUN>}"
fi

# 4. reward terms and per-worker liveness
hr
if [ -n "$LOGDIR" ] && [ -f "$LOGDIR/reward_terms_steps.csv" ]; then
    python3 - "$LOGDIR/reward_terms_steps.csv" <<'PY'
import csv, io, os, sys, time
TERMS = ("balance", "backlog", "badsignal", "satisfaction", "pingpong")
WINDOW, TAIL_BYTES = 200, 600_000
try:
    with open(sys.argv[1], "rb") as fh:
        header = fh.readline().decode("utf8", "replace").rstrip("\r\n")
        start = fh.tell()
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        if size - start > TAIL_BYTES:
            fh.seek(size - TAIL_BYTES)
            body = fh.read().decode("utf8", "replace").split("\n")[1:]
        else:
            fh.seek(start)
            body = fh.read().decode("utf8", "replace").split("\n")
    rows = list(csv.DictReader(io.StringIO(header + "\n" + "\n".join(body))))
except Exception as exc:
    print(f" terms       unreadable: {type(exc).__name__}: {exc}")
    raise SystemExit

steps = [r for r in rows if r.get("phase") == "step"][-WINDOW:]
if not steps:
    print(" terms       no completed steps in the CSV tail yet")
else:
    means = {}
    for t in TERMS:
        vals = []
        for r in steps:
            try:
                vals.append(float(r[t]))
            except (TypeError, ValueError, KeyError):
                pass
        means[t] = sum(vals) / len(vals) if vals else None
    print(" terms       " + " | ".join(
        f"{t} {means[t]:+.3f}" if means[t] is not None else f"{t} -" for t in TERMS)
        + f"   (mean of last {len(steps)} steps)")
    live = {t: v for t, v in means.items() if v is not None}
    if live:
        tot = sum(abs(v) for v in live.values()) or 1.0
        top = max(live, key=lambda t: abs(live[t]))
        print(f" dominant    {top} at {100*abs(live[top])/tot:.0f}% of total |term|")
        zero = [t for t, v in live.items() if v == 0.0]
        if zero:
            print(f" !! zero     {', '.join(zero)} contributed exactly 0.0 all window")

# Per-rank staleness is not a usable signal: StepLogger stamps every row of a
# vec-step with one timestamp, so all ranks always share it. What does diverge
# per rank is phase mix (void = a step with no KPI snapshot), episode_index and
# mean reward, so those are the per-worker health signals.
per = {}
for r in rows:
    rk = r.get("env_rank")
    if rk is None or not str(rk).isdigit():
        continue
    d = per.setdefault(rk, {"step": 0, "terminal": 0, "void": 0, "ep": 0, "rw": []})
    if r.get("phase") in d:
        d[r["phase"]] += 1
    try:
        d["ep"] = max(d["ep"], int(r["episode_index"]))
    except (TypeError, ValueError, KeyError):
        pass
    try:
        d["rw"].append(float(r["reward"]))
    except (TypeError, ValueError, KeyError):
        pass
if per:
    parts = []
    for rk in sorted(per, key=int):
        d = per[rk]
        rw = f"{sum(d['rw']) / len(d['rw']):+.2f}" if d["rw"] else "-"
        parts.append(f"rank{rk} ep{d['ep']} {d['step']}s/{d['terminal']}t/{d['void']}v r{rw}")
    print(" workers     " + " | ".join(parts) + "   (tail window)")
    voids = sorted((rk, d["void"]) for rk, d in per.items() if d["void"])
    if voids:
        print(" !! void      steps with no KPI snapshot: "
              + ", ".join(f"rank{rk}={n}" for rk, n in voids))
    eps = [d["ep"] for d in per.values()]
    if max(eps) - min(eps) > 1:
        print(f" !! episodes  ranks disagree by {max(eps) - min(eps)} episodes;"
              f" a worker may be cycling resets")
stamps = [r["wall_iso"] for r in rows if r.get("wall_iso")]
if stamps:
    try:
        age = (time.time()
               - time.mktime(time.strptime(stamps[-1], "%Y-%m-%dT%H:%M:%S"))) / 60
        print(f" last step   {stamps[-1]}  ({age:.0f} min ago)")
    except Exception:
        pass
PY
else
    echo " terms       no reward_terms_steps.csv yet at ${LOGDIR:-<no LATEST_RUN>}"
fi

# 5. disk
hr
df -h "$GYM/output" | awk 'NR==2 {printf " disk        %s free of %s (%s used)\n", $4, $2, $5}'
python3 - "$GYM/output" <<'PY'
import os, re, sys
root = sys.argv[1]; n = tot = 0
for e in os.scandir(root):
    if e.is_dir() and re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-", e.name):
        n += 1
        for r, _, fs in os.walk(e.path):
            for f in fs:
                try: tot += os.path.getsize(os.path.join(r, f))
                except OSError: pass
print(f" episodes    {n} live episode dirs, {tot/1e9:.2f} GB "
      f"({'normal: one per worker' if n <= 8 else 'HIGH: is the purge still working?'})")
PY
[ -n "$LOGDIR" ] && ls -la "$GYM/output/sac_pilot/mlb_sac_replay_live.pkl" 2>/dev/null | \
    awk '{printf " replay buf  %.0f MB, last written %s %s %s\n", $5/1e6, $6, $7, $8}'

# 6. thermal
hr
PKG=$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_total_time_ms 2>/dev/null)
UP=$(awk '{printf "%d", $1*1000}' /proc/uptime)
[ -n "$PKG" ] && awk -v p="$PKG" -v u="$UP" 'BEGIN{printf " thermal     package throttled %.0f s of %.0f s uptime (%.1f%%)\n", p/1000, u/1000, 100*p/u}'
sensors 2>/dev/null | awk '/Package id 0/ {printf " temp        %s\n", $0}' || \
    awk '{printf " temp        %.0f C\n", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null

# 7. tail
hr
echo " last 6 log lines:"
[ -n "$LOGDIR" ] && tail -6 "$LOGDIR/pilot.log" 2>/dev/null | sed 's/^/   /'
echo "=============================================================================="
