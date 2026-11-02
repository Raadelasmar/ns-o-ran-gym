#!/usr/bin/env bash
# Detached launcher for the MRO SAC pilot. Adapted from run_sac_pilot.sh
# (MLB's launcher) -- same systemd-cgroup protection, same reasoning, applied
# to examples/train_mro_sac_pilot.py instead.
#
# WHY A SEPARATE SERVICE/UNIT: `systemctl --user stop sacpilot.service` must
# never accidentally touch the other agent's run, so this gets its own unit
# name (mrosacpilot), its own log dir (output/mro_sac_pilot/), and its own
# LATEST_RUN pointer. The two can run concurrently (different ports, 2026-10
# machine has 16 cores / ~10 GB free -- see MRO's outage-stress/baseline
# reruns, which already validated 5-way ns-3 parallelism on this box).
#
# N_ENVS DEFAULT IS 2, NOT train_mro_sac_pilot.py's OWN DEFAULT OF 5: that
# script's --n_envs=5 has never been profiled under a hard cgroup MemoryMax
# -- there is no MRO equivalent yet of the memprobe that drove MLB's n_envs
# down from 3 to 2 after two real oomd kills (2026-09-01, 2026-09-06/07, see
# MLB's AGENT_BUILD_LOG.md). Same ns-3 scenario/topology as MLB (7 cells),
# so per-worker memory should be comparable -- starting at MLB's hard-won
# safe value is the conservative choice until MRO has its own profiling
# data. Raise N_ENVS only after watching a real run with watch_mro_sac_pilot.sh.
#
# KillSignal=SIGINT + KillMode=mixed makes `systemctl --user stop` safe: the
# trainer saves model and buffer before exiting. Never kill -9.
set -euo pipefail

N_ENVS=${N_ENVS:-2}
# Same cgroup caps as MLB's launcher, as a starting point -- same scenario
# config drives ns-3's per-episode memory/disk footprint, not which agent
# is acting (see train_mro_sac_pilot.py's own comment on MIN_FREE_GB etc.).
# UNVALIDATED for MRO specifically: watch the first real run closely.
MEM_HIGH=${MEM_HIGH:-infinity}
MEM_MAX=${MEM_MAX:-10.5G}
MEM_SWAP=${MEM_SWAP:-2G}
ANON_CEILING_GB=${ANON_CEILING_GB:-6.0}    # matches train_mro_sac_pilot.py's own MEM_ANON_CEILING_GB
MIN_AVAIL_GB=${MIN_AVAIL_GB:-2.5}
MIN_FREE_GB=${MIN_FREE_GB:-3.0}
# train_freq=1 (ARM_C) makes updates/sample = gradient_steps/N_ENVS. ARM_C's
# own gradient_steps=16 would give 16/2=8.0 at N_ENVS=2 -- a different arm
# from what MLB validated (gs=6 at n_envs=2 -> 3.00, close to arm C's own
# 16/5=3.20 at its native n_envs). Scale gradient_steps down with N_ENVS so
# this run measures "does arm C's update ratio work for MRO", not a new ratio.
GRADIENT_STEPS=${GRADIENT_STEPS:-6}

TOTAL=${TOTAL:-8000}
LEARNING_STARTS=${LEARNING_STARTS:-5000}
WARM_START_STEPS=${WARM_START_STEPS:-0}
PROBE_EVERY=${PROBE_EVERY:-250}
RB_SAVE_EVERY=${RB_SAVE_EVERY:-500}
SEED=${SEED:-555}
BASE_PORT=${BASE_PORT:-5565}    # offset from MLB's 5555 so both can run at once
RESUME_FROM=${RESUME_FROM:-}

UNIT=mrosacpilot
# Real paths on this machine (NOT ~/oran-project/..., the placeholder every
# other script in this repo defaults to and which does not exist here).
GYM="${GYM:-$HOME/son_marl_project/ns-o-ran-gym}"
VENV_PY="${VENV_PY:-$GYM/.venv/bin/python3}"
NS3="${NS3:-$HOME/son_marl_project/ns-o-ran-ns3-mmwave}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$GYM/output/mro_sac_pilot/logs_$STAMP"

echo "=============================================================================="
echo " MRO SAC PILOT LAUNCH  ($STAMP)"
echo "=============================================================================="

# 1. pre-flight: refuse to start rather than fail hours in
[ -x "$VENV_PY" ] || { echo "FATAL: no venv python at $VENV_PY"; exit 1; }
[ -d "$NS3" ] || { echo "FATAL: ns-3 repo not found at $NS3"; exit 1; }
[ -f "$GYM/analysis/mro_probe_obs/probe_obs.npy" ] || {
    echo "FATAL: probe_obs.npy missing. Build it with:"
    echo "       $VENV_PY $GYM/examples/build_mro_probe_obs.py --ns3_path $NS3"; exit 1; }

if systemctl --user is-active --quiet "$UNIT.service"; then
    echo "FATAL: $UNIT.service is ALREADY RUNNING. Stop it first:"
    echo "       systemctl --user stop $UNIT.service"; exit 1
fi
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true

BUSY=""
for p in $(seq "$BASE_PORT" $((BASE_PORT + N_ENVS - 1))); do
    ss -ltn "sport = :$p" 2>/dev/null | grep -q LISTEN && BUSY="$BUSY $p"
done
[ -z "$BUSY" ] || { echo "FATAL: ports busy:$BUSY (another run is holding them)."; exit 1; }

# Gitignored, so absent on a fresh clone; df would fail and abort the launch.
mkdir -p "$GYM/output"
FREE_GB=$(df -B1 --output=avail "$GYM/output" | tail -1 | awk '{printf "%.2f", $1/1e9}')
AVAIL_GB=$(awk '/MemAvailable/ {printf "%.2f", $2/1e6}' /proc/meminfo)
echo "  free disk        $FREE_GB GB   (DiskGuard floor $MIN_FREE_GB GB)"
echo "  MemAvailable     $AVAIL_GB GB  (MemGuard floor $MIN_AVAIL_GB GB)"
awk -v f="$FREE_GB" -v m="$MIN_FREE_GB" 'BEGIN{exit !(f < m + 2.0)}' && {
    echo "FATAL: only $FREE_GB GB free; want at least $(awk -v m=$MIN_FREE_GB 'BEGIN{print m+2}') GB."
    echo "       Old episode dirs are the usual culprit."
    exit 1; }

# 2. linger: the run must outlive the login session
if [ "$(loginctl show-user "$USER" -p Linger --value)" != "yes" ]; then
    echo "  enabling linger for $USER (keeps the user manager up after logout)"
    loginctl enable-linger "$USER"
fi
echo "  linger           $(loginctl show-user "$USER" -p Linger --value)"

mkdir -p "$LOGDIR"
cp "$0" "$LOGDIR/launch.sh"          # the exact invocation, on disk, as it ran

RESUME_ARG=()
[ -n "$RESUME_FROM" ] && RESUME_ARG=(--resume_from "$RESUME_FROM") && \
    echo "  RESUMING FROM    $RESUME_FROM"

echo "  workers          $N_ENVS   ports $BASE_PORT..$((BASE_PORT + N_ENVS - 1))"
echo "  gradient_steps   $GRADIENT_STEPS  -> $(awk -v g=$GRADIENT_STEPS -v n=$N_ENVS 'BEGIN{printf "%.2f", g/n}') updates/sample (arm C's own ratio: 3.20)"
echo "  budget           $TOTAL timesteps, learning_starts $LEARNING_STARTS, warm_start $WARM_START_STEPS"
echo "  cgroup caps      MemoryHigh=$MEM_HIGH  MemoryMax=$MEM_MAX  MemorySwapMax=$MEM_SWAP"
echo "  logs             $LOGDIR"
echo "------------------------------------------------------------------------------"

# 3. launch as a transient user service (not a scope, not nohup)
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "  DRYRUN=1: all pre-flight checks passed, NOT launching."
    rm -rf "$LOGDIR"
    exit 0
fi

systemd-run --user \
    --unit="$UNIT" \
    --description="MRO SAC pilot on real ns-3 ($STAMP)" \
    -p WorkingDirectory="$GYM" \
    -p MemoryHigh="$MEM_HIGH" \
    -p MemoryMax="$MEM_MAX" \
    -p MemorySwapMax="$MEM_SWAP" \
    -p MemoryAccounting=yes \
    -p CPUAccounting=yes \
    -p IOAccounting=yes \
    -p CPUWeight=60 \
    -p KillSignal=SIGINT \
    -p KillMode=mixed \
    -p TimeoutStopSec=600 \
    -p Restart=no \
    -p StandardOutput="append:$LOGDIR/pilot.log" \
    -p StandardError="append:$LOGDIR/pilot.log" \
    --setenv=OMP_NUM_THREADS=1 \
    --setenv=MKL_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 \
    --setenv=PYTHONUNBUFFERED=1 \
    "$VENV_PY" -u "$GYM/examples/train_mro_sac_pilot.py" \
        --n_envs "$N_ENVS" \
        --total_timesteps "$TOTAL" \
        --learning_starts "$LEARNING_STARTS" \
        --warm_start_steps "$WARM_START_STEPS" \
        --probe_every "$PROBE_EVERY" \
        --gradient_steps "$GRADIENT_STEPS" \
        --rb_save_every "$RB_SAVE_EVERY" \
        --seed "$SEED" \
        --base_port "$BASE_PORT" \
        --ns3_path "$NS3" \
        --min_free_gb "$MIN_FREE_GB" \
        --mem_anon_ceiling_gb "$ANON_CEILING_GB" \
        --min_avail_gb "$MIN_AVAIL_GB" \
        --log_dir "$LOGDIR" \
        "${RESUME_ARG[@]}"

echo "$LOGDIR" > "$GYM/output/mro_sac_pilot/LATEST_RUN"
sleep 3

# 4. prove the detachment: "survives the terminal" is not "survives the session".
MAINPID=$(systemctl --user show "$UNIT.service" -p MainPID --value)
echo "------------------------------------------------------------------------------"
echo "  state            $(systemctl --user show "$UNIT.service" -p ActiveState --value)"
echo "  MainPID          $MAINPID"
if [ "$MAINPID" != "0" ] && [ -r "/proc/$MAINPID/cgroup" ]; then
    CG=$(cut -d: -f3 "/proc/$MAINPID/cgroup")
    echo "  cgroup           $CG"
    case "$CG" in
        *vte-spawn*|*session.slice*)
            echo "  !! WARNING: still inside the session cgroup; detachment FAILED." ;;
        *"$UNIT.service")
            echo "  OK: own cgroup under app.slice; a session restart cannot kill this." ;;
    esac
fi
echo "=============================================================================="
echo " MONITOR IT WITH:   $GYM/watch_mro_sac_pilot.sh"
echo " STOP IT SAFELY:    systemctl --user stop $UNIT.service    # SIGINT, checkpoints"
echo "=============================================================================="
