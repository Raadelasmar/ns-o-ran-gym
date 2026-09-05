#!/usr/bin/env bash
# Detached launcher for the SAC pilot. Step 14, after the Step 13 post-mortem.
#
# Step 13 used `setsid nohup ... &`, which detaches the terminal but NOT the
# systemd cgroup: the run stayed in the terminal's scope, drove the user manager
# slice to 76% PSI memory pressure, and oomd (which picks by reclaim activity,
# not size) killed gnome-shell rather than the 9.1 GB pilot, taking the X11
# session and the run down with it.
#
# Hence, below: a transient systemd --user service with its own cgroup under
# app.slice, plus enable-linger; MemoryHigh/Max/SwapMax so the run throttles
# inside its own accounting and never arms oomd; MemGuard checkpointing before
# the ceiling; and ReplayBufferSaver writing the buffer every --rb_save_every
# timesteps, since SAC is off-policy and Step 13's checkpoints were untrained
# weights with nothing to resume from.
#
# KillSignal=SIGINT + KillMode=mixed makes `systemctl --user stop` safe: the
# trainer saves model and buffer and purges episode dirs first. Never kill -9.
set -euo pipefail

# Sizing from analysis/mlb_session_2026-08-31/memprobe (AGENT_BUILD_LOG.md Step
# 14). N_ENVS and the three memory numbers are one decision; change them together.
# 2026-09-01 crash: n_envs=3 died at t=5001, the instant learning_starts=5000 was
# crossed. SB3's off-policy train() has no callback hook, so MemGuard cannot see
# the first gradient-update burst (~1-2 GB) and the cgroup passed MemoryHigh
# unguarded. Hence n_envs=2 plus a wide margin, until train() gets a check inside.
N_ENVS=${N_ENVS:-2}
MEM_HIGH=${MEM_HIGH:-9.0G}           # soft cap: throttle + reclaim here
MEM_MAX=${MEM_MAX:-10.5G}            # hard backstop: kill inside our cgroup only
MEM_SWAP=${MEM_SWAP:-2G}
ANON_CEILING_GB=${ANON_CEILING_GB:-8.5}    # MemGuard: our own anon (below MemoryHigh)
MIN_AVAIL_GB=${MIN_AVAIL_GB:-2.5}            # MemGuard: system MemAvailable
MIN_FREE_GB=${MIN_FREE_GB:-3.0}                        # DiskGuard
# N_ENVS and GRADIENT_STEPS are one decision: train_freq=1 makes updates/sample
# equal gradient_steps/N_ENVS. Arm C ran 16/5 = 3.20; gs=6 at N_ENVS=2 gives
# 3.00. Leaving gs=16 would give 8.0, which is a different arm.
GRADIENT_STEPS=${GRADIENT_STEPS:-6}

TOTAL=${TOTAL:-8000}
LEARNING_STARTS=${LEARNING_STARTS:-5000}
PROBE_EVERY=${PROBE_EVERY:-250}
RB_SAVE_EVERY=${RB_SAVE_EVERY:-500}
SEED=${SEED:-555}
BASE_PORT=${BASE_PORT:-5555}
RESUME_FROM=${RESUME_FROM:-}

UNIT=sacpilot
GYM="${GYM:-$HOME/oran-project/ns-o-ran-gym}"
VENV_PY="${VENV_PY:-$HOME/oran-project/gym-venv/bin/python}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$GYM/output/sac_pilot/logs_$STAMP"

echo "=============================================================================="
echo " SAC PILOT LAUNCH  ($STAMP)"
echo "=============================================================================="

# 1. pre-flight: refuse to start rather than fail four hours in
[ -x "$VENV_PY" ] || { echo "FATAL: no venv python at $VENV_PY"; exit 1; }
[ -f "$GYM/analysis/mlb_session_2026-08-31/probe_obs.npy" ] || {
    echo "FATAL: probe_obs.npy missing. Copy it from a machine that has it, or"
    echo "       build it (needs mlb_pilot_capture/snapshots) with:"
    echo "       $VENV_PY analysis/mlb_session_2026-08-31/build_probe_obs.py"; exit 1; }

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
echo "  gradient_steps   $GRADIENT_STEPS  -> $(awk -v g=$GRADIENT_STEPS -v n=$N_ENVS 'BEGIN{printf "%.2f", g/n}') updates/sample (arm C: 3.20)"
echo "  budget           $TOTAL timesteps, learning_starts $LEARNING_STARTS"
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
    --description="SAC pilot on real ns-3 ($STAMP)" \
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
    "$VENV_PY" -u "$GYM/examples/train_mlb_sac_pilot.py" \
        --n_envs "$N_ENVS" \
        --total_timesteps "$TOTAL" \
        --learning_starts "$LEARNING_STARTS" \
        --probe_every "$PROBE_EVERY" \
        --gradient_steps "$GRADIENT_STEPS" \
        --rb_save_every "$RB_SAVE_EVERY" \
        --seed "$SEED" \
        --base_port "$BASE_PORT" \
        --min_free_gb "$MIN_FREE_GB" \
        --mem_anon_ceiling_gb "$ANON_CEILING_GB" \
        --min_avail_gb "$MIN_AVAIL_GB" \
        --log_dir "$LOGDIR" \
        "${RESUME_ARG[@]}"

echo "$LOGDIR" > "$GYM/output/sac_pilot/LATEST_RUN"
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
echo " MONITOR IT WITH:   $GYM/watch_sac_pilot.sh"
echo " STOP IT SAFELY:    systemctl --user stop $UNIT.service    # SIGINT, checkpoints"
echo "=============================================================================="
