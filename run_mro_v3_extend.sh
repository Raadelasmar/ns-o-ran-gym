#!/bin/bash
set -e
cd /home/vajdad/son_marl_project/ns-o-ran-gym
PY=.venv/bin/python3
NS3=$HOME/son_marl_project/ns-o-ran-ns3-mmwave
LOGDIR="output/mro_v3_pilot_v2_extended/logs_$(date +%Y%m%d_%H%M%S)"
OUTLOG="output/mro_v3_extend_launch.log"

# n_envs=10 MUST match the original pilot run: the saved replay buffer's
# internal arrays are shaped (buffer_size/n_envs, n_envs, ...), and SAC.load()
# + load_replay_buffer() refuses to train on a mismatch (confirmed: n_envs=3
# aborted with "replay_buffer.n_envs: intent 3, actual 10"). This does mean
# 10 more ns-3 workers on top of the 5 already running for
# mro_v3_final_eval.py's comparison -- accepted tradeoff, see conversation:
# user chose to run both in parallel rather than wait.
setsid nohup "$PY" examples/train_mro_sac_pilot.py \
  --n_envs 10 --total_timesteps 60000 --learning_starts 5000 \
  --probe_every 250 --rb_save_every 500 --seed 555 --base_port 7700 \
  --no_zero_init_actor \
  --ns3_path "$NS3" \
  --resume_from output/mro_v3_pilot_v2_extended/mro_sac_final \
  --save_dir output/mro_v3_pilot_v2_extended \
  --log_dir "$LOGDIR" \
  > "$OUTLOG" 2>&1 < /dev/null &
BGPID=$!
disown
echo "launched pid $BGPID -- log dir: $LOGDIR"
sleep 20
echo "--- process alive check ---"
if kill -0 "$BGPID" 2>/dev/null; then
  echo "pid $BGPID is alive"
else
  echo "pid $BGPID is DEAD"
fi
echo "--- launch log so far ---"
cat "$OUTLOG"
