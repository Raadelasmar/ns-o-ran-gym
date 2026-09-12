# Lab PC runbook — ByteTagList leak fix, verification, and 50k sizing

Written on the laptop 2026-09-12. Work through this on the lab PC (`gpu@`, Ubuntu
26.04, i7-8700 6c/12t, 32 GB, no sudo). Everything below runs by hand.

**Always use the venv python** — `~/oran-project/gym-venv/bin/python` (3.12).
System python is 3.14 and breaks ns-3's `./ns3` wrapper.

---

## What the fix is

`ByteTagList` keeps a monotonically non-decreasing `g_maxSize` high-water mark
(`byte-tag-list.cc:388`, in `Deallocate`), and `Allocate` sized every new block
as `max(size, g_maxSize)`.

In the mmWave DL path, RLC AM PDU assembly concatenates N PDCP SDUs via
`Packet::AddAtEnd`, which merges one byte-tag list per constituent SDU
(`packet.cc:357-361`). A PDU therefore carries tens of `PdcpTag`s and its tag
list reaches ~2 kB. From that moment `g_maxSize` is ~2 kB permanently, so *every*
later tag-list allocation — including a fresh packet needing 28 bytes for one
tag — allocated ~2 kB.

The surplus was never even usable: `Allocate` recorded `data->size = size` (the
*requested* size), so the free-list reuse test at `:363` rejected the oversized
block and `Deallocate` at `:392` then `delete[]`'d it.

**The fix is one line**: size the allocation to the request. It changes no tag
semantics. The free-list pop is still guarded by `if (data->size >= size)` at
`:363`, so it cannot return an undersized block.

Measured on the laptop, seed 868986719, single worker, random actions:

| | slope (steps >=10) | projected peak at 300 steps |
|---|---|---|
| before | 12.520 MB/step | 3837 MB |
| after | 0.386 MB/step | 197 MB |
| | **32.4x lower** | |

Anon-per-backlog ratio went from climbing (11.3 -> 22.6 in 20 steps) to flat
(oscillating ~1.0 across 120 steps, no trend).

### Provenance

`byte-tag-list.cc` is **stock upstream ns-3**, last touched here by the ns-3.38
merge `11908797`. This is not a fork bug. `ns-3-mmwave-oran/patches/0001-byte-tag-list-need-based-allocation.patch`
keeps the change as a standalone patch with the full evidence and safety
argument — **its job is to survive an upstream bump**, when the committed source
change may be clobbered by a merge. It is *not* how the server gets the fix;
the server gets it by `git pull` + rebuild.

---

## Step 1 — commit on the laptop (you do this, by hand)

### `ns-3-mmwave-oran`

```
src/network/model/byte-tag-list.cc                            # THE FIX (modified)
patches/0001-byte-tag-list-need-based-allocation.patch        # provenance
```

**`.gitignore:3` is `*.patch`**, so the patch file is ignored and a plain
`git add` will silently skip it. It needs:

```bash
git add -f patches/0001-byte-tag-list-need-based-allocation.patch
```

### `ns-o-ran-gym`

```
analysis/leak_hunt/RUNBOOK.md                  # this file
analysis/leak_hunt/slope.py                    # slope + ratio analyser
analysis/leak_hunt/verify_reward_intact.py     # reward/obs integrity check
analysis/leak_hunt/sizehist.c                  # live-allocation size histogram
analysis/leak_hunt/sizehist_bt.c               # + per-bucket backtrace attribution
analysis/leak_hunt/baseline_rss.csv            # laptop before, 36 steps
analysis/leak_hunt/fix1_rss_120steps.csv       # laptop after, 120 steps
analysis/mlb_session_2026-09-08/memprofile_stage_a.py   # +8 lines: --preload-env
```

Commit `memprofile_stage_a.py` explicitly (not the whole
`analysis/mlb_session_2026-09-08/` directory — `runs/` in there is ~13 MB).
Its only change is a generic `--preload-env KEY=VALUE` passthrough so a
different `.so` can be configured without editing the script again.

### Do NOT commit

| | why |
|---|---|
| `analysis/leak_hunt/*.so` | build artifacts; rebuild from the `.c` with the gcc line in the header |
| `analysis/leak_hunt/bt_bucket11.csv` | 14 MB of raw backtrace dumps, superseded by the findings |
| `analysis/leak_hunt/sizehist_run.csv` | 55 KB intermediate; keep only if you want the raw histogram |
| `**/*.bak*`, `*.pre_fix13`, `*.pre_field3` | `scratch/scenario-marl-zmq.cc.bak_prefield3`, `src/environments/mlb_zmq_env.py.pre_fix13`, `tests/test_mlb_zmq_env.py.pre_fix13` |
| `oom_evidence/`, `oom_evidence.tar.gz` | already captured in the findings |

---

## Step 2 — server baseline, BEFORE you pull

**Order matters.** Pulling brings the fix, so the clean single-worker baseline
has to be taken while the server is still on the old commit. The 87 MB/step
figure you have came from a contended 4-worker run and is not a matched
comparison.

```bash
cd ~/oran-project/ns-o-ran-gym
~/oran-project/gym-venv/bin/python \
  analysis/mlb_session_2026-09-08/memprofile_stage_a.py \
  --seed 868986719 --steps 40 --port 5601 \
  --out_dir /tmp/leak_base --print_every 10 --abort_at_gb 12.0
```

40 steps is enough: at ~87 MB/step that is ~3.5 GB of growth, unmistakable, and
at ~8 s/step it costs about 5 minutes. `--abort_at_gb 12.0` stops it before the
cgroup OOM killer does.

If `memprofile_stage_a.py` is missing on the server, pull first (Step 3), take
the baseline by checking out the previous commit of `byte-tag-list.cc`, rebuild,
measure, then restore. Simpler: just make sure Step 1 included it.

---

## Step 3 — pull, then VERIFY the fix is in the tree

```bash
cd ~/oran-project/ns-3-mmwave-oran && git pull
cd ~/oran-project/ns-o-ran-gym     && git pull
```

Do **not** `git apply` the patch file. The fix arrives as a committed source
change. Verify it landed before spending 90 s on a rebuild:

```bash
cd ~/oran-project/ns-3-mmwave-oran

# 1. the fixed allocation line. Expect EXACTLY 2 matches: 381 and 419.
#    Line 419 is the non-free-list #else variant, which always sized to the
#    request. Before the fix only 419 matched.
grep -n "new uint8_t\[size + sizeof(struct ByteTagListData) - 4\]" \
  src/network/model/byte-tag-list.cc

# 2. the old expression must be GONE. Expect 0.
grep -c "std::max(size, g_maxSize)" src/network/model/byte-tag-list.cc

# 3. the explaining comment should be there. Expect 1 (prose, inside the comment).
grep -c "max(size, g_maxSize)" src/network/model/byte-tag-list.cc

# 4. sanity: the commit is actually checked out
git log --oneline -1 -- src/network/model/byte-tag-list.cc
```

Expected: `2 matches at 381/419`, then `0`, then `1`.

If (2) returns `1`, the pull did not land the fix — stop and sort that out
before rebuilding.

---

## Step 4 — rebuild

```bash
cd ~/oran-project/ns-3-mmwave-oran
~/oran-project/gym-venv/bin/python ./ns3 build scenario-marl-zmq
```

Expect exactly:

```
[  0%] Building CXX object src/network/CMakeFiles/libnetwork-obj.dir/model/byte-tag-list.cc.o
[  0%] Linking CXX shared library .../libns3.38.rc1-network-optimized.so
```

~90 s. If more than `byte-tag-list.cc` compiles, something else came down in the
pull — read the list before continuing.

The scratch binary's mtime will **not** change. It links the `.so` dynamically,
so the fix arrives that way. That is expected, not a failed build.

---

## Step 5 — post-fix measurement

```bash
cd ~/oran-project/ns-o-ran-gym
~/oran-project/gym-venv/bin/python \
  analysis/mlb_session_2026-09-08/memprofile_stage_a.py \
  --seed 868986719 --steps 40 --port 5602 \
  --out_dir /tmp/leak_fix --print_every 10 --abort_at_gb 12.0

~/oran-project/gym-venv/bin/python analysis/leak_hunt/slope.py \
  /tmp/leak_base/rss_seed868986719.csv \
  /tmp/leak_fix/rss_seed868986719.csv
```

Read two things:

* **`SLOPE (>=10)`** on the patched run. Call it **S**, in MB/step. Everything
  downstream keys off it.
* **The anon-per-backlog ratio blocks.** Flat = fixed. Still climbing = an
  allocator high-water mark is still ratcheting, which means the `Buffer`
  sibling (fix 2, in reserve) is contributing materially.

---

## Step 6 — reward integrity

```bash
cd ~/oran-project/ns-o-ran-gym
~/oran-project/gym-venv/bin/python analysis/leak_hunt/verify_reward_intact.py \
  --seed 868986719 --steps 12 --port 5603
```

Must end in `PASS`, with all of these marked OK:

```
prb_utilization   num_active_ues   buffer_bytes   volume_bytes   sinr_db_mean
pdcp delivered_bytes non-zero      sinr_bins non-zero
```

and `terms=5` on every step line (all five reward terms present and finite).

Why this matters: the fix touches allocation sizing only, but the DL `PdcpTag`
added at `mc-enb-pdcp.cc:219-220` has exactly one reader
(`mc-ue-pdcp.cc:247`), feeding only the PDCP delay statistic. Satisfaction
consumes `GetDlRxData` (bytes), and no obs field derives from delay. This check
proves that empirically rather than by argument.

---

## Step 7 — 4-worker SAC smoke through the real launcher

Everything up to here is single-worker with random actions. The 17.7 GB death
was **4 workers under SAC through systemd**. Do not go to 50k without this.

```bash
cd ~/oran-project/ns-o-ran-gym

# 7a. DRYRUN first: runs every pre-flight check (ports, disk, linger,
#     probe_obs.npy) and launches nothing.
DRYRUN=1 N_ENVS=4 GRADIENT_STEPS=13 TOTAL=300 LEARNING_STARTS=100 \
  ./run_sac_pilot.sh

# 7b. the real smoke, ~12 minutes
N_ENVS=4 GRADIENT_STEPS=13 TOTAL=300 LEARNING_STARTS=100 \
  ./run_sac_pilot.sh
```

**`LEARNING_STARTS=100` is the point of the smoke, not a typo.** A 300-sample
run with the production `learning_starts=25000` would never learn and would test
nothing new. At 100, the first gradient-update burst fires ~25 vec-steps in —
the exact event that killed the 2026-09-01 run at `t=5001`, and the one thing
MemGuard cannot see, because SB3's off-policy `train()` has no callback hook.

`GRADIENT_STEPS=13` is not free to choose. `train_freq=1` makes
updates/sample `= gradient_steps / n_envs`, and arm C ran 16/5 = 3.20.
13/4 = 3.25. Changing one without the other changes the arm, which changes
the reward.

### What to watch, and where

```bash
watch -n 30 ~/oran-project/ns-o-ran-gym/watch_sac_pilot.sh
```

That one script surfaces everything:

| what | where it comes from |
|---|---|
| per-worker `RssAnon` (MB) and worker count | `/proc/<pid>/status` for each `scenario-marl-zmq` in the cgroup |
| cgroup `anon` — **the number that matters** | `<cgroup>/memory.stat` `anon` |
| cgroup `current`, `cache`, `swap`, `peak` | `<cgroup>/memory.current`, `memory.peak` |
| memory pressure | `<cgroup>/memory.pressure` `full` |
| system `MemAvailable` | `/proc/meminfo` |
| progress / reward | `$LOGDIR/critic_probe.csv` |

MemGuard's own view is in `$LOGDIR/pilot.log`; it reads cgroup `memory.stat`
`anon` (deliberately *not* `memory.current`, which counts reclaimable ns-3 trace
page cache) and tracks `peak_anon_gb`. A trip prints `!! MemGuard: ...` and
writes `memguard_model` + `memguard_replay` to the log dir.

The cgroup path, if you want it directly:

```bash
CG=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/sacpilot.service
awk '$1=="anon"{printf "%.2f GB\n", $2/1e9}' $CG/memory.stat
```

### Pass criteria, in numbers

300 samples / 4 workers = **75 steps per worker**, so each worker ends ~75 steps
into episode 0. Expected per-worker anon at the end is `85 + 75*S` MB.

| S (MB/step) | expected per-worker | expected cgroup anon |
|---|---|---|
| 3 | 310 MB | ~2.4 GB |
| 5 | 460 MB | ~3.0 GB |
| 10 | 835 MB | ~4.5 GB |

(cgroup anon = 4 x per-worker + ~1.2 GB parent: torch + SAC + the 200k replay
buffer at ~316 B/transition.)

| # | criterion | threshold |
|---|---|---|
| P1 | all workers alive throughout | `(4 workers)` in every watch snapshot |
| P2 | per-worker anon not above projection | max worker <= `1.5 * (85 + 75*S)` MB |
| P3 | workers are not diverging | max/min per-worker anon **< 2.0x** |
| P4 | gradient burst is survivable | cgroup anon jump within 2 min of `t=100` **< 3.0 GB** |
| P5 | headroom against the cap | cgroup `memory.peak` **< 0.6 x MEM_MAX** |
| P6 | no sustained pressure | `memory.pressure full avg10` **< 10** |
| P7 | MemGuard silent | no `!! MemGuard` in `pilot.log` |
| P8 | it finishes | reaches 300 timesteps, `run_status.json` clean |
| P9 | CPU is not the bottleneck | mean step time **< 1.25 x** the Step 5 single-worker step time |

P3 is the important one. The original failure had one worker at ~87 MB/step
while others sat at ~9.2 — **the spread was the tell, not the mean**.

### Decision rules

**Drop to `N_ENVS=3 GRADIENT_STEPS=10`** (ratio 3.33) if **P5**, **P6** or
**P9** fails. Those are contention failures — memory headroom or CPU — and fewer
workers fixes them directly.

**Take fix 2 off reserve** (the `Buffer` `g_maxSize` sibling in `buffer.cc`,
~19% of the remaining growth, see the findings) if **P2** or **P3** fails, or if
the Step 5 ratio was still climbing. Those say the *per-worker* growth is still
wrong, and fewer workers would only postpone it.

**Both fail** -> do fix 2 first, re-run Step 5 and Step 7, and only then decide
the worker count.

---

## Step 8 — size the 50k from the measured S

Per worker, per episode: `peak ~= 0.085 + 0.300*S` GB (85 MB base, 300 steps).

| S (MB/step) | per worker | n=2 | n=3 | n=4 | n=5 |
|---|---|---|---|---|---|
| 3 | 1.0 GB | 2.0 | 3.0 | 4.0 | 4.9 |
| 5 | 1.6 GB | 3.2 | 4.7 | 6.3 | 7.9 |
| 10 | 3.1 GB | 6.2 | 9.3 | 12.3 | 15.4 |
| 20 | 6.1 GB | 12.2 | 18.3 | 24.3 | 30.5 |

Add **parent ~= 1.2 GB** to get steady-state **P**. Then:

```
ANON_CEILING_GB = P + 1.0            # MemGuard, steady-state guard
MEM_HIGH        = ANON_CEILING + 3.0 # the gradient burst lives here
MEM_MAX         = MEM_HIGH + 1.5     # hard backstop
```

The +3.0 GB is deliberately wider than the launcher's current margin.
`run_sac_pilot.sh` records that `n_envs=3` died *at* `t=5001`, the instant
`learning_starts=5000` was crossed, because MemGuard is blind inside `train()`.
With `learning_starts=25000` that burst lands at `t=25000`, and **that phase has
never been reached at scale**. Treat 3 GB as a floor, not a target — and if the
Step 7 smoke shows the burst is bigger, raise it.

**Keep `MEM_MAX <= 24G` regardless.** The Step 13 post-mortem in
`run_sac_pilot.sh` shows `systemd-oomd` picks by reclaim activity, not size, and
killed `gnome-shell` instead of the trainer.

Worked example at **S=10** (the laptop's 32x holding on the server): per worker
3.1 GB, n=4 -> P = 13.5 GB, so `ANON_CEILING_GB=14.5`, `MEM_HIGH=17.5G`,
`MEM_MAX=19G`. That leaves 13 GB of the 32 for OS, desktop and page cache.

### Worker count is now a CPU decision, not a memory one

That is the real change. The i7-8700 has **6 physical cores**; each ns-3 worker
pins one at ~100%, and the parent needs one.

**Start at `N_ENVS=4`** — 4 workers + parent on 6 physical cores, no
hyperthread contention. Drop to 3 or 2 only if S comes back above ~20 MB/step,
or if Step 7 fails P5/P6/P9.

| n_envs | gradient_steps | updates/sample |
|---|---|---|
| 2 | 6 | 3.00 |
| 3 | 10 | 3.33 |
| **4** | **13** | **3.25** |
| 5 | 16 | 3.20 (arm C) |

### Expected wall clock

`50000/n` vec-steps x step time. At ~9.5 s/step:

| n_envs | vec-steps | wall clock |
|---|---|---|
| 2 | 25,000 | ~59 h |
| 3 | 16,667 | ~42 h |
| **4** | **12,500** | **~33 h** |
| 5 | 10,000 | ~29 h |

Gradient time after `t=25000` adds ~1-2% (13 updates x ~10 ms on a
single-threaded [256,256] MLP against a 9.5 s step). ~42 episode resets per
worker at ~8 s each is under 6 minutes. Both negligible — **step time dominates,
so confirm the server's real step time from Step 5 before trusting the 33 h.**

---

## Step 9 — launch 50k

```bash
cd ~/oran-project/ns-o-ran-gym

DRYRUN=1 N_ENVS=4 GRADIENT_STEPS=13 \
  MEM_HIGH=17.5G MEM_MAX=19G ANON_CEILING_GB=14.5 \
  TOTAL=50000 LEARNING_STARTS=25000 \
  ./run_sac_pilot.sh

N_ENVS=4 GRADIENT_STEPS=13 \
  MEM_HIGH=17.5G MEM_MAX=19G ANON_CEILING_GB=14.5 \
  TOTAL=50000 LEARNING_STARTS=25000 \
  ./run_sac_pilot.sh
```

Substitute your own numbers from Step 8. Stop it safely with
`systemctl --user stop sacpilot.service` (SIGINT, checkpoints first) — never
`kill -9`.

---

## Held in reserve, not applied

* **Fix 2** — the `Buffer` `g_maxSize` sibling in `src/network/model/buffer.cc`
  (`:106` ratchet, `:108` free-list retention rule). ~19% of the remaining
  growth, attributed to `Buffer::Create <- Buffer::AddAtStart <-
  Packet::AddHeader` at the 16 kB bucket. Riskier than fix 1: `Buffer::Create(0)`
  relies on free-list headroom via `g_recommendedStart` (`buffer.cc:249`), so
  changing it trades memory for extra reallocations per packet. Measure before
  and after, separately from fix 1.

* **Byte-tag compaction** — worth knowing this is mostly a dead end.
  `ByteTagList::AddAtEnd`/`AddAtStart` already drop out-of-range tags, but
  `Packet::AddAtEnd` calls them with `GetSize()`/`0`, which hit the early-return
  fast paths because no tag ever *is* out of range. The tags that accumulate are
  in-range and semantically distinct (one `PdcpTag` per constituent SDU), so
  dropping them would change what `FindFirstMatchingByteTag` returns at the UE.
  Not semantics-free, unlike fix 1.

* **Suppressing the DL `PdcpTag`** (`mc-enb-pdcp.cc:219-220`) — the reader audit
  is clean (one reader, feeds only the PDCP delay stat, no reward or obs field
  depends on it), but it would cost the `DRB.PdcpSduDelayDl` column in
  `cu-up-cell-N.txt`. Only if fix 1 + fix 2 are not enough.

## Parked, unrelated to the leak

The E2 offline mode writes `du-cell-N.txt`, `cu-up-cell-N.txt` and
`cu-cp-cell-N.txt` per cell at 10 Hz. These are **not** among the 8 `/dev/null`
trace redirects. That is page cache (`RssFile`), not `RssAnon`, so it cannot
cause the OOM — but it is real disk and page-cache pressure over a 33-hour run.
Worth redirecting before 50k if disk headroom is tight.

## Do not touch

The 8 `/dev/null` trace redirects (the sinks stay connected and feed
`prb_utilization`, `volume_bytes` and `sinr_bins`, which drive 2 of 5 obs fields
per cell plus the Balance and BadSignal reward terms), `sys.executable` in
`ns_env.py`, and `MEM_HIGH=infinity` in the memprofile harness.
