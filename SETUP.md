# Setting up and running the MLB training pipeline

This gets you from a clean Ubuntu 24.04 machine to a trained (or at least
training) MLB agent. It assumes no prior knowledge of the project.

What you are building: a reinforcement learning agent that sets a per-cell CIO
(Cell Individual Offset, in dB) on a 7-cell mmWave O-RAN network simulated in
ns-3. Every control step the simulator sends KPIs to Python over ZeroMQ, the
agent replies with seven CIO offsets, and ns-3 applies them and runs on. The
agent is PPO from Stable-Baselines3.

Three pieces have to line up:

- `ns-3-mmwave-oran`, the simulator, which contains the scenario that speaks the
  ZeroMQ protocol
- `ns-o-ran-gym`, the Python side, which holds the Gymnasium environment, the
  reward function and the training scripts
- `oran-e2sim`, a library the simulator links against, installed system-wide as
  a .deb

Budget about an hour for setup, most of it the ns-3 build.

This file lives in the `ns-o-ran-gym` repository, so if you are reading it from
a local checkout you already have that half of step 2 done. Step 2 still lists
both clones, so the sequence works whether you got here by cloning the gym
repository or by reading this on the web.

The commit hashes in step 2 pin the current pushed state of both repositories,
so a setup from this guide is reproducible. If a step below refers to a file your
clone does not have, confirm you checked out the commits in step 2 rather than
the default branch.

---

## 1. Install system packages

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake python3 python3-venv python3-pip \
                        libsctp-dev libzmq3-dev cppzmq-dev nlohmann-json3-dev
```

The last three are easy to miss and each produces a confusing failure much later
in the build:

- `libzmq3-dev` and `cppzmq-dev` provide the ZeroMQ library and its C++ header.
  The bridge between ns-3 and Python is built on them.
- `nlohmann-json3-dev` provides `nlohmann/json.hpp`. ns-3 does not bundle it, and
  nothing in the build checks for it, so without this package you get a missing
  header error with no explanation of what to install.

---

## 2. Create the project directory and clone the two repositories

Everything lives under one directory. The paths below are the defaults the
scripts expect, so use them unless you have a reason not to.

```bash
mkdir -p ~/oran-project && cd ~/oran-project

git clone https://github.com/Raadelasmar/ns-o-ran-ns3-mmwave.git ns-3-mmwave-oran
git -C ns-3-mmwave-oran checkout 21e1b37086d72d289041b0968576d3dd18f292d3

git clone https://github.com/Raadelasmar/ns-o-ran-gym.git ns-o-ran-gym
git -C ns-o-ran-gym checkout 96967fcedf3bc7ae97629f739584bff20340e68f
```

---

## 3. Add the E2 interface module

The simulator needs a module called `oran-interface`, which lives in its own
upstream repository and belongs at `ns-3-mmwave-oran/contrib/oran-interface`.

There is a catch. The ns-3 fork you just cloned already tracks one file inside
that directory (`model/zmq-database-client.h`, the ZeroMQ bridge header), so the
directory already exists and is not empty. A plain `git clone` into it fails with
`destination path already exists and is not an empty directory`.

Clone it somewhere else and move the contents in:

```bash
cd ~/oran-project
git clone https://github.com/o-ran-sc/sim-ns3-o-ran-e2 /tmp/oran-interface-src
git -C /tmp/oran-interface-src checkout 5f547ae8b5ca2d051b9f4d19391d17289ed433d9

cp -r --update=none /tmp/oran-interface-src/. ns-3-mmwave-oran/contrib/oran-interface/
rm -rf /tmp/oran-interface-src
```

`--update=none` means "do not overwrite existing files", which preserves the
header the ns-3 fork already provides. (`cp -rn` does the same thing but warns
that the flag is non-portable on newer coreutils.)

Two things you will notice afterwards. The copy brings the upstream `.git`
directory along, so `contrib/oran-interface` is a git repository nested inside
the ns-3 one. That is expected and matches the reference machine. It does mean
`git status` inside ns-3 can report that directory oddly, and that edits there
belong to the nested repository unless you force-add them into ns-3, which step 4
covers.

---

## 4. Apply the build fix to the oran-interface CMakeLists

The upstream `oran-interface` build file knows nothing about ZeroMQ, so out of
the box the build fails twice. Both failures are real and both are fixed by the
same small edit, which is tracked in the ns-3 fork but excluded from `contrib/`
by a gitignore rule, so it needs to be copied in by hand.

The two failures, so you know what you are avoiding:

1. Without `model/zmq-database-client.h` in the module's `HEADER_FILES` list,
   ns-3 never copies it into `build/include/ns3/`, and compiling the scenario
   dies with `fatal error: ns3/zmq-database-client.h: No such file or directory`.
2. Without `${zmq_LIBRARIES}` in `LIBRARIES_TO_LINK`, the link step fails with
   `undefined reference to zmq_ctx_new` and friends. The cppzmq header is a
   header-only wrapper over libzmq's C API, so the library still has to be
   linked.

Open `ns-3-mmwave-oran/contrib/oran-interface/CMakeLists.txt` and make two
changes.

First, immediately after the `message(STATUS "libraries found: ...")` line near
the top, add:

```cmake
# ZeroMQ (libzmq + cppzmq header) is required by model/zmq-database-client.h
find_external_library(DEPENDENCY_NAME zmq
                      HEADER_NAME zmq.hpp
                      LIBRARY_NAME zmq
                      SEARCH_PATHS /usr/include)

if(NOT ${zmq_FOUND})
    message(WARNING "libzmq/cppzmq is required by oran-interface and was not found")
    return ()
endif()

include_directories(${zmq_INCLUDE_DIRS})
```

Second, inside the `build_lib(...)` block at the bottom, add one line to the end
of `HEADER_FILES` and one to the end of `LIBRARIES_TO_LINK`:

```cmake
    HEADER_FILES ...
                 helper/mmwave-indication-message-helper.h
                 model/zmq-database-client.h
    LIBRARIES_TO_LINK
                    ${libcore}
                    ${e2sim_LIBRARIES}
                    ${zmq_LIBRARIES}
```

If you are working from a checkout that already has this change, note that
`contrib/` is gitignored, so committing it needs a force add:

```bash
git -C ~/oran-project/ns-3-mmwave-oran add --force contrib/oran-interface/CMakeLists.txt
```

---

## 5. Build and install e2sim

This has to happen before any ns-3 build. `src/lte` and `src/mmwave`
unconditionally include `<ns3/oran-interface.h>`, which includes `e2sim.hpp`, so
without e2sim installed the simulator will not compile at all.

```bash
cd ~/oran-project
git clone https://github.com/wineslab/ns-o-ran-e2-sim oran-e2sim
git -C oran-e2sim checkout 2209995a84412b5178f2a49cb92125fd23cb7553

cd oran-e2sim/e2sim
mkdir -p build
./build_e2sim.sh 3
```

The `3` is the log level. The script builds a .deb and installs it with `dpkg`,
so it will ask for your sudo password. Create the `build` directory first as
shown: the script tries to create it itself but does so in a subshell, so it does
not actually work.

Check it landed:

```bash
dpkg -l | grep e2sim        # expect: ii  e2sim-dev  1.0.0  amd64
```

---

## 6. Build ns-3

```bash
cd ~/oran-project/ns-3-mmwave-oran
./ns3 configure -d optimized
./ns3 build
./ns3 show profile          # expect: Build profile: optimized
```

Two things worth knowing:

- `-d optimized` sets the build profile but leaves the output in the default
  `build/` directory. It does not create `build/optimized/`. This is why every
  driver script in `ns-o-ran-gym/examples/` passes `optimized=False`: that flag
  only selects which path goes on `LD_LIBRARY_PATH`, and `optimized=False` with a
  `-d optimized` build is the correct combination.
- Skipping the `-d optimized` step gives you a debug build, which is roughly
  2.8 times slower. Over a multi-day training run that matters.

The first build takes a while. Later builds are incremental.

---

## 7. Set up the Python environment

```bash
cd ~/oran-project
python3 -m venv gym-venv
source gym-venv/bin/activate
pip install --upgrade pip
```

Install PyTorch first, explicitly from the CPU index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use the CPU index deliberately. The default index pulls about 3.4 GB of CUDA
libraries that this project never uses. The policy network is a 64x64 MLP with
around 13,000 parameters, and the bottleneck is the simulator, not the network,
so a GPU buys nothing here. For scale: a venv built this way is about 1.3 GB.

If torch prints `UserWarning: Failed to initialize NumPy` at this point, ignore
it. NumPy arrives with the next command and the warning goes away.

Then the rest:

```bash
pip install -r ~/oran-project/ns-o-ran-gym/requirements.txt
pip install -e ~/oran-project/ns-o-ran-gym
```

The last line installs the gym repository in editable mode, which is what makes
`import nsoran` and `import ns_o_ran_gym.bridge...` resolve to your working copy.

You must activate this venv in every shell you use for this project:

```bash
source ~/oran-project/gym-venv/bin/activate
```

---

## 8. Run the test suite

Do this before touching the simulator. Both test files run entirely on stubs,
with no ns-3 and no ZeroMQ, so they finish in seconds and catch most setup
mistakes immediately.

```bash
cd ~/oran-project/ns-o-ran-gym
python3 tests/test_mlb_zmq_env.py
python3 tests/test_step_logger.py
```

Expect 22 tests passing in the first and 11 in the second. The second prints
`11/11 passed` at the end. If either fails, stop and fix it before going further:
these cover the reward terms, the observation layout and the CSV schema, so a
failure here means every number produced later is suspect.

---

## 9. Run the smoke test

This is the first thing that actually launches ns-3. It runs PPO for 16
timesteps, which is enough to force one policy update and prove the whole loop
turns: action, step, reward, update.

```bash
cd ~/oran-project/ns-o-ran-gym
python3 examples/train_mlb_smoke.py
```

It prints `SMOKE TEST PASSED` on success and exits non-zero on failure.

**It takes about 23 minutes and looks frozen for most of that. Do not kill it.**
Measured end to end on a 12 logical / 6 physical core machine: 1353 seconds, with
the first progress table appearing only after 669 seconds. If you run it under a
timeout, allow at least 40 minutes.

The silence is expected. Every PPO timestep is one simulator control step over
the ZeroMQ bridge, so 16 timesteps is 16 full control steps, and with `simTime=10`
an episode is about 10 steps, meaning the run spans two complete ns-3 launches.
Stable-Baselines3 collects `n_steps=8` before it prints anything at all, so
nothing appears on screen between `PPO constructed` and the first table. It also
reports `fps 0`, which is integer truncation of a real rate well below 1, not a
hang.

For reference, a successful run ends with a table like this and then the pass
line:

```
| rollout/           |      |
|    ep_len_mean     | 10   |
|    ep_rew_mean     | 6.79 |
|    total_timesteps | 16   |
| train/             |      |
|    n_updates       | 10   |
SMOKE TEST PASSED
```

The `n_updates` figure is the thing to look for: it confirms PPO actually
performed policy updates rather than just collecting data.

It is not trying to learn anything. The resulting policy is noise. The only
question this answers is whether the pipeline runs end to end.

---

## 10. Run real training

```bash
cd ~/oran-project/ns-o-ran-gym
python3 examples/train_mlb_parallel.py --n_envs 5 --total_timesteps 1000
```

This starts five simulators side by side, each on its own ZeroMQ port
(5555 to 5559 by default), and trains PPO across all of them. Checkpoints land in
`output/ppo_runs/`, and the per-step reward and action log goes to a timestamped
directory under it.

Be realistic about the cost. Measured on a 12 logical / 6 physical core machine:

- about 285 seconds per sample at 5 workers
- 5 workers is the practical ceiling, not 6. ns-3 is single-threaded at roughly
  0.5 GB resident per instance, and there are only 6 physical cores.
- a 1000-timestep pilot is roughly 14 hours

Do not expect a good policy from 1000 timesteps. It is a pilot, and its purpose
is to show you the machinery works and to let you look for reward hacking. Real
progress needs far more samples: measured on a calibrated synthetic environment,
PPO reaches only about 19 to 25 percent of the achievable gain at 50,000 samples,
which is over 99 days on the real simulator. SAC with `gradient_steps=16` is
about three times more sample efficient and gets to roughly 42 percent at 50,000,
or 33 to 40 days. Read `docs/mlb_training_fixes.md` before committing weeks of
compute.

Useful flags:

- `--n_envs N` number of parallel simulators
- `--total_timesteps N` how long to train
- `--ues 3 --simtime 10` a much faster configuration for trying things out
- `--base_port P` move off 5555 if those ports are busy
- `--log_dir PATH` where the per-step CSV goes

The script refuses to start if any of the ports it wants are already in use,
which is deliberate: a stray run silently stealing a socket is very hard to
diagnose.

---

## 11. Check the results

Point the analysis script at the log directory the training run created:

```bash
cd ~/oran-project/ns-o-ran-gym
python3 analysis/analyse_run.py <log_dir>
```

The script needs only numpy and the standard library, so it runs anywhere the
venv is active.

`<log_dir>` is whatever `--log_dir` was, or the timestamped directory under
`output/ppo_runs/`. It holds `reward_terms_steps.csv`,
`reward_terms_steps_meta.json` and `monitor_rank*.monitor.csv`.

What a healthy run looks like:

- section 0, integrity: the `(env_rank, vec_step)` grid is complete, no term is
  missing on step rows, and StepLogger agrees with Monitor to within about 1e-6.
  A mismatch here means the two independent records disagree, so stop and find
  out why before trusting anything else.
- section 1, reward hacking: read the first line, `backlog TERM rises`, not the
  raw queue bytes line. Around 25 percent is the chance rate. Elevated on the
  term while the other lines stay flat is the signature worth investigating.
- section 3, learning: read the clustered number, not the naive per-episode one.
  The script labels them. Cross-read it against `std` and `clip_fraction` in the
  SB3 output: if `std` is still around 1.0 and `clip_fraction` is 0, the policy
  never moved and any apparent trend is seed noise.

Red flags:

- reward climbing while delivered bytes fall or the queue grows. This project has
  twice been bitten by a reward term improving while the network got worse.
  Reward is the training signal, KPIs are the verdict. Never judge a run on
  reward alone.
- `satisfaction` or `pingpong` blank on many rows. Blank means the term was
  dropped because the simulator did not supply the data, which is different from
  a measured zero, and it means part of the reward was silently absent.
- a large `reset_retry_count`. Roughly one ns-3 seed in ten kills the simulator
  at startup and the environment redraws automatically, so a few are normal, but
  a lot suggests something else is wrong.

---

## Gotchas that will cost you time

**The editable install overrides `sys.path`.** The project is installed with
`pip install -e`, which sets up a finder hook that hardcodes the real `src` path.
Any `sys.path.insert` you write to point at a modified copy of the environment is
ignored, and Python silently imports the original and reports a passing test. If
you want to test a modified copy, edit the real file or uninstall the editable
package first. This one is genuinely hard to spot because nothing fails: you just
get a green result for code you never ran.

**Three simulator flags must stay at their defaults.** Each looks like a free
speed or disk saving and each silently destroys the KPIs instead of raising an
error:

- `enableTraces` (default 1) is the KPI pipeline, not a logging option. It gates
  the only callbacks that ever write the volume and PRB counters. With it off, a
  rollout returns reward exactly 1.000 on every step because every counter reads
  zero.
- `enableE2FileLogging` (default 1) does not control logging. It selects offline
  files instead of connecting to a live RIC. Set to 0 the simulator needs a real
  E2 termination and dies mid-episode without one.
- CU-CP reporting (`e2cuCp`, default on) fills the map that per-UE SINR comes
  from. Disable it and all seven SINR observations pin to -40 dB.

The reasoning is written into the CFG block in `examples/train_mlb_smoke.py`.

**Run directories fill the disk.** ns-3 writes several MB per step, which is
hundreds of GB over a long run, and each 35-UE run directory is about 250 MB. The
training driver passes `purge_sim_path_on_close=True` so each finished episode's
directory is deleted. Nothing in the training loop reads those files back. If you
turn purging off for debugging, watch your free space.

**RLC AM, not UM.** `rlcAmEnabled` is 1 in the training config and should stay
there. The reward was validated in AM. Under UM the same setup produces backlog
and satisfaction figures that differ by 48 and 41 percent, and UM has no ARQ, so
residual failures are permanent loss.

---

## Where to read next

- `docs/mlb_training_fixes.md` in the gym repository: what has been fixed and
  why, the pilot post-mortem, and the PPO versus SAC comparison with measured
  sample budgets. Read this before planning a long run.
- `src/environments/mlb_zmq_env.py`: the reward function, with the reasoning for
  each term written next to it.
