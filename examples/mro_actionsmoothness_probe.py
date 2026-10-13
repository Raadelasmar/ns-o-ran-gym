"""Measure ActionSmoothness's real scale from the actual untrained SAC policy
-- not a hand-picked Monte Carlo distribution.

docs/mro_training_fixes.md's round-2 weight measurement could not derive
ActionSmoothness's weight the way the other four were derived: it has no
signal under an all-zero baseline action by construction (a constant action
has zero step-to-step delta), so the shipped DEFAULT_W_SMOOTH (0.26) came from
a Monte Carlo simulation of a hand-approximated "untrained-PPO-like" action
distribution -- a guess at what exploration looks like, not a measurement of
it.

This script replaces the guess with a real one: a freshly-initialised (NOT
trained) SB3 SAC policy, built with the actual MroZmqEnv action/observation
spaces and the project's own locked SAC hyperparameters (ARM_C, from
train_mlb_sac_pilot.py), acting stochastically against a synthetic-but-
representative KPI stream. No ns-3 needed -- ActionSmoothness is a pure
property of the action sequence, and what matters is what THIS policy class,
at THIS initialisation, actually samples step over step, not how much ns-3
telemetry resembles reality. The reward/action code path under test is the
real one (MroZmqEnv._reward_terms / _margin_from_action), only the KPI source
is synthetic -- same stubbing discipline as tests/test_mro_zmq_env.py.

Usage:
    python examples/mro_actionsmoothness_probe.py
    python examples/mro_actionsmoothness_probe.py --episodes 40 --steps 300
"""
import argparse
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), "..", "src"))
from environments.mro_zmq_env import CELLS, MroZmqEnv

import numpy as np
from stable_baselines3 import SAC

# Same locked SAC hyperparameters as train_mlb_sac_pilot.py's ARM_C -- the
# point is to sample from the SAME policy class MRO will actually train with,
# not SB3's bare defaults. net_arch is left at SB3's own SAC default
# ([256, 256] both nets): no MRO-specific architecture has been locked
# anywhere yet, so there is nothing more specific to mirror.
ARM_C = dict(
    gamma=0.9948838031,
    learning_rate=3e-4,
    batch_size=256,
    ent_coef="auto_0.1",
)

CFG = {"ues": [5], "simTime": [30], "RngRun": [555]}


def _synthetic_snapshot(rng: np.random.Generator, step: int) -> dict:
    """One synthetic KPI snapshot -- plausible magnitudes, not ns-3 fidelity.

    Only the OBSERVATION needs to look like something real so the policy's
    forward pass sees representative inputs; the handover/SINR fields feed
    the reward terms but this script never reads those terms (only
    ActionSmoothness, a pure function of the action sequence, is used), so
    they only need to be well-formed, not physically accurate.
    """
    cells = {}
    for cell in CELLS:
        n_ues = int(rng.integers(0, 8))
        cells[str(cell)] = {
            "sinr_db_mean": float(rng.uniform(-10.0, 30.0)),
            "num_active_ues": n_ues,
            "sinr_bins": [int(rng.integers(0, 5))] + [0] * 5 + [int(rng.integers(0, 10))],
        }
    return {"timestamp": float(step), "cells": cells, "handovers": [],
            "handover_window_s": 1.0}


class _SyntheticMroZmqEnv(MroZmqEnv):
    """Real MroZmqEnv reward/action/obs code, synthetic KPI source.

    Mirrors tests/test_mro_zmq_env.py's _StubMroZmqEnv pattern, duplicated
    rather than imported (examples/ scripts have never depended on tests/,
    and this script needs a longer, randomised episode the test fixture
    was never built for).
    """

    def __init__(self, *a, steps_per_episode: int, seed: int, **kw):
        self._steps_per_episode = steps_per_episode
        self._rng = np.random.default_rng(seed)
        super().__init__(*a, **kw)

    def setup_sim(self):
        self.environment = {}
        self.script_executable = "/bin/true"

    def start_sim(self):
        self.is_open = True
        self.sim_path = "/tmp/synthetic-mro-zmq"
        self._step_count = 0
        self._handover_history = []
        self._last_ue_sinr = {}
        self._last_action = None

        class _Fake:
            def send_control_actions(self_inner, payload):
                pass

            def close(self_inner):
                pass
        self.zmq_db = _Fake()
        self._pending_kpis = None

    def _recv_next_kpis(self):
        if self._step_count >= self._steps_per_episode:
            return None
        kpis = _synthetic_snapshot(self._rng, self._step_count)
        self._step_count += 1
        return kpis

    def is_simulation_over(self):
        return self._step_count >= self._steps_per_episode

    def close(self):
        if self.is_open:
            self.zmq_db = None
            self.is_open = False


def detrended_sd(rows: list) -> tuple:
    """(raw_sd, detrended_sd), per-episode trend removed before pooling.

    Identical method to mro_baseline_experiment.py's own detrended_sd, kept
    as an independent copy for the same file-independence reason the rest of
    this project's analysis scripts already follow.
    """
    by_episode = {}
    for r in rows:
        by_episode.setdefault(r["episode"], []).append((r["step"], r["smoothness"]))

    raw_values, residuals = [], []
    for ep, pairs in by_episode.items():
        if len(pairs) < 2:
            continue
        steps = np.array([p[0] for p in pairs], dtype=float)
        values = np.array([p[1] for p in pairs], dtype=float)
        raw_values.extend(values.tolist())
        slope, intercept = np.polyfit(steps, values, 1)
        residuals.extend((values - (slope * steps + intercept)).tolist())

    raw_values = np.array(raw_values)
    residuals = np.array(residuals)
    return (float(raw_values.std()) if raw_values.size else float("nan"),
            float(residuals.std()) if residuals.size else float("nan"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300,
                    help="steps/episode; 300 matches the locked SAC config's "
                         "simTime=30 at controlInterval=0.1")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--anchor_sd", type=float, default=0.273,
                    help="HandoverSignalQuality's round-2 detrended sd, the "
                         "same anchor every other MRO weight was equalised "
                         "against -- see docs/mro_training_fixes.md")
    args = ap.parse_args()

    probe_env = _SyntheticMroZmqEnv(
        ns3_path="/unused", scenario_configuration=CFG, output_folder="/tmp",
        optimized=False, steps_per_episode=args.steps, seed=args.seed + 1)

    model = SAC("MlpPolicy", probe_env, seed=args.seed, verbose=0, **ARM_C)
    # NOT calling model.learn(): the policy must stay at its random
    # initialisation -- that IS the "untrained SAC policy" this script
    # measures. Any .learn() call here would defeat the entire point.

    rows = []
    for ep in range(args.episodes):
        obs, info = probe_env.reset(seed=args.seed + 1000 + ep)
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = probe_env.step(action)
            if "reward_terms" not in info:
                break
            rows.append({"episode": ep, "step": step,
                        "smoothness": info["reward_terms"]["smoothness"]})
            step += 1
            if terminated or truncated:
                break
    probe_env.close()

    values = [r["smoothness"] for r in rows]
    raw_sd, det_sd = detrended_sd(rows)
    mean = float(np.mean(values))
    print(f"ActionSmoothness, {args.episodes} episodes x up to {args.steps} steps, "
          f"{len(rows)} total steps")
    print(f"  mean           = {mean:.4f}")
    print(f"  raw sd         = {raw_sd:.4f}")
    print(f"  detrended sd   = {det_sd:.4f}")
    print(f"  (prior Monte Carlo estimate was sd ~= 1.06, mean ~= 1.99 -- "
          f"see mro_zmq_env.py's DEFAULT_W_SMOOTH comment)")

    w_smooth = args.anchor_sd / det_sd if det_sd and det_sd > 0 else float("nan")
    print(f"\nw_smooth = anchor_sd({args.anchor_sd}) / detrended_sd({det_sd:.4f}) "
          f"= {w_smooth:.4f}")
    print("(current DEFAULT_W_SMOOTH in mro_zmq_env.py is 0.26, from the old "
          "Monte Carlo estimate -- compare before replacing it)")
