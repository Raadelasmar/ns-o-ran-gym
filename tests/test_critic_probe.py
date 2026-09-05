"""Tests for src/callbacks/critic_probe.py. A stub Gym env and a real SAC with a
tiny buffer, so the whole file runs in seconds.

test_critic_is_fed_scaled_actions_not_db is the one that matters: every other
check here would still pass if the probe fed the critic dB actions.
"""
import os
import sys
import csv
import tempfile

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from callbacks.critic_probe import CriticProbe, RECENT_WINDOW, FIELDS   # noqa: E402

from stable_baselines3 import SAC                                        # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv                 # noqa: E402

OBS_DIM, ACT_DIM, CIO_LIMIT = 35, 7, 6.0
_FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f": {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


class StubEnv(gym.Env):
    """Deterministic rewards so r_bar is exactly predictable. Omits "reward_terms"
    on the terminal step, mirroring MlbZmqEnv's time-limit path."""
    def __init__(self, ep_len=10, reward=0.4):
        self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-CIO_LIMIT, CIO_LIMIT, (ACT_DIM,), dtype=np.float32)
        self.ep_len, self.reward, self.t = ep_len, reward, 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.full(OBS_DIM, 0.5, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        trunc = self.t >= self.ep_len
        obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
        if trunc:
            # Terminal: reward 0.0 and no reward_terms, as MlbZmqEnv does.
            return obs, 0.0, False, True, {"cio_offsets": {}}
        return obs, self.reward, False, False, {"reward_terms": {"balance": 1.0}}


def build(tmp, n_envs=2, total=60, learning_starts=20, every=20, reward=0.4):
    obs_path = os.path.join(tmp, "probe_obs.npy")
    rng = np.random.default_rng(0)
    np.save(obs_path, rng.random((64, OBS_DIM)).astype(np.float32))
    csv_path = os.path.join(tmp, "probe.csv")
    venv = DummyVecEnv([lambda: StubEnv(reward=reward) for _ in range(n_envs)])
    model = SAC("MlpPolicy", venv, gamma=0.9948838031, learning_rate=3e-4,
                batch_size=32, train_freq=1, gradient_steps=16,
                learning_starts=learning_starts, buffer_size=1000,
                ent_coef="auto_0.1", seed=555, verbose=0, device="cpu")
    probe = CriticProbe(csv_path, obs_path, every=every, verbose=0)
    model.learn(total_timesteps=total, callback=probe)
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    return model, probe, rows, obs_path


def main():
    with tempfile.TemporaryDirectory() as tmp:
        model, probe, rows, obs_path = build(tmp)
        X = np.load(obs_path)

        check("test_header_matches_FIELDS",
              list(rows[0].keys()) == FIELDS, f"{list(rows[0].keys())}")
        check("test_init_row_written_before_any_update",
              rows[0]["phase"] == "init" and int(rows[0]["n_updates"]) == 0,
              f"phase={rows[0]['phase']} n_updates={rows[0]['n_updates']}")
        check("test_final_row_written", rows[-1]["phase"] == "final")
        check("test_updates_accumulate",
              int(rows[-1]["n_updates"]) > int(rows[0]["n_updates"]),
              f"{rows[0]['n_updates']} -> {rows[-1]['n_updates']}")

        # The units trap: recompute Q both ways and check which one the probe took.
        with torch.no_grad():
            obs = torch.as_tensor(X)
            a_scaled = model.policy.actor(obs, deterministic=True)
            a_db = torch.as_tensor(model.policy.unscale_action(a_scaled.numpy()))
            q_scaled = torch.min(torch.stack(model.critic(obs, a_scaled)), dim=0).values
            q_db = torch.min(torch.stack(model.critic(obs, a_db)), dim=0).values
        got = float(rows[-1]["q_mean"])
        check("test_critic_is_fed_scaled_actions_not_db",
              abs(got - float(q_scaled.mean())) < 1e-3,
              f"probe {got} vs scaled {float(q_scaled.mean()):.4f} / dB {float(q_db.mean()):.4f}")
        # Mutation proof: the two paths must differ, or the check above is vacuous.
        check("test_the_two_action_scales_give_different_Q",
              abs(float(q_scaled.mean()) - float(q_db.mean())) > 1e-3,
              f"scaled {float(q_scaled.mean()):.4f} == dB {float(q_db.mean()):.4f}")

        check("test_action_reported_in_db_not_normalised",
              abs(float(rows[-1]["action_mean_abs_db"])
                  - float(np.abs(a_db.numpy()).mean())) < 1e-3,
              f"{rows[-1]['action_mean_abs_db']} vs {float(np.abs(a_db.numpy()).mean()):.5f}")
        check("test_action_within_cio_limit",
              float(rows[-1]["action_mean_abs_db"]) <= CIO_LIMIT)
        check("test_drift_zero_on_init_row",
              float(rows[0]["action_drift_db"]) == 0.0)

        # Non-terminal steps pay exactly 0.4; leaked terminals would drag r_bar below it.
        check("test_terminal_rewards_excluded_from_r_bar",
              abs(float(rows[-1]["r_bar_all"]) - 0.4) < 1e-9,
              f"r_bar_all={rows[-1]['r_bar_all']} (0.4 expected; "
              f"leaking terminals would give < 0.4)")
        check("test_value_target_is_r_over_one_minus_gamma",
              abs(float(rows[-1]["v_target_all"])
                  - 0.4 / (1 - 0.9948838031)) < 0.01,
              f"{rows[-1]['v_target_all']}")
        check("test_q_pct_is_q_over_target",
              abs(float(rows[-1]["q_pct_of_target_all"])
                  - 100 * float(rows[-1]["q_mean"]) / float(rows[-1]["v_target_all"])) < 0.02)

    # A different reward level must move r_bar_recent, proving it reads the live stream.
    with tempfile.TemporaryDirectory() as tmp:
        _, _, rows_lo, _ = build(tmp, reward=0.1)
    check("test_r_bar_recent_follows_the_reward_stream",
          abs(float(rows_lo[-1]["r_bar_recent"]) - 0.1) < 1e-9,
          f"{rows_lo[-1]['r_bar_recent']}")

    check("test_recent_window_is_a_window", RECENT_WINDOW == 1000)
    with tempfile.TemporaryDirectory() as tmp:
        _, probe, _, _ = build(tmp, total=60)
        # float32 round-trip through the VecEnv makes 0.4 land as 0.4000000059604645.
        check("test_all_nonterminal_rewards_retained",
              len(probe._rewards) > 0
              and all(abs(r - 0.4) < 1e-6 for r in probe._rewards),
              f"n={len(probe._rewards)} set={set(probe._rewards)}")

    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.npy")
        np.save(bad, np.zeros((8, OBS_DIM - 1), dtype=np.float32))
        venv = DummyVecEnv([lambda: StubEnv()])
        m = SAC("MlpPolicy", venv, buffer_size=100, learning_starts=10,
                seed=1, verbose=0, device="cpu")
        raised = False
        try:
            m.learn(total_timesteps=12,
                    callback=CriticProbe(os.path.join(tmp, "p.csv"), bad, every=5))
        except ValueError:
            raised = True
        check("test_wrong_obs_width_is_rejected", raised)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {_FAILURES}")
        sys.exit(1)
    print("all critic-probe tests passed")


if __name__ == "__main__":
    main()
