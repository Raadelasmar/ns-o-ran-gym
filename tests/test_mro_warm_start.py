"""Tests for the v3 warm-start recipe helpers in examples/train_mro_sac_pilot.py
(zero_init_actor_mean, critic_only_prefill). A stub Gym env and a real SAC
with a tiny buffer, same pattern as tests/test_critic_probe.py, so this runs
in seconds with no ns-3 dependency.

These two are the "bundled" fixes from the 2026-10-25 v3 plan: zero-init
makes the untrained policy reproduce the rule-based baseline exactly (see
RuleBasedMroControllerV3 and MroZmqEnv's action_space docstring), and
critic-only prefill lets the critic settle against the warm-start buffer
before any policy-gradient update uses it.
"""
import os
import sys

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from train_mro_sac_pilot import critic_only_prefill, zero_init_actor_mean   # noqa: E402

from stable_baselines3 import SAC                                            # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv                     # noqa: E402

OBS_DIM, ACT_DIM, ROLE_DELTA_MAX = 35, 2, 3.0


class StubEnv(gym.Env):
    """Deterministic, non-trivial rewards so the critic has something real
    to fit during prefill -- mirrors test_critic_probe.py's StubEnv."""

    def __init__(self, ep_len=10):
        self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-ROLE_DELTA_MAX, ROLE_DELTA_MAX, (ACT_DIM,),
                                           dtype=np.float32)
        self.ep_len, self.t = ep_len, 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.full(OBS_DIM, 0.5, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
        reward = float(np.sum(np.asarray(action)))  # non-constant, fittable
        trunc = self.t >= self.ep_len
        return obs, reward, False, trunc, {}


def _build_model(n_envs=2):
    venv = DummyVecEnv([lambda: StubEnv() for _ in range(n_envs)])
    model = SAC("MlpPolicy", venv, gamma=0.95, learning_rate=3e-4, batch_size=16,
               buffer_size=2000, learning_starts=0, train_freq=1, gradient_steps=1,
               device="cpu", seed=0, verbose=0)
    return model, venv


def _fill_buffer(model, venv, n_transitions=200):
    obs = venv.reset()
    model._last_obs = obs
    rng = np.random.default_rng(0)
    collected = 0
    while collected < n_transitions:
        actions = rng.uniform(-ROLE_DELTA_MAX, ROLE_DELTA_MAX, size=(venv.num_envs, ACT_DIM)
                              ).astype(np.float32)
        next_obs, rewards, dones, infos = venv.step(actions)
        model._store_transition(model.replay_buffer, actions, next_obs, rewards, dones, infos)
        obs = next_obs
        collected += venv.num_envs


def test_zero_init_actor_mean_makes_untrained_policy_output_zero():
    model, venv = _build_model()
    zero_init_actor_mean(model)
    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)
    # Box(-3, 3) is symmetric, so the tanh-squashed mean's midpoint (0 in
    # [-1, 1] space) rescales to exactly 0 in actual action units -- the
    # property v3's "action=0 reproduces the base-gain rule" design depends on.
    assert np.allclose(action, 0.0, atol=1e-5)
    mu = model.policy.actor.mu
    assert torch.all(mu.weight == 0.0)
    assert torch.all(mu.bias == 0.0)
    venv.close()


def test_zero_init_actor_mean_rejects_sde():
    """Guard against the untested use_sde=True path silently zeroing the
    wrong thing (see zero_init_actor_mean's own docstring)."""
    venv = DummyVecEnv([lambda: StubEnv() for _ in range(1)])
    model = SAC("MlpPolicy", venv, use_sde=True, learning_starts=0, device="cpu", seed=0)
    raised = False
    try:
        zero_init_actor_mean(model)
    except TypeError:
        raised = True
    assert raised
    venv.close()


def test_critic_only_prefill_moves_critic_not_actor():
    model, venv = _build_model()
    zero_init_actor_mean(model)
    _fill_buffer(model, venv, n_transitions=200)

    mu_before = model.policy.actor.mu.weight.clone()
    critic_before = next(model.policy.critic.parameters()).clone()

    critic_only_prefill(model, n_updates=5)

    mu_after = model.policy.actor.mu.weight
    critic_after = next(model.policy.critic.parameters())

    assert torch.equal(mu_before, mu_after), "actor.mu must be untouched by critic-only prefill"
    assert not torch.equal(critic_before, critic_after), "critic must actually have moved"
    venv.close()


def test_critic_only_prefill_restores_requires_grad():
    """The actor must be trainable again after prefill ends, so normal
    training (model.learn()) can update it -- a leaked frozen state would
    silently produce a policy that never learns."""
    model, venv = _build_model()
    zero_init_actor_mean(model)
    _fill_buffer(model, venv, n_transitions=200)
    assert all(p.requires_grad for p in model.policy.actor.parameters())
    critic_only_prefill(model, n_updates=3)
    assert all(p.requires_grad for p in model.policy.actor.parameters())
    venv.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
