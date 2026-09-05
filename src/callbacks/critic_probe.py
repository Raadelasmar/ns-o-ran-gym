"""Per-checkpoint measurement of whether SAC's critic and actor are moving.

Records the SAC equivalents of the PPO pilot's numbers (Step 7/8, re-confirmed in
Step 11: 130 updates over 16.5 h, critic 6.212 at sd 0.416, actor 0.055 to
0.129 dB against a +/-6 dB range) on a fixed batch of real ns-3 observations
(analysis/mlb_session_2026-08-31/probe_obs.npy). Fixed, so a change in mean Q is a
change in the critic and not in what it was asked about, and the same batch the
PPO number was measured on so the two are comparable.

UNITS TRAP: the critic's action domain is [-1, 1], not dB. SAC.train() feeds it
scaled buffer_actions and tanh actor output, so handing it the +/-6 dB action that
model.predict() returns scores it 6x outside its training domain and yields a
confident, meaningless number. Score with actor(obs) directly and unscale only for
reporting. PPO is the mirror case (it emits dB and needs no unscale), so the two
conversions are kept apart deliberately.

The value target r_bar/(1-gamma) is written twice: v_target_all, and
v_target_recent over the last RECENT_WINDOW samples, which is the one to judge the
critic against because the learning_starts warm-up is uniform random and scores
near this reward's worst. Terminal steps are excluded from both: MlbZmqEnv's
time-limit path returns 0.0 with no "reward_terms" key, a shape rather than a
sample of the reward distribution.
"""
import csv
import json
import os
import re
import shutil
import time
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

FIELDS = [
    "wall_iso", "elapsed_s", "num_timesteps", "n_updates", "samples_per_hour",
    # critic
    "q_mean", "q_sd", "q_min", "q_max",
    "r_bar_all", "r_bar_recent", "v_target_all", "v_target_recent",
    "q_pct_of_target_all", "q_pct_of_target_recent",
    # actor
    "action_mean_abs_db", "action_sd_db", "action_drift_db",
    "log_std_mean", "log_std_sd", "ent_coef",
    # SB3's own training scalars, blank between logger dumps
    "critic_loss", "actor_loss", "ent_coef_loss",
]

# Resource telemetry sampled at probe time. cg_mem_peak_gb is the kernel's own
# high-water mark, which unlike MemGuard's per-vec-step sampling does capture the
# gradient-update burst inside SB3's train() (2026-09-01 crash). Blank if a read
# fails: telemetry must never stop a run.
_RESOURCE_FIELDS = ["cg_mem_peak_gb", "cg_mem_current_gb", "cg_swap_gb",
                    "free_disk_gb", "episode_dirs"]

FIELDS = FIELDS + _RESOURCE_FIELDS + ["phase"]

# ns-3 episode dirs are uuid4-named, created directly inside output_folder.
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-")


def _cgroup_dir():
    """This process's cgroup v2 directory, or None. Never raises."""
    try:
        with open("/proc/self/cgroup", encoding="utf8") as fh:
            for line in fh:
                if line.startswith("0::"):
                    d = os.path.join("/sys/fs/cgroup", line.strip()[3:].lstrip("/"))
                    return d if os.path.isdir(d) else None
    except Exception:                                  # noqa: BLE001 - telemetry only
        pass
    return None


def _gb(file_path):
    """A single-integer cgroup file in GB, or blank. Never raises."""
    try:
        with open(file_path, encoding="utf8") as fh:
            return round(int(fh.read().strip()) / 1e9, 3)
    except Exception:                                  # noqa: BLE001 - telemetry only
        return ""


def _free_disk_gb(folder):
    """Free space at folder in GB, or blank. Never raises."""
    try:
        return round(shutil.disk_usage(folder).free / 1e9, 2)
    except Exception:                                  # noqa: BLE001 - telemetry only
        return ""


def _episode_dirs(folder):
    """Count of live ns-3 episode dirs, or blank. Never raises."""
    try:
        n = 0
        with os.scandir(folder) as it:
            for e in it:
                if e.is_dir() and _UUID_RE.match(e.name):
                    n += 1
        return n
    except Exception:                                  # noqa: BLE001 - telemetry only
        return ""

# ~3.3 episodes per worker at 5 workers: long enough to average out the
# within-episode backlog ramp (Step 7), short enough to track the policy.
RECENT_WINDOW = 1000


class CriticProbe(BaseCallback):
    """Writes one CSV row every `every` samples with the critic/actor state.

    `every` counts total timesteps aggregated across workers, so its meaning does
    not shift when n_envs does. csv_path is flushed per row, so a killed run keeps
    its tail.
    """

    def __init__(self, csv_path: str, probe_obs_path: str, every: int = 250,
                 meta_path: Optional[str] = None, output_folder: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.probe_obs_path = probe_obs_path
        self.output_folder = output_folder      # None: disk columns stay blank
        try:
            self._cg = _cgroup_dir()
        except Exception:                              # noqa: BLE001 - telemetry only
            self._cg = None
        self.meta_path = meta_path or (csv_path.rsplit(".", 1)[0] + "_meta.json")
        self.every = int(every)
        self.rows_written = 0
        self._fh = None
        self._w = None
        self._obs = None
        self._act_init = None       # deterministic action of the untrained policy, dB
        self._next_at = 0
        self._t0 = None
        self._rewards = []          # every non-terminal reward, in order

    def _on_training_start(self) -> None:
        self._obs = np.load(self.probe_obs_path).astype(np.float32)
        expected = self.training_env.observation_space.shape[0]
        if self._obs.shape[1] != expected:
            raise ValueError(f"probe batch has {self._obs.shape[1]} features, "
                             f"env observation_space has {expected}")
        self._t0 = time.time()
        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=FIELDS, extrasaction="raise")
        self._w.writeheader()
        self._fh.flush()

        # Baseline before any gradient step; also the anchor for action_drift_db.
        stats = self._measure()
        self._act_init = stats.pop("_actions_db")
        self._write(stats, phase="init")
        # Relative to where the run starts: on a resume num_timesteps is already
        # at the checkpoint's count, and a bare `= self.every` fired every
        # vec-step until it caught up (20 junk rows on the 2026-09-01 resume).
        # Identical to the old behaviour on a fresh run, where num_timesteps is 0.
        self._next_at = self.num_timesteps + self.every

        with open(self.meta_path, "w") as fh:
            json.dump({
                "probe_obs": os.path.abspath(self.probe_obs_path),
                "probe_rows": int(self._obs.shape[0]),
                "every_timesteps": self.every,
                "recent_window": RECENT_WINDOW,
                "n_envs": self.training_env.num_envs,
                "gamma": float(self.model.gamma),
                "v_target_formula": "r_bar / (1 - gamma)",
                "ppo_pilot_baseline": {
                    "mean_V": 6.212, "sd_V": 0.416,
                    "v_target": [42.0, 49.0], "pct_of_target": [12.7, 14.8],
                    "total_gradient_updates": 130,
                    "action_mean_abs_db": [0.0545, 0.1293],
                    "std": [0.9844, 1.0181],
                    "explained_variance_mean": -0.576,
                    "source": "AGENT_BUILD_LOG.md Step 11a-11c",
                },
            }, fh, indent=2)
        if self.verbose:
            print(f"CriticProbe -> {self.csv_path} "
                  f"({self._obs.shape[0]} held-out obs, every {self.every} samples)",
                  flush=True)

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._write(self._measure(), phase="final")
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        infos = self.locals.get("infos") or []
        for i, r in enumerate(rewards):
            info = infos[i] if i < len(infos) else {}
            # No "reward_terms" means a terminal or KPI-less step, which is not a
            # draw from the reward distribution.
            if isinstance(info, dict) and info.get("reward_terms") is not None:
                self._rewards.append(float(r))

        if self.num_timesteps >= self._next_at:
            self._next_at += self.every
            stats = self._measure()
            stats.pop("_actions_db", None)
            self._write(stats, phase="train")
        return True

    def _measure(self) -> dict:
        policy = self.model.policy
        was_training = policy.training
        policy.set_training_mode(False)
        try:
            with torch.no_grad():
                obs = torch.as_tensor(self._obs, device=self.model.device)
                # The critic's domain is [-1, 1]. Do not unscale before it.
                a_scaled = policy.actor(obs, deterministic=True)
                q_heads = self.model.critic(obs, a_scaled)
                # min over the twin critics: what the actor is trained against.
                q = torch.min(torch.stack(q_heads, dim=0), dim=0).values
                q = q.cpu().numpy().ravel()
                # Only now convert to the human-readable +/-6 dB.
                a_db = policy.unscale_action(a_scaled.cpu().numpy())
                _, log_std, _ = policy.actor.get_action_dist_params(obs)
                log_std = log_std.cpu().numpy()
        finally:
            policy.set_training_mode(was_training)

        gamma = float(self.model.gamma)
        r_all = float(np.mean(self._rewards)) if self._rewards else float("nan")
        recent = self._rewards[-RECENT_WINDOW:]
        r_recent = float(np.mean(recent)) if recent else float("nan")
        v_all = r_all / (1.0 - gamma)
        v_recent = r_recent / (1.0 - gamma)

        drift = (float(np.abs(a_db - self._act_init).mean())
                 if self._act_init is not None else 0.0)
        elapsed = time.time() - self._t0
        logged = self.model.logger.name_to_value

        return {
            "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(elapsed, 1),
            "num_timesteps": int(self.num_timesteps),
            "n_updates": int(getattr(self.model, "_n_updates", 0)),
            "samples_per_hour": round(self.num_timesteps / elapsed * 3600.0, 1) if elapsed > 0 else "",
            "q_mean": round(float(q.mean()), 4),
            "q_sd": round(float(q.std()), 4),
            "q_min": round(float(q.min()), 4),
            "q_max": round(float(q.max()), 4),
            "r_bar_all": round(r_all, 5),
            "r_bar_recent": round(r_recent, 5),
            "v_target_all": round(v_all, 3),
            "v_target_recent": round(v_recent, 3),
            "q_pct_of_target_all": round(100.0 * float(q.mean()) / v_all, 2) if v_all else "",
            "q_pct_of_target_recent": round(100.0 * float(q.mean()) / v_recent, 2) if v_recent else "",
            "action_mean_abs_db": round(float(np.abs(a_db).mean()), 5),
            "action_sd_db": round(float(a_db.std()), 5),
            "action_drift_db": round(drift, 5),
            "log_std_mean": round(float(log_std.mean()), 5),
            "log_std_sd": round(float(log_std.std(axis=0).mean()), 5),
            "ent_coef": round(self._ent_coef(), 6),
            "critic_loss": _opt(logged.get("train/critic_loss")),
            "actor_loss": _opt(logged.get("train/actor_loss")),
            "ent_coef_loss": _opt(logged.get("train/ent_coef_loss")),
            "_actions_db": a_db,
            **self._resources(),
        }

    def _resources(self) -> dict:
        """Resource telemetry at probe time. Never raises: every field degrades to
        blank, because a failed telemetry read must not stop a training run."""
        out = {k: "" for k in _RESOURCE_FIELDS}
        try:
            if self._cg:
                out["cg_mem_peak_gb"] = _gb(os.path.join(self._cg, "memory.peak"))
                out["cg_mem_current_gb"] = _gb(os.path.join(self._cg, "memory.current"))
                out["cg_swap_gb"] = _gb(os.path.join(self._cg, "memory.swap.current"))
            if self.output_folder:
                out["free_disk_gb"] = _free_disk_gb(self.output_folder)
                out["episode_dirs"] = _episode_dirs(self.output_folder)
        except Exception:                              # noqa: BLE001 - telemetry only
            pass
        return out

    def _ent_coef(self) -> float:
        """auto ent_coef lives in log_ent_coef; a fixed one in ent_coef_tensor."""
        if getattr(self.model, "ent_coef_optimizer", None) is not None:
            return float(self.model.log_ent_coef.detach().exp().item())
        tensor = getattr(self.model, "ent_coef_tensor", None)
        return float(tensor.item()) if tensor is not None else float("nan")

    def _write(self, stats: dict, phase: str) -> None:
        stats = {k: v for k, v in stats.items() if not k.startswith("_")}
        stats["phase"] = phase
        self._w.writerow(stats)
        self.rows_written += 1
        self._fh.flush()
        if self.verbose:
            print(f"  probe t={stats['num_timesteps']:>6} upd={stats['n_updates']:>6} "
                  f"Q={stats['q_mean']:>9.3f} (sd {stats['q_sd']:>7.3f}) "
                  f"{stats['q_pct_of_target_recent']}% of {stats['v_target_recent']} "
                  f"| act {stats['action_mean_abs_db']:.3f} dB "
                  f"drift {stats['action_drift_db']:.3f} "
                  f"| alpha {stats['ent_coef']:.4f}", flush=True)


def _opt(v):
    """Blank between SB3 logger dumps; 0.0 would read as a real measurement."""
    return round(float(v), 5) if v is not None else ""
