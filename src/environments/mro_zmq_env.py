import os
import selectors
import shutil
import subprocess
import time
import uuid
from typing import Optional

import gymnasium as gym
import numpy as np

from nsoran.ns_env import NsOranEnv
from ns_o_ran_gym.bridge.zmq_database import ZmqStateDatabase

# Same 7-cell mmWave topology as MlbZmqEnv. Kept as an independent copy rather
# than an import from mlb_zmq_env.py: that file is under active, concurrent
# development (the PPO -> SAC migration and reward rate-normalisation fixes),
# and this env is deliberately NOT built as a shared base class or subclass of
# it, purely to avoid two people editing the same file at once. Revisit this
# duplication once both environments have stabilised -- see the module
# docstring on MroZmqEnv.
CELLS = [2, 3, 4, 5, 6, 7, 8]

# One block of OBS_FIELDS per cell, flattened, mirroring MlbZmqEnv's layout
# convention (flat Box, not (7, N)) for the same reason: SB3/VecNormalize/replay
# buffers all want a 1-D Box, and the per-cell structure is recoverable through
# obs_index(). The field choice here is MRO-specific, not copied from MLB:
#   sinr_db_mean      -- link quality per cell, the core mobility-quality signal
#   num_active_ues    -- context / normaliser
#   badsignal_frac    -- per-cell share of near-outage transmissions, gives the
#                        agent visibility into which cells are outage risks
#   handover_activity -- handovers touching this cell (as source OR target)
#                        this window, per active user network-wide
#   last_margin_db    -- the agent's own previous action for this cell, so a
#                        smoothness-penalised policy can see what it last set
# prb_utilization is deliberately NOT included: it is MLB's primary decision
# variable, not MRO's, and the coordinator-blueprint design keeps each agent's
# observation scoped to its own function (MobilityScore) rather than pulling in
# LoadScore inputs it does not act on.
OBS_FIELDS = ("sinr_db_mean", "num_active_ues", "badsignal_frac",
              "handover_activity", "last_margin_db")

# Same bounds and same empirical reason as MlbZmqEnv: this is the identical
# physical SINR quantity from the identical simulator. The ceiling was widened
# from 40 to 100 dB after MLB's own validation found 4.41% of real L3 serving
# SINR values exceeding 40 dB (max 99.1 dB) and being silently clipped.
SINR_FLOOR_DB = -40.0
SINR_CEIL_DB = 100.0
SINR_SCALE_DB = 40.0

# Per-cell handover touches, normalised by network-wide active UEs, can exceed
# 1.0 in principle (a busy cell with churn) -- bounded loosely rather than at
# exactly 1.0 so a real spike clips rather than silently saturating at a value
# that looks like "every UE handed over exactly once".
HANDOVER_ACTIVITY_CEIL = 3.0

OBS_LOW_PER_CELL = np.array([SINR_FLOOR_DB / SINR_SCALE_DB, 0.0, 0.0, 0.0, -1.0],
                            dtype=np.float32)
OBS_HIGH_PER_CELL = np.array([SINR_CEIL_DB / SINR_SCALE_DB, 1.0, 1.0,
                              HANDOVER_ACTIVITY_CEIL, 1.0], dtype=np.float32)
OBS_LOW = np.tile(OBS_LOW_PER_CELL, len(CELLS))
OBS_HIGH = np.tile(OBS_HIGH_PER_CELL, len(CELLS))

# action[i] is a per-cell handover-margin OFFSET in dB for cell CELLS[i], added
# to (not replacing) the network's base HoSinrDifference. +-3 dB is a
# conservative starting bound -- unlike CIO's +-6 dB, this has not yet been
# validated against a live controller and should be revisited once it has, the
# same way CIO's own limit was locked from early placeholder-heuristic use
# rather than derived from first principles.
MARGIN_LIMIT_DB = 3.0

# RLF proxy threshold. Real RLF detection does not exist in this ns-3 fork
# (LteUeRrc::DoNotifyRadioLinkFailure is an empty stub and nothing ever calls
# NotifyRadioLinkFailure -- confirmed by source audit, not assumed). -5 dB
# reuses the exact convention EnergySavingEnv already established for its own
# RLF proxy (L3servingSINR < -5), rather than introducing a third threshold; see
# es_env.py's getRLFCounter.
OUTAGE_SINR_DB = -5.0

# Ping-pong window, in seconds. Identical value and identical caveat as
# MlbZmqEnv.PINGPONG_Y_S: locked from a small sample, not empirically derived,
# and reused here rather than imported so the two envs do not depend on each
# other's internals. If MLB's value is ever recalibrated, this one should be
# reconsidered too since it targets the same underlying handover-return
# phenomenon.
PINGPONG_Y_S = 0.8

# Reward weights. ALL PLACEHOLDERS at 1.0 -- deliberately not tuned. MLB's own
# weights came from measuring the *detrended* standard deviation of each term
# across nine seeded do-nothing baseline runs; hand-picked weights were tried
# once for MLB and rejected after one term ended up dominating 85% of the
# reward's variance. These five must go through the same measurement once this
# env can run against real ns-3 traffic. Do not train with these values as-is.
DEFAULT_W_QUALITY = 1.0
DEFAULT_W_PINGPONG = 1.0
DEFAULT_W_CHURN = 1.0
DEFAULT_W_OUTAGE = 1.0
DEFAULT_W_SMOOTH = 1.0


def obs_index(cell: int, field: str) -> int:
    """Index of `field` for `cell` in the flat observation vector."""
    return CELLS.index(cell) * len(OBS_FIELDS) + OBS_FIELDS.index(field)


def _parse_handovers(kpis: dict) -> Optional[list]:
    """[(t, imsi, src, dst), ...] for this step, or None if the field is absent.

    Pure function of the payload, no side effects -- shared by _get_obs() (for
    the per-cell handover_activity field) and _reward_terms() (for PingPong,
    HandoverRate and HandoverQuality), so the same events are parsed once and
    used consistently by both rather than risking two independent parses
    disagreeing. None (field absent, e.g. an ns-3 build predating Field 3) is
    kept distinct from an empty list (field present, genuinely zero handovers
    this step) -- callers must not conflate the two.
    """
    raw = (kpis or {}).get("handovers")
    if raw is None:
        return None
    events = []
    for h in raw:
        try:
            events.append((float(h["t"]), int(h["imsi"]), int(h["src"]), int(h["dst"])))
        except (KeyError, TypeError, ValueError):
            continue
    return events


def _extract_ue_sinr(kpis: dict) -> dict:
    """{imsi: l3_serving_sinr_db} for every UE reported anywhere in the
    payload, network-wide. Pure function, shared by reset() (to seed the
    "before" state for the first step's handovers) and _reward_terms() (to
    both read the "after" state and roll it forward for the next step) -- the
    two must agree on exactly what counts as "this UE's current SINR".
    """
    sinr_by_imsi = {}
    for cell_kpis in (kpis or {}).get("cells", {}).values():
        for imsi_str, ue in (cell_kpis.get("ues") or {}).items():
            sinr = ue.get("l3_serving_sinr_db")
            if sinr is None:
                continue
            try:
                sinr_by_imsi[int(imsi_str)] = float(sinr)
            except (TypeError, ValueError):
                continue
    return sinr_by_imsi


def _sinr_bad_fraction(cell_kpis: dict) -> float:
    """Fraction of this cell's DL transmissions at <= -6 dB SINR.

    Same source and same aggregation as MlbZmqEnv's BadSignal term
    (sinr_bins[0] / sum(sinr_bins)), duplicated locally rather than imported
    for the same file-independence reason as the other constants above. No
    transmissions at all gives 0.0, never NaN.
    """
    bins = cell_kpis.get("sinr_bins")
    if not isinstance(bins, (list, tuple)) or len(bins) != 7:
        return 0.0
    total = float(sum(float(v or 0.0) for v in bins))
    if total <= 0.0:
        return 0.0
    return float(bins[0] or 0.0) / total


class MroZmqEnv(NsOranEnv):
    """MRO/handover-margin Gym environment for scenario-marl-zmq, over ZeroMQ.

    Second of the three planned SON agents under Architecture A (MLB, MRO,
    COC), trained independently first with MLB's own action (CIO) held fixed
    -- see docs/mro_reward_design.pdf for the full term-by-term rationale and
    the source-level audit behind every substitution made versus the original
    reward-design brief.

    ACTION IS NOT YET WIRED ON THE ns-3 SIDE.
    -----------------------------------------
    step() sends {"cells": {cellId: {"ho_margin_db": ...}}} over the same
    ZeroMQ bridge MLB uses, but LteEnbNetDevice::ApplyControlPayload
    (lte-enb-net-device.cc) currently only recognises "cio_offset" -- an entry
    containing only "ho_margin_db" is silently skipped, so THIS ACTION IS
    CURRENTLY A NO-OP AGAINST THE REAL SIMULATOR. Training against real ns-3
    today would be training against a fixed base margin regardless of the
    agent's output. The ns-3 side (converting LteEnbRrc::m_sinrThresholdDifference
    into a per-cell map and adding the ApplyControlPayload branch) is tracked
    as separate work on the ns-3 repo's own feature/mro-agent branch. Reward
    and observation logic here does not depend on that landing first, so this
    file, and stub-driven tests against it, can be built and verified now.

    SB3-TRAINABLE SURFACE
    ----------------------
    observation_space: Box(35,), float32 -- one 5-field OBS_FIELDS block per
        cell in the fixed CELLS order. See obs_index().
    action_space: Box(-3, 3, (7,)), float32 -- per-cell handover-margin offset
        in dB, action[i] -> cell CELLS[i].
    reward: the five-term MRO reward (see _reward_terms): HandoverQuality
        (replaces the brief's literal HO-success rate, which the ns-3 source
        audit found would read a constant 1.0 in this simulator), PingPong,
        HandoverRate, OutageExposure (the RLF proxy), and ActionSmoothness.

    Deliberately independent of MlbZmqEnv: no shared base class, no subclass
    relationship. The ns-3/ZeroMQ lifecycle plumbing below (setup_sim,
    start_sim, _recv_next_kpis, reset, close, _purge_sim_path) is therefore a
    near-duplicate of MlbZmqEnv's own. That duplication is intentional for now
    -- see the CELLS comment at module scope -- and should be collapsed into a
    shared base once both environments are stable and no longer being edited
    concurrently by two people.
    """

    def __init__(self, ns3_path: str, scenario_configuration: dict, output_folder: str,
                 optimized: bool = False, zmq_port: int = 5555, history_maxlen: int = 10,
                 sinr_agg: str = "mean", recv_timeout_ms: int = 2000,
                 w_quality: float = DEFAULT_W_QUALITY, w_pingpong: float = DEFAULT_W_PINGPONG,
                 w_churn: float = DEFAULT_W_CHURN, w_outage: float = DEFAULT_W_OUTAGE,
                 w_smooth: float = DEFAULT_W_SMOOTH,
                 purge_sim_path_on_close: bool = False,
                 vary_rng_run_per_episode: bool = True,
                 reset_max_retries: int = 5,
                 build_ns3: bool = True):
        # MUST be set BEFORE super().__init__(): NsOranEnv.__init__ calls
        # setup_sim(), which reads this flag (see setup_sim() docstring).
        self.build_ns3 = bool(build_ns3)
        super().__init__(ns3_path=ns3_path, scenario='scenario-marl-zmq',
                          scenario_configuration=scenario_configuration,
                          output_folder=output_folder, optimized=optimized,
                          control_header=['timestamp', 'cellId', 'hoMarginDb'],
                          log_file='unused', control_file='unused')

        assert sinr_agg in ("min", "p10", "mean"), sinr_agg
        self.zmq_port = zmq_port
        # ns-3 must dial the port we bind. See MlbZmqEnv's identical comment:
        # two separate settings here used to mean a silent deadlock, both sides
        # alive at ~0% CPU with no error anywhere. The env owns the port.
        self.scenario_configuration['zmqPort'] = int(zmq_port)
        self.history_maxlen = history_maxlen
        self.sinr_agg = sinr_agg
        self.recv_timeout_ms = recv_timeout_ms
        self.total_ues = int(self.scenario_configuration['ues']) * len(CELLS)

        self.w_quality = float(w_quality)
        self.w_pingpong = float(w_pingpong)
        self.w_churn = float(w_churn)
        self.w_outage = float(w_outage)
        self.w_smooth = float(w_smooth)

        self.purge_sim_path_on_close = bool(purge_sim_path_on_close)
        self.vary_rng_run_per_episode = bool(vary_rng_run_per_episode)
        self.reset_max_retries = int(reset_max_retries)
        self.reset_retry_count = 0

        self.observation_space = gym.spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-MARGIN_LIMIT_DB, high=MARGIN_LIMIT_DB,
                                           shape=(len(CELLS),), dtype=np.float32)

        # Rolling handover history for cross-step ping-pong detection -- same
        # mechanics as MlbZmqEnv._handover_history (a bounce can straddle two
        # control steps, so a single payload is not enough to detect it).
        self._handover_history: list = []
        # Per-IMSI SINR from the previous step, for HandoverQuality's
        # before/after comparison. Cleared on every start_sim() like the
        # handover history: stale cross-episode data must not be compared
        # against a fresh episode's handovers.
        self._last_ue_sinr: dict = {}
        # The agent's own previous action, for ActionSmoothness. None until the
        # first step of an episode -- the very first action has no prior action
        # to be compared against, so smoothness is defined as 0.0 there rather
        # than treated as an unmeasured/dropped term.
        self._last_action: Optional[np.ndarray] = None

        self.zmq_db: Optional[ZmqStateDatabase] = None
        self._pending_kpis: Optional[dict] = None

    def setup_sim(self):
        """As NsOranEnv.setup_sim(), but optionally without rebuilding ns-3.

        Identical rationale to MlbZmqEnv.setup_sim(): under SubprocVecEnv, N
        workers each calling configure_and_build_ns3() contend for one CMake
        cache and one lock file. Parallel training builds once in the parent
        and constructs workers with build_ns3=False.
        """
        if self.build_ns3:
            return super().setup_sim()
        original = self.configure_and_build_ns3
        self.configure_and_build_ns3 = lambda *a, **k: None
        try:
            super().setup_sim()
        finally:
            self.configure_and_build_ns3 = original

    def start_sim(self):
        """ZeroMQ counterpart of NsOranEnv.start_sim(). No semaphores, no
        ActionController, no SQLiteDatabaseAPI -- see MlbZmqEnv.start_sim()."""
        if self.is_open:
            raise ValueError('The environment is open and a new start_sim has been called.')
        self.is_open = True

        parameters = self.scenario_configuration
        self.sim_result = {'params': dict(parameters), 'meta': {}}
        sim_uuid = str(uuid.uuid4())
        self.sim_result['meta']['id'] = sim_uuid
        self.sim_path = os.path.join(self.output_folder, sim_uuid)
        os.makedirs(self.sim_path)

        self.zmq_db = ZmqStateDatabase(port=self.zmq_port, history_maxlen=self.history_maxlen)
        self.zmq_db.start()
        self._pending_kpis = None
        # A new episode is a new simulation: stale handovers/SINR history from
        # the previous one would otherwise pair with fresh events across the
        # reset boundary.
        self._handover_history = []
        self._last_ue_sinr = {}
        self._last_action = None

        command = [self.script_executable] + [f'--{param}={value}' for param, value in parameters.items()]
        self.sim_result['meta']['start_time'] = time.time()
        self.sim_process = subprocess.Popen(command, cwd=self.sim_path, env=self.environment,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.selector = selectors.DefaultSelector()
        self._set_nonblocking(self.sim_process.stdout)
        self._set_nonblocking(self.sim_process.stderr)
        self.selector.register(self.sim_process.stdout, selectors.EVENT_READ)
        self.selector.register(self.sim_process.stderr, selectors.EVENT_READ)

    def _recv_next_kpis(self) -> Optional[dict]:
        """Poll for the next KPI snapshot; see MlbZmqEnv's identical method for
        why this cannot simply block on recv() forever."""
        while True:
            kpis = self.zmq_db.recv_kpi_update(timeout_ms=self.recv_timeout_ms)
            if kpis is not None:
                return kpis
            if self.is_simulation_over():
                return None

    def reset(self, *, seed: int = None, options: dict = None):
        gym.Env.reset(self, seed=seed)
        # Per-episode ns-3 seed, and the crash-retry budget: see MlbZmqEnv's
        # identical logic and its docstring for the full rationale (roughly 1
        # ns-3 seed in 10 kills the simulator during start-up; retry with a
        # fresh seed rather than losing the episode, but only when the seed is
        # actually allowed to change).
        attempts = self.reset_max_retries + 1 if self.vary_rng_run_per_episode else 1
        for attempt in range(attempts):
            if self.vary_rng_run_per_episode:
                self.scenario_configuration['RngRun'] = int(
                    self.np_random.integers(1, 2 ** 31 - 1))
            self.close()
            self.start_sim()

            self.terminated = False
            self.truncated = False
            self._pending_kpis = self._recv_next_kpis()
            if self._pending_kpis is not None:
                # Seed the "before" state from the episode's first snapshot.
                # Without this, any handover reported alongside the very next
                # step's KPIs would find no prior SINR to compare against and
                # be dropped from HandoverQuality even though the UE's
                # pre-handover SINR was actually observed at reset() time.
                self._last_ue_sinr = _extract_ue_sinr(self._pending_kpis)
                return self._get_obs(self._pending_kpis), {
                    "kpis": self._pending_kpis,
                    "reset_attempts": attempt + 1,
                    "rng_run": self.scenario_configuration.get('RngRun'),
                }
            self.reset_retry_count += 1

        self.terminated = True
        self.truncated = True
        return self._zero_obs(), {"reset_failed": True, "reset_attempts": attempts}

    def _margin_from_action(self, action) -> dict:
        """{cellId: {"ho_margin_db": dB}} for the action reply ns-3 is waiting on.

        action[i] -> cell CELLS[i], clipped to +-MARGIN_LIMIT_DB. Only cells
        present in the pending snapshot are addressed, mirroring
        MlbZmqEnv._cio_from_action. See the class docstring: ns-3 does not yet
        act on this key, so sending it is currently a no-op.
        """
        cells = self._pending_kpis.get("cells", {})
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != len(CELLS):
            raise ValueError(f'action must have {len(CELLS)} entries (one margin offset per '
                             f'cell {CELLS}), got {values.size}')
        values = np.clip(values, -MARGIN_LIMIT_DB, MARGIN_LIMIT_DB)
        return {str(cell): {"ho_margin_db": float(values[i])}
                for i, cell in enumerate(CELLS) if str(cell) in cells}, values

    def step(self, action):
        if self._pending_kpis is None:
            return self._zero_obs(), 0.0, self.terminated, self.truncated, {}

        cell_actions, margin = self._margin_from_action(action)
        action_payload = {
            "timestamp": self._pending_kpis.get("timestamp", 0.0),
            "cells": cell_actions,
        }
        self.zmq_db.send_control_actions(action_payload)

        next_kpis = self._recv_next_kpis()
        if next_kpis is None:
            # Time-limit end: truncated, not terminated. Same SB3 bootstrapping
            # reason as MlbZmqEnv.step() -- see that docstring.
            self.terminated = False
            self.truncated = True
            last_kpis = self._pending_kpis
            obs = self._get_obs(last_kpis)
            self._pending_kpis = None
            return obs, 0.0, self.terminated, self.truncated, {"kpis": last_kpis, "margin_offsets": cell_actions}

        self._pending_kpis = next_kpis
        terms = self._reward_terms(next_kpis, margin)
        info = {"kpis": next_kpis, "margin_offsets": cell_actions, "reward_terms": terms}
        return self._get_obs(next_kpis), terms["reward"], False, False, info

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _aggregate_sinr(self, ues: dict) -> Optional[float]:
        sinrs = [u.get("l3_serving_sinr_db") for u in ues.values() if u.get("l3_serving_sinr_db") is not None]
        if not sinrs:
            return None
        if self.sinr_agg == "min":
            return min(sinrs)
        if self.sinr_agg == "mean":
            return sum(sinrs) / len(sinrs)
        return float(np.quantile(sinrs, 0.10))

    def _cell_sinr_mean(self, cell_kpis: dict) -> Optional[float]:
        """Per-cell SINR in dB, or None if the cell reported no SINR at all.
        Same aggregation convention as MlbZmqEnv._cell_sinr_mean."""
        if cell_kpis.get("sinr_db_mean") is not None:
            return float(cell_kpis["sinr_db_mean"])
        return self._aggregate_sinr(cell_kpis.get("ues", {}))

    def _get_obs(self, kpis: dict) -> np.ndarray:
        """Flat float32 vector matching observation_space: one OBS_FIELDS block
        per cell, cells in the fixed CELLS order (2..8).

        A cell absent from the snapshot contributes an all-zero block except
        SINR, which gets SINR_FLOOR_DB like MlbZmqEnv -- 0 dB is a good link,
        so zero-filling it would read an absent/silent cell as healthy.
        last_margin_db reads self._last_action, which is None until the first
        step() of an episode; reset()'s observation therefore reports 0.0 dB
        for every cell, correctly reflecting that no margin has been applied yet.
        """
        cells = (kpis or {}).get("cells", {})
        row = np.zeros(self.observation_space.shape, dtype=np.float32)
        ue_scale = float(self.total_ues) if self.total_ues else 1.0

        handovers = _parse_handovers(kpis) or []
        active_ues_network = 0.0
        for cell_kpis in cells.values():
            active_ues_network += float(cell_kpis.get("num_active_ues", 0) or 0)

        for i, cell in enumerate(CELLS):
            cell_kpis = cells.get(str(cell)) or {}
            sinr = self._cell_sinr_mean(cell_kpis) if cell_kpis else None
            touches = sum(1 for _, _, src, dst in handovers if src == cell or dst == cell)
            activity = (touches / active_ues_network) if active_ues_network > 0 else 0.0
            last_margin = float(self._last_action[i]) if self._last_action is not None else 0.0
            row[i * len(OBS_FIELDS):(i + 1) * len(OBS_FIELDS)] = (
                (SINR_FLOOR_DB if sinr is None else sinr) / SINR_SCALE_DB,
                cell_kpis.get("num_active_ues", 0) / ue_scale,
                _sinr_bad_fraction(cell_kpis),
                activity,
                last_margin / MARGIN_LIMIT_DB,
            )
        return np.clip(row, OBS_LOW, OBS_HIGH).astype(np.float32)

    def _reward_terms(self, kpis: dict, action: np.ndarray) -> dict:
        """The five-term MRO reward, plus its breakdown and diagnostics.

            reward = w_quality   * HandoverQuality
                   - w_pingpong  * PingPong
                   - w_churn     * HandoverRate
                   - w_outage    * OutageExposure
                   - w_smooth    * ActionSmoothness

        See docs/mro_reward_design.pdf for the full rationale and the ns-3
        source audit behind each term, in particular why HandoverQuality
        replaces a literal RRC handover-success rate (the mmWave secondary-cell
        handover path does not go through the generic RRC state machine that
        can report failure, so a literal success rate would read a constant
        1.0 in this simulator) and why OutageExposure replaces a literal RLF
        rate (RLF detection is confirmed dead code in this ns-3 fork).

        HandoverQuality and PingPong/HandoverRate are dropped from the sum
        (not scored 0) when they cannot be measured this step, matching
        MlbZmqEnv's three-zeros discipline: a missing measurement must never
        be confused with a measured zero.
        """
        cells = (kpis or {}).get("cells", {})
        per_cell = [cells.get(str(cell), {}) for cell in CELLS]
        n_ues = np.array([c.get("num_active_ues", 0) or 0 for c in per_cell], dtype=float)
        active_ues = float(n_ues.sum())

        # --- gather every currently-reported UE's SINR, network-wide --------
        current_ue_sinr = _extract_ue_sinr(kpis)

        # --- OutageExposure --------------------------------------------------
        # A genuinely idle network (no UEs reporting SINR yet, e.g. warm-up)
        # scores 0.0 here -- nobody is in outage because nobody is connected --
        # rather than being dropped. This mirrors Balance's low-load guard in
        # MlbZmqEnv: an empty network is a real, well-defined state, not a
        # missing measurement.
        if current_ue_sinr:
            n_outage = sum(1 for s in current_ue_sinr.values() if s < OUTAGE_SINR_DB)
            outage = n_outage / len(current_ue_sinr)
        else:
            outage = 0.0

        # --- handovers this window -------------------------------------------
        handovers = _parse_handovers(kpis)
        handovers_unavailable = handovers is None
        handovers = handovers or []
        n_handovers = len(handovers)

        # --- HandoverRate ------------------------------------------------------
        if handovers_unavailable:
            handover_rate = None
        else:
            handover_rate = (n_handovers / active_ues) if active_ues > 0 else 0.0

        # --- PingPong (identical mechanics to MlbZmqEnv._reward_terms) -------
        if handovers_unavailable:
            pingpong = None
            pingpong_count = 0
        else:
            self._handover_history.extend(handovers)
            if self._handover_history:
                newest = max(t for t, *_ in self._handover_history)
                self._handover_history = [e for e in self._handover_history
                                          if newest - e[0] <= 2.0 * PINGPONG_Y_S]
            pingpong_count = 0
            for t, imsi, src, dst in handovers:
                for tp, ip, sp, dp in self._handover_history:
                    if ip != imsi or tp >= t:
                        continue
                    if t - tp <= PINGPONG_Y_S and sp == dst and dp == src:
                        pingpong_count += 1
                        break
            pingpong = (pingpong_count / active_ues) if active_ues > 0 else 0.0

        # --- HandoverQuality ---------------------------------------------------
        # Did the destination cell's SINR actually beat the source cell's? Only
        # scored for handovers where both a before-SINR (this env's own memory
        # of the previous step) and an after-SINR (this step's payload) exist;
        # a UE that disappears before its outcome can be checked is dropped
        # from the ratio entirely, not counted as a failure.
        if handovers_unavailable:
            quality = None
            quality_good = 0
            quality_measured = 0
        else:
            quality_good = 0
            quality_measured = 0
            for t, imsi, src, dst in handovers:
                before = self._last_ue_sinr.get(imsi)
                after = current_ue_sinr.get(imsi)
                if before is None or after is None:
                    continue
                quality_measured += 1
                if after > before:
                    quality_good += 1
            quality = (quality_good / quality_measured) if quality_measured > 0 else None

        # --- ActionSmoothness ---------------------------------------------------
        margin = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._last_action is None:
            smoothness = 0.0
        else:
            smoothness = float(np.mean((margin - self._last_action) ** 2))

        reward = -self.w_outage * outage - self.w_smooth * smoothness
        if quality is not None:
            reward += self.w_quality * quality
        if pingpong is not None:
            reward -= self.w_pingpong * pingpong
        if handover_rate is not None:
            reward -= self.w_churn * handover_rate

        # Roll state forward for the next step's before/after and Δaction.
        self._last_ue_sinr = current_ue_sinr
        self._last_action = margin.copy()

        return {
            "reward": float(reward),
            "quality": None if quality is None else float(quality),
            "pingpong": None if pingpong is None else float(pingpong),
            "handover_rate": None if handover_rate is None else float(handover_rate),
            "outage": float(outage),
            "smoothness": float(smoothness),
            "weights": {"quality": self.w_quality, "pingpong": self.w_pingpong,
                        "churn": self.w_churn, "outage": self.w_outage,
                        "smooth": self.w_smooth},
            "diagnostics": {
                "active_ues": active_ues,
                "n_ues_reporting_sinr": len(current_ue_sinr),
                "handovers_unavailable": bool(handovers_unavailable),
                "handovers_this_step": n_handovers,
                "handover_history_len": len(self._handover_history),
                "pingpong_count": pingpong_count,
                "pingpong_y_s": PINGPONG_Y_S,
                "quality_good": quality_good,
                "quality_measured": quality_measured,
                "outage_sinr_db": OUTAGE_SINR_DB,
                "margin_action_db": dict(zip(CELLS, margin.tolist())),
                # Built from what is actually in the sum, so a dropped term is
                # never mistaken for one that scored zero.
                "terms_present": tuple(
                    ["outage", "smoothness"]
                    + (["quality"] if quality is not None else [])
                    + (["pingpong"] if pingpong is not None else [])
                    + (["handover_rate"] if handover_rate is not None else [])),
            },
        }

    def _compute_reward(self, kpis: dict) -> float:
        # Required by NsOranEnv's interface, but this env computes reward
        # inside step() where the current action is available; _reward_terms
        # needs that action, so this entry point cannot be used standalone.
        raise NotImplementedError('MroZmqEnv computes reward in step(), which has '
                                  'the action _reward_terms needs; use step() instead')

    def _purge_sim_path(self):
        """Delete this env's own run directory, if purging is enabled.
        Identical conservative guards to MlbZmqEnv._purge_sim_path."""
        if not self.purge_sim_path_on_close:
            return
        sim_path = getattr(self, 'sim_path', None)
        if not sim_path:
            return

        target = os.path.realpath(sim_path)
        parent = os.path.realpath(self.output_folder)
        if os.path.dirname(target) != parent:
            return
        if os.path.islink(sim_path) or not os.path.isdir(target):
            return
        try:
            uuid.UUID(os.path.basename(target))
        except ValueError:
            return

        shutil.rmtree(target, ignore_errors=True)

    def close(self):
        if self.is_open:
            if self.zmq_db is not None:
                self.zmq_db.close()
                self.zmq_db = None
            self.sim_process.kill()
            try:
                self.sim_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            self.is_open = False
            self._purge_sim_path()

    # NsOranEnv's file/datalake hooks are unused by this ZMQ-native env.
    def _compute_action(self, action):
        raise NotImplementedError('MroZmqEnv builds its action payload directly in step()')

    def _init_datalake_usecase(self):
        pass

    def _fill_datalake_usecase(self):
        pass
