import os
import selectors
import shutil
import subprocess
import time
import uuid
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from nsoran.ns_env import NsOranEnv
from ns_o_ran_gym.bridge.zmq_database import ZmqStateDatabase

# NR cells in scenario-marl-zmq.cc's default topology. basicCellId=1 is the LTE
# anchor; the mmWave eNBs get 2..8. Mirrors tests/test_server.py's EXPECTED_CELLS.
CELLS = [2, 3, 4, 5, 6, 7, 8]

# One block of OBS_FIELDS per cell in CELLS order, flattened to a 1-D vector of
# 7 * 5 = 35. Flat rather than (7, 5) because a 1-D Box is what every SB3
# algorithm, VecNormalize and the replay buffers handle without an extra
# extractor. The per-cell structure is still recoverable through obs_index().
OBS_FIELDS = ("prb_utilization", "num_active_ues", "buffer_bytes",
              "volume_bytes", "sinr_db_mean")

# Per-field bounds, in OBS_FIELDS order. Deliberately loose rather than tight: a
# tight bound would silently clip real traffic.
SINR_FLOOR_DB = -40.0           # also used when a cell reports no SINR
# The ceiling used to be 40 dB, which clipped 4.41% of real L3 serving SINR
# values (max 99.1 dB), so a 79 dB cell read as a 40 dB one.
SINR_CEIL_DB = 100.0
MAX_BYTES = 1e9

# The raw fields differ by seven orders of magnitude: on a 35-UE run buffer_bytes
# sd is 21.8 million times prb_utilization's. MlpPolicy does not normalise
# observations, so without scaling the byte fields dominate the first Linear
# layer and the [0,1] fields contribute almost nothing. Fixed physical divisors
# rather than VecNormalize, so scaling is deterministic, needs no saved state and
# keeps the observation interpretable.
#   prb_utilization : already a fraction (capped ~0.849 by the TDD frame)
#   num_active_ues  : divided by the run's total UE count, giving a per-cell
#                     share that stays scale-free across `ues` settings
#   buffer_bytes    : 10 MB, LteRlcUm::MaxTxBufferSize, above which ns-3 drops
#   volume_bytes    : 1 MB per 0.1 s E2 window, about 80 Mbps for one cell
#   sinr_db_mean    : 40 dB, the nominal top of the usable range
BUFFER_SCALE_BYTES = 1e7
VOLUME_SCALE_BYTES = 1e6
SINR_SCALE_DB = 40.0

OBS_LOW_PER_CELL = np.array([0.0, 0.0, 0.0, 0.0, SINR_FLOOR_DB / SINR_SCALE_DB],
                            dtype=np.float32)
OBS_HIGH_PER_CELL = np.array([1.2, 1.0,
                              MAX_BYTES / BUFFER_SCALE_BYTES,
                              MAX_BYTES / VOLUME_SCALE_BYTES,
                              SINR_CEIL_DB / SINR_SCALE_DB], dtype=np.float32)
OBS_LOW = np.tile(OBS_LOW_PER_CELL, len(CELLS))
OBS_HIGH = np.tile(OBS_HIGH_PER_CELL, len(CELLS))

# action[i] is the CIO offset in dB for cell CELLS[i], so action[0] is cell 2 and
# action[6] is cell 8. The 6 dB limit matches what the placeholder heuristic and
# tests/test_server.py have always clamped to.
CIO_LIMIT_DB = 6.0

# Reward weights, tuned on 9 do-nothing baseline worlds (seeds 1001-1010, one
# crashed) at the locked config: ues=5, RLC AM, 500 us, simTime=20.
#
# At w=1.0 everywhere the reward was 85% Backlog, and 52% of its variance came
# from the step index alone, since queues fill with time whatever the agent does.
# The weights equalise each term's detrended spread. Detrended matters: Backlog's
# raw sd is inflated 1.70x by that drift, so equalising on raw sd would punish it
# for drifting and under-weight the term that actually discriminates offload
# quality (4/4 offline, -26% under a good action).
#
#   term shares  85/6/5/4  becomes  32/25/21/21  (backlog/balance/satisfaction/pingpong)
#   reward R^2 vs clock  0.518  becomes  0.194
#
# BadSignal is deliberately not equalised. It is a near-silent overshoot detector
# by design and equalising it would need w=30.
#
# All of this came from do-nothing runs. A learning agent explores and will drive
# larger swings, especially in PingPong and BadSignal. Revisit after the first
# real training run.
DEFAULT_W_BALANCE = 1.0
DEFAULT_W_SATISFACTION = 1.1
DEFAULT_W_BACKLOG = 0.1
DEFAULT_W_BADSIGNAL = 1.0
DEFAULT_W_PINGPONG = 1.5

# The busiest cell must use at least this fraction of its PRBs before Jain's
# index counts as meaningful. Below it the network is near-empty and Balance is
# neutralised to BALANCE_GUARD_VALUE. Same convention as analysis/mlb_reward.py.
LOW_LOAD_PRB_THRESHOLD = 0.05
BALANCE_GUARD_VALUE = 1.0

# Window the byte counters in the ZMQ payload actually cover, in seconds.
#
# volume_bytes comes from MmWaveEnbNetDevice::GetMacVolumeCellSpecific(). Its
# backing field m_macVolumeCellSpecific is written inside the E2 DU report
# builder (mmwave-enb-net-device.cc:1212), and the per-UE source counters are
# zeroed by ResetPhyTracesForRntiCellId() at the end of every build (:1377). So
# the value read at a control step is one E2 indication window of traffic, not
# everything since the previous control step. That window is the scenario's
# indicationPeriodicity, 0.1 s.
#
# This is not T_control, which defaults to 1.0 s and comes from the scenario's
# controlInterval GlobalValue. Using T_control as the divisor would understate
# the delivery rate 10x and inflate Backlog by the same factor. Keep this in sync
# with the CFG's indicationPeriodicity if that ever changes.
KPI_WINDOW_S = 0.1

# Full-buffer UDP DL UEs are node index u % 4 == 0 in scenario-marl-zmq.cc's
# trafficModel=3 (UdpClient, 1280 B payload every 500 us), and IMSI = u + 1,
# hence the (imsi - 1) % 4 == 0 rule.
#
# n_udp is run-level, derived once from the configured UE count and never
# recounted per step. A UDP UE that drops off still wanted its data, so it stays
# in the denominator and scores 0. Recounting would shrink the denominator
# exactly when the network fails a user, which the agent could exploit via CIO.

# Y, the ping-pong window, in seconds.
#
# This value is arbitrary and should not be presented as empirically derived. It
# was locked at 0.8 s on "observed return gaps 0.12-0.66 s" from 85 handovers in
# one run. At 137x that sample (11,666 handover legs) the range does not survive:
# 0.8 s sits essentially on the median consecutive same-UE handover gap
# (0.7845 s), the point of maximum sensitivity to its own threshold. Coverage of
# immediate A-to-B-to-A reversals decays smoothly with no knee anywhere: 0.5 s
# gives 38.5%, 0.8 gives 55.2%, 1.0 gives 63.3%, 1.5 gives 75.2%, 2.0 gives
# 82.9%. Reverse-pair gaps at any distance have a long tail (p50 1.05 s,
# p75 3.31 s, p90 9.05 s).
#
# The term is not dead: it flags 28.9% of legs and carries 19% of reward SD. It
# is simply not calibrated to anything. Treat the value as a pending decision.
PINGPONG_Y_S = 0.8

# PDCP SDU size of one full-buffer UDP packet: 1280 B payload plus 30 B of
# UDP/IP headers. DlE2PdcpStats.txt reports PduSize 1310 for these flows.
UDP_PDCP_SDU_BYTES = 1310.0
# ns-3's own default for the UdpClient "Interval" attribute.
DEFAULT_UDP_INTERVAL_US = 500.0


def udp_ue_rate_kbps(interval_us: float = DEFAULT_UDP_INTERVAL_US) -> float:
    """Offered DL rate of one full-buffer UDP UE, from the configured interval.

    This is the Satisfaction denominator, so it has to come from configuration
    rather than measurement: it is what the UEs asked for. A measured denominator
    would shrink exactly when the network fails a user.
    """
    interval_us = float(interval_us) or DEFAULT_UDP_INTERVAL_US
    return UDP_PDCP_SDU_BYTES * 8.0 * 1000.0 / interval_us


SATISFACTION_CLIP_MAX = 1.2


def obs_index(cell: int, field: str) -> int:
    """Index of `field` for `cell` in the flat observation vector."""
    return CELLS.index(cell) * len(OBS_FIELDS) + OBS_FIELDS.index(field)


def _udp_imsis(total_ues: int) -> tuple:
    """IMSIs of the full-buffer UDP DL UEs, derived from the run's UE count.

    Mirrors scenario-marl-zmq.cc trafficModel=3, `u % 4 == 0`, with IMSI = u+1.
    Derived, never hardcoded: an earlier note in the build log quoted "9 UDP
    UEs", which was true of a different UE count -- ues=3 gives 21 UEs and
    therefore 6 UDP DL UEs.
    """
    return tuple(u + 1 for u in range(int(total_ues)) if u % 4 == 0)


def _sinr_bins(cell_kpis: dict) -> list:
    """The 7 L1M.RS-SINR bin counters for one cell, as floats.

    Bin order matches scenario-marl-zmq.cc's MarlControlStep, which mirrors
    MmWavePhyTrace::UpdateTraces (mmwave-phy-trace.cc:306-333):
        [0] <= -6 dB   [1] <= 0   [2] <= 6   [3] <= 12   [4] <= 18
        [5] <= 24      [6] > 24
    Index 0 is the "bin34" of the offline analysis -- the one BadSignal counts.
    A cell that is absent, or that predates the payload field, yields all-zero
    (which the caller's denominator guard turns into badsignal 0.0, not NaN).
    """
    values = cell_kpis.get("sinr_bins")
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        return [0.0] * 7
    return [float(v or 0.0) for v in values]


def _jain(values: np.ndarray) -> float:
    """Jain's fairness index of a non-negative vector; 1.0 for an all-zero vector."""
    total = float(values.sum())
    if total <= 0.0:
        return 1.0
    return float(total ** 2 / (len(values) * float((values ** 2).sum())))


class MlbZmqEnv(NsOranEnv):
    """MLB/CIO Gym environment for scenario-marl-zmq, driven purely over ZeroMQ.

    This has nothing to do with EnergySavingEnv/es_env.py: that class is the
    separate ES sleep-mode (heuristicType/bsOn/.../bsOff) use case, which
    scenario-marl-zmq.cc runs with heuristicType=-1 (inert) by default. This
    class targets the CIO/MLB lever instead.

    scenario-marl-zmq.cc's MarlControlStep (scheduled every T_control, which
    defaults to 1.0 s and is set by the controlInterval GlobalValue) sends one
    KPI snapshot per real cell over a ZeroMQ
    REQ socket and blocks for the reply (ZmqDatabaseClient::StepSync) before
    applying it via LteEnbNetDevice::ApplyControlPayload. This env binds the
    matching REP side (bridge/zmq_database.py's ZmqStateDatabase, same class
    tests/test_server.py exercises standalone) and does the identical
    recv-KPIs / compute / send-CIO round trip inside step(), instead of
    NsOranEnv's semaphore + CSV datalake + ActionController pipeline: no
    cu-up/cu-cp/du-cell CSVs or control file are read or written here. As long
    as scenario_configuration never sets controlFileName, ns-3's legacy
    semaphore-gated control path (lte-enb-net-device.cc, gated on
    m_controlFilename != "") stays inert, so nothing on the ns-3 side is
    waiting on a semaphore this env doesn't create.

    SB3-TRAINABLE SURFACE
    ---------------------
    observation_space: Box(35,), float32 -- one 5-field block per cell in the
        fixed CELLS order (see OBS_FIELDS / obs_index). Read straight off the
        live JSON snapshot; absent cells are zero-filled and values are clipped
        into the declared bounds.
    action_space: Box(-6, 6, (7,)), float32 -- per-cell CIO offset in dB,
        action[i] -> cell CELLS[i]. step() sends exactly this back to ns-3.
    reward: the five-term MLB reward (see _reward_terms).

    total_ues is derived from scenario_configuration['ues'] * len(CELLS) rather
    than assuming a fixed UE count, so the per-cell UE share in the observation
    stays correct across `ues` settings.
    """

    def __init__(self, ns3_path: str, scenario_configuration: dict, output_folder: str,
                 optimized: bool = False, zmq_port: int = 5555, history_maxlen: int = 10,
                 sinr_agg: str = "mean",
                 recv_timeout_ms: int = 2000, control_period_s: float = KPI_WINDOW_S,
                 w_balance: float = 1.0, w_backlog: float = 0.1,
                 w_badsignal: float = 1.0, w_satisfaction: float = 1.1,
                 w_pingpong: float = 1.5,
                 low_load_prb_threshold: float = LOW_LOAD_PRB_THRESHOLD,
                 purge_sim_path_on_close: bool = False,
                 vary_rng_run_per_episode: bool = True,
                 udp_interval_us_range: Optional[tuple] = None,
                 reset_max_retries: int = 5,
                 build_ns3: bool = True):
        # NsOranEnv.__init__ -> setup_sim() builds ns-3 and resolves
        # self.script_executable; that's the only part of the base class this
        # env reuses. control_header/log_file/control_file are required by
        # the signature but otherwise unused: start_sim()/reset()/step()/
        # close() are fully overridden below to bypass ActionController and
        # the SQLite datalake.
        # MUST be set BEFORE super().__init__(): NsOranEnv.__init__ calls
        # setup_sim(), which reads this flag. Assigning it after the super call
        # raises AttributeError.
        self.build_ns3 = bool(build_ns3)
        super().__init__(ns3_path=ns3_path, scenario='scenario-marl-zmq',
                          scenario_configuration=scenario_configuration,
                          output_folder=output_folder, optimized=optimized,
                          control_header=['timestamp', 'cellId', 'cioDb'],
                          log_file='unused', control_file='unused')

        assert sinr_agg in ("min", "p10", "mean"), sinr_agg
        self.zmq_port = zmq_port
        # ns-3 must dial the port we bind. These used to be two separate
        # settings, zmq_port here and the scenario's zmqPort GlobalValue, and a
        # caller that set only the first got a silent deadlock: Python listening
        # on one port, ns-3 blocked forever on another, both alive at ~0% CPU
        # with no error anywhere. The env now owns the port and passes it down.
        self.scenario_configuration['zmqPort'] = int(zmq_port)
        self.history_maxlen = history_maxlen
        self.sinr_agg = sinr_agg
        self.recv_timeout_ms = recv_timeout_ms
        self.total_ues = int(self.scenario_configuration['ues']) * len(CELLS)

        # control_period_s is the window the byte counters cover, the 0.1 s E2
        # indication window, not the T_control step period. See KPI_WINDOW_S. An
        # earlier revision used T_control on the reasoning that one step() covers
        # that much simulated time, which is true of the step but not of
        # volume_bytes, and it inflated Backlog 10x.
        self.control_period_s = float(control_period_s)
        self.w_balance = float(w_balance)
        self.w_backlog = float(w_backlog)
        self.w_badsignal = float(w_badsignal)
        self.w_satisfaction = float(w_satisfaction)
        # Run-level, computed once. See the Satisfaction notes at module scope
        # for why this must not be recounted per step.
        self.udp_imsis = _udp_imsis(self.total_ues)
        self.n_udp = len(self.udp_imsis)
        # Per-UE offered rate follows the scenario's udpFullBufferIntervalUs
        # knob. Absent from the config it falls back to ns-3's own 500 us
        # default, which is 20960 kbps.
        self.udp_ue_rate_kbps = udp_ue_rate_kbps(
            self.scenario_configuration.get('udpFullBufferIntervalUs',
                                            DEFAULT_UDP_INTERVAL_US))
        # Per-episode offered load. None, the default, writes no key at all, so
        # the command line and udp_ue_rate_kbps are identical to a build without
        # this feature. See _draw_udp_interval for the two traps.
        self.udp_interval_us_range = None
        if udp_interval_us_range is not None:
            lo, hi = (int(v) for v in udp_interval_us_range)
            if lo <= 0 or hi < lo:
                raise ValueError(f'udp_interval_us_range must be 0 < lo <= hi, got {(lo, hi)}')
            self.udp_interval_us_range = (lo, hi)
        self.low_load_prb_threshold = float(low_load_prb_threshold)

        # Delete this env's own run directory in close(). Off by default,
        # because a run dir is a deliverable for the one-off analysis scripts
        # (see the ordering hazard on _purge_sim_path). Training turns it on,
        # where the dirs are pure waste: ns-3 writes several MB per step, which
        # is hundreds of GB over a 100k-step run.
        self.purge_sim_path_on_close = bool(purge_sim_path_on_close)
        # See reset(). Turn OFF for a controlled A/B where both arms must run
        # the identical realization (that is how the Move 2/3 arms were run).
        self.vary_rng_run_per_episode = bool(vary_rng_run_per_episode)
        # See reset(): budget for re-drawing a seed when ns-3 dies at start-up.
        # 5 retries makes an episode-start failure ~1e-5 at the measured ~10%
        # per-seed crash rate, while still terminating rather than looping.
        self.reset_max_retries = int(reset_max_retries)
        self.reset_retry_count = 0        # cumulative, for diagnostics

        # See the OBS_FIELDS / CIO_LIMIT_DB blocks at module scope for the
        # layout and the bound rationale.
        self.observation_space = gym.spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-CIO_LIMIT_DB, high=CIO_LIMIT_DB,
                                           shape=(len(CELLS),), dtype=np.float32)

        self.w_pingpong = float(w_pingpong)
        # Rolling handover history for cross-step ping-pong detection; see the
        # PingPong docstring on why a single payload is not enough.
        self._handover_history: list = []

        self.zmq_db: Optional[ZmqStateDatabase] = None
        self._pending_kpis: Optional[dict] = None

    def setup_sim(self):
        """As NsOranEnv.setup_sim(), but optionally without rebuilding ns-3.

        NsOranEnv.setup_sim() calls configure_and_build_ns3() on every env
        construction. Under SubprocVecEnv that is N workers launching an ns-3
        build at the same time, contending for one CMake cache, one .lock-ns3_*
        status file and one output binary. Even a no-op build still rewrites the
        lock file. Parallel training therefore builds once in the parent and
        constructs its workers with build_ns3=False. Only the build is skipped:
        LD_LIBRARY_PATH, reading the build-status file and resolving
        script_executable all still run, and that is what locates the binary.
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
        """ZeroMQ counterpart of NsOranEnv.start_sim(): binds the KPI/action
        bridge, then launches scenario-marl-zmq. No semaphores, no
        ActionController, no SQLiteDatabaseAPI.
        """
        if self.is_open:
            raise ValueError('The environment is open and a new start_sim has been called.')
        self.is_open = True

        parameters = self.scenario_configuration
        self.sim_result = {'params': dict(parameters), 'meta': {}}
        sim_uuid = str(uuid.uuid4())
        self.sim_result['meta']['id'] = sim_uuid
        self.sim_path = os.path.join(self.output_folder, sim_uuid)
        os.makedirs(self.sim_path)

        # Bind before launching ns-3, so ZmqDatabaseClient::Connect() has a
        # listener as soon as scenario-marl-zmq tries to reach it.
        self.zmq_db = ZmqStateDatabase(port=self.zmq_port, history_maxlen=self.history_maxlen)
        self.zmq_db.start()
        self._pending_kpis = None
        # A new episode is a new simulation: stale handovers from the previous one
        # would otherwise pair with fresh ones across the reset boundary.
        self._handover_history = []

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
        """Poll for the next KPI snapshot, timing out repeatedly so we can
        check whether ns-3 has exited (in which case no snapshot is coming
        and blocking forever on recv() would hang the caller)."""
        while True:
            kpis = self.zmq_db.recv_kpi_update(timeout_ms=self.recv_timeout_ms)
            if kpis is not None:
                return kpis
            if self.is_simulation_over():
                return None

    def _draw_udp_interval(self) -> Optional[int]:
        """Draw this episode's offered DL load and rebuild the demand with it.

        Returns the drawn interval, or None when randomisation is off.

        The denominator has to move with the knob. udp_ue_rate_kbps is the
        Satisfaction denominator: n_udp * udp_ue_rate_kbps is what the UEs are
        deemed to have asked for. It is computed once in __init__ from the static
        config, which is only correct while the interval never changes. Randomise
        the interval without recomputing it and every episode is scored against
        the demand of a different episode. Satisfaction still stays in range and
        still moves when the network moves, so nothing downstream can detect it.

        The value must be a scalar, not a list. ns_env.py:61 stores {k: v[0]}, so
        everything in scenario_configuration is a scalar by the time we get here,
        and start_sim renders the command line as f'--{param}={value}'. Assigning
        [480] would emit "--udpFullBufferIntervalUs=[480]", which ns-3's
        UintegerValue parser rejects. RngRun documents the same trap.
        """
        if self.udp_interval_us_range is None:
            return None
        lo, hi = self.udp_interval_us_range
        # ns-3 declares this GlobalValue as UintegerValue, so it must be an int.
        # integers(lo, hi + 1) makes the range INCLUSIVE of hi.
        interval = int(self.np_random.integers(lo, hi + 1))
        self.scenario_configuration['udpFullBufferIntervalUs'] = interval
        self.udp_ue_rate_kbps = udp_ue_rate_kbps(interval)
        return interval

    def reset(self, *, seed: int = None, options: dict = None):
        gym.Env.reset(self, seed=seed)
        # Per-episode ns-3 seed. Without this every episode replayed the same
        # world: start_sim() builds its command line from scenario_configuration,
        # whose RngRun is fixed at construction, and gym.Env.reset(seed=) only
        # seeds this object's np_random, which never reaches ns-3. Drawing from
        # the seeded np_random means a given reset(seed=) still reproduces the
        # same episode sequence.
        #
        # Roughly 1 ns-3 seed in 10 kills the simulator during start-up: it exits
        # with empty stdout/stderr before its first control step (seed 1003 did
        # this reproducibly on two different ZMQ ports, so it is the seed, not the
        # setup). With a fresh seed every episode a long run is guaranteed to hit
        # it, so retry with a different seed rather than losing the run.
        #
        # Only retry when the seed is allowed to change. Under
        # vary_rng_run_per_episode=False (controlled A/B runs) a retry would
        # replay the identical failing simulation, so surface the failure instead.
        attempts = self.reset_max_retries + 1 if self.vary_rng_run_per_episode else 1
        for attempt in range(attempts):
            if self.vary_rng_run_per_episode:
                # NOTE ns_env.py:61 stores {k: v[0]} -- the values here are
                # SCALARS, not the single-element lists the caller passes in.
                # Assigning a list would emit "--RngRun=[123]".
                self.scenario_configuration['RngRun'] = int(
                    self.np_random.integers(1, 2 ** 31 - 1))
            # Redrawn on every ATTEMPT, not once per reset: a retry launches a
            # new simulation, and its Satisfaction must be scored against the
            # load that simulation was actually given.
            udp_interval = self._draw_udp_interval()
            self.close()
            self.start_sim()

            self.terminated = False
            self.truncated = False
            self._pending_kpis = self._recv_next_kpis()
            if self._pending_kpis is not None:
                # Do not add `attempt` to reset_retry_count here: each failed
                # attempt already incremented it below.
                return self._get_obs(self._pending_kpis), {
                    "kpis": self._pending_kpis,
                    "reset_attempts": attempt + 1,
                    "rng_run": self.scenario_configuration.get('RngRun'),
                    # None when randomisation is off. Logged per episode so the
                    # offered load this episode was scored against is on record
                    # next to the reward it produced.
                    "udp_interval_us": udp_interval,
                    "udp_ue_rate_kbps": self.udp_ue_rate_kbps,
                }
            # ns-3 exited before its first control step -> bad seed, try another.
            self.reset_retry_count += 1

        # Every attempt failed. Do NOT pretend the episode started: return a
        # terminated episode so the caller sees it rather than training on zeros.
        self.terminated = True
        self.truncated = True
        return self._zero_obs(), {"reset_failed": True, "reset_attempts": attempts}

    def _cio_from_action(self, action) -> dict:
        """{cellId: {"cio_offset": dB}} for the CIO reply ns-3 is waiting on.

        The agent's action drives this: action[i] -> cell CELLS[i], clipped to
        the action_space's +/-6 dB. Only cells present in the pending snapshot
        are addressed, so a cell ns-3 didn't report is never sent an offset.

        action=None keeps the ORIGINAL placeholder heuristic (test_server.py's:
        push UEs away from an overloaded cell, pull them onto an idle one),
        which is what examples/cio_zmq_experiment.py still drives the round trip
        with. That path is a fallback for the no-agent scripts, not the training
        path -- a policy always supplies an action.
        """
        cells = self._pending_kpis.get("cells", {})

        if action is None:
            return {
                cell_id: {"cio_offset": max(-CIO_LIMIT_DB, min(
                    CIO_LIMIT_DB, -12.0 * (cell_kpis.get("prb_utilization", 0.5) - 0.5)))}
                for cell_id, cell_kpis in cells.items()
            }

        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != len(CELLS):
            raise ValueError(f'action must have {len(CELLS)} entries (one CIO offset per '
                             f'cell {CELLS}), got {values.size}')
        values = np.clip(values, -CIO_LIMIT_DB, CIO_LIMIT_DB)

        return {str(cell): {"cio_offset": float(values[i])}
                for i, cell in enumerate(CELLS) if str(cell) in cells}

    def step(self, action):
        if self._pending_kpis is None:
            return self._zero_obs(), 0.0, self.terminated, self.truncated, {}

        # Reply to the snapshot obs was last computed from. ns-3's
        # MarlControlStep/StepSync is blocked waiting for this. Same "cells"
        # shape as the KPI payload: LteEnbNetDevice::ApplyControlPayload reads
        # actionPayload["cells"][cellId]["cio_offset"] for each cell present.
        cell_actions = self._cio_from_action(action)

        action_payload = {
            "timestamp": self._pending_kpis.get("timestamp", 0.0),
            "cells": cell_actions,
        }
        self.zmq_db.send_control_actions(action_payload)

        next_kpis = self._recv_next_kpis()
        if next_kpis is None:
            # Time-limit end: truncated, not terminated. This used to set both
            # True, and SB3 computes
            #   info["TimeLimit.truncated"] = truncated and not terminated
            # (dummy_vec_env.py), so the flag came out False and SB3 never
            # bootstrapped the terminal value. The value function was trained
            # toward 0 at the end of every episode even though the episode was
            # only cut off by simTime. Running out of configured duration is a
            # time limit, not an MDP terminal state.
            self.terminated = False
            self.truncated = True
            last_kpis = self._pending_kpis
            obs = self._get_obs(last_kpis)
            self._pending_kpis = None
            return obs, 0.0, self.terminated, self.truncated, {"kpis": last_kpis, "cio_offsets": cell_actions}

        self._pending_kpis = next_kpis
        terms = self._reward_terms(next_kpis)
        info = {"kpis": next_kpis, "cio_offsets": cell_actions, "reward_terms": terms}
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

        The ZMQ payload carries SINR PER UE (cells[id]["ues"][imsi]
        ["l3_serving_sinr_db"], see tests/test_server.py), so the cell-level
        number the observation needs is an aggregate over that dict -- the mean
        by default, hence the field name sinr_db_mean; `sinr_agg` can switch it
        to min/p10. A cell-level "sinr_db_mean" sent directly by ns-3 is used
        as-is if it ever appears in the payload.
        """
        if cell_kpis.get("sinr_db_mean") is not None:
            return float(cell_kpis["sinr_db_mean"])
        return self._aggregate_sinr(cell_kpis.get("ues", {}))

    def _get_obs(self, kpis: dict) -> np.ndarray:
        """Flat float32 vector matching observation_space: one OBS_FIELDS block
        per cell, cells in the fixed CELLS order (2..8).

        A cell absent from the snapshot contributes an all-zero block. A cell
        that IS present but reports no SINR (no UEs attached, or no L3 report
        yet) gets SINR_FLOOR_DB rather than 0.0 -- 0 dB is a perfectly good
        link, so zero-filling it would read as a healthy cell instead of an
        empty one. Values are clipped into the declared bounds so the
        observation is always inside observation_space even if a KPI overshoots
        (e.g. prb_utilization slightly above 1.0).
        """
        cells = (kpis or {}).get("cells", {})
        row = np.zeros(self.observation_space.shape, dtype=np.float32)
        ue_scale = float(self.total_ues) if self.total_ues else 1.0
        for i, cell in enumerate(CELLS):
            cell_kpis = cells.get(str(cell)) or {}
            sinr = self._cell_sinr_mean(cell_kpis) if cell_kpis else None
            # An absent cell used to leave the whole block at 0.0, which put
            # 0 dB, a perfectly good link, in the SINR slot. Absent and
            # present-but-silent now both get the floor.
            row[i * len(OBS_FIELDS):(i + 1) * len(OBS_FIELDS)] = (
                cell_kpis.get("prb_utilization", 0.0),
                cell_kpis.get("num_active_ues", 0) / ue_scale,
                cell_kpis.get("buffer_bytes", 0.0) / BUFFER_SCALE_BYTES,
                cell_kpis.get("volume_bytes", 0.0) / VOLUME_SCALE_BYTES,
                (SINR_FLOOR_DB if sinr is None else sinr) / SINR_SCALE_DB,
            )
        return np.clip(row, OBS_LOW, OBS_HIGH).astype(np.float32)

    @staticmethod
    def _pdcp_totals(per_cell):
        """(total_bytes, window_s, field_present, per_imsi) from the PDCP accumulator.

        One parse shared by Backlog (drain rate) and Satisfaction (delivered vs
        demanded). That is the point of keeping it in one place: these two terms
        used to read delivery from two different sources over two different
        windows, which is what FIX 13 removed.

        Keyed by IMSI and summed across cells, not maxed. mmwave-helper.cc:2206
        assigns one shared MmWaveBearerStatsCalculator to every device, but each
        cell drains only the bytes it banked, so a UE that moved mid-period has a
        genuine partial contribution in two cells. The old raw per-UE field was a
        global counter read redundantly, which is why that one had to be deduped
        by max instead. Opposite rule, same-looking data.
        """
        delivered_by_imsi = {}
        window_s = 0.0
        field_present = False
        for cell_kpis in per_cell:
            drained = cell_kpis.get("pdcp_delivered_bytes")
            if drained is None:
                continue                       # cell absent, or pre-accumulator ns-3
            field_present = True               # key present, even if empty
            window_s = max(window_s, float(cell_kpis.get("pdcp_window_s") or 0.0))
            for imsi_str, value in drained.items():
                try:
                    imsi = int(imsi_str)
                except (TypeError, ValueError):
                    continue
                delivered_by_imsi[imsi] = delivered_by_imsi.get(imsi, 0.0) + float(value)
        return (float(sum(delivered_by_imsi.values())), window_s,
                field_present, delivered_by_imsi)

    def _reward_terms(self, kpis: dict) -> dict:
        """The five-term MLB reward, plus its breakdown and diagnostics.

            reward = w_balance * Balance
                   + w_satisfaction * Satisfaction
                   - w_backlog * Backlog
                   - w_badsignal * BadSignal
                   - w_pingpong * PingPong

        Satisfaction and PingPong are dropped from the sum rather than scored 0
        when the payload cannot supply them, so a missing measurement is never
        confused with a measured zero. Every term is scale-free, so none of them
        drags the scenario's byte or count magnitudes into the reward.

        Backlog, seconds of delay (mlb_reward.py's TERM 3, online form):

            backlog_sec = sum_cells buffer_bytes
                          / (sum_cells volume_bytes / control_period_s)

        Network-level rather than per-cell: the per-cell ratio has a heavy tail,
        because a nearly-idle cell with a non-empty buffer divides by almost
        nothing. control_period_s is the 0.1 s E2 indication window the byte
        counters accumulate over, the same window mlb_reward.KPM_PERIOD_S
        assumes, not the T_control step period. See KPI_WINDOW_S. If nothing was
        delivered this step the term is 0.0 rather than inf: no traffic means no
        queueing delay to charge for.

        Balance, Jain's index over per-cell congestion (TERM 1, 'jain_guarded'):

            CI_i = prb_utilization_i * (n_ues_i / sum_j n_ues_j)
            Balance = (sum CI)^2 / (N * sum CI^2)

        One deliberate difference from the offline term: there is no
        `* 1[median_mcs_i > tau]` gate. Median MCS is not in the ZMQ payload (it
        comes from the du table's per-UE MCS rows offline), so the gate cannot be
        evaluated live. A cell carrying UEs on an unusably low MCS therefore
        still counts as congested here where the offline term would zero it out.
        Restore the gate if a per-cell MCS field is ever added to the payload.

        The n_ues denominator is this step's network total rather than the
        offline TOTAL_UES constant. Jain's index is invariant under a constant
        rescaling of the whole CI vector, so this changes no value.

        Low-load guard: an idle network is maximally unbalanced by Jain (one
        trickling cell and six zeros scores 1/7) even though there is nothing
        worth balancing. So if no UEs are attached, or sum(CI) <= 0, or the
        busiest cell is under low_load_prb_threshold of its PRBs, Balance is
        BALANCE_GUARD_VALUE and balance_guarded is set, which keeps it distinct
        from a genuine index that happens to land on 1.0.
        """
        cells = (kpis or {}).get("cells", {})
        per_cell = [cells.get(str(cell), {}) for cell in CELLS]
        prb = np.array([c.get("prb_utilization", 0.0) or 0.0 for c in per_cell], dtype=float)
        n_ues = np.array([c.get("num_active_ues", 0) or 0 for c in per_cell], dtype=float)
        buffer_bytes = np.array([c.get("buffer_bytes", 0.0) or 0.0 for c in per_cell], dtype=float)
        vol_bytes = np.array([c.get("volume_bytes", 0.0) or 0.0 for c in per_cell], dtype=float)

        # --- Backlog ------------------------------------------------------
        # FIX 13. Backlog is a drain time: queued bytes over the rate they leave
        # at. The denominator used to be volume_bytes / control_period_s, the E2
        # indication volume measured over a 0.1 s window, which is a 10% sample
        # of the control step. On the AM discrimination runs (30 steps, 3 arms)
        # that sampled rate correlated only r = +0.257 with the PDCP full-step
        # delivery Satisfaction uses, with a CV of 16-28% against 8-18%. Backlog
        # was being divided by noise, and the noise was large enough to reorder
        # arms whose true delivery differed by ~8%.
        #
        # The PDCP accumulator covers the whole control step and is the source
        # Satisfaction already trusts, so both terms now describe the same
        # seconds of the same network. Falls back to the old E2 volume when the
        # PDCP field is absent (older ns-3 builds) so behaviour degrades rather
        # than crashing. Scale is unchanged, since a 0.1 s window divided by
        # 0.1 s already estimates the same bytes per second, so the tuned weights
        # carry over and only the variance drops.
        backlog_bytes = float(buffer_bytes.sum())
        pdcp_total_bytes, pdcp_window_s, pdcp_field_present, delivered_by_imsi = \
            self._pdcp_totals(per_cell)
        if pdcp_field_present and pdcp_window_s > 0.0:
            delivered_bytes = pdcp_total_bytes
            delivery_rate_bps = pdcp_total_bytes / pdcp_window_s
            backlog_rate_source = "pdcp"
        else:
            delivered_bytes = float(vol_bytes.sum())
            delivery_rate_bps = delivered_bytes / self.control_period_s
            backlog_rate_source = "e2_volume"
        backlog_denominator_zero = delivery_rate_bps <= 0.0
        backlog = 0.0 if backlog_denominator_zero else backlog_bytes / delivery_rate_bps

        # --- Balance ------------------------------------------------------
        active_ues = float(n_ues.sum())
        ci = prb * (n_ues / active_ues) if active_ues > 0.0 else np.zeros_like(prb)
        max_prb = float(prb.max()) if prb.size else 0.0
        balance_guarded = (active_ues <= 0.0 or float(ci.sum()) <= 0.0
                           or max_prb < self.low_load_prb_threshold)
        balance = BALANCE_GUARD_VALUE if balance_guarded else _jain(ci)

        # --- BadSignal ---------------------------------------------------
        # Fraction of DL transmissions at <= -6 dB SINR, network-level: the
        # <=-6 dB bin summed over cells, divided by all 7 bins summed over cells.
        # Same aggregation as analysis/mlb_reward.py:799-805, so live and offline
        # scores are comparable. No transmissions at all gives 0.0, never NaN.
        bins = np.array([_sinr_bins(c) for c in per_cell], dtype=float)
        badsignal_bad = float(bins[:, 0].sum())
        badsignal_total_tx = float(bins.sum())
        badsignal_denominator_zero = badsignal_total_tx <= 0.0
        badsignal = 0.0 if badsignal_denominator_zero else badsignal_bad / badsignal_total_tx

        # --- Satisfaction --------------------------------------------------
        # Delivered UDP DL throughput as a fraction of what those UEs asked for:
        #     sum_over_UDP_UEs(delivered_kbits) / pdcp_window_s
        #     / (n_udp * udp_ue_rate_kbps)
        #
        # Three different zeros are kept distinct, because zero is the maximum
        # penalty here and a zero meaning "not measured" would train the agent on
        # a lie:
        #   1. UE present, delivered 0 bytes. Counts 0 in the numerator and stays
        #      in the run-level denominator. A real failure, penalised.
        #   2. UE absent from the payload (dropped or handed over). Also 0 in the
        #      numerator, and the denominator is still run-level n_udp, so it is
        #      penalised without letting the denominator shrink.
        #   3. Field absent everywhere (calculator did not resolve, or an older
        #      ns-3 build). satisfaction is None, flagged satisfaction_unavailable
        #      and dropped from the reward sum rather than contributing 0.
        #
        # delivered_by_imsi, pdcp_window_s and pdcp_field_present all come from
        # the single _pdcp_totals() call in the Backlog block above, so the two
        # terms cannot end up describing different windows. See that docstring
        # for why the per-cell values are summed rather than maxed.
        satisfaction_unavailable = (not pdcp_field_present) or pdcp_window_s <= 0.0
        if satisfaction_unavailable or self.n_udp <= 0:
            satisfaction = None
            delivered_udp_bytes = 0.0
            satisfaction_demand_kbps = 0.0
        else:
            delivered_udp_bytes = float(sum(delivered_by_imsi.get(i, 0.0) for i in self.udp_imsis))
            # The window comes from ns-3 rather than being assumed. It is
            # T_control, not the 0.1 s E2 window volume_bytes uses, and it stays
            # correct if T_control is retuned.
            delivered_kbps = (delivered_udp_bytes * 8.0 / 1e3) / pdcp_window_s
            satisfaction_demand_kbps = self.n_udp * self.udp_ue_rate_kbps
            satisfaction = float(np.clip(delivered_kbps / satisfaction_demand_kbps,
                                         0.0, SATISFACTION_CLIP_MAX))

        # --- PingPong ------------------------------------------------------
        raw_handovers = (kpis or {}).get("handovers")
        pingpong_unavailable = raw_handovers is None
        handovers_this_step = []
        if not pingpong_unavailable:
            for h in raw_handovers:
                try:
                    handovers_this_step.append((float(h["t"]), int(h["imsi"]),
                                                int(h["src"]), int(h["dst"])))
                except (KeyError, TypeError, ValueError):
                    continue
        n_handovers = len(handovers_this_step)

        if pingpong_unavailable:
            pingpong = None
            pingpong_count = 0
        else:
            self._handover_history.extend(handovers_this_step)
            if self._handover_history:
                newest = max(t for t, *_ in self._handover_history)
                # 2*Y is the longest span that can still form a pair.
                self._handover_history = [e for e in self._handover_history
                                          if newest - e[0] <= 2.0 * PINGPONG_Y_S]
            pingpong_count = 0
            for t, imsi, src, dst in handovers_this_step:
                # A return leg: the same UE went dst to src earlier, within Y.
                for tp, ip, sp, dp in self._handover_history:
                    if ip != imsi or tp >= t:
                        continue
                    if t - tp <= PINGPONG_Y_S and sp == dst and dp == src:
                        pingpong_count += 1
                        break
            pingpong = (pingpong_count / active_ues) if active_ues > 0 else 0.0

        reward = (self.w_balance * balance
                  - self.w_backlog * backlog
                  - self.w_badsignal * badsignal)
        if satisfaction is not None:
            reward += self.w_satisfaction * satisfaction
        if pingpong is not None:
            reward -= self.w_pingpong * pingpong
        return {
            "reward": float(reward),
            "balance": float(balance),
            "backlog": float(backlog),
            "badsignal": float(badsignal),
            "satisfaction": None if satisfaction is None else float(satisfaction),
            "pingpong": None if pingpong is None else float(pingpong),
            "weights": {"balance": self.w_balance, "backlog": self.w_backlog,
                        "badsignal": self.w_badsignal,
                        "satisfaction": self.w_satisfaction,
                        "pingpong": self.w_pingpong},
            "diagnostics": {
                "ci": dict(zip(CELLS, ci.round(6))),
                "prb_utilization": dict(zip(CELLS, prb)),
                "n_ues": dict(zip(CELLS, n_ues)),
                "active_ues": active_ues,
                "max_prb_utilization": max_prb,
                "backlog_bytes": backlog_bytes,
                "delivered_bytes": delivered_bytes,
                "delivery_rate_bytes_per_s": delivery_rate_bps,
                "control_period_s": self.control_period_s,
                "balance_guarded": bool(balance_guarded),
                "backlog_denominator_zero": bool(backlog_denominator_zero),
                "backlog_rate_source": backlog_rate_source,
                "badsignal_bad_count": badsignal_bad,
                "badsignal_total_tx": badsignal_total_tx,
                "badsignal_denominator_zero": bool(badsignal_denominator_zero),
                "sinr_bins": {c: list(bins[i]) for i, c in enumerate(CELLS)},
                "satisfaction_unavailable": bool(satisfaction_unavailable),
                "udp_imsis": list(self.udp_imsis),
                "n_udp": self.n_udp,
                "delivered_udp_bytes": delivered_udp_bytes,
                "pdcp_window_s": pdcp_window_s,
                "pdcp_field_present": bool(pdcp_field_present),
                "satisfaction_demand_kbps": satisfaction_demand_kbps,
                "udp_ue_rate_kbps": self.udp_ue_rate_kbps,
                "delivered_by_imsi": dict(sorted(delivered_by_imsi.items())),
                "low_load_prb_threshold": self.low_load_prb_threshold,
                "pingpong_unavailable": bool(pingpong_unavailable),
                "pingpong_count": pingpong_count,
                "handovers_this_step": n_handovers,
                "handover_history_len": len(self._handover_history),
                "pingpong_y_s": PINGPONG_Y_S,
                # No median-MCS gate on CI: median MCS is not in the live
                # payload. See the Balance notes in this method's docstring.
                "median_mcs_gate_applied": False,
                # Built from what is actually in the sum, so a dropped term is
                # never mistaken for a term that scored zero.
                "terms_present": tuple(
                    ["balance", "backlog", "badsignal"]
                    + (["satisfaction"] if satisfaction is not None else [])
                    + (["pingpong"] if pingpong is not None else [])),
            },
        }

    def _compute_reward(self, kpis: dict) -> float:
        return self._reward_terms(kpis)["reward"]

    def _purge_sim_path(self):
        """Delete this env's own run directory, if purging is enabled.

        Deliberately conservative. This is an rmtree driven by an attribute, so
        it refuses anything it did not clearly create in start_sim(): the path
        must exist, be a real directory rather than a symlink, sit directly
        inside output_folder, and carry the uuid4 name start_sim() generated.
        Anything else is left on disk rather than guessed at.

        Ordering hazard: close() runs this, so anything writing into sim_path
        must do so before close(). examples/cio_zmq_experiment.py calls
        env.close() and only then saves its figures into env.sim_path, which is
        why purging defaults to off.
        """
        if not self.purge_sim_path_on_close:
            return
        sim_path = getattr(self, 'sim_path', None)
        if not sim_path:
            return

        target = os.path.realpath(sim_path)
        parent = os.path.realpath(self.output_folder)
        if os.path.dirname(target) != parent:
            return                                  # not directly in our output folder
        if os.path.islink(sim_path) or not os.path.isdir(target):
            return
        try:
            uuid.UUID(os.path.basename(target))     # only a run dir we named
        except ValueError:
            return

        shutil.rmtree(target, ignore_errors=True)

    def close(self):
        if self.is_open:
            if self.zmq_db is not None:
                self.zmq_db.close()
                self.zmq_db = None
            self.sim_process.kill()
            # Reap it: close() runs on every reset(), so without a wait() a
            # long training run leaves one zombie per episode.
            try:
                self.sim_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            self.is_open = False
            self._purge_sim_path()

    # NsOranEnv's file/datalake hooks are unused by this ZMQ-native env.
    def _compute_action(self, action):
        raise NotImplementedError('MlbZmqEnv builds its action payload directly in step()')

    def _init_datalake_usecase(self):
        pass

    def _fill_datalake_usecase(self):
        pass
