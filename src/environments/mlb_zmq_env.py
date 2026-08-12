import os
import selectors
import subprocess
import time
import uuid
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from nsoran.ns_env import NsOranEnv
from ns_o_ran_gym.bridge.zmq_database import ZmqStateDatabase

# Real NR cells in scenario-marl-zmq.cc's default topology (basicCellId=1 is
# the LTE anchor; mmWave eNBs get cell ids 2..8) -- mirrors
# tests/test_server.py's EXPECTED_CELLS.
CELLS = [2, 3, 4, 5, 6, 7, 8]

DEFAULT_WEIGHTS = {"prb": 0.4, "ues": 0.2, "vol": 0.2, "qual": 0.2}


class MlbZmqEnv(NsOranEnv):
    """MLB/CIO Gym environment for scenario-marl-zmq, driven purely over ZeroMQ.

    This has nothing to do with EnergySavingEnv/es_env.py: that class is the
    separate ES sleep-mode (heuristicType/bsOn/.../bsOff) use case, which
    scenario-marl-zmq.cc runs with heuristicType=-1 (inert) by default. This
    class targets the CIO/MLB lever instead.

    scenario-marl-zmq.cc's MarlControlStep (scheduled every T_control = 1.0 s,
    hardcoded in the .cc for now) sends one KPI snapshot per real cell over a ZeroMQ
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

    Observation/reward port analysis/load_metric.py's compute_load() formula
    to run online, per step, directly off the JSON snapshot instead of an
    offline whole-run dataframe. Two adaptations were necessary:
      - TOTAL_UES is derived from scenario_configuration['ues'] * len(CELLS)
        instead of load_metric.py's hardcoded 21 (that constant assumed the
        ues=3 config from cio_experiment.py/scenario-three specifically).
      - VOL_P95 (the volume-normalization denominator) can't be the whole-run
        95th percentile load_metric.py computes offline -- a live env doesn't
        have the full run yet. It's instead a running 95th percentile over
        ZmqStateDatabase's buffered KPI history (history_maxlen snapshots),
        which is inherently noisier early in an episode; widen history_maxlen
        if that instability matters for your use.
    """

    def __init__(self, ns3_path: str, scenario_configuration: dict, output_folder: str,
                 optimized: bool = False, zmq_port: int = 5555, history_maxlen: int = 10,
                 sinr_agg: str = "min", weights: Optional[dict] = None,
                 recv_timeout_ms: int = 2000):
        # NsOranEnv.__init__ -> setup_sim() builds ns-3 and resolves
        # self.script_executable; that's the only part of the base class this
        # env reuses. control_header/log_file/control_file are required by
        # the signature but otherwise unused: start_sim()/reset()/step()/
        # close() are fully overridden below to bypass ActionController and
        # the SQLite datalake.
        super().__init__(ns3_path=ns3_path, scenario='scenario-marl-zmq',
                          scenario_configuration=scenario_configuration,
                          output_folder=output_folder, optimized=optimized,
                          control_header=['timestamp', 'cellId', 'cioDb'],
                          log_file='unused', control_file='unused')

        assert sinr_agg in ("min", "p10", "mean"), sinr_agg
        self.zmq_port = zmq_port
        self.history_maxlen = history_maxlen
        self.sinr_agg = sinr_agg
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.recv_timeout_ms = recv_timeout_ms
        self.total_ues = int(self.scenario_configuration['ues']) * len(CELLS)

        self.zmq_db: Optional[ZmqStateDatabase] = None
        self._pending_kpis: Optional[dict] = None

    def start_sim(self):
        """ZeroMQ counterpart of NsOranEnv.start_sim(): binds the KPI/action
        bridge, then launches scenario-marl-zmq -- no semaphores, no
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

    def reset(self, *, seed: int = None, options: dict = None):
        gym.Env.reset(self, seed=seed)
        self.close()
        self.start_sim()

        self.terminated = False
        self.truncated = False
        self._pending_kpis = self._recv_next_kpis()
        if self._pending_kpis is None:
            # ns-3 exited before its first control step.
            self.terminated = True
            self.truncated = True
            return self._zero_obs(), {}

        return self._get_obs(self._pending_kpis), {"kpis": self._pending_kpis}

    def step(self, action):
        # action is currently unused: the CIO reply is computed straight from
        # the pending KPI snapshot's prb_utilization (test_server.py's
        # heuristic), not from the caller-supplied action. Kept in the
        # signature for Gym API compatibility and for a future trained policy
        # to take over this computation.
        # THE action WILL BE COMPUTED BY THE AGENT AND LATER THE COORDINATOR/
        # AGGREGATOR AND PUSHED TO THE SIMULATOR
        if self._pending_kpis is None:
            return self._zero_obs(), 0.0, self.terminated, self.truncated, {}

        # Reply to the KPI snapshot obs was last computed from -- this is the
        # action ns-3's MarlControlStep/StepSync is blocked waiting for.
        #
        # Send back control actions for every real cell, in the same "cells" shape
        # as the KPI payload. LteEnbNetDevice::ApplyControlPayload now reads
        # actionPayload["cells"][cellId]["cio_offset"] for each cell present.
        #
        # This is a placeholder load-balancing heuristic, not a trained MARL
        # policy: it just proves the round-trip works by reacting to real PRB
        # utilization -- push UEs away from an overloaded cell (negative CIO)
        # and make an underloaded cell more attractive (positive CIO).
        cell_actions = {}
        for cell_id, cell_kpis in self._pending_kpis.get("cells", {}).items():
            prb_utilization = cell_kpis.get("prb_utilization", 0.5)
            cio_offset = max(-6.0, min(6.0, -12.0 * (prb_utilization - 0.5)))
            cell_actions[cell_id] = {"cio_offset": cio_offset}

        action_payload = {
            "timestamp": self._pending_kpis.get("timestamp", 0.0),
            "cells": cell_actions,
        }
        self.zmq_db.send_control_actions(action_payload)

        next_kpis = self._recv_next_kpis()
        if next_kpis is None:
            # Time-limit end: same terminated/truncated convention as
            # NsOranEnv.is_simulation_over()'s return_code==0 branch.
            self.terminated = True
            self.truncated = True
            last_kpis = self._pending_kpis
            obs = self._get_obs(last_kpis)
            self._pending_kpis = None
            return obs, 0.0, self.terminated, self.truncated, {"kpis": last_kpis, "cio_offsets": cell_actions}

        self._pending_kpis = next_kpis
        info = {"kpis": next_kpis, "cio_offsets": cell_actions}
        return self._get_obs(next_kpis), self._compute_reward(next_kpis), False, False, info

    def _zero_obs(self):
        return [tuple(0.0 for _ in range(len(CELLS) * 4))]

    def _cell_loads(self, kpis: dict) -> dict:
        """Per-cell load components/score for one live KPI snapshot -- the
        online counterpart of analysis/load_metric.py's compute_load(),
        which runs the same formula over a whole completed run.
        """
        cells = kpis.get("cells", {})
        vol_p95 = self._running_vol_p95()
        w = self.weights

        loads = {}
        for cell in CELLS:
            cell_kpis = cells.get(str(cell), {})
            prb_pct = cell_kpis.get("prb_utilization", 0.0) * 100.0
            n_ues = cell_kpis.get("num_active_ues", 0)
            vol_bytes = cell_kpis.get("volume_bytes", 0.0)
            buffer_bytes = cell_kpis.get("buffer_bytes", 0.0)
            sinr = self._aggregate_sinr(cell_kpis.get("ues", {}))

            c_prb = prb_pct / 100.0
            c_ues = n_ues / self.total_ues if self.total_ues else 0.0
            c_vol = min(max(vol_bytes / vol_p95, 0.0), 1.0) if vol_p95 > 0 else 0.0
            if n_ues == 0 or sinr is None:
                c_qual = 0.0
            else:
                c_qual = min(max((30.0 - sinr) / 35.0, 0.0), 1.0)

            load = w["prb"] * c_prb + w["ues"] * c_ues + w["vol"] * c_vol + w["qual"] * c_qual
            loads[cell] = {
                "prb_pct": prb_pct, "n_ues": n_ues, "vol_bytes": vol_bytes,
                "buffer_bytes": buffer_bytes, "sinr": sinr,
                "c_prb": c_prb, "c_ues": c_ues, "c_vol": c_vol, "c_qual": c_qual,
                "load": load,
            }
        return loads

    def _aggregate_sinr(self, ues: dict) -> Optional[float]:
        sinrs = [u.get("l3_serving_sinr_db") for u in ues.values() if u.get("l3_serving_sinr_db") is not None]
        if not sinrs:
            return None
        if self.sinr_agg == "min":
            return min(sinrs)
        if self.sinr_agg == "mean":
            return sum(sinrs) / len(sinrs)
        return float(np.quantile(sinrs, 0.10))

    def _running_vol_p95(self) -> float:
        """95th percentile of vol_bytes across all cells in the buffered KPI
        history. Substitutes for load_metric.py's whole-run VOL_P95, which
        isn't available mid-episode. Falls back to 1.0 (so c_vol reads 0
        while vol_bytes is 0) until at least one snapshot has been buffered.
        """
        samples = [
            cell_kpis.get("volume_bytes", 0.0)
            for snapshot in self.zmq_db.kpi_history
            for cell_kpis in snapshot.get("cells", {}).values()
        ]
        if not samples:
            return 1.0
        return float(np.quantile(samples, 0.95)) or 1.0

    def _get_obs(self, kpis: dict):
        loads = self._cell_loads(kpis)
        row = []
        for cell in CELLS:
            c = loads[cell]
            row.extend([c["prb_pct"], c["n_ues"], c["vol_bytes"], c["sinr"] if c["sinr"] is not None else 0.0])
        return [tuple(row)]

    def _compute_reward(self, kpis: dict) -> float:
        # MLB objective: keep cells balanced -- penalize spread in load.
        loads = self._cell_loads(kpis)
        values = np.array([c["load"] for c in loads.values()])
        return float(-values.std())

    def close(self):
        if self.is_open:
            if self.zmq_db is not None:
                self.zmq_db.close()
                self.zmq_db = None
            self.sim_process.kill()
            self.is_open = False

    # NsOranEnv's file/datalake hooks are unused by this ZMQ-native env.
    def _compute_action(self, action):
        raise NotImplementedError('MlbZmqEnv builds its action payload directly in step()')

    def _init_datalake_usecase(self):
        pass

    def _fill_datalake_usecase(self):
        pass
