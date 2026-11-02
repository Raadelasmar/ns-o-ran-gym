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
#   last_pair_margin_db -- mean |offset| over every directed cell-pair margin
#                        currently applied that touches this cell (as serving
#                        OR candidate), so the policy can see roughly how much
#                        correction is active nearby. v3 replaces the old
#                        per-cell last_margin_db (a single signed scalar per
#                        cell) with this: under the pairwise action (see
#                        PairMarginState) a cell can be touched by several
#                        directed-pair offsets of different signs at once, so
#                        there is no single signed "this cell's margin" left
#                        to report -- this is a magnitude proxy instead,
#                        always >= 0 (also feeds the ActionSmoothness
#                        diagnostic -- not reward-weighted, see
#                        REWARD_FORMULA_VERSION)
# prb_utilization is deliberately NOT included: it is MLB's primary decision
# variable, not MRO's, and the coordinator-blueprint design keeps each agent's
# observation scoped to its own function (MobilityScore) rather than pulling in
# LoadScore inputs it does not act on.
OBS_FIELDS = ("sinr_db_mean", "num_active_ues", "badsignal_frac",
              "handover_activity", "last_pair_margin_db")

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

OBS_LOW_PER_CELL = np.array([SINR_FLOOR_DB / SINR_SCALE_DB, 0.0, 0.0, 0.0, 0.0],
                            dtype=np.float32)
OBS_HIGH_PER_CELL = np.array([SINR_CEIL_DB / SINR_SCALE_DB, 1.0, 1.0,
                              HANDOVER_ACTIVITY_CEIL, 1.0], dtype=np.float32)
OBS_LOW = np.tile(OBS_LOW_PER_CELL, len(CELLS))
OBS_HIGH = np.tile(OBS_HIGH_PER_CELL, len(CELLS))

# Ceiling on any single directed cell-pair handover-margin OFFSET in dB,
# added to (not replacing) the network's base HoSinrDifference for that
# (serving, candidate) pair -- see PairMarginState. +-3 dB is a conservative
# starting bound -- unlike CIO's +-6 dB, this has not yet been validated
# against a live controller and should be revisited once it has, the same way
# CIO's own limit was locked from early placeholder-heuristic use rather than
# derived from first principles. v2's action_space bound (a flat per-cell
# margin) used this same constant directly; v3's action_space bounds the
# AGENT'S own additive correction instead (see ROLE_DELTA_MAX_DB) -- this
# constant now only bounds the final, combined, per-pair offset actually
# applied.
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

# Reference window, in seconds, for PingPong AND HandoverRate. Shared by both
# rather than two separately-named constants: they count the same underlying
# event type (handovers) over the same ns-3-reported window, and are meant to
# mean "activity per UE per reference second" -- keeping them on one constant
# is what stops the two from drifting out of sync if this is ever retuned.
#
# WHY THIS EXISTS. c576d6c (MlbZmqEnv) found that MLB's own PingPong
# (`pingpong_count / active_ues`) is a COUNT PER CONTROL STEP in disguise, not
# a rate: every other term there is a level or ratio (a Jain index, kbps/kbps,
# a bin fraction, seconds of drain time) and does not move if the control
# period changes, but a per-step count silently rescales with it. That is not
# a scaling nuisance, it is a ranking bug -- see c576d6c's own commit message
# for the reward-inversion example it found.
#
# MRO's PingPong was copied from MLB's PRE-FIX code and has the identical
# bug. HandoverRate (`n_handovers / active_ues`, MRO-only, added after
# c576d6c landed) is structurally the same shape and had never been checked
# for this at all until docs/mro_training_fixes.md flagged it. Quality and
# OutageExposure are pure ratios/fractions (good/measured, UEs-in-outage/
# UEs-reporting) and are immune, same as MLB's non-PingPong terms.
#
# REF = 1.0 s because that is the control period every baseline measured in
# docs/mro_training_fixes.md ran at. At that period the scale factor is
# HANDOVER_RATE_REF_S / handover_window_s = 1.0 / 1.0 = exactly 1.0, an
# IEEE-754 exact multiply, so this fix changes nothing about the round-1
# baseline numbers already on record -- it only matters once controlInterval
# is ever set to something other than 1.0 (e.g. the colleague's locked SAC
# config, which runs at 0.1).
HANDOVER_RATE_REF_S = 1.0

# Floor on handover_window_s before it becomes a divisor, and the resulting
# cap on handover_rate_scale. Found during the boundedness audit prompted by
# 20260915_Dani_RewardWeightMethodologyReport.pdf Section 5.1: the scale
# factor guards against handover_window_s <= 0.0 (falls back to REF) but NOT
# against a small positive value ns-3 might report near an episode boundary
# or a partial step -- an unguarded window of e.g. 0.001 s would inflate
# PingPong/HandoverRate 1000x, an unbounded term neither weight nor training
# could recover from. Floored at 1% of REF (100x cap) rather than something
# tighter: the existing rate-invariance test
# (test_pingpong_and_handover_rate_are_rates_not_per_step_counts) exercises a
# genuine 10x case at HANDOVER_RATE_REF_S/10, which must keep passing
# unchanged, so the floor sits a full decade below that.
HANDOVER_WINDOW_FLOOR_S = HANDOVER_RATE_REF_S / 100.0

# Which formula produced a given reward, stamped into every diagnostics
# block. Bump whenever a term's FORMULA changes, not when a weight is
# retuned (weights are already recorded per run under "weights"). Mirrors
# MlbZmqEnv.REWARD_FORMULA_VERSION -- see c576d6c and its own Step 10i
# postscript for why comparing a raw reward field across a formula change,
# without this stamp, can look like a real physical effect when it is not.
# v1 -> v2: ActionSmoothness removed from the sum (see DEFAULT_W_OUTAGE's
# neighbouring comment for why) -- any reward measured under v1 is not
# comparable to one measured under v2.
REWARD_FORMULA_VERSION = 2

# Reward weights. Re-derived (3rd time) from a 5-seed x 3-arm, simTime=30,
# 1,495-steps/arm do-nothing baseline measured AFTER the lte-enb-rrc.cc
# TttBasedHandover wiring fix (the `sinrDifference > GetEffectiveSinrThreshold
# (maxSinrCellId)` term added to the primary trigger condition -- see
# mro_investigation_log.md's [01:35]-[06:35] entries for the full trace and
# validation). These weights are only valid together with that fix: it
# changes the do-nothing baseline's own term distributions (restoring the
# 3GPP-standard 3dB base hysteresis that TttBasedHandover's primary branch had
# been skipping), not only MRO's controllable range, so every prior weight
# measurement in this file's git history was taken under the un-wired
# baseline and is stale. Same detrended-sd equalisation method as always,
# quality anchored at 1.0. Do NOT re-derive these by re-running the
# equalisation formula blindly -- OUTAGE needed a documented deviation from
# the literal measured value; see below.
#
# QUALITY / PINGPONG / CHURN: the literal equalised values under the fixed
# wiring. The trajectory-ranking check (examples/mro_weight_ranking_check.py)
# confirmed both the tighten (+3dB) and loosen (-3dB) margin arms now rank
# worse than do-nothing under these weights -- loosen's margin is large and
# decisive (-0.232 vs +0.153), not a marginal pass; it had ranked BETTER than
# do-nothing (wrongly) under the pre-fix weights below it in this file's
# history, which is exactly what the wiring fix was meant to correct.
DEFAULT_W_QUALITY = 1.00
DEFAULT_W_PINGPONG = 9.39
DEFAULT_W_CHURN = 1.21

# OUTAGE: the literal equalised value is 159.7 -- NOT usable. OutageExposure
# fired on only 0.01% of UE-steps in round 2; matching its variance to
# HandoverQuality's would make the reward swing enormously on the rare step
# it fires, the exact same failure shape MLB's own BadSignal term hit (its
# own literal equalised value needed w~30 and was deliberately never used).
# 20260915_Dani_RewardWeightMethodologyReport.pdf's stated rule for exactly
# this case (Section 5, "rare terms"): a fixed, modest, NON-equalised guard
# weight instead of the literal variance-matched one. Set to 1.0, matching
# BadSignal's own MLB precedent and sitting inside the other MRO weights'
# natural range (0.80-1.51) rather than the 160x-scaled outlier.
#
# UPDATE: the recommended adversarial test has now been run for real
# (examples/mro_outage_stress_experiment.py, fixed +3 dB "sticky" margin on
# every cell -- a well-grounded stress, not an arbitrary one: a MORE
# restrictive margin makes a UE linger on a degrading serving cell longer,
# the textbook too-late-handover mechanism that should drive real outage).
# 10 seeds, pooled (examples/mro_outage_stress_pool.py), 2,990 steps at the
# same locked config: OutageExposure's firing rate rose from 0.01% (round-2
# do-nothing) to 0.234% -- ~20x more frequent, confirming the mechanism is
# real -- but its literal equalised weight is STILL unusable (198.7, in the
# same 160-215 range as every measurement of this term so far, do-nothing or
# stressed). The fixed-guard decision was therefore correct, now backed by a
# real stress measurement instead of only an analogy to BadSignal.
#
# Also validated: examples/mro_weight_ranking_check.py confirmed this full
# weight set (quality/pingpong/churn above, this guard) ranks the sticky arm
# WORSE than do-nothing, as a real MRO failure mode must (mean reward 0.1232
# vs 0.1628) -- a real but modest per-step margin (well under 0.1 sd), not a
# dramatic one; closes the methodology report's Section 5.3 trajectory-
# ranking gap for this term rather than leaving it unchecked.
DEFAULT_W_OUTAGE = 1.0

# SMOOTH: dropped from the reward sum entirely (was DEFAULT_W_SMOOTH=0.086,
# measured from an untrained SAC policy's action distribution -- see git
# history on this block for that derivation). The first full MRO SAC pilot
# (docs/mro_training_fixes.md) showed why that measurement couldn't work as a
# shaping weight: it was calibrated against RANDOM noise, not a learning
# signal, and ActionSmoothness turned out to be the only term easy enough to
# improve that the agent fixated on it -- collapsing smoothness ~5x while
# quality/handover_rate/pingpong/outage never moved. The term also traces to
# an MLB-specific failure mode (an undamped controller oscillating 3 dB/step)
# imported by analogy, never confirmed as an MRO problem of its own.
# ActionSmoothness is still computed and logged every step (see
# _reward_terms's returned dict) -- useful as a diagnostic, just no longer
# part of the sum.


# =============================================================================
# v3: pairwise handover-margin action, 3GPP TR 36.839-style roles
# =============================================================================
# Replaces v2's flat per-cell margin action with a per-DIRECTED-cell-pair
# margin, decided via two roles grounded in 3GPP TR 36.839's MRO failure
# taxonomy (too-late HO, too-early HO, ping-pong, handover-to-wrong-cell) --
# see the 2026-10-25 plan discussion for the literature trail (TR 36.839;
# Weaver & Monogioudis, "Self-Organizing MRO in LTE Networks with eICIC";
# FuzAMP-style fuzzy-MRO combination controllers).
#
#   too-late(s -> c), DIRECTIONAL: a UE stays on serving cell s while
#   candidate c's SINR has been clearly better for a sustained fraction of
#   the window -- lowers margin(s -> c) only.
#
#   churn(a, b), PAIR-LEVEL: cells a and b have returned a UE to each other
#   within PINGPONG_Y_S more than CHURN_RATE_THRESHOLD of the window --
#   raises margin both directions. Merges the literature's too-early-HO and
#   ping-pong categories: current telemetry (handover events as
#   (t, imsi, src, dst), no RLF flag) cannot distinguish an RLF-tagged
#   too-early return from a plain short-stay return. A genuinely new
#   literature category -- handover-to-wrong-cell (fails at the target,
#   re-establishes on a THIRD cell) -- is a neighbor-relation/CIO problem,
#   not a 2-cell margin problem, and is NOT addressed here.
#
# Conflicting signals on the same pair are not resolved by one role
# overriding the other (that is exactly the "race condition" Weaver &
# Monogioudis warn independently-set, uncoordinated biases on a pair cause):
# both contribute a signed, rate-excess-weighted correction to the same
# directed offset, which is then rate-limited and clamped -- see
# _pairwise_margin_targets and PairMarginState.
#
# The agent's action is a 2-dim ADDITIVE correction to a FIXED per-role base
# gain, not the gain itself -- mirrors the manuscript's MLB design
# (b_p = b_p^rule + delta*s_p*c_role): zero-initialising the actor's mean
# output layer (see train_mro_sac_pilot.py) then makes the untrained policy
# reproduce the pure rule-based baseline (BASE_*_GAIN_DB) exactly, the same
# warm-start property MLB's own recipe relies on.
PAIR_RATE_WINDOW_STEPS = 5          # same window length v2a validated for
                                     # its own per-cell rolling rates
TOO_LATE_SINR_MARGIN_DB = 3.0       # candidate must beat serving by this much
                                     # to count as a qualifying sample
# TOO_LATE_RATE_THRESHOLD / CHURN_RATE_THRESHOLD were both 0.3 until
# 2026-10-27, when a direct empirical measurement (examples/
# mro_v3_role_rate_diagnostic.py, examples/mro_v3_sinr_gap_diagnostic.py; 10
# real episodes, ues=5, simTime=10) found 0.3 was structurally unreachable
# for either role under PAIR_RATE_WINDOW_STEPS=5's quantization (rates only
# take values in {0, 0.2, 0.4, ..., 1.0}): too_late_rate was nonzero in 0 of
# 990 observed steps, and churn_rate's only observed nonzero value, across
# 10 seeds, was exactly 0.2 (15 samples, 3/10 seeds). With the 0.3 threshold
# every pair's `excess = rate - threshold` was <= 0.0 everywhere, every step
# -- the entire role-gated margin mechanism (_pairwise_margin_targets) was a
# structural no-op, independent of any agent action (confirmed separately: a
# trained policy's actions produced a per-step trajectory byte-identical to
# the zero-action rule baseline across all tested seeds).
#
# too_late's near-zero firing rate is NOT a threshold problem: the SINR-gap
# diagnostic found a UE is reported under more than one cell's measurement
# (the prerequisite for ANY serving-vs-candidate comparison) in only 11 of
# 34 055 UE-steps (0.03%) under this ues=5 scenario, and even those 11 rarely
# show the candidate ahead (mean gap -6.56 dB, max +0.77 dB, i.e. still well
# short of TOO_LATE_SINR_MARGIN_DB=3.0). Lowering TOO_LATE_RATE_THRESHOLD
# cannot fix a qualifying condition that the KPI stream almost never has the
# raw data to evaluate in the first place; this is a known, documented
# limitation of the role under this scenario's sparse per-UE cell-overlap
# reporting, not a bug. Left at 0.0 (consistent with churn, below) so the
# role is live as designed if its prerequisite condition does fire.
#
# churn's threshold is lowered to 0.0 (full excess = rate on any qualifying
# window), since window=5's coarsest step (0.2, a single ping-pong event in
# the window) is the ONLY value this session ever observed firing -- any
# threshold at or above that is equally unreachable, and splitting hairs
# below it (e.g. 0.1) only shrinks the resulting gain for no benefit.
TOO_LATE_RATE_THRESHOLD = 0.0
CHURN_RATE_THRESHOLD = 0.0
# Starting points, same order of magnitude as v2a's own validated step_db
# (0.5) and dead_band (0.3) -- NOT yet empirically validated for v3, same
# caveat MARGIN_LIMIT_DB's own comment carries.
BASE_TOO_LATE_GAIN_DB = 1.0
BASE_CHURN_GAIN_DB = 1.0
ROLE_DELTA_MAX_DB = 3.0             # agent action_space bound (the additive
                                     # correction, not the combined gain)
PAIR_RATE_LIMIT_DB_PER_STEP = 0.5   # hard rate limiter on the APPLIED
                                     # per-pair offset -- bundled per the
                                     # 2026-10-25 plan (Python side; the ns-3
                                     # SetHandoverMarginOffset clamp to
                                     # [-3, +3] is the C++-side safety net)

ALL_CELL_PAIRS_UNDIRECTED = tuple(
    (CELLS[i], CELLS[j]) for i in range(len(CELLS)) for j in range(i + 1, len(CELLS)))
ALL_CELL_PAIRS_DIRECTED = tuple((a, b) for a in CELLS for b in CELLS if a != b)


def _extract_ue_cell_sinr_matrix(kpis: dict) -> dict:
    """{imsi: {cell: l3_sinr_db}}, every (UE, cell) measurement reported
    anywhere in the payload -- unlike _extract_ue_sinr, nothing is collapsed.

    A UE measured by more than one cell in the same step is the expected
    case near a cell boundary, not an edge case: ns-3 source audit confirms
    LteEnbRrc::m_imsiCellSinrMap is a genuine per-UE candidate-cell map (the
    handover-ranking code scans ALL of a UE's entries in it to find
    maxSinrCellId), and scenario-marl-zmq.cc emits "l3_serving_sinr_db"
    under EVERY cell that has a measurement for that imsi -- the field name
    is legacy from when only the serving cell's value was consumed; it is
    really "this UE's L3 SINR to this cell". _extract_ue_sinr's flat
    {imsi: sinr} dict silently collapses this to whichever cell iterates
    last for a given imsi, which is fine for its own callers (Quality/Outage
    want a single current reading, not a matrix) but wrong for this: the
    too-late diagnostic needs the full per-candidate picture.
    """
    matrix: dict = {}
    for cell_str, cell_kpis in (kpis or {}).get("cells", {}).items():
        try:
            cell = int(cell_str)
        except (TypeError, ValueError):
            continue
        for imsi_str, ue in (cell_kpis.get("ues") or {}).items():
            sinr = ue.get("l3_serving_sinr_db")
            if sinr is None:
                continue
            try:
                imsi = int(imsi_str)
            except (TypeError, ValueError):
                continue
            try:
                matrix.setdefault(imsi, {})[cell] = float(sinr)
            except (TypeError, ValueError):
                continue
    return matrix


class PairRoleTracker:
    """Stateful: rolling per-pair role-rate tracking for MRO v3's two roles.

    One instance per episode (own per-pair windows and best-effort
    UE->serving-cell map reset at episode start, same discipline as
    MroZmqEnv._handover_history).
    """

    def __init__(self, window: int = PAIR_RATE_WINDOW_STEPS):
        self.window = window
        self._serving_cell: dict = {}
        self._too_late_hist = {pair: [] for pair in ALL_CELL_PAIRS_DIRECTED}
        self._churn_hist = {pair: [] for pair in ALL_CELL_PAIRS_UNDIRECTED}

    def _update_serving_cell(self, handovers: list, sinr_matrix: dict) -> None:
        for _, imsi, _src, dst in handovers:
            self._serving_cell[imsi] = dst
        for imsi, cell_sinrs in sinr_matrix.items():
            known = self._serving_cell.get(imsi)
            if known not in cell_sinrs and len(cell_sinrs) == 1:
                # Self-healing best-effort: an unambiguous single-cell
                # report (or no prior knowledge at all) is taken as the
                # serving cell. A UE reported under >1 cell with no prior
                # knowledge is left unresolved for this step rather than
                # guessed -- it will typically resolve on its next handover
                # event or the next unambiguous single-cell snapshot.
                self._serving_cell[imsi] = next(iter(cell_sinrs))

    def update(self, kpis: dict, handovers: list, handover_history: list) -> tuple:
        """Advances every pair's rolling window by one step using THIS
        step's kpis/handovers, and returns the current
        (too_late_rate, churn_rate) dicts -- each pair -> fraction of the
        window with a qualifying sample, INCLUDING this step.
        """
        sinr_matrix = _extract_ue_cell_sinr_matrix(kpis)
        self._update_serving_cell(handovers, sinr_matrix)

        qualifies_too_late = set()
        for imsi, cell_sinrs in sinr_matrix.items():
            serving = self._serving_cell.get(imsi)
            if serving is None or serving not in cell_sinrs:
                continue
            serving_sinr = cell_sinrs[serving]
            for cell, sinr in cell_sinrs.items():
                if cell != serving and sinr - serving_sinr >= TOO_LATE_SINR_MARGIN_DB:
                    qualifies_too_late.add((serving, cell))

        # Identical pairing rule to _reward_terms's own PingPong detector
        # (a return leg for the SAME imsi, within PINGPONG_Y_S, swapping src
        # and dst), just bucketed per undirected pair instead of summed
        # globally. handover_history is the SAME rolling buffer
        # _reward_terms maintains -- see MroZmqEnv._reward_terms for why
        # this method is called from there (ordering: the buffer must
        # already include this step's own handovers for the pairing to see
        # a same-step return leg, exactly like PingPong's own logic).
        qualifies_churn = set()
        for t, imsi, src, dst in handovers:
            for tp, ip, sp, dp in handover_history:
                if ip != imsi or tp >= t:
                    continue
                if t - tp <= PINGPONG_Y_S and sp == dst and dp == src:
                    qualifies_churn.add((src, dst) if src < dst else (dst, src))
                    break

        too_late_rate = {}
        for pair in ALL_CELL_PAIRS_DIRECTED:
            h = self._too_late_hist[pair]
            h.append(pair in qualifies_too_late)
            if len(h) > self.window:
                h.pop(0)
            too_late_rate[pair] = sum(h) / len(h)

        churn_rate = {}
        for pair in ALL_CELL_PAIRS_UNDIRECTED:
            h = self._churn_hist[pair]
            h.append(pair in qualifies_churn)
            if len(h) > self.window:
                h.pop(0)
            churn_rate[pair] = sum(h) / len(h)

        return too_late_rate, churn_rate


def _pairwise_margin_targets(too_late_rate: dict, churn_rate: dict,
                              too_late_gain: float, churn_gain: float) -> dict:
    """{(src, dst): target_db}, pre-rate-limit pre-clip, for every directed
    pair touched by either role this step.

    Both roles contribute to the SAME directed offset when they disagree
    (see the module-level docstring above on why this is additive-and-bounded
    rather than one role overriding the other): churn raises both directions
    of a pair, too-late lowers one specific direction, and a pair hit by both
    nets out to whichever pulls harder rather than either being silently
    discarded.
    """
    targets: dict = {}

    def add(key, delta):
        targets[key] = targets.get(key, 0.0) + delta

    for (a, b), rate in churn_rate.items():
        excess = rate - CHURN_RATE_THRESHOLD
        if excess > 0.0:
            delta = churn_gain * excess
            add((a, b), delta)
            add((b, a), delta)

    for (s, c), rate in too_late_rate.items():
        excess = rate - TOO_LATE_RATE_THRESHOLD
        if excess > 0.0:
            add((s, c), -too_late_gain * excess)

    return targets


class PairMarginState:
    """Stateful: the actually-applied (rate-limited, clamped) per-directed-
    pair margin offset. One instance per episode.

    A pair that converges to exactly 0.0 dB is dropped from internal state
    after being returned once more at that value: ns-3's own
    m_hysteresisMarginOffset map has by then received the explicit 0.0 reset
    (map entries never expire on their own -- see LteEnbRrc::
    SetHandoverMarginOffset), so nothing further needs to be sent for that
    pair, and if it is targeted again later it correctly starts from 0.0
    either way.
    """

    def __init__(self):
        self._offsets: dict = {}

    def offsets_touching(self, cell: int) -> list:
        return [v for (s, d), v in self._offsets.items() if s == cell or d == cell]

    def step(self, targets: dict) -> dict:
        keys = set(self._offsets) | set(targets)
        result = {}
        next_state = {}
        for key in keys:
            cur = self._offsets.get(key, 0.0)
            tgt = targets.get(key, 0.0)
            delta = float(np.clip(tgt - cur, -PAIR_RATE_LIMIT_DB_PER_STEP,
                                  PAIR_RATE_LIMIT_DB_PER_STEP))
            new = float(np.clip(cur + delta, -MARGIN_LIMIT_DB, MARGIN_LIMIT_DB))
            result[key] = new
            if new != 0.0:
                next_state[key] = new
        self._offsets = next_state
        return result


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

    ACTION: v3, PAIRWISE handover-margin offsets.
    -----------------------------------------
    step() sends {"cells": {srcCellId: {"ho_margin_pairs": {dstCellId:
    marginDb, ...}}}} over the same ZeroMQ bridge MLB uses.
    LteEnbNetDevice::ApplyControlPayload (lte-enb-net-device.cc) parses this
    into LteEnbRrc::SetHandoverMarginOffset(servingCellId, targetCellId,
    offsetDb), keyed by the (serving, target) PAIR -- see
    LteEnbRrc::GetEffectiveSinrThreshold. Both the ns-3-side pairwise keying
    and this file's action-shaping were landed together (v3, 2026-10-25); the
    7-cell topology makes GetUeMap()/m_imsiCellSinrMap already report a
    genuine per-candidate-cell SINR matrix per UE (confirmed by source audit,
    not assumed -- see _extract_ue_cell_sinr_matrix), which is what makes the
    too-late role diagnostic possible without any further ns-3 instrumentation.

    SB3-TRAINABLE SURFACE
    ----------------------
    observation_space: Box(35,), float32 -- one 5-field OBS_FIELDS block per
        cell in the fixed CELLS order. See obs_index().
    action_space: Box(-ROLE_DELTA_MAX_DB, +ROLE_DELTA_MAX_DB, (2,)), float32
        -- the agent's ADDITIVE correction to two FIXED per-role base gains
        (BASE_TOO_LATE_GAIN_DB, BASE_CHURN_GAIN_DB), combined then clamped to
        [0, MARGIN_LIMIT_DB] (see _pairwise_payload_from_action). action=0
        reproduces the pure rule-based baseline exactly -- see the v3 module
        docstring above PairRoleTracker for why this is a deliberate
        warm-start property, not a coincidence. The 2 scalars are NOT a
        per-cell margin any more: the env internally expands them into
        per-directed-pair offsets via PairRoleTracker + _pairwise_margin_targets
        + PairMarginState's rate limiter.
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

        self.purge_sim_path_on_close = bool(purge_sim_path_on_close)
        self.vary_rng_run_per_episode = bool(vary_rng_run_per_episode)
        self.reset_max_retries = int(reset_max_retries)
        self.reset_retry_count = 0

        self.observation_space = gym.spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-ROLE_DELTA_MAX_DB, high=ROLE_DELTA_MAX_DB,
                                           shape=(2,), dtype=np.float32)

        # Rolling handover history for cross-step ping-pong/churn detection --
        # same mechanics as MlbZmqEnv._handover_history (a bounce can straddle
        # two control steps, so a single payload is not enough to detect it).
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
        # v3 pairwise-action state: PairRoleTracker computes this step's
        # too-late/churn rates from the snapshot _reward_terms just saw
        # (cached below as self._pair_rates for the NEXT step's action
        # decision -- see _reward_terms and _pairwise_payload_from_action for
        # the full ordering rationale); PairMarginState holds the
        # rate-limited, clamped per-pair offsets actually applied.
        self._pair_tracker = PairRoleTracker()
        self._pair_margin_state = PairMarginState()
        self._pair_rates: tuple = ({}, {})

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
        self._pair_tracker = PairRoleTracker()
        self._pair_margin_state = PairMarginState()
        self._pair_rates = ({}, {})

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

    def _pairwise_payload_from_action(self, action) -> tuple:
        """({srcCellId: {"ho_margin_pairs": {dstCellId: dB}}}, the 2-element
        role-gain array actually applied, {(src,dst): offset_db}) for the
        action reply ns-3 is waiting on.

        action is the agent's ADDITIVE correction to the two fixed base
        gains, clamped to +-ROLE_DELTA_MAX_DB; the combined gain is then
        clamped to [0, MARGIN_LIMIT_DB] before being used as a role gain
        (never negative -- role correction DIRECTION is fixed by the
        literature-grounded role definition, see the v3 module docstring;
        only the magnitude is learned/tunable). Reacts to
        self._pair_rates, the too-late/churn rates computed from the most
        recently RECEIVED snapshot (cached by _reward_terms at the end of
        the previous step) -- same "react to what has been observed so far"
        timing v2c's own window+cadence design used.
        """
        delta = np.clip(np.asarray(action, dtype=np.float32).reshape(-1),
                        -ROLE_DELTA_MAX_DB, ROLE_DELTA_MAX_DB)
        if delta.size != 2:
            raise ValueError(f'action must have 2 entries (too_late_gain_delta, '
                             f'churn_gain_delta), got {delta.size}')
        gains = np.clip(np.array([BASE_TOO_LATE_GAIN_DB, BASE_CHURN_GAIN_DB],
                                 dtype=np.float32) + delta, 0.0, MARGIN_LIMIT_DB)

        too_late_rate, churn_rate = self._pair_rates
        targets = _pairwise_margin_targets(too_late_rate, churn_rate,
                                           float(gains[0]), float(gains[1]))
        offsets = self._pair_margin_state.step(targets)

        cells_payload: dict = {}
        for (src, dst), offset_db in offsets.items():
            cells_payload.setdefault(str(src), {}).setdefault(
                "ho_margin_pairs", {})[str(dst)] = offset_db
        return cells_payload, delta, offsets

    def step(self, action):
        if self._pending_kpis is None:
            return self._zero_obs(), 0.0, self.terminated, self.truncated, {}

        cell_actions, action_delta, pair_offsets = self._pairwise_payload_from_action(action)
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
            # ns3_exit_code / crashed: a clean simTime finish and a crash
            # used to be indistinguishable here -- both land on this exact
            # branch, since _recv_next_kpis() returns None either way. Found
            # live during the OutageExposure stress run: one seed logged a
            # nonzero ns-3 exit yet still produced a full, matching step
            # count -- harmless that time, but nothing in the collected data
            # could have proven that after the fact. is_simulation_over()
            # (called inside the _recv_next_kpis() loop above, via the base
            # class) already computes and stores the real exit code on
            # self.sim_result['meta']['exitcode'] before returning; this
            # just surfaces it instead of discarding it.
            exit_code = self.sim_result.get('meta', {}).get('exitcode')
            info = {"kpis": last_kpis, "margin_offsets": cell_actions,
                    "pair_margins_db": pair_offsets,
                    "ns3_exit_code": exit_code, "crashed": exit_code not in (0, None)}
            return obs, 0.0, self.terminated, self.truncated, info

        self._pending_kpis = next_kpis
        terms = self._reward_terms(next_kpis, action_delta)
        info = {"kpis": next_kpis, "margin_offsets": cell_actions,
                "pair_margins_db": pair_offsets, "reward_terms": terms}
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
        last_pair_margin_db reads self._pair_margin_state, which is empty
        until the first step() of an episode applies a nonzero pair offset;
        reset()'s observation therefore reports 0.0 dB for every cell,
        correctly reflecting that no margin has been applied yet.
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
            touching = self._pair_margin_state.offsets_touching(cell)
            last_pair_margin = (sum(abs(v) for v in touching) / len(touching)) if touching else 0.0
            row[i * len(OBS_FIELDS):(i + 1) * len(OBS_FIELDS)] = (
                (SINR_FLOOR_DB if sinr is None else sinr) / SINR_SCALE_DB,
                cell_kpis.get("num_active_ues", 0) / ue_scale,
                _sinr_bad_fraction(cell_kpis),
                activity,
                last_pair_margin / MARGIN_LIMIT_DB,
            )
        return np.clip(row, OBS_LOW, OBS_HIGH).astype(np.float32)

    def _reward_terms(self, kpis: dict, action_delta: np.ndarray) -> dict:
        """The four-term MRO reward, plus its breakdown and diagnostics.

            reward = w_quality   * HandoverQuality
                   - w_pingpong  * PingPong
                   - w_churn     * HandoverRate
                   - w_outage    * OutageExposure

        See docs/mro_reward_design.pdf for the full rationale and the ns-3
        source audit behind each term, in particular why HandoverQuality
        replaces a literal RRC handover-success rate (the mmWave secondary-cell
        handover path does not go through the generic RRC state machine that
        can report failure, so a literal success rate would read a constant
        1.0 in this simulator) and why OutageExposure replaces a literal RLF
        rate (RLF detection is confirmed dead code in this ns-3 fork).

        ActionSmoothness is NOT in the sum (REWARD_FORMULA_VERSION 2) -- see
        DEFAULT_W_OUTAGE's neighbouring module-level comment for why. It is
        still computed and returned below, diagnostic-only.

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

        # Shared by HandoverRate and PingPong -- see HANDOVER_RATE_REF_S for
        # why one scale factor, computed once, rather than two.
        #
        # Bound on both branches: the diagnostics block below reads these
        # unconditionally, and a term that is merely UNAVAILABLE must not
        # raise NameError on the way out. handover_window_s is what ns-3
        # actually drained (handover_window_s in the payload, set to
        # stepInterval by scenario-marl-zmq.cc's MarlControlStep), so the
        # window is REPORTED, never inferred. Absent, or non-positive, falls
        # back to REF, which makes the factor exactly 1.0 and reproduces the
        # old per-step-count behaviour rather than dividing by zero.
        if handovers_unavailable:
            handover_window_s = HANDOVER_RATE_REF_S
            handover_rate_scale = 1.0
        else:
            handover_window_s = float((kpis or {}).get("handover_window_s") or 0.0)
            if handover_window_s <= 0.0:
                handover_window_s = HANDOVER_RATE_REF_S
            # Floored, not just guarded against <=0.0: see
            # HANDOVER_WINDOW_FLOOR_S for why a small POSITIVE window is also
            # a boundedness hazard, found by the reward-weight-methodology
            # audit rather than by a training incident.
            elif handover_window_s < HANDOVER_WINDOW_FLOOR_S:
                handover_window_s = HANDOVER_WINDOW_FLOOR_S
            handover_rate_scale = HANDOVER_RATE_REF_S / handover_window_s

        # --- HandoverRate ------------------------------------------------------
        if handovers_unavailable:
            handover_rate = None
        else:
            handover_rate = ((n_handovers / active_ues) * handover_rate_scale
                             if active_ues > 0 else 0.0)

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
            pingpong = ((pingpong_count / active_ues) * handover_rate_scale
                        if active_ues > 0 else 0.0)

        # --- v3 pair-role rates, cached for the NEXT step's action decision -
        # Must run AFTER self._handover_history has been extended with THIS
        # step's handovers (just above) -- PairRoleTracker's churn pairing
        # needs the history to already include them, identical ordering
        # requirement to PingPong's own detector just above. Runs even when
        # handovers_unavailable (degrades to all-zero churn rates, same
        # graceful-degradation spirit as the rest of this function; NOT
        # added to terms_present since it is not a reward term).
        self._pair_rates = self._pair_tracker.update(kpis, handovers, self._handover_history)

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

        # --- ActionSmoothness (diagnostic only -- not in the sum, see
        # DEFAULT_W_OUTAGE's neighbouring comment and REWARD_FORMULA_VERSION).
        # v3: compares the 2-dim role-gain DELTA step to step, not a 7-dim
        # per-cell margin -- same mean-squared-delta shape, smaller vector. ---
        margin = np.asarray(action_delta, dtype=np.float32).reshape(-1)
        if self._last_action is None:
            smoothness = 0.0
        else:
            smoothness = float(np.mean((margin - self._last_action) ** 2))

        reward = -self.w_outage * outage
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
            "smoothness": float(smoothness),  # diagnostic only -- not in the sum
            "weights": {"quality": self.w_quality, "pingpong": self.w_pingpong,
                        "churn": self.w_churn, "outage": self.w_outage},
            "diagnostics": {
                "active_ues": active_ues,
                "n_ues_reporting_sinr": len(current_ue_sinr),
                "handovers_unavailable": bool(handovers_unavailable),
                "handovers_this_step": n_handovers,
                "handover_history_len": len(self._handover_history),
                "handover_window_s": handover_window_s,
                "handover_rate_scale": handover_rate_scale,
                "handover_rate_ref_s": HANDOVER_RATE_REF_S,
                "reward_formula_version": REWARD_FORMULA_VERSION,
                "pingpong_count": pingpong_count,
                "pingpong_y_s": PINGPONG_Y_S,
                "quality_good": quality_good,
                "quality_measured": quality_measured,
                "outage_sinr_db": OUTAGE_SINR_DB,
                "role_gain_delta_db": {"too_late": float(margin[0]), "churn": float(margin[1])},
                # Built from what is actually in the sum, so a dropped term is
                # never mistaken for one that scored zero. smoothness is
                # deliberately absent -- it's logged (see "smoothness" above)
                # but never summed, so it was never "present" in this sense.
                "terms_present": tuple(
                    ["outage"]
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
