"""The traditional (non-RL) MRO SON function.

v2: windowed rate-based decisions on a slower cadence, replacing v1's
instantaneous-per-step thresholding, per a closer literature pass on
hysteresis-margin-specific (not CIO) SON algorithms: patents/papers on
this exact problem compute a too-late RATE over a window per cell and
compare it to a threshold (not a single noisy sample), and state that
hysteresis/TTT "change by one step per optimization only" -- i.e.
decide on a slower cadence, not every control step. v1 was tested
against the do-nothing baseline and scored BELOW it (+0.0625 vs
+0.1526 under current weights, measured from real warm-start replay
data) -- the instantaneous-reaction, no-dead-band design is the
leading suspect (see mro_investigation_log.md's 2026-10-19 entry).

Reads directly off MroZmqEnv's own observation vector (badsignal_frac
as the too-late/outage-risk proxy, handover_activity as the
too-early/churn proxy) -- the controller and the RL agent see the same
information.

v3 (2026-10-25): per-cell-pair action, grounded in 3GPP TR 36.839's MRO
failure taxonomy (too-late HO, too-early HO/ping-pong), replacing this
file's per-cell margin design above. See RuleBasedMroControllerV3 below
and environments/mro_zmq_env.py's module docstring for the full design --
the per-pair rule now lives in the env itself (PairRoleTracker +
_pairwise_margin_targets + PairMarginState), not in a controller class,
since the RL action and the rule-based baseline share the exact same
per-pair machinery and differ only in the 2-dim role-gain correction fed
into it. v1/v2's controllers above are kept for reference/comparison, not
wired into v3's pairwise action -- they predate the ns-3-side pairwise
margin keying and cannot be used to warm-start a v3 SAC agent.
"""
import numpy as np

from environments.mro_zmq_env import CELLS, MARGIN_LIMIT_DB, obs_index

STEP_DB = 0.5


class RuleBasedMroController:
    """Stateful: one instance per environment/worker (each needs its own
    rolling window and decision-cadence counter).
    """

    def __init__(self, window: int = 5, cadence: int = 5,
                 badsignal_threshold: float = 0.05, activity_threshold: float = 0.25,
                 dead_band: float = 0.3, step_db: float = STEP_DB, max_repeat: int = 1,
                 multi_cell: bool = False, top_k: int = 0):
        self.window = window
        self.cadence = cadence
        self.badsignal_threshold = badsignal_threshold
        self.activity_threshold = activity_threshold
        self.dead_band = dead_band  # fraction of threshold required to RELEASE a trigger
        self.step_db = step_db
        self.max_repeat = max_repeat  # consecutive same-direction steps allowed per episode
        # v2c: cross-cell ARBITRATION. v2b (per-cell threshold scaled by
        # network-relative severity, all triggering cells still act every
        # cadence tick) tested WORSE than plain v2a over the full 24 seeds
        # (mean diff vs baseline -0.021 vs v2a's -0.011, same 9/24 win
        # count) -- and every variant tried today that made the controller
        # act MORE (bigger steps, more cells per tick, continuous
        # re-triggering) scored worse, while v2a -- the most conservative,
        # least-active design -- stayed closest to baseline. That pattern,
        # plus the reward's heavy ping-pong weight (9.39x quality) relative
        # to a static baseline that pays zero ping-pong risk by
        # construction, suggests the lever to pull is fewer actions, not
        # more. top_k, when > 0, ranks all cells exceeding their threshold
        # by normalized excess each cadence tick and acts on only the
        # top_k worst (cross-cell arbitration/prioritization, the
        # literature's per-neighbor decision framing adapted to "most
        # urgent first" instead of "all simultaneously") -- 0 disables this
        # and keeps v2a's original behavior of every triggering cell acting.
        self.top_k = top_k
        # v2b: multi-cell trigger sensitivity. Three parameterizations of a
        # separate graduated-STEP-SIZE design (ProportionalMroController)
        # all underperformed both baseline and this flat-step v2a (mean
        # reward 0.122-0.144 vs baseline 0.163 / v2a 0.158, seeds 2001-2009)
        # -- consistent across gain settings and with/without hold state,
        # so the flat, infrequent, trigger-once step itself looks like the
        # robust part of this design, not the thing to graduate. Instead of
        # scaling STEP SIZE by network-relative severity, this scales the
        # trigger THRESHOLD: a cell notably worse than its peers becomes
        # MORE sensitive (lower effective threshold), a cell merely
        # reflecting network-wide load is unaffected -- genuine multi-cell
        # context without touching the step mechanics already validated.
        self.multi_cell = multi_cell
        self._badsignal_hist = {c: [] for c in CELLS}
        self._activity_hist = {c: [] for c in CELLS}
        self._step = 0
        self._action = np.zeros(len(CELLS), dtype=np.float32)
        self._triggered = {c: None for c in CELLS}  # None / "tighten" / "loosen"
        self._repeat_count = {c: 0 for c in CELLS}

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._step += 1
        for i, cell in enumerate(CELLS):
            badsignal = float(obs[obs_index(cell, "badsignal_frac")])
            activity = float(obs[obs_index(cell, "handover_activity")])
            h = self._badsignal_hist[cell]
            h.append(badsignal)
            if len(h) > self.window:
                h.pop(0)
            a = self._activity_hist[cell]
            a.append(activity)
            if len(a) > self.window:
                a.pop(0)

        if self._step % self.cadence != 0:
            return np.clip(self._action, -MARGIN_LIMIT_DB, MARGIN_LIMIT_DB)

        bs_rates = {c: np.mean(self._badsignal_hist[c]) for c in CELLS}
        act_rates = {c: np.mean(self._activity_hist[c]) for c in CELLS}
        if self.multi_cell:
            bs_network_avg = max(np.mean(list(bs_rates.values())), 1e-6)
            act_network_avg = max(np.mean(list(act_rates.values())), 1e-6)

        allowed = set(CELLS)
        if self.top_k > 0:
            severities = {}
            for cell in CELLS:
                bs_thresh = self.badsignal_threshold
                act_thresh = self.activity_threshold
                if self.multi_cell:
                    bs_thresh /= max(1.0, bs_rates[cell] / bs_network_avg)
                    act_thresh /= max(1.0, act_rates[cell] / act_network_avg)
                bs_excess = (bs_rates[cell] - bs_thresh) / bs_thresh
                act_excess = (act_rates[cell] - act_thresh) / act_thresh
                sev = max(bs_excess, act_excess)
                if sev > 0.0:
                    severities[cell] = sev
            ranked = sorted(severities, key=severities.get, reverse=True)
            allowed = set(ranked[:self.top_k])

        for i, cell in enumerate(CELLS):
            bs_rate = bs_rates[cell]
            act_rate = act_rates[cell]
            bs_thresh = self.badsignal_threshold
            act_thresh = self.activity_threshold
            if self.multi_cell:
                bs_sev = max(1.0, bs_rate / bs_network_avg)
                act_sev = max(1.0, act_rate / act_network_avg)
                bs_thresh /= bs_sev
                act_thresh /= act_sev
            state = self._triggered[cell]
            can_act = cell in allowed  # top_k arbitration: cells not among the
            # worst-ranked this tick still get their dead-band RELEASE check
            # below (state cleanup costs nothing), just not a new trigger.
            # v3 tried stepping every cycle the condition holds (integral-
            # style) -- tested WORSE than this (+0.1520 vs +0.1750, 6
            # seeds), most likely integral windup: the window lags the
            # true state by ~window steps, so repeated stepping overshoots
            # before the window reflects the correction already made. Back
            # to step-once-per-episode (self-limiting), but allow up to
            # max_repeat consecutive steps for genuinely persistent
            # problems instead of v2's hard cap of exactly one.
            if can_act and bs_rate > bs_thresh:
                if state != "loosen" or self._repeat_count[cell] < self.max_repeat:
                    self._action[i] -= self.step_db
                    self._repeat_count[cell] = (self._repeat_count[cell] + 1
                                                 if state == "loosen" else 1)
                    self._triggered[cell] = "loosen"
            elif can_act and act_rate > act_thresh:
                if state != "tighten" or self._repeat_count[cell] < self.max_repeat:
                    self._action[i] += self.step_db
                    self._repeat_count[cell] = (self._repeat_count[cell] + 1
                                                 if state == "tighten" else 1)
                    self._triggered[cell] = "tighten"
            elif (bs_rate < bs_thresh * (1 - self.dead_band)
                  and act_rate < act_thresh * (1 - self.dead_band)):
                self._triggered[cell] = None  # clearly resolved
                self._repeat_count[cell] = 0
        return np.clip(self._action, -MARGIN_LIMIT_DB, MARGIN_LIMIT_DB)


class ProportionalMroController:
    """v4: proportional (graduated) correction, and genuinely MULTI-CELL.

    Two gaps identified after v2/v3 landed at a statistical tie with
    baseline (24 seeds, t=-1.31): (1) the literature's own recommendation
    is a GRADUATED step size (correction scales with how far past
    threshold the signal is), not v2/v3's flat step whenever a fixed
    threshold is crossed; (2) real MRO (Jansen et al., D-HCP) decides
    PER NEIGHBOR, i.e. with cross-cell context, not each cell in total
    isolation. Our action space is per-cell, not per-neighbor-pair, so
    true pair-level coordination isn't directly expressible -- but
    genuine multi-cell awareness still is: each cell's correction here is
    relative to the NETWORK-WIDE average rate at that decision point, not
    an absolute per-cell threshold alone. A cell that is merely
    reflecting network-wide conditions (all cells similarly loaded) is
    corrected less than one that is uniquely bad relative to its peers.

    Gains calibrated so that an error the size of v2a's own thresholds
    (badsignal 0.05, activity 0.25) produces roughly v2a's own step
    (0.5dB) -- same order of magnitude as the validated starting point,
    now scaling continuously instead of in one flat jump.
    """

    def __init__(self, window: int = 5, cadence: int = 5,
                 kp_loosen: float = 10.0, kp_tighten: float = 2.0,
                 badsignal_threshold: float = 0.05, activity_threshold: float = 0.25,
                 max_step: float = 1.0, dead_band: float = 0.3, max_repeat: int = 1):
        self.window = window
        self.cadence = cadence
        self.kp_loosen = kp_loosen
        self.kp_tighten = kp_tighten
        # ABSOLUTE thresholds gate whether a cell acts at all -- first version
        # compared each cell only to the live network average with no floor,
        # so ~half the cells were "above average" at any instant even when
        # everything was fine, constantly over-correcting (tested: worse
        # than both baseline and the bangbang controller, not better).
        self.badsignal_threshold = badsignal_threshold
        self.activity_threshold = activity_threshold
        self.max_step = max_step
        # v4a (this version) carries over the trigger-once/hold/dead-band
        # discipline already validated in RuleBasedMroController -- the
        # first cut of this class re-applied delta every cadence tick for
        # as long as error exceeded threshold, with no hold state at all.
        # That's the same integral-windup failure mode RuleBasedMroController
        # v3 hit (and reverted) when it allowed repeated same-direction
        # steps: the window lags the true state, so continuous re-application
        # overshoots before the correction already made is reflected back.
        # Tested three gain settings with the un-gated continuous version,
        # all three landed below both baseline and the bangbang controller
        # (mean reward 0.124-0.144 vs baseline 0.163 / bangbang 0.158,
        # seeds 2001-2009) -- consistent across settings, pointing to this
        # structural gap rather than gain magnitude.
        self.dead_band = dead_band
        self.max_repeat = max_repeat
        self._badsignal_hist = {c: [] for c in CELLS}
        self._activity_hist = {c: [] for c in CELLS}
        self._step = 0
        self._action = np.zeros(len(CELLS), dtype=np.float32)
        self._triggered = {c: None for c in CELLS}
        self._repeat_count = {c: 0 for c in CELLS}

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._step += 1
        for cell in CELLS:
            h = self._badsignal_hist[cell]
            h.append(float(obs[obs_index(cell, "badsignal_frac")]))
            if len(h) > self.window:
                h.pop(0)
            a = self._activity_hist[cell]
            a.append(float(obs[obs_index(cell, "handover_activity")]))
            if len(a) > self.window:
                a.pop(0)

        if self._step % self.cadence != 0:
            return np.clip(self._action, -MARGIN_LIMIT_DB, MARGIN_LIMIT_DB)

        bs_rates = {c: np.mean(self._badsignal_hist[c]) for c in CELLS}
        act_rates = {c: np.mean(self._activity_hist[c]) for c in CELLS}
        bs_network_avg = max(np.mean(list(bs_rates.values())), 1e-6)
        act_network_avg = max(np.mean(list(act_rates.values())), 1e-6)

        for i, cell in enumerate(CELLS):
            # absolute threshold GATES whether this cell acts at all --
            # fixes the over-triggering bug above. Once gated, the network
            # average still matters, but only as a SEVERITY multiplier: a
            # cell that's notably worse than its peers gets a bigger
            # correction than one whose elevated rate just reflects
            # network-wide load -- genuine multi-cell context without the
            # "always tinkering" failure mode.
            bs_error = max(0.0, bs_rates[cell] - self.badsignal_threshold)
            act_error = max(0.0, act_rates[cell] - self.activity_threshold)
            if bs_error > 0.0:
                bs_error *= max(1.0, bs_rates[cell] / bs_network_avg)
            if act_error > 0.0:
                act_error *= max(1.0, act_rates[cell] / act_network_avg)
            state = self._triggered[cell]
            if bs_error > 0.0:
                if state != "loosen" or self._repeat_count[cell] < self.max_repeat:
                    delta = -self.kp_loosen * bs_error
                    delta = float(np.clip(delta, -self.max_step, self.max_step))
                    self._action[i] += delta
                    self._repeat_count[cell] = (self._repeat_count[cell] + 1
                                                 if state == "loosen" else 1)
                    self._triggered[cell] = "loosen"
            elif act_error > 0.0:
                if state != "tighten" or self._repeat_count[cell] < self.max_repeat:
                    delta = self.kp_tighten * act_error
                    delta = float(np.clip(delta, -self.max_step, self.max_step))
                    self._action[i] += delta
                    self._repeat_count[cell] = (self._repeat_count[cell] + 1
                                                 if state == "tighten" else 1)
                    self._triggered[cell] = "tighten"
            elif (bs_rates[cell] < self.badsignal_threshold * (1 - self.dead_band)
                  and act_rates[cell] < self.activity_threshold * (1 - self.dead_band)):
                self._triggered[cell] = None
                self._repeat_count[cell] = 0
        return np.clip(self._action, -MARGIN_LIMIT_DB, MARGIN_LIMIT_DB)


class RuleBasedMroControllerV3:
    """v3: the pure rule-based baseline for the pairwise too-late/churn
    action (see environments/mro_zmq_env.py's module docstring on
    PairRoleTracker for the full role definitions and literature grounding).

    Unlike v1/v2's controllers above, this one does essentially nothing: the
    env itself now owns the entire per-pair rule (PairRoleTracker,
    _pairwise_margin_targets, PairMarginState -- rate-limited, clamped,
    literature-grounded role detection and combination), and the action
    space is the agent's ADDITIVE correction on top of two FIXED base gains
    (BASE_TOO_LATE_GAIN_DB, BASE_CHURN_GAIN_DB). A constant zero action
    therefore IS the traditional/rule-based controller -- not an
    approximation of it, the exact same thing an untrained, zero-initialised
    SAC policy produces (see train_mro_sac_pilot.py's zero_init_actor_mean).
    Kept as a tiny class (not a bare lambda) purely to match
    RuleBasedMroController's .act(obs) call-site interface used by
    warm_start_collect() and other callers.
    """

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


def rule_based_action(obs: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    """Stateless-looking v1 shim, kept for any caller still using the old
    function interface. NOT windowed/cadenced -- prefer
    RuleBasedMroController directly for anything new.
    """
    action = np.array(prev_action, dtype=np.float32, copy=True)
    for i, cell in enumerate(CELLS):
        badsignal = obs[obs_index(cell, "badsignal_frac")]
        activity = obs[obs_index(cell, "handover_activity")]
        if badsignal > 0.08:
            action[i] -= STEP_DB
        elif activity > 0.35:
            action[i] += STEP_DB
    return np.clip(action, -MARGIN_LIMIT_DB, MARGIN_LIMIT_DB)
