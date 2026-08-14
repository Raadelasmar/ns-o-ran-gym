"""MLB reward terms over a completed run's database.db (read-only).

    R = w1*Balance + w2*Satisfaction - w3*Backlog - w4*BadSignal - w5*PingPong

STATUS: all five terms are built here in their final SCALE-FREE forms, and
`compute_reward` returns a combined total. Term 2 (formerly the deferred "Data"
term) is now UDP downlink user satisfaction -- delivered over a demand
denominator that is provably exogenous to the agent's actions. See
"TERM 2 -- SATISFACTION" below.

The only step at which `reward` is still None is one where Satisfaction is
undefined (no UDP UEs in the scenario, or no UDP du rows at that step); a
4-of-5 total would be silently incomparable with the others, so it is withheld.

All terms are computed over the fixed mmWave cell set load_metric.NR_CELLS
= [2..8]. Columns that build_load_view already provides (prb_pct, n_ues,
buffer_bytes, vol_bytes) are taken from it rather than re-queried; median MCS
comes from mlb_observation. The extra du columns this module needs
(rruprbuseddl, dlavailableprbs, the L1M.RS-SINR bins) are read here by
build_du_extras(), since neither of those modules reads them.

Every du column read by build_du_extras() is CELL-LEVEL -- repeated identically
on each of the cell's UE rows -- so it is taken with .first() per (timestamp,
cellid), never summed. Summing would multiply the value by the cell's UE count.
The same disagreement guard load_metric.build_load_view uses is applied.

The ONE exception is the Satisfaction term, which is genuinely PER-UE:
du.drbuethpdlpdcpbasedueid holds a distinct value on every UE row, and du has
UNIQUE(timestamp, ueimsicomplete), so it is SUMMED over UE rows -- never
.first()-ed. build_udp_satisfaction() is the only place this happens, and it
does not go through build_du_extras(). Confusing the two conventions in either
direction silently produces a number that is wrong by a factor of the UE count.

SCALE-FREE FORMS
----------------
Every term below is a ratio of like quantities (or already a [0,1] index), so
none of them carries the run's byte/count magnitudes into the reward:

  Balance       Jain's index over a per-cell congestion index -> [0, 1]
  Satisfaction  UDP DL delivered / UDP DL demanded -> [0, 1.2] (clipped)
  Backlog       seconds-of-delay = backlog bytes / delivery bytes-per-second
  BadSignal     fraction of DL transmissions below -6 dB SINR -> [0, 1]
  PingPong      ping-pong legs per active UE

TERM 1 -- BALANCE, and the three modes
--------------------------------------
Per-cell congestion index, all three factors in [0,1]:

    CI_i = (rruprbuseddl_i / dlavailableprbs_i) * (n_ues_i / TOTAL_UES)
           * 1[median_mcs_i > tau]

The PRB factor is the RAW PRB ratio: rruprbuseddl (average PRBs allocated,
a real number) over dlavailableprbs (the cell's PRB budget, a du COLUMN --
139 in the current scenario, but read, never hardcoded, so a run with a
different numerology scores correctly). This replaces the earlier
dlprbusage/100, which is the same quantity after ns-3 has truncated it to an
integer percentage and clamped it at 100 (mmwave-enb-net-device.cc:1377).

TOTAL_UES only ever scales the whole CI vector by a constant, and Jain's index
is invariant under that scaling -- so `total_ues` changes no Balance value. It
is exposed as a parameter for readability, not because it moves the number.

Jain's fairness index over the CI vector:

    J = (sum CI)^2 / (N * sum CI^2)

Raw J is a poor reward on its own: an idle network is maximally "unbalanced"
(one cell with a trickle of traffic, six at zero) even though there is nothing
worth balancing. Four ways to handle that, all implemented:

  balance_mode='jain_guarded' (DEFAULT)
      Balance = J, the UNGATED Jain's index over the full CI vector across
      NR_CELLS -- no threshold, no multiplier -- EXCEPT on a nearly-empty
      network, where the term is explicitly neutralised:

          if sum(CI) <= 0  or  max_i prb_frac_i < low_load_prb_threshold:
              Balance = BALANCE_GUARD_VALUE   (1.0)

      The guard returns early and sets diagnostics['balance_guarded'] = True, so
      a guarded step is never confused with a genuine J that happens to land on
      1.0. Default low_load_prb_threshold = 0.05, i.e. "the busiest cell in the
      network is using under 5% of its PRBs".

      WHY 1.0 IS THE NEUTRAL VALUE: it is the same convention _jain() already
      uses for an all-zero vector, and it matches 'gate' -- a free maximum means
      "nothing to balance, so no gradient here". The guard simply widens that
      existing all-zero convention from "exactly empty" to "nearly empty", which
      is what stops the one-trickling-cell pathology (six zeros and one small
      CI gives J = 1/7 ~ 0.14, a severe penalty for a network doing nothing).

      WHY THIS REPLACED 'weighted': the load_weight multiplier turned out to be
      FROZEN. Across every scored run the busiest cell's PRB fraction sits in a
      narrow band around ~0.78 (measured range 0.61-0.81), so load_weight was
      very nearly a constant -- it rescaled every Balance value by the same
      factor and changed no ordering, no direction, and no discrimination. The
      only thing it actually did was drive Balance toward 0 on a near-empty
      network, i.e. act as a light-load guard. Replacing a frozen multiplier
      with an explicit guard is arithmetically equivalent over the operating
      range and states the intent directly instead of hiding it inside a
      product. It also restores the natural reading of the number: Balance is
      now Jain's index, so 0.62 means "the load split is 62% fair", not "62% of
      an unstated load-weighted maximum".

      load_weight is still computed and reported in diagnostics under every mode
      so historical run records stay comparable, but nothing multiplies by it.

      *** SIGN / SEMANTICS -- READ BEFORE TUNING WEIGHTS ***
      Under 'jain_guarded' a CALM network reads HIGH (1.0 via the guard), which
      is the 'gate' convention and the OPPOSITE of 'weighted', where calm read
      ~0. Balance is low only when the network is loaded AND lopsided. Any
      weight tuned against 'weighted' numbers must be re-tuned: the term's mean
      rises by roughly 1/0.78 ~ 1.28x on a busy run, and much more on a calm one.

      Known edge: if every CI_i is 0 while the network is genuinely busy (e.g.
      every cell's median_mcs <= tau), the sum(CI) <= 0 guard fires and Balance
      reads 1.0. This is the same edge 'weighted' had (there it collapsed to
      load_weight); diagnostics expose n_loaded and balance_guarded so the case
      is visible rather than silent.

  balance_mode='weighted' (PREVIOUS default, preserved for comparison)
      Balance = load_weight * J, with

          load_weight = min(max_i prb_frac_i, 1.0)

      i.e. the busiest cell's PRB fraction, clamped to 1.0 in case a data quirk
      ever pushes rruprbuseddl above dlavailableprbs. Both factors are in [0,1],
      so Balance is in [0,1]. See above for why the multiplier was dropped.

  balance_mode='gate' (OLD primary, preserved for comparison)
      Balance switches on only when some cell is actually overloaded.
      If no cell has prb_pct > overload_prb_threshold, Balance = 1.0
      (a free maximum: nothing to balance, so no gradient). Otherwise
      Balance = J over all N = len(NR_CELLS) cells.
      NOTE: the GATE reads prb_pct (dlprbusage, a percentage) so the threshold
      keeps its documented units.

      WHY IT STOPPED BEING THE DEFAULT: the 80% threshold sat above the
      network's actual operating range (peak cell prb_frac 0.61-0.81 across the
      scored runs), so 'gate' degenerated into a binary n_overloaded flag --
      balance == 1.0 was equivalent to n_overloaded == 0 on 100% of steps -- and
      reported a free maximum of 1.0 on 48 (44dadaab) and 45 (5e89652b) steps
      whose backlog ran 2.4-5.5 s and whose ungated Jain was 0.28-0.64, i.e.
      visibly uneven. 'jain_guarded' keeps the free-maximum convention but fires
      it only on a genuinely near-empty network (max prb_frac < 0.05), which no
      loaded step reaches.

  balance_mode='subset' (OLD fallback, preserved)
      Balance = J computed over only the loaded cells, i.e. those with CI_i > 0,
      with N = the number of such cells. Idle cells never enter the index, so
      they cannot manufacture unfairness. With 0 or 1 loaded cells the index is
      trivially 1.0.
      NOTE: "loaded" is read here as "carrying traffic" (CI_i > 0), the standard
      subset-Jain formulation, NOT as "exceeding overload_prb_threshold". If the
      other reading is wanted, change the `loaded` mask in _balance().

`overload_prb_threshold` is used only by 'gate'; it is accepted and reported in
diagnostics under every mode so run records stay comparable.

TERM 2 -- SATISFACTION (UDP downlink delivered / demanded)
----------------------------------------------------------
    sat_udp = (sum over UDP UEs of drbuethpdlpdcpbasedueid)
              / (n_udp * per_ue_rate_kbps)

Both sides are kbps, so the result is a dimensionless satisfaction ratio in
[0,1] -- "what fraction of what the users actually wanted did they get".

NUMERATOR: du.drbuethpdlpdcpbasedueid is PDCP-based DL throughput per UE in
kbps -- rxBytes / m_e2Periodicity, where rxBytes comes from
m_e2PdcpStatsCalculator->GetDlRxData(imsi, 3), i.e. PDCP SDUs actually
DELIVERED TO THE UE (mmwave-enb-net-device.cc:798). It is a genuinely per-UE
column, so it is SUMMED over UE rows.

  *** DO NOT SUBSTITUTE qosflowpdcppduvolumedl_filter[ueid] ***
  That column is macVolume, accumulated in MmWavePhyTrace::UpdateTraces
  (mmwave-phy-trace.cc:262) as `+= params.m_tbSize` for EVERY transport block
  seen at the PHY Rx trace -- with no filter on m_corrupt and no filter on m_rv.
  It counts corrupted TBs and every HARQ retransmission, so it measures air-
  interface bytes ATTEMPTED, not payload delivered. A bad offload raises
  retransmissions, so MAC volume RISES while user-delivered bytes FALL: scored
  on the seedC pair (a known-bad offload, backlog +105%) it reports +7.8%
  "better". That is exactly how the earlier raw-volume Data term was fooled.
  drbuethpdlpdcpbasedueid tracks the raw DlE2PdcpStatsLte.txt RxBytes to within
  0.0-0.4% on all ten validation runs; macVolume does not.

DENOMINATOR: the demand side, and the reason this term is safe. In
scenario-three.cc under trafficModel=3 the UEs with u%4 == 0 (IMSI 1,5,9,...)
are the only ones with DOWNLINK traffic: a UdpClient on the remote host firing
1280 B every 500 us with MaxPackets=UINT32_MAX (scenario-three.cc:916-937), i.e.
20.48 Mbps of payload = 20.96 Mbps at PDCP once headers are added. Every other
UE class runs an OnOff TCP client installed ON THE UE (scenario-three.cc:940-951)
-- those are UPLINK, and their downlink is nothing but TCP ACKs (5-60 kbps,
Rx/Tx == 1.00 always).

That matters because it sidesteps the TCP-STARVATION TRAP. A congested TCP flow
backs off and ASKS for less, so its measured demand reflects what the network
ALLOWED, not what the user WANTED, and delivered/demanded would flatter a
network it should be punishing. The UDP class is open-loop: it never backs off,
so its demand is the config constant and is knowable. Measured directly, the
per-UE, per-100ms DL PDCP arrival series for these nine UEs is BIT-IDENTICAL
across all nine trafficModel=3/0 validation runs -- four seeds x two CIO arms
plus a different traffic model, 88-182 handovers each, 891/891 cells populated,
zero dropouts. The denominator is provably immune to anything the agent does,
which is what makes this term un-hackable in the way spectral efficiency was
not. (The uplink is the counter-example: measured UL demand MOVES with the
action, +4.3% on the seedC pair. Never build the denominator from it.)

n_udp and per_ue_rate_kbps are PARAMETERS, not literals inside the formula, so
the term survives a change to `ues` or `trafficModel`:
  - n_udp defaults to the number of DISTINCT UDP IMSIs PRESENT IN THE RUN, so
    scaling the UE count rescales the denominator automatically.
  - per_ue_rate_kbps defaults to UDP_UE_RATE_KBPS_TM3 (20960.0), correct for
    trafficModel=3 / configuration=0. A run with configuration=2 uses a 250 us
    interval (40 Mbps) and MUST pass its own value.

  *** n_udp IS RUN-LEVEL, DELIBERATELY, AND MUST NOT BECOME PER-STEP ***
  The denominator uses the run's UDP POPULATION, not the count of UDP rows
  present at that step. A UDP UE that vanishes from du -- dropped, in outage,
  starved to silence -- still WANTED its 20.96 Mbps, so it must keep its place
  in the denominator and score 0 in the numerator. Recomputing n_udp per step
  would shrink the denominator exactly when the network fails a user, turning a
  failure into a higher score. That is a reward-hacking hole; the run-level
  count closes it.

Clipped to [0, SATISFACTION_CLIP_MAX] (1.2). It should never exceed 1.0; the
0.2 of slack absorbs header-overhead accounting between the payload rate used in
the denominator and the PDCP-layer bytes counted in the numerator, and the clip
stops any residual mismatch from producing an unbounded term.

Guards, both of which yield NaN for the step (excluded from aggregates, and the
combined reward is withheld rather than computed from four terms):
  - n_udp == 0: the scenario has no open-loop DL class, so there is no honest
    demand denominator available. Flagged as satisfaction_no_udp_ues.
  - no UDP du rows at this step: nothing was measured, which is not the same as
    "nothing was delivered". Flagged as satisfaction_no_rows_at_step.

Observed range: 0.41-0.65 on the healthy validation runs, 0.29 on t2_saturated
(4x the DL load -> correctly starved), and it moves the right way on 4 of 4
offload pairs including the seedC disaster that spectral efficiency and raw
volume both got wrong. It never approaches 1.0 in this scenario because 189 Mbps
of DL intent genuinely exceeds the network's capacity, so the term keeps
headroom at all times and never saturates.

The OLD raw byte-sum Data term is not lost: it is still reported, unweighted and
unused, as raw['data_bytes'].

TERM 3 -- BACKLOG, as seconds of delay
--------------------------------------
Network-level, not per-cell:

    backlog_sec = (sum_cells drbbuffersizeqos)
                  / ((sum_cells qosflowpdcppduvolumedl_filter) * 10)

Numerator is queued bytes (RLC tx buffer occupancy; LteRlc*::GetTxBufferSize
returns bytes). Denominator converts the 0.1 s KPM window's delivered byte
volume into bytes/second, so the ratio is a time: how long the current queue
would take to drain at the current delivery rate. Bytes cancel; the result is
seconds and carries no scenario magnitude.

Deliberately NETWORK-level. The per-cell version of the same ratio has a heavy
tail (p99 ~50 s, max ~131 s on b3efb1ee) because a nearly-idle cell with a
non-empty buffer divides by almost nothing. Aggregating first removes that.

Guard: denominator == 0 (no cell delivered anything this step) -> 0.0.
No traffic means no queueing delay to charge for. Flagged in diagnostics as
backlog_denominator_zero.

TERM 4 -- BADSIGNAL, as a fraction
----------------------------------
    badsignal_frac = (sum_cells l1mrssinrbin34) / (sum_cells total_transmissions)

*** WHAT THIS TERM IS FOR -- IT IS AN OVERSHOOT GUARD, NOT AN OFFLOAD METRIC ***
BadSignal is a LOW-FREQUENCY OVER-AGGRESSION DETECTOR. It is near-zero in normal
operation (mean 0.0005-0.003 across the validation runs, median exactly 0 on
almost all of them) and rises only when a CIO action is aggressive enough to
push UEs onto a cell whose signal cannot carry them -- e.g. the -6 dB arm
(7cb3494f) runs ~2x the -3 dB arm on the same seed.

This is INTENDED and is not a defect to be tuned away. Do not read a small
BadSignal as evidence that an offload was good, and do not expect it to rank
good offloads against bad ones -- it does not, and it is not meant to. It says
only "this action did not shove anyone off a cliff". The counter-example is
t3_churn (c2f36f99), the run with 1078 handovers: it is the QUIETEST of all
thirteen runs on BadSignal (0.00007) precisely because UEs that hand over
constantly stay well-served. Offload QUALITY is carried by Satisfaction and
Backlog; BadSignal is the guard rail that stops the agent from buying either of
them with an unserviceable link. Keep it small-weighted and one-sided.

Numerator: L1M.RS-SINR.Bin34 = DL transmissions whose SINR was <= -6 dB
(mmwave-phy-trace.cc:306).

Denominator: the sum of ALL SEVEN L1M.RS-SINR bins, not tbtotnbrdl1. The seven
bins are an exhaustive if/else-if partition over exactly the same events as the
numerator -- one increment per DL transport-block report, thresholds at
-6/0/6/12/18/24 dB (mmwave-phy-trace.cc:305-333) -- so the ratio is guaranteed
to be a true fraction in [0,1] by construction, whatever else the run does.
tbtotnbrdl1 is numerically identical (verified: equal on 677/677 + 693/693 +
693/693 cell-steps of the three runs scored here, since it counts the same MAC
PDUs), but that equality is an empirical fact about two separately-maintained
counters, not a structural guarantee. The bin sum is used because it cannot
drift; SINR_BIN_TOTAL_CROSSCHECK keeps tbtotnbrdl1 in the diagnostics so any
future divergence is visible.

Guard: denominator == 0 (no transmissions this step) -> 0.0.

TERM 5 -- PINGPONG, per active UE
---------------------------------
A handover (imsi X, src=A, dst=B) at t_now is a ping-pong if the same X made the
reverse handover (src=B, dst=A) at some t_prev with 0 < t_now - t_prev <= Y.
The completing leg is counted, never the outbound one, and each leg counts at
most once however many reverse legs sit in its window. Window math uses time_s
(exact seconds); each detected bounce is attributed to the stored KPM
`timestamp`, so it aligns with the step t of the other terms.
A missing or empty handovers table yields PingPong = 0 rather than an error.

The count is then divided by the number of active UEs in the network at that
step (sum over cells of numactiveues), making it a per-UE bounce rate that is
comparable across scenarios with different UE populations:

    pingpong_norm = pingpong_count / (sum_cells n_ues)

Guard: denominator == 0 -> 0.0.
"""

import sqlite3
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from load_metric import NR_CELLS, TOTAL_UES, build_load_view
from mlb_observation import median_mcs_at

DEFAULT_WEIGHTS = {
    "balance": 1.0,
    "satisfaction": 1.0,
    "backlog": 1.0,
    "badsignal": 1.0,
    "pingpong": 1.0,
}

# Accepted in a caller's `weights` dict and mapped onto the current name, so run
# scripts written against the deferred-Data era keep working.
WEIGHT_ALIASES = {"data": "satisfaction"}

# Terms that enter the total with a minus sign.
PENALTY_TERMS = ("backlog", "badsignal", "pingpong")

# --- TERM 1 (Balance) -----------------------------------------------------
BALANCE_MODES = ("jain_guarded", "weighted", "gate", "subset")

# Value returned by 'jain_guarded' on a nearly-empty network. 1.0 = a free
# maximum, the same convention _jain() uses for an all-zero vector and the same
# one balance_mode='gate' uses: "nothing to balance, so no gradient here".
BALANCE_GUARD_VALUE = 1.0

# The busiest cell must use at least this fraction of its PRBs before Jain's
# index is treated as meaningful. Loaded steps in this scenario run 0.61-0.81,
# so 0.05 separates "empty" from "operating" with two decades of margin.
LOW_LOAD_PRB_THRESHOLD = 0.05

# --- TERM 2 (Satisfaction) ------------------------------------------------
# PDCP-based DL throughput per UE, kbps. PER-UE (one distinct value per du row),
# so it is SUMMED, never .first()-ed. See the module docstring for why
# qosflowpdcppduvolumedl_filter[ueid] is not an acceptable substitute.
UDP_SAT_COL = "drbuethpdlpdcpbasedueid"

# Open-loop DL class in scenario-three.cc: node index u with u % 4 == 0, and
# IMSI == u + 1, so IMSI 1, 5, 9, ... Verified empirically on every validation
# run: exactly these UEs carry ~20.96 Mbps DL and exactly 0.000 Mbps UL.
UDP_IMSI_MODULUS = 4
UDP_IMSI_REMAINDER = 1          # (imsi % 4) == 1  <=>  ((imsi - 1) % 4) == 0

# Per-UE PDCP-layer offered rate for trafficModel=3 / configuration=0:
# 1280 B payload + 30 B of UDP/IP/PDCP headers every 500 us = 20960 kbps.
# configuration=2 halves the interval (40 Mbps) and must override this.
UDP_UE_RATE_KBPS_TM3 = 20960.0

# Satisfaction is a ratio that should not exceed 1.0. The slack absorbs header-
# accounting differences between numerator and denominator; the clip stops any
# residual mismatch from producing an unbounded term.
SATISFACTION_CLIP_MAX = 1.2

BADSIGNAL_COL = "l1mrssinrbin34"   # DL transmissions at SINR <= -6 dB, cell-level

# All seven L1M.RS-SINR bins, in threshold order (<=-6, <=0, <=6, <=12, <=18,
# <=24, else). Their sum is the total number of DL transmissions reported.
SINR_BIN_COLS = [
    "l1mrssinrbin34", "l1mrssinrbin46", "l1mrssinrbin58", "l1mrssinrbin70",
    "l1mrssinrbin82", "l1mrssinrbin94", "l1mrssinrbin127",
]

# Kept in diagnostics only: an independent count of the same MAC PDUs, so a
# future divergence from sum(SINR_BIN_COLS) is visible rather than silent.
SINR_BIN_TOTAL_CROSSCHECK = "tbtotnbrdl1"

# Cell-level du columns this module reads directly (build_load_view reads none
# of them). Every one must be identical across a cell's UE rows.
DU_EXTRA_COLS = (["rruprbuseddl", "dlavailableprbs", SINR_BIN_TOTAL_CROSSCHECK]
                 + SINR_BIN_COLS)


def _jain(values: np.ndarray) -> float:
    """Jain's fairness index of a non-negative vector; 1.0 for an all-zero vector."""
    total = float(values.sum())
    if total <= 0.0:
        return 1.0
    return float(total ** 2 / (len(values) * float((values ** 2).sum())))


def build_du_extras(db_path: str) -> pd.DataFrame:
    """One row per (cellid, timestamp) with the extra cell-level du columns.

    Columns: cellid, timestamp, rruprbuseddl, dlavailableprbs, the seven
    l1mrssinrbin* columns, tbtotnbrdl1, and the derived sinr_tx_total
    (= sum of the seven bins).

    Run-level and hoistable, like load_metric.build_load_view: pass the result
    into compute_reward() as `du_extras` to keep a stepwise loop from
    re-querying the whole run at every timestamp.

    These are cell-level columns repeated across the cell's UE rows, so they are
    collapsed with .first() -- never summed -- and every group is checked for
    agreement first, mirroring build_load_view's guard.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        du = pd.read_sql_query(
            "SELECT timestamp, ueimsicomplete, nrcellid AS cellid, "
            + ", ".join(DU_EXTRA_COLS)
            + " FROM du WHERE nrcellid BETWEEN 2 AND 8",
            con,
        )
    finally:
        con.close()

    if du.empty:
        return pd.DataFrame(columns=(["cellid", "timestamp"] + DU_EXTRA_COLS
                                     + ["sinr_tx_total"]))

    nunique = du.groupby(["cellid", "timestamp"])[DU_EXTRA_COLS].nunique(dropna=False)
    disagreeing = nunique[(nunique > 1).any(axis=1)]
    if not disagreeing.empty:
        raise ValueError(
            "cell-level du columns disagree within (cellid, timestamp) groups "
            f"({len(disagreeing)} groups):\n{disagreeing}")

    cell = du.groupby(["cellid", "timestamp"], as_index=False)[DU_EXTRA_COLS].first()
    cell["sinr_tx_total"] = cell[SINR_BIN_COLS].fillna(0.0).sum(axis=1)
    return cell


def udp_imsis(db_path: str) -> list:
    """The open-loop (UDP, downlink) IMSIs present in this run, sorted.

    Selects on IMSI arithmetic rather than on measured traffic, so the class is
    defined by the SCENARIO CONFIG and cannot drift with what the network
    happened to deliver. ueimsicomplete is a zero-padded TEXT column ('00005'),
    hence the CAST.

    verify_udp_class() checks the arithmetic against the traffic the run
    actually carried; this function does not, on purpose -- a UDP UE that was
    starved to silence must still be counted.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT CAST(ueimsicomplete AS INTEGER) AS imsi FROM du "
            "WHERE CAST(ueimsicomplete AS INTEGER) % ? = ? ORDER BY imsi",
            (UDP_IMSI_MODULUS, UDP_IMSI_REMAINDER),
        ).fetchall()
    finally:
        con.close()
    return [int(r[0]) for r in rows]


def build_udp_satisfaction(db_path: str, per_ue_rate_kbps: float = UDP_UE_RATE_KBPS_TM3,
                           n_udp: int | None = None) -> pd.DataFrame:
    """One row per timestamp with the UDP downlink satisfaction ratio.

    Columns: timestamp, udp_delivered_kbps, udp_demand_kbps, n_udp_rows,
    satisfaction (NaN where undefined).

    Run-level and hoistable like build_du_extras(): pass the result into
    compute_reward() as `udp_sat` to keep a stepwise loop from rescanning du.

    `n_udp` defaults to len(udp_imsis(db_path)) -- the run's UDP POPULATION, not
    a per-step row count. See the module docstring: making this per-step would
    let a starved UE shrink its own demand out of the denominator.

    `per_ue_rate_kbps` is the open-loop per-UE offered rate from the scenario
    config; the default is correct for trafficModel=3 / configuration=0 only.

    NOTE the aggregation: UDP_SAT_COL is PER-UE, so rows are SUMMED within a
    timestamp. Every other du column this module reads is cell-level and taken
    with .first(). Do not copy the wrong convention across.
    """
    imsis = udp_imsis(db_path)
    if n_udp is None:
        n_udp = len(imsis)
    n_udp = int(n_udp)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        du = pd.read_sql_query(
            "SELECT timestamp, CAST(ueimsicomplete AS INTEGER) AS imsi, "
            f"{UDP_SAT_COL} AS udp_kbps FROM du "
            "WHERE CAST(ueimsicomplete AS INTEGER) % ? = ?",
            con, params=(UDP_IMSI_MODULUS, UDP_IMSI_REMAINDER))
        all_ts = pd.read_sql_query("SELECT DISTINCT timestamp FROM du", con)
    finally:
        con.close()

    cols = ["timestamp", "udp_delivered_kbps", "udp_demand_kbps", "n_udp_rows",
            "satisfaction"]
    if all_ts.empty:
        return pd.DataFrame(columns=cols)

    # du is UNIQUE(timestamp, ueimsicomplete), so this sums each UE exactly once.
    per_ts = (du.groupby("timestamp")
                .agg(udp_delivered_kbps=("udp_kbps", lambda s: float(s.fillna(0.0).sum())),
                     n_udp_rows=("imsi", "size"))
                .reset_index())

    # Timestamps with no UDP rows at all must survive as NaN rows, not vanish.
    out = all_ts.merge(per_ts, on="timestamp", how="left").sort_values("timestamp")
    out["udp_delivered_kbps"] = out["udp_delivered_kbps"].fillna(0.0)
    out["n_udp_rows"] = out["n_udp_rows"].fillna(0).astype(int)
    out["udp_demand_kbps"] = float(n_udp) * float(per_ue_rate_kbps)

    demand_ok = out["udp_demand_kbps"] > 0.0          # n_udp == 0 -> undefined
    rows_ok = out["n_udp_rows"] > 0                   # nothing measured -> undefined
    sat = out["udp_delivered_kbps"] / out["udp_demand_kbps"].where(demand_ok)
    out["satisfaction"] = sat.where(demand_ok & rows_ok).clip(0.0, SATISFACTION_CLIP_MAX)
    return out[cols].reset_index(drop=True)


def verify_udp_class(db_path: str, per_ue_rate_kbps: float = UDP_UE_RATE_KBPS_TM3,
                     tolerance: float = 0.25) -> dict:
    """Check that the IMSI-arithmetic UDP class matches the run's actual traffic.

    Confirms the two properties the Satisfaction denominator rests on:
      1. the selected UEs carry a DL PDCP-arrival rate close to per_ue_rate_kbps,
      2. the non-selected UEs carry essentially none (they are uplink TCP).

    Uses lte_cu_up.m_pdcpbytesdlcelldltxvolume (cell-level DL PDCP ARRIVAL
    volume, kbit per KPM window) for the network total, since the per-UE CU-UP
    columns are NULL in these databases. Returns a dict of measurements plus
    `ok`; a caller that wants a hard failure should assert on it.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        udp = con.execute(
            f"SELECT AVG(t) FROM (SELECT timestamp, SUM({UDP_SAT_COL}) AS t FROM du "
            "WHERE CAST(ueimsicomplete AS INTEGER) % ? = ? GROUP BY timestamp)",
            (UDP_IMSI_MODULUS, UDP_IMSI_REMAINDER)).fetchone()[0]
        other = con.execute(
            f"SELECT AVG(t) FROM (SELECT timestamp, SUM({UDP_SAT_COL}) AS t FROM du "
            "WHERE CAST(ueimsicomplete AS INTEGER) % ? != ? GROUP BY timestamp)",
            (UDP_IMSI_MODULUS, UDP_IMSI_REMAINDER)).fetchone()[0]
        try:
            demand = con.execute(
                "SELECT AVG(v) FROM (SELECT timestamp, MAX(m_pdcpbytesdlcelldltxvolume) AS v "
                "FROM lte_cu_up GROUP BY timestamp)").fetchone()[0]
        except sqlite3.OperationalError:
            demand = None
    finally:
        con.close()

    imsis = udp_imsis(db_path)
    n = len(imsis)
    expected = n * per_ue_rate_kbps
    # m_pdcpbytesdlcelldltxvolume is kbit per 0.1 s window -> x10 for kbps.
    measured_demand = None if demand is None else float(demand) * 10.0
    ratio = None if not measured_demand else measured_demand / expected if expected else None
    return {
        "n_udp": n,
        "udp_imsis": imsis,
        "expected_demand_kbps": expected,
        "measured_dl_arrival_kbps": measured_demand,
        "measured_over_expected": ratio,
        "udp_delivered_kbps_mean": None if udp is None else float(udp),
        "non_udp_delivered_kbps_mean": None if other is None else float(other),
        "ok": bool(n > 0 and ratio is not None and abs(ratio - 1.0) <= tolerance),
    }


def badsignal_at(db_path: str, timestamp: int,
                 du_extras: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-cell bad-signal numerator and denominator at one timestamp.

    Returns a frame indexed by cellid (reindexed onto NR_CELLS, zero-filled)
    with columns `bad` (l1mrssinrbin34) and `total` (sum of all seven SINR
    bins). Cells absent from du at this timestamp contribute zero to both.
    """
    if du_extras is None:
        du_extras = build_du_extras(db_path)

    step = du_extras[du_extras["timestamp"] == timestamp]
    out = pd.DataFrame(0.0, index=NR_CELLS, columns=["bad", "total"])
    out.index.name = "cellid"
    if step.empty:
        return out

    step = step.set_index("cellid")
    out["bad"] = step[BADSIGNAL_COL].reindex(NR_CELLS).fillna(0.0)
    out["total"] = step["sinr_tx_total"].reindex(NR_CELLS).fillna(0.0)
    return out


def pingpong_per_timestamp(db_path: str, Y: float = 0.8) -> dict:
    """{kpm_timestamp: number of ping-pong legs completed in that step}.

    Returns {} if the handovers table is absent or empty, so a run ingested
    without handover events still produces a reward.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        try:
            rows = con.execute(
                "SELECT timestamp, time_s, imsi, src_cell, dst_cell "
                "FROM handovers ORDER BY time_s").fetchall()
        except sqlite3.OperationalError:
            return {}          # no handovers table in this database
    finally:
        con.close()

    counts = Counter()
    seen = defaultdict(list)   # imsi -> [(time_s, src, dst)] already processed
    for kpm_ts, time_s, imsi, src, dst in rows:
        for prev_t, prev_src, prev_dst in reversed(seen[imsi]):
            if prev_t < time_s - Y:
                break          # sorted by time: everything older is out of window
                               # (strict, so a leg exactly Y back still counts)
            if prev_src == dst and prev_dst == src and time_s - prev_t > 0:
                counts[kpm_ts] += 1
                break          # one bounce per completing leg
        seen[imsi].append((time_s, src, dst))
    return dict(counts)


def _balance(prb_pct: np.ndarray, prb_frac: np.ndarray, ci: np.ndarray,
             mode: str, overload_prb_threshold: float,
             low_load_prb_threshold: float = LOW_LOAD_PRB_THRESHOLD) -> tuple:
    """Return (balance, jain_all, load_weight, n_overloaded, n_loaded, guarded).

    See the module docstring for the four modes. `load_weight` is computed under
    every mode so it stays visible in diagnostics, but only the retired
    'weighted' mode multiplies by it. `guarded` is True only when
    'jain_guarded' took its nearly-empty-network early return.
    """
    if mode not in BALANCE_MODES:
        raise ValueError(
            f"balance_mode must be one of {BALANCE_MODES}, got {mode!r}")

    overloaded = prb_pct > overload_prb_threshold
    loaded = ci > 0.0
    jain_all = _jain(ci)

    # Busiest cell's PRB fraction. Clamped to 1.0: prb_frac is a fraction by
    # construction, but a data quirk (rruprbuseddl > dlavailableprbs) must not
    # be allowed to push Balance above 1.
    max_prb_frac = float(prb_frac.max()) if prb_frac.size else 0.0
    load_weight = float(min(max(max_prb_frac, 0.0), 1.0))

    guarded = False
    if mode == "jain_guarded":
        # Nearly-empty network: there is nothing to balance, so return the
        # neutral value EXPLICITLY rather than letting Jain's index report a
        # near-empty grid as severe unfairness (one trickling cell and six
        # zeros scores 1/7). Flagged so a guarded 1.0 is never mistaken for a
        # genuine one.
        if float(ci.sum()) <= 0.0 or max_prb_frac < low_load_prb_threshold:
            guarded = True
            balance = BALANCE_GUARD_VALUE
        else:
            balance = jain_all
    elif mode == "weighted":
        balance = load_weight * jain_all
    elif mode == "gate":
        balance = jain_all if overloaded.any() else 1.0
    else:   # 'subset'
        balance = _jain(ci[loaded]) if loaded.sum() > 1 else 1.0

    return (balance, jain_all, load_weight,
            int(overloaded.sum()), int(loaded.sum()), guarded)


# Bytes delivered in one KPM window -> bytes per second. The KPM indication
# period is 0.1 s (scenario_configurations/*.json: indicationPeriodicity), so
# the window volume is multiplied by 1/0.1 = 10.
KPM_PERIOD_S = 0.1


def compute_reward(db_path: str, timestamp: int, view: pd.DataFrame | None = None,
                   weights: dict | None = None, tau: float = 1.0,
                   overload_prb_threshold: float = 80.0,
                   balance_mode: str = "jain_guarded", Y: float = 0.8,
                   pingpong_counts: dict | None = None,
                   du_extras: pd.DataFrame | None = None,
                   total_ues: int | None = None,
                   udp_sat: pd.DataFrame | None = None,
                   udp_rate_kbps: float = UDP_UE_RATE_KBPS_TM3,
                   n_udp: int | None = None,
                   low_load_prb_threshold: float = LOW_LOAD_PRB_THRESHOLD) -> dict:
    """All five reward terms for one step, with the full breakdown and total.

    Each term is reported both RAW (the underlying counts/bytes) and in its
    SCALE-FREE form, so the two can be compared.

    `reward` is the signed weighted total, or None at a step where Satisfaction
    is undefined (see the module docstring's TERM 2 guards) -- a 4-of-5 total
    would not be comparable with the rest of the run.

    `view`, `du_extras`, `pingpong_counts` and `udp_sat` let a caller hoist the
    four whole-run scans out of a stepwise loop; each is rebuilt on demand when
    omitted. `udp_rate_kbps` and `n_udp` are ignored when `udp_sat` is supplied
    -- pass them to build_udp_satisfaction() instead.

    `total_ues` defaults to load_metric.TOTAL_UES (21). It only rescales the CI
    vector by a constant and Jain's index is invariant to that, so it does not
    change Balance -- see the module docstring.
    """
    if weights is None:
        w = dict(DEFAULT_WEIGHTS)
    else:
        renamed = {WEIGHT_ALIASES.get(k, k): v for k, v in weights.items()}
        w = {**DEFAULT_WEIGHTS, **renamed}

    n_total_ues = TOTAL_UES if total_ues is None else int(total_ues)

    if view is None:
        view = build_load_view(db_path)
    if du_extras is None:
        du_extras = build_du_extras(db_path)
    if udp_sat is None:
        udp_sat = build_udp_satisfaction(db_path, per_ue_rate_kbps=udp_rate_kbps,
                                         n_udp=n_udp)

    step = view[view["timestamp"] == timestamp]
    if step.empty:
        raise ValueError(f"timestamp {timestamp} not present in {db_path}")
    step = step.set_index("cellid").reindex(NR_CELLS)

    prb_pct = step["prb_pct"].fillna(0.0).to_numpy(dtype=float)
    n_ues = step["n_ues"].fillna(0.0).to_numpy(dtype=float)
    median_mcs = (median_mcs_at(db_path, timestamp)
                  .reindex(NR_CELLS).fillna(0.0).to_numpy(dtype=float))

    extras_step = (du_extras[du_extras["timestamp"] == timestamp]
                   .set_index("cellid").reindex(NR_CELLS))
    prb_used = extras_step["rruprbuseddl"].fillna(0.0).to_numpy(dtype=float)
    prb_avail = extras_step["dlavailableprbs"].to_numpy(dtype=float)

    # --- Term 1: Balance (scale-free: Jain's index, [0,1]) ----------------
    # PRB fraction from the RAW du columns. dlavailableprbs is read, never
    # assumed: a cell absent from du this step has NaN here, and an absent cell
    # served no UEs, so its PRB fraction is 0 either way.
    with np.errstate(divide="ignore", invalid="ignore"):
        prb_frac = np.where(np.isfinite(prb_avail) & (prb_avail > 0),
                            prb_used / prb_avail, 0.0)
    ci = prb_frac * (n_ues / n_total_ues) * (median_mcs > tau)
    balance, jain_all, load_weight, n_overloaded, n_loaded, balance_guarded = _balance(
        prb_pct, prb_frac, ci, balance_mode, overload_prb_threshold,
        low_load_prb_threshold)

    # --- Term 2: Satisfaction (UDP DL delivered / demanded) ---------------
    # Denominator is the run-level UDP population times the config offered rate,
    # both exogenous to the agent. NaN -> the step is undefined and the combined
    # reward below is withheld. The old raw byte sum is kept, unweighted and
    # unused, as raw['data_bytes'].
    data_raw = float(step["vol_bytes"].fillna(0.0).sum())   # kept for inspection
    sat_step = udp_sat[udp_sat["timestamp"] == timestamp]
    if sat_step.empty:
        satisfaction = float("nan")
        udp_delivered_kbps = 0.0
        udp_demand_kbps = 0.0
        n_udp_rows = 0
    else:
        row = sat_step.iloc[0]
        satisfaction = float(row["satisfaction"])
        udp_delivered_kbps = float(row["udp_delivered_kbps"])
        udp_demand_kbps = float(row["udp_demand_kbps"])
        n_udp_rows = int(row["n_udp_rows"])

    satisfaction_defined = bool(np.isfinite(satisfaction))
    satisfaction_no_udp_ues = udp_demand_kbps <= 0.0
    satisfaction_no_rows_at_step = n_udp_rows <= 0

    # --- Term 3: Backlog, as seconds of delay -----------------------------
    backlog_bytes = float(step["buffer_bytes"].fillna(0.0).sum())
    delivered_bytes = float(step["vol_bytes"].fillna(0.0).sum())
    delivery_rate_bps = delivered_bytes / KPM_PERIOD_S      # bytes per second
    backlog_denominator_zero = delivery_rate_bps <= 0.0
    # Nothing delivered this step means there is no drain rate to divide by --
    # and no traffic means no queueing delay to charge for, so 0.0, not inf.
    backlog_sec = (0.0 if backlog_denominator_zero
                   else backlog_bytes / delivery_rate_bps)

    # --- Term 4: BadSignal, as a fraction of DL transmissions -------------
    bad_per_cell = badsignal_at(db_path, timestamp, du_extras=du_extras)
    badsignal_count = float(bad_per_cell["bad"].sum())
    badsignal_total_tx = float(bad_per_cell["total"].sum())
    badsignal_denominator_zero = badsignal_total_tx <= 0.0
    badsignal_frac = (0.0 if badsignal_denominator_zero
                      else badsignal_count / badsignal_total_tx)

    # --- Term 5: PingPong, per active UE ----------------------------------
    if pingpong_counts is None:
        pingpong_counts = pingpong_per_timestamp(db_path, Y=Y)
    pingpong_count = float(pingpong_counts.get(int(timestamp), 0))
    active_ues = float(np.nansum(n_ues))
    pingpong_denominator_zero = active_ues <= 0.0
    pingpong_norm = (0.0 if pingpong_denominator_zero
                     else pingpong_count / active_ues)

    # The scale-free value of each term -- what the reward actually uses.
    scale_free = {
        "balance": balance,
        "satisfaction": satisfaction,    # NaN where undefined
        "backlog": backlog_sec,
        "badsignal": badsignal_frac,
        "pingpong": pingpong_norm,
    }
    # The underlying un-normalised quantity, kept for inspection only.
    raw = {
        "balance_jain_all_cells": jain_all,     # ungated Jain, pre-guard
        "balance_load_weight": load_weight,     # clamped max_i prb_frac, unused
        "balance_max_prb_frac": float(prb_frac.max()) if prb_frac.size else 0.0,
        "satisfaction_delivered_kbps": udp_delivered_kbps,
        "satisfaction_demand_kbps": udp_demand_kbps,
        "data_bytes": data_raw,                 # retired raw Data term, unused
        "backlog_bytes": backlog_bytes,
        "badsignal_count": badsignal_count,
        "pingpong_count": pingpong_count,
    }
    # Signed, weighted terms. An undefined Satisfaction is dropped here AND
    # suppresses the total, so a 4-term sum can never masquerade as a 5-term one.
    weighted = {name: (-w[name] * value if name in PENALTY_TERMS else w[name] * value)
                for name, value in scale_free.items()
                if value is not None and np.isfinite(value)}
    reward = float(sum(weighted.values())) if satisfaction_defined else None

    return {
        "reward": reward,
        "timestamp": int(timestamp),
        **scale_free,
        **raw,
        "terms_weighted": weighted,
        "weights": w,
        "diagnostics": {
            "ci": dict(zip(NR_CELLS, ci.round(6))),
            "prb_frac": dict(zip(NR_CELLS, prb_frac.round(6))),
            "prb_used": dict(zip(NR_CELLS, prb_used)),
            "prb_avail": dict(zip(NR_CELLS, prb_avail)),
            "prb_pct": dict(zip(NR_CELLS, prb_pct)),
            "n_ues": dict(zip(NR_CELLS, n_ues)),
            "median_mcs": dict(zip(NR_CELLS, median_mcs)),
            "badsignal_per_cell": bad_per_cell["bad"].to_dict(),
            "badsignal_total_tx": badsignal_total_tx,
            "badsignal_total_tx_crosscheck": float(
                extras_step[SINR_BIN_TOTAL_CROSSCHECK].fillna(0.0).sum()),
            "delivered_bytes": delivered_bytes,
            "delivery_rate_bytes_per_s": delivery_rate_bps,
            "active_ues": active_ues,
            "jain_all_cells": jain_all,
            "load_weight": load_weight,
            "max_prb_frac": float(prb_frac.max()) if prb_frac.size else 0.0,
            "n_overloaded": n_overloaded,
            "n_loaded": n_loaded,
            "balance_mode": balance_mode,
            # True only when 'jain_guarded' took its nearly-empty-network early
            # return, so a guarded 1.0 is distinguishable from a genuine one.
            "balance_guarded": bool(balance_guarded),
            "low_load_prb_threshold": low_load_prb_threshold,
            "udp_delivered_kbps": udp_delivered_kbps,
            "udp_demand_kbps": udp_demand_kbps,
            "n_udp_rows": n_udp_rows,
            "satisfaction_defined": satisfaction_defined,
            "satisfaction_no_udp_ues": bool(satisfaction_no_udp_ues),
            "satisfaction_no_rows_at_step": bool(satisfaction_no_rows_at_step),
            # Only meaningful under balance_mode='gate'; reported under every
            # mode so run records stay comparable.
            "gate_open": bool(n_overloaded > 0),
            "tau": tau,
            "total_ues": n_total_ues,
            "overload_prb_threshold": overload_prb_threshold,
            "Y": Y,
            "kpm_period_s": KPM_PERIOD_S,
            "backlog_denominator_zero": bool(backlog_denominator_zero),
            "badsignal_denominator_zero": bool(badsignal_denominator_zero),
            "pingpong_denominator_zero": bool(pingpong_denominator_zero),
        },
    }
