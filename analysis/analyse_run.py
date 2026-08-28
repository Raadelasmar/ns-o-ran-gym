"""POST-TRAINING ANALYSIS — run this on the next real run, unedited.

Defined BEFORE the run on purpose: the pilot showed two ways to fool yourself
with this data, and both are easier to avoid than to undo once you have started
improvising against numbers you have already seen.

    python3 analyse_run.py <log_dir> [--tailer /path/reward_terms.csv]

<log_dir> is the --log_dir of examples/train_mlb_parallel.py: it holds
reward_terms_steps.csv, reward_terms_steps_meta.json and monitor_rank*.monitor.csv.

THE TWO TRAPS, both measured on the 2026-08-21/22 pilot:

 1. CORRELATE AGAINST THE BACKLOG TERM, NOT RAW QUEUE BYTES. Backlog is a DRAIN
    TIME (buf / delivery rate, FIX 13). On the pilot, 226 of 690 steps had reward
    UP while raw buffer bytes rose -- 32.8 % against a 25 % chance rate, which
    reads as reward-hacking. It was not: on exactly those steps the backlog TERM
    FELL, because the drain rate rose faster than the queue. Judging a drain-time
    penalty by raw bytes manufactures a false positive.

 2. CLUSTER THE LEARNING TEST. vary_rng_run_per_episode=True redraws the ns-3
    seed every episode, so episodes are not independent samples of a policy. On
    the pilot, a per-episode regression gave t(23)=2.19 "significant"; the
    clustered test over the 5 episode-waves gave t(3)=1.49, not significant --
    and std=1.0 / clip_fraction=0 confirmed the policy had never moved. Always
    report the clustered number, and always read it next to the SB3 log's std
    and clip_fraction.

 3. (New.) Use phase == "step" rows for anything about the terms. The phase ==
    "terminal" row is the time-limit end: reward 0.0 and NO terms. Including it
    drags every episode mean down by ~1/30.
"""
import argparse, csv, glob, json, math, sys
from collections import defaultdict
from os import path

import numpy as np

CELLS = [2, 3, 4, 5, 6, 7, 8]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def tstat(r, n):
    if not np.isfinite(r) or n < 3 or abs(r) >= 1:
        return np.nan
    return r * math.sqrt((n - 2) / (1 - r * r))


def col(rows, name):
    return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])


def _fmt(x):
    return f"{x:+.3f}" if np.isfinite(x) else " (n/a)"


def lever_report(live, meta):
    """Did the agent steer cells RELATIVE to each other, or just move them together?

    A CIO vector decomposes into a common mode (the mean across cells) and a
    differential (the deviation from it). Handover decisions compare cells, so
    ONLY the differential can move a UE: adding +3 dB to all seven cells is an
    exact no-op. An agent whose action variance is nearly all common mode is
    doing nothing, however busy its action trace looks -- and the scalar reward
    cannot tell you that, which is why the per-cell columns exist.
    """
    cio = np.column_stack([col(live, f"cio_offset_cell{c}") for c in CELLS])
    raw = np.column_stack([col(live, f"action_raw_cell{c}") for c in CELLS])
    n_ues = np.column_stack([col(live, f"n_ues_cell{c}") for c in CELLS])
    prb = np.column_stack([col(live, f"prb_utilization_cell{c}") for c in CELLS])
    lim = max(meta.get("action_space_high", [6.0]))

    common = np.nanmean(cio, axis=1)
    diff = cio - common[:, None]
    v_cio = float(np.nanvar(cio))
    v_diff = float(np.nanvar(diff))
    ratio = v_diff / v_cio if v_cio > 0 else np.nan

    # THE NULL IS NOT ZERO. For k cells acting independently, var(diff)/var(cio)
    # = (k-1)/k = 0.857 at k=7 -- an UNTRAINED Gaussian policy scores ~0.9 here
    # and reads as "genuinely steering" if the ratio is judged against 0. What
    # distinguishes a policy is whether the differential is STRUCTURED (persists
    # step to step, tracks load), not whether it exists.
    null = (len(CELLS) - 1) / len(CELLS)
    if not np.isfinite(ratio):
        verdict_class = "constant"
        verdict = "no action variance at all -- the policy is constant"
    elif ratio < 0.10:
        verdict_class = "noop"
        verdict = ("NO-OP: >90% of the action variance is common mode. The agent is "
                   "moving all cells together, which cannot move a single UE.")
    elif ratio < 0.50:
        verdict_class = "mixed"
        verdict = (f"MIXED: {100*ratio:.0f}% differential vs a {100*null:.0f}% iid null. "
                   "Actively suppressed steering -- most of the action does nothing.")
    else:
        verdict_class = "differential"
        verdict = (f"{100*ratio:.0f}% differential, iid null is {100*null:.0f}% -- "
                   "differential EXISTS, but see persistence/load-tracking below for "
                   "whether it is STRUCTURED or just per-cell noise.")

    # Pair each step's differential with the NEXT step's change in n_ues, within
    # an episode only -- a reset would otherwise manufacture a huge fake delta.
    eps = defaultdict(list)
    for i, r in enumerate(live):
        eps[(int(r["env_rank"]), int(r["episode_index"]))].append(i)
    d_list, n_list, p_list, cos = [], [], [], []
    for idx in eps.values():
        idx = sorted(idx, key=lambda j: int(live[j]["episode_step"]))
        if len(idx) < 2:
            continue
        D, N = diff[idx], n_ues[idx]
        d_list.append(D[:-1].ravel())
        n_list.append((N[1:] - N[:-1]).ravel())
        p_list.append(prb[idx].ravel())
        for a, b in zip(D[:-1], D[1:]):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-9 and nb > 1e-9:
                cos.append(float(a @ b / (na * nb)))

    # NUMERICAL-DUST GUARD. When the policy moves all cells together the
    # differential is zero in principle but ~1e-16 in floating point, and any
    # quantity derived from it correlates perfectly with that dust -- MEASURED:
    # a pure common-mode action reported lever_corr = +0.59 before this guard.
    # A differential below 1e-6 dB cannot influence an A3 comparison (hysteresis
    # is order 1 dB), so there is nothing there to correlate with.
    dust = float(np.nanstd(diff)) < 1e-6
    lever = (np.nan if (dust or not d_list)
             else corr(np.concatenate(d_list), np.concatenate(n_list)))
    return {
        "dust": dust,
        "null_ratio": null,
        "cio_limit": lim,
        "raw_abs_mean": float(np.nanmean(np.abs(raw))),
        "raw_abs_max": float(np.nanmax(np.abs(raw))),
        "clip_frac": float(np.nanmean(np.abs(raw) > lim + 1e-9)),
        "common_abs_mean": float(np.nanmean(np.abs(common))),
        "common_sd": float(np.nanstd(common)),
        "diff_abs_mean": float(np.nanmean(np.abs(diff))),
        "diff_sd": float(np.nanstd(diff)),
        "diff_ratio": ratio,
        "spread": float(np.nanmean(np.nanstd(cio, axis=1))),
        "verdict": verdict,
        # Machine-readable, so a test never has to match on prose.
        "verdict_class": verdict_class,
        "cio_by_cell": np.nanmean(cio, axis=0),
        "diff_by_cell": np.nanmean(diff, axis=0),
        "nues_by_cell": np.nanmean(n_ues, axis=0),
        "prb_by_cell": np.nanmean(prb, axis=0),
        "lever_corr": lever,
        "lever_n": int(sum(len(x) for x in d_list)),
        "load_response": np.nan if dust else corr(diff.ravel(), prb.ravel()),
        "persistence": float(np.mean(cos)) if cos else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--tailer", default=None, help="the tailer's reward_terms.csv")
    args = ap.parse_args()

    csv_path = path.join(args.log_dir, "reward_terms_steps.csv")
    rows = list(csv.DictReader(open(csv_path)))
    meta = json.load(open(path.join(args.log_dir, "reward_terms_steps_meta.json")))
    live = [r for r in rows if r["phase"] == "step"]
    print(f"{len(rows)} rows  ({len(live)} phase=step, "
          f"{sum(1 for r in rows if r['phase'] == 'terminal')} terminal, "
          f"{sum(1 for r in rows if r['phase'] == 'void')} void)  "
          f"{meta['n_envs']} envs")

    # ---- 0. INTEGRITY -------------------------------------------------------
    print("\n" + "=" * 74 + "\n0. INTEGRITY\n" + "=" * 74)
    grid = {(int(r["env_rank"]), int(r["vec_step"])) for r in rows}
    n_vec = max(int(r["vec_step"]) for r in rows) + 1
    print(f"  (env_rank, vec_step) grid complete : "
          f"{len(grid) == n_vec * meta['n_envs']}  ({len(grid)}/{n_vec * meta['n_envs']})")
    for f in ("balance", "backlog", "badsignal", "satisfaction", "pingpong"):
        miss = sum(1 for r in live if r[f] == "")
        print(f"  {f:13s} missing on {miss:5d} / {len(live)} step rows"
              + ("   <-- TERM WAS DROPPED FROM THE SUM" if miss else ""))
    for f in ("pingpong_unavailable", "satisfaction_unavailable", "pdcp_field_present"):
        vals = {r[f] for r in live}
        print(f"  {f:24s} values seen: {sorted(vals)}")

    # StepLogger vs Monitor -- the independent per-episode cross-check.
    sums, lens, closed = defaultdict(float), defaultdict(int), set()
    for r in rows:
        k = (int(r["env_rank"]), int(r["episode_index"]))
        sums[k] += float(r["reward"]) if r["reward"] != "" else 0.0
        lens[k] += 1
        if r["done"] == "1":
            closed.add(k)
    worst, n_cmp = 0.0, 0
    for f in sorted(glob.glob(path.join(args.log_dir, "monitor_rank*.monitor.csv"))):
        rank = int(path.basename(f).split("monitor_rank")[1].split(".")[0])
        with open(f) as fh:
            next(fh)
            for ep, m in enumerate(csv.DictReader(fh)):
                if (rank, ep) not in closed:
                    continue
                worst = max(worst, abs(sums[(rank, ep)] - float(m["r"])))
                n_cmp += 1
    print(f"  StepLogger vs Monitor: {n_cmp} episodes, worst |diff| {worst:.2e} "
          f"({'OK' if worst < 1e-5 else 'MISMATCH -- STOP AND INVESTIGATE'})")

    # ---- 1. REWARD-HACKING --------------------------------------------------
    print("\n" + "=" * 74 + "\n1. REWARD-HACKING  (trap 1: judge by the TERM, not raw bytes)\n" + "=" * 74)
    rew = col(live, "reward")
    phys = {"backlog TERM (drain s)": col(live, "backlog"),
            "delivery_rate_bytes_per_s": col(live, "delivery_rate_bytes_per_s"),
            "delivered_bytes": col(live, "delivered_bytes"),
            "backlog_bytes (RAW -- see trap 1)": col(live, "backlog_bytes"),
            "active_ues": col(live, "active_ues"),
            "max_prb_utilization": col(live, "max_prb_utilization"),
            "handovers_this_step": col(live, "handovers_this_step"),
            "pingpong_count": col(live, "pingpong_count"),
            "episode_step": col(live, "episode_step")}
    print("  pooled corr(reward, .):")
    for k, v in phys.items():
        c = corr(rew, v)
        # A constant column has no correlation to report -- say so rather than
        # printing nan, which reads like a broken column.
        print(f"     {k:36s} " + (f"{c:+.3f}" if np.isfinite(c)
                                  else "  (constant)" if np.nanstd(v) == 0 else "  (n/a)"))

    eps = defaultdict(list)
    for r in live:
        eps[(int(r["env_rank"]), int(r["episode_index"]))].append(r)
    def within(a, b):
        v = np.array([corr(col(rs, a), col(rs, b)) for rs in eps.values()])
        v = v[np.isfinite(v)]
        return v
    print(f"\n  within-episode corr (mean over {len(eps)} episodes, sign consistency):")
    for a, b in [("reward", "backlog"), ("reward", "delivery_rate_bytes_per_s"),
                 ("reward", "balance"), ("reward", "satisfaction"),
                 ("reward", "pingpong"), ("reward", "episode_step"),
                 ("backlog_bytes", "episode_step"), ("backlog", "episode_step")]:
        v = within(a, b)
        if not len(v):
            continue
        neg = int((v < 0).sum())
        print(f"     {a:12s} vs {b:26s} {v.mean():+.3f}  sd {v.std():.3f}  "
              f"neg {neg}/{len(v)}")

    print("\n  SIGNATURE TEST — reward UP while a physical quantity WORSENS (chance ~25%):")
    tests = {"backlog TERM rises (the real test)": ("backlog", +1),
             "delivery rate falls": ("delivery_rate_bytes_per_s", -1),
             "raw queue bytes rise (FALSE POSITIVE PRONE)": ("backlog_bytes", +1)}
    for label, (field, sign) in tests.items():
        hit = tot = 0
        for rs in eps.values():
            dr, dx = np.diff(col(rs, "reward")), np.diff(col(rs, field))
            m = np.isfinite(dr) & np.isfinite(dx)
            hit += int(((dr[m] > 0) & (sign * dx[m] > 0)).sum()); tot += int(m.sum())
        if tot:
            print(f"     {label:46s} {hit:5d}/{tot} = {100*hit/tot:5.1f}%")
    print("     >> Only the FIRST line is evidence of hacking. If it is elevated and")
    print("        the others are not, decompose those steps by weighted term delta.")

    # ---- 2. TERM SHARES -----------------------------------------------------
    print("\n" + "=" * 74 + "\n2. WHAT MOVES THE REWARD\n" + "=" * 74)
    W = {"balance": meta.get("w_balance", 1.0), "backlog": -meta.get("w_backlog", 0.1),
         "badsignal": -meta.get("w_badsignal", 1.0),
         "satisfaction": meta.get("w_satisfaction", 1.1),
         "pingpong": -meta.get("w_pingpong", 1.5)}
    contrib = {k: w * col(live, k) for k, w in W.items()}
    tot_sd = sum(np.nanstd(v) for v in contrib.values())
    for k, v in sorted(contrib.items(), key=lambda x: -np.nanstd(x[1])):
        print(f"  {k:13s} weighted mean {np.nanmean(v):+7.3f}  sd {np.nanstd(v):6.3f}  "
              f"share {100*np.nanstd(v)/tot_sd:5.1f}%  corr w/ reward {corr(v, rew):+.3f}")

    # ---- 3. LEARNING --------------------------------------------------------
    print("\n" + "=" * 74 + "\n3. LEARNING  (trap 2: cluster, do not regress on episodes)\n" + "=" * 74)
    keys = sorted(eps)
    ep_mean = np.array([np.nanmean(col(eps[k], "reward")) for k in keys])
    ep_idx = np.array([k[1] for k in keys])
    r_naive = corr(ep_idx, ep_mean)
    print(f"  NAIVE per-episode  : r={r_naive:+.3f}  t({len(keys)-2})={tstat(r_naive, len(keys)):.2f}"
          f"   <-- DO NOT REPORT THIS")
    waves = sorted(set(ep_idx))
    wave_mean = np.array([ep_mean[ep_idx == w].mean() for w in waves])
    r_cl = corr(np.array(waves, float), wave_mean)
    n_w = len(waves)
    t_cl = tstat(r_cl, n_w)
    print(f"  CLUSTERED by wave  : r={r_cl:+.3f}  t({n_w-2})={t_cl:.2f}  n={n_w} waves"
          f"   <-- REPORT THIS")
    print(f"     wave means: {np.round(wave_mean, 3).tolist()}")
    print("  >> Cross-read against the SB3 log: if std is still ~1.0 and clip_fraction")
    print("     is 0, the policy did not move and any trend here is seed noise.")

    # ---- 4. THE LEVER ------------------------------------------------------
    print("\n" + "=" * 74 + "\n4. WHAT THE AGENT DID WITH THE LEVER\n" + "=" * 74)
    rep = lever_report(live, meta)
    lim = rep["cio_limit"]
    print(f"  raw action  |mean| {rep['raw_abs_mean']:.3f} dB   max |.| {rep['raw_abs_max']:.3f} dB")
    print(f"  clipped at +/-{lim} dB on {100*rep['clip_frac']:.2f}% of cell-actions"
          + ("   <-- SATURATING: the policy wants more range than it has"
             if rep["clip_frac"] > 0.05 else ""))
    print()
    print("  DIFFERENTIAL vs COMMON MODE  (only DIFFERENCES between cells move UEs:")
    print("  adding the same dB to every cell leaves every A3 comparison unchanged)")
    print(f"     common-mode  |mean| {rep['common_abs_mean']:.4f} dB   sd {rep['common_sd']:.4f}")
    print(f"     differential |mean| {rep['diff_abs_mean']:.4f} dB   sd {rep['diff_sd']:.4f}")
    print(f"     DIFFERENTIAL RATIO  var(diff)/var(cio) = {rep['diff_ratio']:.3f}"
          f"   (iid-policy null = {rep['null_ratio']:.3f} -- NOT 0)")
    print(f"     spread across cells within a step      = {rep['spread']:.4f} dB")
    print(f"     VERDICT: {rep['verdict']}")
    print()
    print("  per-cell mean APPLIED cio / mean DIFFERENTIAL / mean n_ues / mean prb:")
    for i, c in enumerate(CELLS):
        print(f"     cell {c}: cio {rep['cio_by_cell'][i]:+.3f}   diff {rep['diff_by_cell'][i]:+.3f}"
              f"   n_ues {rep['nues_by_cell'][i]:6.2f}   prb {rep['prb_by_cell'][i]:.3f}")
    print()
    print("  DOES THE LEVER ACTUALLY MOVE UEs?  (differential CIO on a cell at step t")
    print("  vs that cell's CHANGE in n_ues from t to t+1; higher CIO = more attractive,")
    print("  so a working lever is POSITIVE)")
    if rep["dust"]:
        print("     >> DIFFERENTIAL IS NUMERICALLY ZERO. There is no bias to correlate")
        print("        with, so the two correlations below are suppressed, not missing.")
    print(f"     corr(diff_cio[t], d_n_ues[t->t+1]) = {_fmt(rep['lever_corr'])}   "
          f"n={rep['lever_n']} cell-steps")
    print(f"     corr(diff_cio[t], prb[t])          = {_fmt(rep['load_response'])}   "
          f"(negative = biasing AWAY from busy cells, i.e. sensible offload)")
    print(f"     step-to-step persistence of the differential (cosine) = "
          f"{_fmt(rep['persistence'])}")
    print("     >> Near 0 persistence means the agent re-rolls its bias every step;")
    print("        Step 6-final measured that churn costs more than it buys.")
    if not rep["dust"] and rep["diff_ratio"] < 0.10:
        print("     >> Verdict is NO-OP, so read the lever correlation with suspicion:")
        print("        it is computed over a bias too small to move a handover.")

    # ---- 5. TAILER CROSS-CHECK ---------------------------------------------
    if args.tailer and path.exists(args.tailer):
        print("\n" + "=" * 74 + "\n5. INDEPENDENT CROSS-CHECK vs THE TAILER\n" + "=" * 74)
        # The tailer keys rows by ns-3 run uuid, StepLogger by rng_run, and
        # nothing currently records the mapping -- so this compares
        # DISTRIBUTIONS, not paired rows. That is enough to catch a scorer that
        # has drifted, which is all this cross-check is for.
        trows = list(csv.DictReader(open(args.tailer)))
        a = np.array([float(t["reward"]) for t in trows])
        b = col(live, "reward")
        print(f"  tailer   n={len(a):5d}  mean {a.mean():+.4f}  sd {a.std():.4f}")
        print(f"  in-proc  n={len(b):5d}  mean {np.nanmean(b):+.4f}  sd {np.nanstd(b):.4f}")
        print(f"  >> The tailer misses steps by construction (it starts late and the run")
        print(f"     dir is purged), so expect n to differ. What must NOT differ is the")
        print(f"     mean/sd by more than a few percent. A real gap means one of the two")
        print(f"     scorers is wrong -- and faithful.py is the one with a frozen oracle.")

    print()


if __name__ == "__main__":
    main()
