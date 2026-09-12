#!/usr/bin/env python3
"""Report the RssAnon slope and the anon-per-backlog ratio from a stage-A CSV.

Usage:
  slope.py BASELINE.csv [PATCHED.csv]

Slope is least-squares over steps [--from, --to], defaulting to 10..end so the
start-up fill (the first few steps, where ns-3 is still populating fixed-size
caches) does not flatten the number.  The ratio is what actually matters: if
anon per retained backlog byte climbs across the run, an allocator high-water
mark is still ratcheting; if it stays flat, memory is merely proportional to
the packets the network is holding, which is expected and bounded.
"""
import argparse
import csv
import sys


def load(path):
    with open(path) as fh:
        return [(int(r["step"]),
                 int(r["rss_anon_kb"]) / 1024.0,
                 float(r["backlog_bytes_sum"]) / 1048576.0,
                 float(r["step_ms"])) for r in csv.DictReader(fh)]


def slope(rows, lo, hi):
    pts = [(s, a) for s, a, _, _ in rows if lo <= s <= hi]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts)
    denom = n * sxx - sx * sx
    return float("nan") if denom == 0 else (n * sxy - sx * sy) / denom


def report(label, rows, lo, hi):
    hi = min(hi, rows[-1][0])
    s = slope(rows, lo, hi)
    first, last = rows[0], rows[-1]
    step_ms = [t for st, _, _, t in rows if lo <= st <= hi]
    print(f"\n=== {label} ===")
    print(f"  steps            {first[0]}..{last[0]}")
    print(f"  anon             {first[1]:.1f} MB -> {last[1]:.1f} MB")
    print(f"  backlog          {first[2]:.2f} MB -> {last[2]:.2f} MB")
    print(f"  SLOPE (>={lo})    {s:.3f} MB/step")
    if step_ms:
        print(f"  step time        {sum(step_ms)/len(step_ms):.0f} ms mean")
    print(f"  projected peak   {first[1] + s*300:.0f} MB at 300 steps "
          f"(one episode, one worker)")
    print("  anon-per-backlog ratio, 10-step blocks "
          "(flat = fixed, climbing = still ratcheting):")
    d = {st: (a, b) for st, a, b, _ in rows}
    out = []
    for end in range(lo + 10, last[0] + 1, 10):
        if end in d and end - 10 in d:
            da = d[end][0] - d[end - 10][0]
            db = d[end][1] - d[end - 10][1]
            out.append(f"{end-10:>3}-{end:<3} {da/db:6.1f}" if db > 0.01
                       else f"{end-10:>3}-{end:<3} {'n/a':>6}")
    for i in range(0, len(out), 4):
        print("     " + "   ".join(out[i:i + 4]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--from", dest="lo", type=int, default=10)
    ap.add_argument("--to", dest="hi", type=int, default=10 ** 9)
    args = ap.parse_args()

    slopes = []
    for path in args.csvs:
        rows = load(path)
        if not rows:
            print(f"{path}: empty", file=sys.stderr)
            return 1
        slopes.append(report(path.split("/")[-1], rows, args.lo, args.hi))

    if len(slopes) >= 2 and slopes[0] and slopes[1]:
        print(f"\n=== REDUCTION ===")
        print(f"  {slopes[0]:.3f} -> {slopes[1]:.3f} MB/step "
              f"= {slopes[0]/slopes[1]:.1f}x lower" if slopes[1] else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
