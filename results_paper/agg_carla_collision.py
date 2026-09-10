#!/usr/bin/env python3
"""
Pool several CARLA suppression runs into one summary.

The unsafe rate is a threshold (TTC < 1.5 s) on a continuous quantity whose
distribution sits close to that threshold, so a single 10-case run is a noisy
estimator. Pooling runs, and reporting the mean TTC reduction alongside the rate,
gives a number that does not move much between runs.

    python results_paper/agg_carla_collision.py [run.json ...]

With no arguments it pools every results_paper/carla_collision*.json.
"""
import glob
import json
import statistics as st
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main(paths):
    if not paths:
        paths = sorted(glob.glob(os.path.join(ROOT, "carla_collision*.json")))
    if not paths:
        print("no run files found; run carla_demo/run_collision_cases.py first")
        return 1

    allc = []
    print("%-34s %4s %7s %8s %9s %9s %7s" %
          ("run", "K", "unsafe", "collide", "baseTTC", "atkTTC", "dTTC"))
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        cs = d["cases"]
        if not cs:
            continue
        allc += cs
        b = [c["baseline_min_ttc_s"] for c in cs]
        a = [c["attack_min_ttc_s"] for c in cs]
        print("%-34s %4s %6.0f%% %7.0f%% %8.2f %9.2f %7.2f" %
              (os.path.basename(p), d.get("attack_frames", "?"),
               d["unsafe_rate_pct"], d["collision_rate_pct"],
               st.mean(b), st.mean(a), st.mean(b) - st.mean(a)))

    n = len(allc)
    if not n:
        print("no cases in the given runs")
        return 1
    unsafe = sum(bool(c["unsafe"]) for c in allc)
    coll = sum(bool(c["collided"]) for c in allc)
    b = [c["baseline_min_ttc_s"] for c in allc]
    a = [c["attack_min_ttc_s"] for c in allc]
    mb, ma = st.mean(b), st.mean(a)
    near = [x for x in a if abs(x - 1.5) <= 0.3]

    print("\nPOOLED over %d cases" % n)
    print("  unsafe (TTC < 1.5 s or collision) : %d/%d = %.0f%%" % (unsafe, n, 100.0 * unsafe / n))
    print("  collisions                        : %d" % coll)
    print("  mean min-TTC  %.2f s -> %.2f s     : drop %.2f s (%.0f%%)"
          % (mb, ma, mb - ma, 100.0 * (mb - ma) / mb))
    print("  attacked TTC  min %.2f  median %.2f  max %.2f" % (min(a), st.median(a), max(a)))
    print("  within +/-0.3 s of the threshold  : %d/%d cases" % (len(near), n))
    if len(near) > n / 4:
        print("\n  A large share of cases sits within a few tenths of the 1.5 s cut, which is why"
              "\n  the rate swings between runs. Quote the pooled rate and the TTC drop together.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
