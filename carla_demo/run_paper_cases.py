"""
Closed-loop CARLA braking cases reported in the paper.

Screens straight roads for baseline-safe ("cruise-clean") scenarios and runs the
faithful voxelwise scenario attack on each one, reporting the paired danger rate
and the per-case braking depth. This is `run_eval.py` pinned to the paper's case
count with per-case results written to disk.

Requires a running CARLA server (see README.md).

    python carla_demo/run_paper_cases.py                 # 12 braking cases
    python carla_demo/run_paper_cases.py --n_clean 6     # quicker smoke run

NOTE: the paper's collision cases are NOT produced by this script. The closed-loop
harness scores braking severity (ego min-speed drop) and has no collision metric;
see the "CARLA closed-loop" section of the top-level README.
"""
import argparse
import json
import os

import numpy as np
import mvp_carla                      # bootstraps env (paths + CUDA)
from mvp_carla import sim, stack, runner
from mvp_carla.config import ATTACK_FRAMES, SLOWDOWN_MARGIN
import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--n_clean", type=int, default=12,
                    help="number of baseline-safe scenarios to attack (paper: 12)")
    ap.add_argument("--max_screen", type=int, default=40,
                    help="max roads to screen; raise if fewer than n_clean are found")
    ap.add_argument("--steps", type=int, default=56)
    ap.add_argument("--beta", type=float, default=None,
                    help="omit for the faithful default (beta=2 + PertNet)")
    ap.add_argument("--out", default="results_paper/carla_braking_cases.json")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    orig = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    world.apply_settings(s)
    try:
        perc = stack.build_perception()
        predictor = stack.build_predictor()
        attacker = stack.build_attacker(perc, beta=args.beta)
        roads = sim.find_straight_roads(world.get_map())
        print("found %d straight roads; screening for %d baseline-safe scenarios"
              % (len(roads), args.n_clean))

        results, danger, mean_slow = runner.evaluate(
            client, world, roads, perc, predictor, attacker,
            n_clean=args.n_clean, max_screen=args.max_screen, steps=args.steps)

        if len(results) < args.n_clean:
            print("WARNING: only %d/%d cruise-clean scenarios found; raise --max_screen"
                  % (len(results), args.n_clean))

        print("\n=== CARLA BRAKING CASES (K=%d, real voxelwise exec) ===" % ATTACK_FRAMES)
        print("cases: %d  |  paired danger rate (slow>=%.0f km/h): %.0f%%  |  mean slowdown: %.1f km/h"
              % (len(results), SLOWDOWN_MARGIN, danger, mean_slow))

        rows = []
        for i, r in enumerate(results):
            base_v = r["baseline"]["min_speed"]
            atk_v = r["attack"]["min_speed"]
            rows.append(dict(case=i,
                             baseline_min_speed_kmh=round(float(base_v), 2),
                             attack_min_speed_kmh=round(float(atk_v), 2),
                             slowdown_kmh=round(float(base_v - atk_v), 2),
                             dangerous=bool(base_v - atk_v >= SLOWDOWN_MARGIN),
                             predicted_intrusion_m=round(float(r["attack"]["min_intrusion"]), 3),
                             realized_shift_m=round(float(r["attack"]["realized_shift"]), 3)))
            print("  case %2d: %5.1f -> %5.1f km/h (drop %4.1f)%s"
                  % (i, base_v, atk_v, base_v - atk_v, "  DANGER" if rows[-1]["dangerous"] else ""))

        if results:
            print("mean predicted intrusion: %.2f m  |  mean realized track shift: %.2f m"
                  % (np.mean([r["attack"]["min_intrusion"] for r in results]),
                     np.mean([r["attack"]["realized_shift"] for r in results])))

        out = os.path.join(mvp_carla.MVP_ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(attack_frames=ATTACK_FRAMES,
                           slowdown_margin_kmh=SLOWDOWN_MARGIN,
                           beta=args.beta,
                           n_cases=len(results),
                           danger_rate_pct=danger,
                           mean_slowdown_kmh=mean_slow,
                           cases=rows), f, indent=2)
        print("saved %s" % out)
    finally:
        world.apply_settings(orig)


if __name__ == "__main__":
    main()
