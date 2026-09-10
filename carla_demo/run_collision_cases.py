"""
Closed-loop CARLA suppression (collision) cases reported in the paper.

A stopped vehicle marginally blocks the ego lane. The clean ego perceives it in lane and
brakes to a safe stop; under attack the collaborator shifts its perceived pose out of the
lane, so the ACC sees no lead and braking is delayed or suppressed.

Reports the paper's three suppression metrics: unsafe rate (collision or min TTC < 1.5 s),
collision rate, and worst-case impact speed. TTC and collisions are measured from
ground-truth actor state, never from the attacked perception.

Requires a running CARLA server (see README.md).

    python carla_demo/run_collision_cases.py                  # 10 cases
    python carla_demo/run_collision_cases.py --n_cases 3      # quicker smoke run

Straight roads with an adjacent lane are scarce on a single map, so scenarios are built by
perturbing each road's blocker distance and lane encroachment; see build_moveout_scenarios.
"""
import argparse
import json
import os

import mvp_carla                      # bootstraps env (paths + CUDA)
from mvp_carla import sim, stack, runner
from mvp_carla.config import MOVEOUT_ATTACK_FRAMES, TTC_UNSAFE
import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--n_cases", type=int, default=10,
                    help="number of baseline-safe suppression scenarios to attack (paper: 10)")
    ap.add_argument("--max_screen", type=int, default=40,
                    help="max scenarios to screen; raise if fewer than n_cases come back safe")
    ap.add_argument("--beta", type=float, default=None,
                    help="omit for the faithful default (beta=2 + PertNet)")
    ap.add_argument("--attack_frames", type=int, default=None,
                    help="perception steps the suppression window stays open once triggered "
                         "(default MOVEOUT_ATTACK_FRAMES). Large values keep the perturbation "
                         "live all the way to the blocker; the per-frame stealth bound is "
                         "unaffected either way.")
    ap.add_argument("--out", default="results_paper/carla_collision_cases.json")
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
        scenarios = sim.build_moveout_scenarios(roads, n_wanted=args.n_cases)
        print("found %d straight roads -> %d candidate scenarios; screening for %d safe baselines"
              % (len(roads), len(scenarios), args.n_cases))

        results, unsafe_rate, coll_rate, worst_impact = runner.evaluate_moveout(
            client, world, scenarios, perc, predictor, attacker,
            n_clean=args.n_cases, max_screen=args.max_screen,
            attack_frames=args.attack_frames)

        if len(results) < args.n_cases:
            print("WARNING: only %d/%d scenarios had a safe clean baseline; raise --max_screen"
                  % (len(results), args.n_cases))

        K = args.attack_frames if args.attack_frames is not None else MOVEOUT_ATTACK_FRAMES
        print("\n=== CARLA SUPPRESSION CASES (K=%d, real voxelwise exec) ===" % K)
        print("cases: %d  |  unsafe (collision or TTC<%.1fs): %.0f%%  |  collision: %.0f%%  |  worst impact: %.1f km/h"
              % (len(results), TTC_UNSAFE, unsafe_rate, coll_rate, worst_impact))

        rows = []
        for i, r in enumerate(results):
            b, a, sc = r["baseline"], r["attack"], r["scenario"]
            rows.append(dict(case=i,
                             blocker_ahead_m=round(float(sc["blocker_ahead"]), 2),
                             encroach_m=round(float(sc["encroach"]), 2),
                             baseline_min_ttc_s=round(float(b["min_ttc"]), 3),
                             attack_min_ttc_s=round(float(a["min_ttc"]), 3),
                             collided=bool(a["collided"]),
                             impact_kmh=(round(float(a["impact_kmh"]), 2)
                                         if a["impact_kmh"] is not None else None),
                             attack_min_speed_kmh=round(float(a["min_speed"]), 2),
                             realized_shift_m=round(float(a["realized_shift"]), 3),
                             unsafe=bool(r["unsafe"])))
            print("  case %2d: TTC %5.2f -> %5.2f s%s%s"
                  % (i, b["min_ttc"], a["min_ttc"],
                     "  COLLISION @ %.1f km/h" % a["impact_kmh"] if a["collided"] and a["impact_kmh"] is not None else "",
                     "  UNSAFE" if r["unsafe"] and not a["collided"] else ""))

        out = os.path.join(mvp_carla.MVP_ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(attack_frames=(args.attack_frames if args.attack_frames is not None
                                          else MOVEOUT_ATTACK_FRAMES),
                           ttc_unsafe_s=TTC_UNSAFE,
                           beta=args.beta,
                           n_cases=len(results),
                           unsafe_rate_pct=unsafe_rate,
                           collision_rate_pct=coll_rate,
                           worst_impact_kmh=worst_impact,
                           cases=rows), f, indent=2)
        print("saved %s" % out)
    finally:
        world.apply_settings(orig)


if __name__ == "__main__":
    main()
