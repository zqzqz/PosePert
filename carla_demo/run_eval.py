"""
Faithful scenario-attack evaluation in CARLA closed-loop (the headline danger-rate number).

Online GRIP-optimized planning (<= STEALTH m/frame) executed by the REAL voxelwise feature
attack over CRUISE-CLEAN scenarios (baseline min-speed >= CLEAN_MIN_SPEED, screened), reporting
the paired danger rate (the attack drops the ego's OWN min-speed >= SLOWDOWN_MARGIN km/h below its
baseline). The attack window is config.ATTACK_FRAMES (locked-in K=5). Run with the CARLA server up.

    python run_eval.py [--n_clean 6] [--max_screen 24] [--steps 56] [--beta B]

Omit --beta for the faithful default (beta=2 + PertNet); see README.md.
"""
import argparse
import numpy as np
import mvp_carla                      # bootstraps env (paths + CUDA)
from mvp_carla import sim, stack, runner
import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--n_clean", type=int, default=6, help="number of baseline-safe scenarios to attack")
    ap.add_argument("--max_screen", type=int, default=24, help="max roads to screen for clean scenarios")
    ap.add_argument("--steps", type=int, default=56)
    ap.add_argument("--beta", type=float, default=None, help="None => faithful beta=2 + PertNet")
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
        print("found %d straight roads; screening for %d baseline-safe scenarios" % (len(roads), args.n_clean))
        results, danger, mean_slow = runner.evaluate(client, world, roads, perc, predictor, attacker,
                                          n_clean=args.n_clean, max_screen=args.max_screen, steps=args.steps)
        from mvp_carla.config import ATTACK_FRAMES, SLOWDOWN_MARGIN
        print("\n=== FAITHFUL SCENARIO ATTACK (K=%d, real voxelwise exec) ===" % ATTACK_FRAMES)
        print("clean scenarios: %d  |  ATTACK danger rate (paired, slow>=%.0f km/h): %.0f%%  |  mean slowdown: %.1f km/h"
              % (len(results), SLOWDOWN_MARGIN, danger, mean_slow))
        if results:
            print("attack mean predicted intrusion: %.2f m  |  mean realized track shift: %.2f m"
                  % (np.mean([r['attack']['min_intrusion'] for r in results]),
                     np.mean([r['attack']['realized_shift'] for r in results])))
    finally:
        world.apply_settings(orig)


if __name__ == "__main__":
    main()
