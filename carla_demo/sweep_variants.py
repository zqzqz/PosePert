"""
Sweep attack/scenario variants on the SAME cruise-clean scenarios to show which levers move
the (paired) end-to-end danger rate. Each variant screens cruise-clean roads (baseline min-speed
>= CLEAN_MIN_SPEED) then runs the faithful attack; danger = the attack drops the ego's own
min-speed >= SLOWDOWN_MARGIN km/h below its baseline (calibration-free). Run with the CARLA server up.

This reproduces the two findings locked into config.py:
  * attack DEPTH: K=5 gives the same danger RATE as the minimal K=3 but ~2x deeper braking;
  * slow-lead REQUIREMENT: a faster (18 km/h) target is a safe-gap in-lane lead -> no braking,
    so the danger rate collapses even though the perceived shift is identical.
"""
import numpy as np
import mvp_carla
from mvp_carla import sim, stack, runner
import carla

FAST = 18.0 / 3.6   # m/s: a "fast" target the 25 km/h ego does not overtake (vs the default 9 km/h)


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(60.0)
    world = client.get_world(); orig = world.get_settings()
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05; world.apply_settings(s)
    try:
        perc = stack.build_perception(); predictor = stack.build_predictor(); attacker = stack.build_attacker(perc)
        roads = sim.find_straight_roads(world.get_map())
        variants = [
            ("K3 slow target (faithful-v1)", {"attack_frames": 3}),
            ("K5 slow target (locked-in)",   {"attack_frames": 5}),
            ("K5 fast target (18 km/h)",     {"attack_frames": 5, "target_speed": FAST}),
            ("K3 fast target (18 km/h)",     {"attack_frames": 3, "target_speed": FAST}),
        ]
        print("=== VARIANT SWEEP (cruise-clean screen, paired danger, n_clean=8) ===")
        print("%-32s | danger | mean_slow | intr | shift | n" % "variant")
        for name, var in variants:
            results, rate, slow = runner.evaluate(client, world, roads, perc, predictor, attacker,
                                                  n_clean=8, max_screen=45, log=lambda *a, **k: None, **var)
            intr = np.mean([r["attack"]["min_intrusion"] for r in results]) if results else 0
            sh = np.mean([r["attack"]["realized_shift"] for r in results]) if results else 0
            print("%-32s |  %3.0f%% |  %4.1f km/h | %.2f | %.2f  | %d" % (name, rate, slow, intr, sh, len(results)))
    finally:
        world.apply_settings(orig)


if __name__ == "__main__":
    main()
