"""
Single illustrative episode: baseline vs attack on one baseline-safe scenario.
Online-optimized PosePert planning executed by the real voxelwise feature attack
(attack window = config.ATTACK_FRAMES). Run with the CARLA 0.9.16 server up.

    python run_demo.py [--steps 56] [--beta B]   # omit --beta for the faithful beta=2 + PertNet
"""
import argparse
import mvp_carla                      # bootstraps env
from mvp_carla import sim, stack, runner
import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
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

        # pick the first scenario the ego cruises through without slowing (baseline-safe)
        road, base = None, None
        for r in roads[:12]:
            b = runner.run_episode(client, world, r, perc, predictor, None, attack=False, steps=args.steps)
            if not b["decelerated"]:
                road, base = r, b
                break
        if road is None:
            print("no baseline-safe road found; using first road")
            road = roads[0]
            base = runner.run_episode(client, world, road, perc, predictor, None, attack=False, steps=args.steps)

        atk = runner.run_episode(client, world, road, perc, predictor, attacker, attack=True, steps=args.steps)
        print("\n--- move-in scenario (baseline vs attack) ---")
        print("baseline | min_speed=%4.1f km/h | pred_lat_intrusion=%.1f m | decel=%s"
              % (base["min_speed"], base["min_intrusion"], base["decelerated"]))
        print("attack   | min_speed=%4.1f km/h | pred_lat_intrusion=%.1f m | decel=%s | realized_shift=%.2f m"
              % (atk["min_speed"], atk["min_intrusion"], atk["decelerated"], atk["realized_shift"]))
    finally:
        world.apply_settings(orig)


if __name__ == "__main__":
    main()
