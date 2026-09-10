"""
Integration tests for the mvp_carla package (REQUIRES the CARLA 0.9.16 server running).
Runs short closed-loop episodes (baseline + attack) and checks they complete sanely.
Run:  python tests/test_integration.py
"""
import os, sys, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mvp_carla
from mvp_carla import sim, stack, runner
import carla

PASS, FAIL = [], []
def check(name, fn):
    try:
        info = fn()
        PASS.append(name); print("  [PASS] %-40s %s" % (name, info or ""))
    except Exception as e:
        FAIL.append((name, repr(e))); print("  [FAIL] %-40s %r" % (name, e)); traceback.print_exc()


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(60.0)
    world = client.get_world()
    orig = world.get_settings()
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05
    world.apply_settings(s)
    state = {}
    try:
        check("connect + sync", lambda: "server=%s" % client.get_server_version())
        def t_roads():
            r = sim.find_straight_roads(world.get_map())
            assert len(r) > 0, "no straight roads"
            state["roads"] = r
            return "%d straight roads" % len(r)
        check("find straight roads", t_roads)

        def t_build():
            state["perc"] = stack.build_perception()
            state["pred"] = stack.build_predictor()
            state["atk"] = stack.build_attacker(state["perc"])      # faithful: beta=2 + PertNet
            return "perception + predictor + attacker"
        check("build stack components", t_build)

        def t_baseline():
            road = state["roads"][0]
            b = runner.run_episode(client, world, road, state["perc"], state["pred"], None, attack=False, steps=24)
            assert set(b) >= {"min_intrusion", "min_speed", "decelerated", "realized_shift"}, b
            state["base"] = b
            return "baseline episode: min_speed=%.1f intr=%.1f decel=%s" % (b["min_speed"], b["min_intrusion"], b["decelerated"])
        check("baseline episode runs (ACC)", t_baseline)

        def t_attack():
            road = state["roads"][0]
            a = runner.run_episode(client, world, road, state["perc"], state["pred"], state["atk"], attack=True, steps=24)
            assert set(a) >= {"min_intrusion", "min_speed", "decelerated", "realized_shift"}, a
            assert a["realized_shift"] >= 0.0          # voxelwise attack perturbs the tracked target
            return "attack episode: min_speed=%.1f intr=%.1f decel=%s shift=%.2f" % (
                a["min_speed"], a["min_intrusion"], a["decelerated"], a["realized_shift"])
        check("attack episode runs (real voxelwise, ACC)", t_attack)
    finally:
        world.apply_settings(orig)

    print("\nPASS=%d FAIL=%d" % (len(PASS), len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
