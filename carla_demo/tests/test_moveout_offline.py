"""
Offline checks for the move-out (suppression) scenario. No CARLA server required, and no
GPU: this covers the geometry and bookkeeping that decide whether a case is scored unsafe.

The closed-loop behaviour itself still has to be verified against a live server with
carla_demo/run_collision_cases.py.

Run:  python carla_demo/tests/test_moveout_offline.py
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


# ---- minimal carla stub, so sim.py imports without the simulator ----------
def _install_carla_stub():
    if "carla" in sys.modules:
        return
    m = types.ModuleType("carla")

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

    class Vector3D(Location):
        pass

    class Rotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch, self.yaw, self.roll = pitch, yaw, roll

    class Transform:
        def __init__(self, location=None, rotation=None):
            self.location = location or Location()
            self.rotation = rotation or Rotation()

    m.Location, m.Vector3D, m.Rotation, m.Transform = Location, Vector3D, Rotation, Transform
    m.VehicleControl = lambda **kw: types.SimpleNamespace(**kw)
    m.LaneType = types.SimpleNamespace(Driving="Driving")
    m.command = types.SimpleNamespace(DestroyActor=lambda a: a)
    sys.modules["carla"] = m


_install_carla_stub()
from mvp_carla import sim                                    # noqa: E402
from mvp_carla.config import BLOCKER_AHEAD, BLOCKER_ENCROACH, LANE_HALF, TTC_UNSAFE  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        info = fn()
        PASS.append(name)
        print("  [PASS] %-42s %s" % (name, info or ""))
    except Exception as e:
        FAIL.append((name, repr(e)))
        print("  [FAIL] %-42s %r" % (name, e))


def _actor(x, y, vx=0.0, vy=0.0, half_len=2.4, half_wid=1.0):
    return types.SimpleNamespace(
        get_transform=lambda: sim.carla.Transform(sim.carla.Location(x, y, 0.0)),
        get_velocity=lambda: sim.carla.Vector3D(vx, vy, 0.0),
        bounding_box=types.SimpleNamespace(
            extent=types.SimpleNamespace(x=half_len, y=half_wid, z=0.7)))


FWD = np.array([1.0, 0.0])


def t_ttc_closing():
    """Stopped blocker 20 m ahead, ego at 10 m/s: gap is box-to-box, so TTC < 20/10."""
    ego, blk = _actor(0, 0, vx=10.0), _actor(20, 0)
    ttc = sim.longitudinal_ttc(ego, blk, FWD)
    gap = 20 - (2.4 + 2.4)
    assert abs(ttc - gap / 10.0) < 1e-6, ttc
    assert ttc < 2.0
    return "%.2f s (box gap %.1f m)" % (ttc, gap)


def t_ttc_behind_is_inf():
    ego, blk = _actor(0, 0, vx=10.0), _actor(-20, 0)
    assert sim.longitudinal_ttc(ego, blk, FWD) == float("inf")
    return "blocker behind -> inf"


def t_ttc_lateral_clear_is_inf():
    """A blocker displaced far enough sideways is not on a collision course."""
    ego, blk = _actor(0, 0, vx=10.0), _actor(20, 5.0)
    assert sim.longitudinal_ttc(ego, blk, FWD) == float("inf")
    return "5 m lateral -> inf"


def t_ttc_stopped_ego_is_inf():
    ego, blk = _actor(0, 0, vx=0.0), _actor(20, 0)
    assert sim.longitudinal_ttc(ego, blk, FWD) == float("inf")
    return "ego stopped -> inf"


def t_ttc_never_negative():
    """Overlapping boxes clamp to 0, so an unsafe case never reads as safe."""
    ego, blk = _actor(0, 0, vx=10.0), _actor(3.0, 0)
    assert sim.longitudinal_ttc(ego, blk, FWD) == 0.0
    return "overlap -> 0 s"


def t_encroachment_is_marginal():
    """The default blocker must gate in-lane when clean, and clear the lane once shifted."""
    assert BLOCKER_ENCROACH < LANE_HALF, "blocker would start outside the lane"
    from mvp_carla.config import MAX_OFFSET
    assert BLOCKER_ENCROACH + MAX_OFFSET > LANE_HALF, "max shift cannot clear the lane"
    return "encroach %.2f m, lane half %.2f m, max shift %.2f m" % (
        BLOCKER_ENCROACH, LANE_HALF, MAX_OFFSET)


def t_scenarios_expand_and_vary():
    roads = ["r%d" % i for i in range(3)]
    sc = sim.build_moveout_scenarios(roads, n_wanted=10)
    assert len(sc) >= 10, len(sc)
    keys = {(s["road"], s["blocker_ahead"], round(s["encroach"], 3)) for s in sc}
    assert len(keys) == len(sc), "scenarios are not distinct"
    assert len({s["road"] for s in sc[:3]}) == 3, "roads should vary fastest"
    assert all(s["encroach"] > 0 for s in sc)
    return "%d scenarios from %d roads, all distinct" % (len(sc), len(roads))


def t_scenarios_survive_one_road():
    """One usable road still yields enough perturbed cases to screen."""
    sc = sim.build_moveout_scenarios(["only"], n_wanted=10)
    assert len(sc) >= 9, len(sc)
    return "%d scenarios from 1 road" % len(sc)


def t_collision_only_counts_the_blocker():
    """Clipping scenery or the collaborator must not be scored as a suppression collision."""
    scene = sim.MoveOutScene.__new__(sim.MoveOutScene)
    scene.collisions = []
    scene.target = types.SimpleNamespace(id=42)
    ev = lambda oid: types.SimpleNamespace(other_actor=types.SimpleNamespace(id=oid))
    scene.on_collision(ev(7))          # collaborator
    scene.on_collision(ev(None))       # static scenery
    assert scene.collided() is False, "non-blocker impact counted"
    scene.on_collision(ev(42))         # the blocker
    assert scene.collided() is True
    return "1 blocker impact kept, 2 others dropped"


def t_gap_to_blocker_is_box_to_box():
    """The proximity trigger measures a box-to-box gap, not a centre distance."""
    scene = sim.MoveOutScene.__new__(sim.MoveOutScene)
    scene.ego = _actor(0, 0)
    scene.target = _actor(30, 0)
    scene.ego.get_transform = lambda: sim.carla.Transform(
        sim.carla.Location(0, 0, 0), sim.carla.Rotation(yaw=0.0))
    scene.target.get_transform = lambda: sim.carla.Transform(
        sim.carla.Location(30, 0, 0), sim.carla.Rotation(yaw=0.0))
    gap = scene.gap_to_blocker()
    assert abs(gap - (30 - 4.8)) < 1e-6, gap
    return "%.1f m for 30 m centres" % gap


def t_attack_range_covers_the_braking_decision():
    """The suppression window must open before the ego would gate the blocker in lane."""
    from mvp_carla.config import MOVEOUT_ATTACK_RANGE, LOOKAHEAD
    assert MOVEOUT_ATTACK_RANGE >= LOOKAHEAD, (MOVEOUT_ATTACK_RANGE, LOOKAHEAD)
    return "range %.0f m >= lookahead %.0f m" % (MOVEOUT_ATTACK_RANGE, LOOKAHEAD)


def t_out_ward_points_away_from_lane():
    """The move-out direction must increase the blocker's lateral offset, whichever side it leans."""
    from mvp_carla.stack import AVStack
    ego_xy = np.array([0.0, 0.0])
    for lean in (+1.0, -1.0):
        tgt = np.array([20.0, lean * BLOCKER_ENCROACH, 0, 0, 0, 0, 0.0])
        d = AVStack._out_ward(tgt, ego_xy, FWD)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-9
        moved = tgt[:2] + d * 0.5
        assert abs(moved[1]) > abs(tgt[1]), "shift reduced the lateral offset"
    return "both lean directions push outward"


for name, fn in sorted(globals().items()):
    if name.startswith("t_"):
        check(name[2:], fn)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
