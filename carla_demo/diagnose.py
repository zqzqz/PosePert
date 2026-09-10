"""
Diagnose where the attack chain loses signal:
  plan (desired offset, <=0.5 m/frame)  ->  voxelwise realized DETECTION shift  ->
  AB3DMOT realized TRACK shift  ->  track lateral velocity  ->  GRIP predicted intrusion.
Runs the attack on a few baseline-safe roads and prints the per-frame chain.
"""
import math
import numpy as np
import mvp_carla
from mvp_carla import sim, stack
from mvp_carla.sim import MoveInScene, actor_world_bbox
from mvp_carla.stack import AVStack, acc_target_speed
from mvp_carla.config import EGO_KMH, ATTACK_START, ATTACK_FRAMES
from agents.navigation.controller import VehiclePIDController
import carla

_LAT = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.05}
_LON = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.05, 'dt': 0.05}


def run_logged(client, world, road, perc, predictor, attacker, attack, steps=56):
    scene = MoveInScene(world, road).spawn()
    try:
        ctrl = VehiclePIDController(scene.ego, args_lateral=_LAT, args_longitudinal=_LON)
        av = AVStack(perc, predictor, attacker if attack else None)
        route = scene.ego_route()
        idx = 0
        for _ in range(40):
            scene.drive_constant(); world.tick()
            eloc = scene.ego.get_location()
            while idx < len(route) - 1 and eloc.distance(route[idx].transform.location) < 3.5:
                idx += 1
            scene.ego.apply_control(ctrl.run_step(EGO_KMH, route[idx]))
        cur_lead, log = None, []
        for step in range(steps):
            scene.drive_constant(); world.tick()
            eloc = scene.ego.get_location(); spd = 3.6 * scene.ego.get_velocity().length()
            while idx < len(route) - 1 and eloc.distance(route[idx].transform.location) < 3.5:
                idx += 1
            twp = route[idx]
            if step % 2 == 0 and scene.sensors_ready():
                frame, ep, cp = scene.build_frame(spd)
                tw = actor_world_bbox(scene.target)
                ego_xy = np.array([eloc.x, eloc.y])
                ego_fwd = np.array([math.cos(math.radians(ep[4])), math.sin(math.radians(ep[4]))])
                pstep = step // 2
                in_window = attack and (ATTACK_START <= pstep < ATTACK_START + ATTACK_FRAMES)
                r = av.step(frame, ep, cp, tw, ego_xy, ego_fwd, attack=in_window)
                cur_lead = r["lead"]
                d = av.dbg
                log.append((pstep, in_window, spd, d.get("desired_off"), d.get("det_lat_shift"),
                            d.get("track_lat_shift"), d.get("track_lat_vel"), d.get("intrusion")))
            scene.ego.apply_control(ctrl.run_step(acc_target_speed(EGO_KMH, spd, cur_lead), twp))
        return log
    finally:
        scene.destroy(client)


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(60.0)
    world = client.get_world(); orig = world.get_settings()
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05; world.apply_settings(s)
    try:
        perc = stack.build_perception(); predictor = stack.build_predictor(); attacker = stack.build_attacker(perc)
        roads = sim.find_straight_roads(world.get_map())
        for ri in [0, 5, 6]:
            log = run_logged(client, world, roads[ri], perc, predictor, attacker, attack=True)
            print("\n=== road %d (attack) === ATTACK window psteps %d..%d" % (ri, ATTACK_START, ATTACK_START + ATTACK_FRAMES - 1))
            print(" pstep win  spd | desired  det_shift  track_shift  track_vel | intrusion")
            for (ps, win, spd, des, det, tr, vel, intr) in log:
                if ps < ATTACK_START - 1 or ps > ATTACK_START + 8:
                    continue
                f = lambda x: ("%6.2f" % x) if x is not None else "   -  "
                print("  %3d  %s %5.1f | %s   %s     %s     %s | %s"
                      % (ps, "A" if win else " ", spd, f(des), f(det), f(tr), f(vel), f(intr)))
    finally:
        world.apply_settings(orig)


if __name__ == "__main__":
    main()
