"""
Windowed CARLA viewer for the move-in attack (run the server WITHOUT -RenderOffScreen).
Chase camera follows the ego; in-world overlays show the chain:
  GREEN box  = real target (ground truth)
  RED box    = perceived/tracked target (under attack it drifts toward the ego lane)
  line       = GRIP++ predicted target path (RED if it enters the ego lane)
  text on ego= mode + ego speed + ACC target speed

Episode 1 BASELINE: ego cruises past the parallel target.
Episode 2 ATTACK (idealized cut-in, upper bound): perceived target moves into the ego
  lane -> GRIP predicts a cut-in -> the ACC smoothly slows the ego.
(The realistic voxelwise attack realizes only ~0.5 m/frame and is probabilistic -- see diagnose.py.)
"""
import math, time
import numpy as np
import mvp_carla
from mvp_carla import sim, stack
from mvp_carla.sim import MoveInScene, actor_world_bbox
from mvp_carla.stack import acc_target_speed, predicted_lead_distance, lane_intrusion
from mvp_carla.config import EGO_KMH, ATTACK_START, OBS_LEN, PRED_LEN, DT, LANE_HALF
from mvp.tools.object_tracking import Ab3dmotTracker
from mvp.attack.scenario_attacker_util import tracking_ab3dmot, prediction_grip
from mvp.data.util import bbox_sensor_to_map
from agents.navigation.controller import VehiclePIDController
import carla

_LAT = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.05}
_LON = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.05, 'dt': 0.05}
CUTIN_RATE = 2.5            # idealized perceived lateral drift (m/s)


def chase_cam(world, ego):
    t = ego.get_transform(); f = t.get_forward_vector()
    loc = t.location + carla.Location(x=-9 * f.x, y=-9 * f.y, z=20)
    world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-58, yaw=t.rotation.yaw)))


def box(dbg, x, y, z, yaw, color):
    dbg.draw_box(carla.BoundingBox(carla.Location(float(x), float(y), float(z)), carla.Vector3D(2.3, 1.0, 0.8)),
                 carla.Rotation(yaw=float(math.degrees(yaw))), 0.15, color, 0.15)


def run(client, world, road, perc, predictor, attack, label, steps=80):
    scene = MoveInScene(world, road).spawn()
    dbg = world.debug
    try:
        ctrl = VehiclePIDController(scene.ego, args_lateral=_LAT, args_longitudinal=_LON)
        route = scene.ego_route()
        tracker = Ab3dmotTracker(); hist = {}; sim_t = 0.0; tgt_tid = None; cur_lead = None
        idx = 0
        for _ in range(40):                                     # warmup: drive to cruise
            scene.drive_constant(); world.tick(); chase_cam(world, scene.ego)
            eloc = scene.ego.get_location()
            while idx < len(route) - 1 and eloc.distance(route[idx].transform.location) < 3.5:
                idx += 1
            scene.ego.apply_control(ctrl.run_step(EGO_KMH, route[idx]))
        for step in range(steps):
            scene.drive_constant(); world.tick(); chase_cam(world, scene.ego)
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
                pred, _ = perc.run(frame, "ego")
                detsw = np.array([bbox_sensor_to_map(b, ep) for b in pred]) if len(pred) else np.zeros((0, 7))
                if attack and pstep >= ATTACK_START and len(detsw):
                    off = min(CUTIN_RATE * DT * (pstep - ATTACK_START), 4.0)
                    latd = ego_xy - tw[:2]; latd = latd / (np.linalg.norm(latd) + 1e-6)
                    i = int(np.argmin(np.hypot(detsw[:, 0] - tw[0], detsw[:, 1] - tw[1])))
                    detsw[i, 0] += latd[0] * off; detsw[i, 1] += latd[1] * off
                _, indexed = tracking_ab3dmot(tracker, sim_t, detsw); sim_t += DT
                tgt_tid, bd = None, 4.0
                for tid, bb in indexed.items():
                    d = math.hypot(bb[0] - tw[0], bb[1] - tw[1])
                    if d < bd:
                        bd, tgt_tid = d, tid
                for tid, bb in indexed.items():
                    hist.setdefault(tid, []).append(np.asarray(bb)[:7]); hist[tid] = hist[tid][-OBS_LEN:]
                z = eloc.z + 0.5
                box(dbg, tw[0], tw[1], z, tw[6], carla.Color(0, 255, 0))                 # real target (green)
                if tgt_tid is not None:
                    pb = indexed[tgt_tid]
                    box(dbg, pb[0], pb[1], z, pb[6], carla.Color(255, 0, 0))             # perceived (red)
                cur_lead = None
                obs = {tid: np.stack(h) for tid, h in hist.items()
                       if len(h) >= 6 and math.hypot(h[-1][0] - eloc.x, h[-1][1] - eloc.y) < 40}
                if obs:
                    preds = prediction_grip(obs, model_args={"model_api": predictor, "obs_length": OBS_LEN, "pred_length": PRED_LEN})
                    if tgt_tid in preds:
                        pt = np.asarray(preds[tgt_tid]); intr = lane_intrusion(pt, ego_xy, ego_fwd)
                        s = predicted_lead_distance(pt, ego_xy, ego_fwd)
                        if s is not None:
                            h = hist[tgt_tid]
                            v_lead = float((np.asarray(h[-1][:2]) - np.asarray(h[-2][:2])) @ ego_fwd / DT) if len(h) >= 2 else 0.0
                            cur_lead = {"distance": s, "lead_speed": v_lead}
                        col = carla.Color(255, 0, 0) if intr < LANE_HALF else carla.Color(255, 200, 0)
                        for k in range(len(pt) - 1):
                            dbg.draw_line(carla.Location(float(pt[k, 0]), float(pt[k, 1]), z),
                                          carla.Location(float(pt[k + 1, 0]), float(pt[k + 1, 1]), z), 0.2, col, 0.15)
            v_target = acc_target_speed(EGO_KMH, spd, cur_lead)
            scene.ego.apply_control(ctrl.run_step(v_target, twp))
            slowing = cur_lead is not None and v_target < EGO_KMH - 2
            dbg.draw_string(eloc + carla.Location(z=3),
                            "%s | %4.1f km/h%s" % (label, spd, "  SLOWING" if slowing else ""),
                            False, carla.Color(255, 80, 80) if slowing else carla.Color(255, 255, 255), 0.12)
            time.sleep(0.04)
    finally:
        scene.destroy(client); world.tick()


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(60.0)
    world = client.get_world(); orig = world.get_settings()
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05; world.apply_settings(s)
    try:
        perc = stack.build_perception(); predictor = stack.build_predictor()
        road = sim.find_straight_roads(world.get_map())[0]
        print(">>> Watch the CARLA window. Episode 1: BASELINE (ego cruises past parallel target).")
        run(client, world, road, perc, predictor, attack=False, label="BASELINE")
        time.sleep(1.0)
        print(">>> Episode 2: ATTACK (perceived RED target cuts in; ACC slows the ego).")
        run(client, world, road, perc, predictor, attack=True, label="ATTACK")
        print(">>> Demo done.")
    finally:
        world.apply_settings(orig)


if __name__ == "__main__":
    main()
