"""
Benchmark the PERCEPTION attack alone in our CARLA demo (move-in) geometry, with the paper's
ablation: ray-cast only (beta=1), +beta scaling, my beta=8, and FULL (beta=2 + PertNet).
For each scenario/shift we attack a target to a shifted pose and measure IoU(detection, target box)
and %Success(IoU>0.5) -- the paper's perception metrics -- plus the realized shift.
"""
import math
import numpy as np
import mvp_carla
from mvp_carla import sim, stack
from mvp_carla.sim import MoveInScene, actor_world_bbox
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
import torch, carla
from shapely.geometry import box as shbox
from shapely.affinity import rotate, translate


def bev_iou(a, b):
    def poly(bb):
        x, y, z, l, w, h, yaw = bb[:7]
        p = shbox(-l / 2, -w / 2, l / 2, w / 2)
        return translate(rotate(p, yaw, use_radians=True), x, y)
    pa, pb = poly(a), poly(b)
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter + 1e-9)


def load_pertnet(perc):
    from mvp.attack.perturbation_network import PerturbationNetwork
    ck = torch.load(mvp_carla.MVP_ROOT + "/models/perturbation_net_paper_pointpillar/perturbation_net_best.pt",
                    map_location=perc.device)
    net = PerturbationNetwork(feature_channels=ck["feature_channels"], geo_channels=ck["geo_channels"]).to(perc.device).eval()
    net.load_state_dict(ck["model_state"])
    return net, float(ck.get("beta", 2.0))


def capture(scene, world):
    for _ in range(20):
        scene.drive_constant(); world.tick()
    world.tick()
    spd = 3.6 * scene.ego.get_velocity().length()
    return scene.build_frame(spd)


def attack_iou(attacker, frame, ep, cp, target_true, shifted):
    res = attacker.run_multi_vehicle(frame, {
        "attacker_vehicle_id": "collab", "victim_vehicle_id": "ego",
        "bbox_to_remove": bbox_map_to_sensor(target_true, cp),
        "bbox_to_spoof": bbox_map_to_sensor(shifted, cp)})
    pred = res["pred_bboxes"]
    if not len(pred):
        return 0.0, 0.0
    dets = np.array([bbox_sensor_to_map(b, ep) for b in pred])
    i = int(np.argmin(np.hypot(dets[:, 0] - shifted[0], dets[:, 1] - shifted[1])))
    return bev_iou(dets[i], shifted), bev_iou(dets[i], target_true)


def main():
    client = carla.Client("127.0.0.1", 2000); client.set_timeout(60.0)
    world = client.get_world(); orig = world.get_settings()
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05; world.apply_settings(s)
    try:
        perc = stack.build_perception()
        attacker = stack.build_attacker(perc, beta=2.0)
        pertnet, paper_beta = load_pertnet(perc)
        print("PertNet loaded (paper beta=%.1f, %d params)" % (paper_beta, sum(p.numel() for p in pertnet.parameters())))
        roads = sim.find_straight_roads(world.get_map())[:6]
        shifts = [0.5, 1.0, 1.5]
        variants = [("raycast(b1)", 1.0, None), ("+beta2", 2.0, None),
                    ("beta8", 8.0, None), ("FULL(b2+PertNet)", paper_beta, pertnet)]
        agg = {v[0]: {sh: [] for sh in shifts} for v in variants}
        base_iou_true = []
        for ri, road in enumerate(roads):
            scene = MoveInScene(world, road).spawn()
            try:
                frame, ep, cp = capture(scene, world)
                tw = actor_world_bbox(scene.target)
                # baseline: is the target detected at its true pose?
                pb, _ = perc.run(frame, "ego")
                if len(pb):
                    dw = np.array([bbox_sensor_to_map(b, ep) for b in pb])
                    j = int(np.argmin(np.hypot(dw[:, 0] - tw[0], dw[:, 1] - tw[1])))
                    base_iou_true.append(bev_iou(dw[j], tw))
                latd = np.array([1.0, 0.0])  # lateral (perpendicular to target heading)
                cy, sy = math.cos(tw[6]), math.sin(tw[6])
                latd = np.array([-sy, cy])
                for sh in shifts:
                    shifted = tw.copy(); shifted[0] += latd[0] * sh; shifted[1] += latd[1] * sh
                    for name, beta, pn in variants:
                        attacker.beta = beta; attacker.pertnet = pn
                        try:
                            iou_s, iou_o = attack_iou(attacker, frame, ep, cp, tw, shifted)
                        except Exception as e:
                            iou_s, iou_o = -1.0, -1.0
                            print("  err road%d %s sh%.1f: %r" % (ri, name, sh, e))
                        agg[name][sh].append(iou_s)
                print("road %d done" % ri)
            finally:
                scene.destroy(client)

        print("\n=== PERCEPTION ATTACK ALONE (move-in geometry, %d scenarios) ===" % len(roads))
        print("baseline IoU(detection, true target) = %.2f  (perception quality)" % (np.mean(base_iou_true) if base_iou_true else 0))
        print("\n%-18s | %s" % ("variant", "  ".join("sh=%.1f: avgIoU / %%succ(>0.5)" % sh for sh in shifts)))
        for name, _, _ in variants:
            cells = []
            for sh in shifts:
                v = [x for x in agg[name][sh] if x >= 0]
                avg = np.mean(v) if v else 0
                succ = 100 * np.mean([x > 0.5 for x in v]) if v else 0
                cells.append("%.2f / %3.0f%%" % (avg, succ))
            print("%-18s | %s" % (name, "        ".join(cells)))
    finally:
        world.apply_settings(orig)


if __name__ == "__main__":
    main()
