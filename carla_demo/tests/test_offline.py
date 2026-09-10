"""
Offline tests for the mvp_carla package (no CARLA server needed; uses the GPU).
Builds perception/predictor/attacker ONCE up front (as production does), then checks.
Run:  python tests/test_offline.py
"""
import os, sys, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # find mvp_carla
import numpy as np
import mvp_carla                       # bootstraps env (paths + CUDA)
from mvp_carla import sim, stack, runner, config           # noqa
from mvp.attack.scenario_attacker_util import prediction_grip
import torch

PASS, FAIL = [], []
def check(name, fn):
    try:
        info = fn()
        PASS.append(name); print("  [PASS] %-38s %s" % (name, info or ""))
    except Exception as e:
        FAIL.append((name, repr(e))); print("  [FAIL] %-38s %r" % (name, e)); traceback.print_exc()


print("=== OFFLINE TESTS ===")
print("  building perception / predictor / attacker (once)...")
PERC = stack.build_perception()
PRED = stack.build_predictor()
ATK = stack.build_attacker(PERC)            # faithful: beta=2 + PertNet


def t_lane_intrusion():
    from mvp_carla.stack import lane_intrusion
    ego_xy = np.array([0.0, 0.0]); fwd = np.array([1.0, 0.0])
    adj = np.array([[10, 3.5], [12, 3.5], [14, 3.5]])          # parallel adjacent lane
    cut = np.array([[8, 1.0], [12, 0.4], [16, -0.2]])          # crossing into ego lane
    behind = np.array([[-5, 0.0], [-10, 0.0]])                 # behind ego -> ignored
    a, c, b = lane_intrusion(adj, ego_xy, fwd), lane_intrusion(cut, ego_xy, fwd), lane_intrusion(behind, ego_xy, fwd)
    assert 3.0 < a < 4.0 and c < 0.5 and b > 90, (a, c, b)
    return "adjacent=%.1f cutin=%.1f behind=%.0f" % (a, c, b)

def t_acc():
    from mvp_carla.stack import acc_target_speed, predicted_lead_distance
    cruise = 25.0
    assert acc_target_speed(cruise, 25.0, None) == cruise                       # no obstacle -> cruise
    far = {"distance": 60.0, "lead_speed": 2.5}
    assert acc_target_speed(cruise, 25.0, far) == cruise                        # far -> clipped to cruise
    close = {"distance": 6.0, "lead_speed": 2.5}
    v_close = acc_target_speed(cruise, 25.0, close)
    assert 0.0 <= v_close < cruise, v_close                                     # close -> slow down
    ego = np.array([0.0, 0.0]); fwd = np.array([1.0, 0.0])
    assert predicted_lead_distance(np.array([[10, 0.5], [14, 0.6]]), ego, fwd) is not None   # in-lane
    assert predicted_lead_distance(np.array([[10, 3.5], [14, 3.5]]), ego, fwd) is None       # adjacent
    return "cruise=%.0f far=%.0f close=%.1f km/h" % (cruise, acc_target_speed(cruise, 25, far), v_close)

def t_sort_vertices():
    from mvp.perception.iou_util import oriented_box_intersection_2d
    from shapely.geometry import Polygon
    def corners(cx, cy, l, w, yaw):
        c = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return c @ R.T + np.array([cx, cy])
    rng = np.random.default_rng(0); N = 300
    b1 = rng.uniform([-3, -3, 1, 1, 0], [3, 3, 5, 3, np.pi], size=(N, 5))
    b2 = rng.uniform([-3, -3, 1, 1, 0], [3, 3, 5, 3, np.pi], size=(N, 5))
    c1 = np.stack([corners(*b) for b in b1]); c2 = np.stack([corners(*b) for b in b2])
    t1 = torch.tensor(c1, dtype=torch.float32).unsqueeze(0); t2 = torch.tensor(c2, dtype=torch.float32).unsqueeze(0)
    area, _ = oriented_box_intersection_2d(t1, t2)
    sh = np.array([Polygon(c1[i]).intersection(Polygon(c2[i])).area for i in range(N)])
    err = float(np.abs(area[0].numpy() - sh).max())
    assert err < 1e-2, err
    return "max_abs_err=%.1e" % err

def t_perception():
    n = sum(x.numel() for x in PERC.model.parameters())
    return "%.2fM params on %s" % (n / 1e6, next(PERC.model.parameters()).device)

def t_predict():
    traj = np.zeros((20, 7))
    for i in range(20):
        traj[i] = [10 + i, 2.0, 0, 4, 2, 1.5, 0]
    out = prediction_grip({1: traj}, model_args={"model_api": PRED, "obs_length": 20, "pred_length": 20})
    assert 1 in out and np.asarray(out[1]).shape == (20, 2), out
    return "GRIP predicts %s, continues +x (dx=%.1f)" % (np.asarray(out[1]).shape,
                                                          float(out[1][-1][0] - out[1][0][0]))

def t_attacker_stack():
    av = stack.AVStack(PERC, PRED, ATK)
    assert av.tracker is not None and av.prev_off == 0.0 and av.vox is ATK
    assert ATK.pertnet is not None, "PertNet not loaded"
    assert abs(ATK.beta - 2.0) < 1e-6, ATK.beta
    return "AVStack + faithful attacker (PertNet loaded, beta=%.1f)" % ATK.beta


check("lane_intrusion metric", t_lane_intrusion)
check("ACC + predicted-lead", t_acc)
check("sort_vertices vs shapely", t_sort_vertices)
check("perception (AttFusion) loaded", t_perception)
check("predictor (GRIP++) predicts", t_predict)
check("attacker + AVStack", t_attacker_stack)
print("\nPASS=%d FAIL=%d" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
