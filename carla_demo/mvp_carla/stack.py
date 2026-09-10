"""The custom AV stack (perception -> tracking -> prediction) and the online attack."""
import os
import math
import numpy as np

from . import MVP_ROOT, reassert_cuda_device
from .config import (OBS_LEN, PRED_LEN, DT, STEALTH, MAX_OFFSET, PLAN_GRID, BETA, LANE_HALF,
                     LOOKAHEAD, ACC_S0, ACC_TGAP, ACC_KGAP, PERTNET_CKPT)

from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.tools.object_tracking import Ab3dmotTracker
from mvp.attack.scenario_attacker_util import tracking_ab3dmot, prediction_grip
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor


# ---- builders -------------------------------------------------------------
def build_perception():
    return OpencoodPerception(fusion_method="intermediate", model_name="pointpillar", dataset_name="OPV2V")


def build_predictor():
    # Lock the CUDA context to device 0 BEFORE importing GRIP (its main.py sets
    # CUDA_VISIBLE_DEVICES='1', which would otherwise hide our single GPU).
    reassert_cuda_device()
    import torch
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")          # force CUDA init with the '0' mask
    from prediction.model.GRIP import GRIPInterface
    reassert_cuda_device()
    ckpt = os.path.join(MVP_ROOT, "models/GRIP/OPV2V/checkpoint.pt")
    return GRIPInterface(OBS_LEN, PRED_LEN, pre_load_model=ckpt)


def load_pertnet(perception):
    """Load the trained PertNet (learned per-voxel correction)."""
    import torch
    from mvp.attack.perturbation_network import PerturbationNetwork
    ck = torch.load(os.path.join(MVP_ROOT, PERTNET_CKPT), map_location=perception.device)
    net = PerturbationNetwork(feature_channels=ck["feature_channels"],
                              geo_channels=ck["geo_channels"]).to(perception.device).eval()
    net.load_state_dict(ck["model_state"])
    return net, float(ck.get("beta", BETA))


def build_attacker(perception, beta=None, use_pertnet=True):
    """Full PosePert perception attack = ray-cast init + beta scaling + PertNet correction.
    Faithful default: beta=2 + PertNet (beta>3 corrupts features; PertNet ~doubles the IoU at 1 m)."""
    atk = LidarShiftVoxelwiseAttacker(perception=perception, beta=beta if beta is not None else BETA)
    if use_pertnet:
        net, paper_beta = load_pertnet(perception)
        atk.pertnet = net
        if beta is None:
            atk.beta = paper_beta
    return atk


# ---- danger metric --------------------------------------------------------
def _lon_lat(pred_xy, ego_xy, ego_fwd):
    """Longitudinal (along ego heading) and |lateral| (perpendicular) offsets of points."""
    rel = np.asarray(pred_xy) - np.asarray(ego_xy)
    lon = rel @ ego_fwd
    lat = np.abs(rel[:, 0] * ego_fwd[1] - rel[:, 1] * ego_fwd[0])
    return lon, lat


def lane_intrusion(pred_xy, ego_xy, ego_fwd):
    """Min |lateral| distance of a predicted trajectory's *ahead* points to the ego path
    (reporting metric). < LANE_HALF means the predicted object enters the ego lane."""
    lon, lat = _lon_lat(pred_xy, ego_xy, ego_fwd)
    m = (lon > 0) & (lon < LOOKAHEAD)
    return float(lat[m].min()) if m.any() else 99.0


def predicted_lead_distance(pred_xy, ego_xy, ego_fwd):
    """Longitudinal distance (m) to the nearest predicted point that lies IN the ego lane and
    ahead -- i.e. the predicted trajectory treated as a lead obstacle. None if no such point."""
    lon, lat = _lon_lat(pred_xy, ego_xy, ego_fwd)
    m = (lat < LANE_HALF) & (lon > 0) & (lon < LOOKAHEAD)
    return float(lon[m].min()) if m.any() else None


def acc_target_speed(cruise_kmh, ego_kmh, lead):
    """Constant-time-gap ACC: target speed (km/h) for a lead obstacle.
    lead = dict(distance s [m], lead_speed v_lead [m/s]) or None (-> free cruise).
        v_target = clip( v_lead + K * (s - (s0 + T*v_ego)),  0,  cruise )
    The same speed PID then tracks v_target, so braking/acceleration is smooth and principled."""
    if lead is None:
        return cruise_kmh
    cruise, ego_v = cruise_kmh / 3.6, ego_kmh / 3.6
    s, v_lead = lead["distance"], max(0.0, lead["lead_speed"])
    s_des = ACC_S0 + ACC_TGAP * ego_v
    v_follow = v_lead + ACC_KGAP * (s - s_des)
    return max(0.0, min(cruise, v_follow)) * 3.6


# ---- AV stack -------------------------------------------------------------
class AVStack:
    """Per-tick detect -> track -> predict, with an optional online-optimized PosePert attack
    realized through the voxelwise feature attack. Holds tracker/history state across the episode."""

    def __init__(self, perception, predictor, vox_attacker=None, max_offset=None, mode="movein"):
        self.perc = perception
        self.predictor = predictor
        self.vox = vox_attacker
        self.max_offset = MAX_OFFSET if max_offset is None else max_offset
        # "movein": drag the target toward the ego lane, minimizing predicted lane clearance.
        # "moveout": push it away from the lane, maximizing clearance so no lead is reported.
        assert mode in ("movein", "moveout")
        self.mode = mode
        self.tracker = Ab3dmotTracker()
        self.hist = {}
        self.sim_t = 0.0
        self.tgt_tid = None
        self.prev_off = 0.0
        self.realized_shift = 0.0
        self.dbg = {}            # per-step diagnostics (desired vs realized shifts, track vel, intrusion)
        self.n_attack_ok = 0     # attack steps where the voxelwise attack actually ran
        self.n_attack_err = 0    # attack steps that fell back to clean detections
        self.last_attack_err = None
        self.n_lead_suppressed = 0   # attack steps that left the ACC with no in-lane lead

    def _clean_detections(self, frame, ego_pose):
        pred, _ = self.perc.run(frame, "ego")
        return np.array([bbox_sensor_to_map(b, ego_pose) for b in pred]) if len(pred) else np.zeros((0, 7))

    def _plan_offset(self, target_true, latd, ego_xy, ego_fwd):
        """Online query-based plan: pick this-frame total offset in [prev_off, prev_off+STEALTH]
        minimizing GRIP's predicted lane intrusion (observe-predict-plan, re-planned each frame)."""
        base = np.asarray(self.hist[self.tgt_tid])[-(OBS_LEN - 1):]
        want_max = (self.mode == "moveout")
        best_off, best_intr = self.prev_off, (-1.0 if want_max else 99.0)
        hi = min(self.prev_off + STEALTH, self.max_offset)   # don't ask beyond the realizable shift
        for off in np.linspace(self.prev_off, hi, PLAN_GRID):
            cur = target_true.copy().astype(np.float64)
            cur[0] += latd[0] * off
            cur[1] += latd[1] * off
            track = np.vstack([base, cur[:7][None, :]])
            if len(track) < 6:
                return off
            pred = prediction_grip({99999: track}, model_args={
                "model_api": self.predictor, "obs_length": OBS_LEN, "pred_length": PRED_LEN})
            if 99999 in pred:
                intr = lane_intrusion(pred[99999], ego_xy, ego_fwd)
                if (intr > best_intr) if want_max else (intr < best_intr):
                    best_off, best_intr = off, intr
        return best_off

    @staticmethod
    def _out_ward(target_true, ego_xy, ego_fwd):
        """Lateral unit vector pointing further OUT of the ego lane (the move-out direction).

        Perpendicular to the ego heading rather than the target heading: a blocker sitting in
        the ego lane has no meaningful "toward the ego" side, so _lane_ward's sign is unstable
        there. The sign follows the blocker's existing lateral lean, so the attack exaggerates
        the encroachment it already has.
        """
        perp = np.array([-ego_fwd[1], ego_fwd[0]], dtype=np.float64)
        lat = (np.asarray(target_true[:2], dtype=np.float64) - np.asarray(ego_xy)) @ perp
        return perp if lat >= 0 else -perp

    def _lateral_dir(self, target_true, ego_xy, ego_fwd):
        return (self._out_ward(target_true, ego_xy, ego_fwd) if self.mode == "moveout"
                else self._lane_ward(target_true, ego_xy))

    @staticmethod
    def _lane_ward(target_true, ego_xy):
        """Lateral unit vector perpendicular to the target's heading, pointing toward the ego's
        lane (the move-in direction) -- NOT toward the ego's position (which is diagonally back)."""
        yaw = float(target_true[6])
        perp = np.array([-math.sin(yaw), math.cos(yaw)])
        if (ego_xy - target_true[:2]) @ perp < 0:
            perp = -perp
        return perp

    def _attacked_detections(self, frame, ego_pose, collab_pose, target_true, ego_xy, ego_fwd):
        """Plan the next stealthy target offset, then REALIZE it via the voxelwise feature attack."""
        latd = self._lateral_dir(target_true, ego_xy, ego_fwd)
        off = self._plan_offset(target_true, latd, ego_xy, ego_fwd)
        self.prev_off = off
        desired = target_true.copy().astype(np.float64)
        desired[0] += latd[0] * off
        desired[1] += latd[1] * off
        res = self.vox.run_multi_vehicle(frame, {
            "attacker_vehicle_id": "collab", "victim_vehicle_id": "ego",
            "bbox_to_remove": bbox_map_to_sensor(target_true, collab_pose),
            "bbox_to_spoof": bbox_map_to_sensor(desired, collab_pose)})
        pred = res["pred_bboxes"]
        dets = np.array([bbox_sensor_to_map(b, ego_pose) for b in pred]) if len(pred) else np.zeros((0, 7))
        # diagnostics: realized DETECTION-level lateral shift (toward ego) of the target-nearest det
        if len(dets):
            i = int(np.argmin(np.hypot(dets[:, 0] - target_true[0], dets[:, 1] - target_true[1])))
            self.dbg.update(desired_off=float(off), det_lat_shift=float((dets[i, :2] - target_true[:2]) @ latd))
        return dets

    def step(self, frame, ego_pose, collab_pose, target_true, ego_xy, ego_fwd, attack=False):
        """One perception step. Returns dict(intrusion, hazard, n_det)."""
        can_attack = (attack and self.vox is not None and self.tgt_tid is not None
                      and self.tgt_tid in self.hist and len(self.hist[self.tgt_tid]) >= 6)
        if can_attack:
            try:
                dets = self._attacked_detections(frame, ego_pose, collab_pose, target_true, ego_xy, ego_fwd)
                self.n_attack_ok += 1
            except Exception as e:
                # Silent fallback would make a failed attack look like a robust victim.
                self.n_attack_err += 1
                self.last_attack_err = repr(e)
                dets = self._clean_detections(frame, ego_pose)
        else:
            dets = self._clean_detections(frame, ego_pose)

        _, indexed = tracking_ab3dmot(self.tracker, self.sim_t, dets)
        self.sim_t += DT

        # target track = tracked box nearest the REAL target position
        self.tgt_tid, best_d = None, 4.0
        for tid, bb in indexed.items():
            d = math.hypot(bb[0] - target_true[0], bb[1] - target_true[1])
            if d < best_d:
                best_d, self.tgt_tid = d, tid
        if self.tgt_tid is not None:
            self.realized_shift = max(self.realized_shift, best_d)
        for tid, bb in indexed.items():
            self.hist.setdefault(tid, []).append(np.asarray(bb)[:7])
            self.hist[tid] = self.hist[tid][-OBS_LEN:]

        intrusion, lead = 99.0, None
        obs = {tid: np.stack(h) for tid, h in self.hist.items()
               if len(h) >= 6 and math.hypot(h[-1][0] - ego_xy[0], h[-1][1] - ego_xy[1]) < 40}
        if obs:
            preds = prediction_grip(obs, model_args={
                "model_api": self.predictor, "obs_length": OBS_LEN, "pred_length": PRED_LEN})
            if self.tgt_tid in preds:
                pt = np.asarray(preds[self.tgt_tid])
                intrusion = lane_intrusion(pt, ego_xy, ego_fwd)
                s = predicted_lead_distance(pt, ego_xy, ego_fwd)
                if s is not None:
                    # lead longitudinal speed from the target track (along the ego heading)
                    h = self.hist.get(self.tgt_tid, [])
                    v_lead = 0.0
                    if len(h) >= 2:
                        v_lead = float((np.asarray(h[-1][:2]) - np.asarray(h[-2][:2])) @ ego_fwd / DT)
                    lead = {"distance": s, "lead_speed": v_lead}
        if attack and lead is None:
            self.n_lead_suppressed += 1
        # diagnostics: realized TRACK-level lateral shift + lateral velocity (toward ego)
        self.dbg["intrusion"] = intrusion
        self.dbg["lead_dist"] = None if lead is None else lead["distance"]
        if self.tgt_tid is not None:
            latd = self._lateral_dir(target_true, ego_xy, ego_fwd)
            tb = np.asarray(indexed[self.tgt_tid][:2])
            self.dbg["track_lat_shift"] = float((tb - target_true[:2]) @ latd)
            h = self.hist.get(self.tgt_tid, [])
            self.dbg["track_lat_vel"] = (float((np.asarray(h[-1][:2]) - np.asarray(h[-2][:2])) @ latd / DT)
                                         if len(h) >= 2 else 0.0)
        return dict(intrusion=intrusion, lead=lead, n_det=len(dets))
