"""
Voxel-wise feature manipulation attacker for object shifting.

Instead of gradient-based optimization, this attacker:
1. Constructs multi-view F_spoof and F_remove from precomputed point cloud bases
2. Replaces the attacker's spatial features at object-zone voxels with
   amplified malicious features (beta * F_attack)
3. Runs the perception pipeline with the modified features

This is lightweight (no gradient optimization) and achieves ~77% strong
success rate on OPV2V with PointPillar + AttFusion.
"""

import os
import copy
import pickle
import logging
import numpy as np
import torch
from collections import OrderedDict

from .attacker import Attacker
from mvp.data.util import (
    bbox_map_to_sensor, bbox_sensor_to_map,
    pcd_sensor_to_map, pcd_map_to_sensor,
    get_point_indices_in_bbox,
)
from mvp.config import data_root
from mvp.attack.shift_rotation import apply_shift
from mvp.util import set_seed


class LidarShiftVoxelwiseAttacker(Attacker):
    def __init__(self, perception, dataset=None, beta=4.0, gamma=0.0,
                 extension=1.0, multiview=True, bases_dir=None, debug=False):
        """
        Args:
            perception: OpencoodPerception instance
            dataset: OPV2VDataset instance
            beta: feature amplification factor for active voxels
            gamma: benign feature projection strength (projected onto benign
                   unit direction at each voxel). Default 0 (disabled).
                   When gamma > 0, f_atk = beta * F_attack + gamma * ê_benign
            extension: bbox extension (meters) for zone mask computation
            multiview: if True, use multi-view F_attack (best per-voxel from all vehicles)
            bases_dir: path to precomputed spoof/remove bases
                       (default: data/OPV2V/multi_frame/attack/lidar_shift_bases)
            debug: if True, return extra info (features, zones, etc.)
        """
        super().__init__()
        self.dataset = dataset
        self.name = "lidar_shift"
        self.load_benchmark_meta()
        self.name = "lidar_shift_voxelwise"

        self.name += f"_b{beta:.0f}"
        if gamma > 0:
            self.name += f"_g{gamma:.0f}"
        if multiview:
            self.name += "_mv"
        if debug:
            self.name += "_debug"

        self.perception = perception
        self.beta = beta
        self.gamma = gamma
        self.extension = extension
        self.multiview = multiview
        self.debug = debug
        self.pertnet = None  # Set externally to enable PertNet corrections
        self.pertnet_epsilon = 10.0  # PertNet correction clamp bound

        if bases_dir is None:
            self.bases_dir = os.path.join(data_root,
                "OPV2V/multi_frame/attack/lidar_shift_bases")
        else:
            self.bases_dir = bases_dir

        # Cache BEV grid parameters
        self.lidar_range = perception.dataset.pre_processor.params["cav_lidar_range"]
        self.voxel_size = perception.dataset.pre_processor.params["args"]["voxel_size"]

    def _get_bboxes(self, multi_vehicle_case, attack_opts, frame_id):
        """Get original and shifted bboxes in attacker's sensor frame."""
        attacker_id = attack_opts["attacker_vehicle_id"]
        object_id = attack_opts["object_id"]
        sv = multi_vehicle_case[frame_id][attacker_id]
        object_index = sv["object_ids"].index(object_id)
        bbox_original = np.copy(sv["gt_bboxes"][object_index])
        bbox_target = apply_shift(bbox_original, attack_opts)
        return bbox_original, bbox_target, object_index

    def _bbox_to_voxel_mask(self, bbox_ego):
        """Convert a bbox (in ego frame) to a 2D BEV mask."""
        from matplotlib.path import Path
        lr = self.lidar_range
        vs = self.voxel_size
        H = round((lr[4] - lr[1]) / vs[1])
        W = round((lr[3] - lr[0]) / vs[0])
        x, y, z, l, w, h, yaw = bbox_ego
        ext = self.extension
        hl, hw = (l + ext) / 2, (w + ext) / 2
        corners = np.array([[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]])
        c, s = np.cos(yaw), np.sin(yaw)
        corners = corners @ np.array([[c, s], [-s, c]]) + np.array([x, y])
        cv = np.zeros_like(corners)
        cv[:, 0] = (corners[:, 0] - lr[0]) / vs[0]
        cv[:, 1] = (corners[:, 1] - lr[1]) / vs[1]
        polygon = Path(cv)
        w_min = max(0, int(cv[:, 0].min()) - 1)
        w_max = min(W, int(cv[:, 0].max()) + 2)
        h_min = max(0, int(cv[:, 1].min()) - 1)
        h_max = min(H, int(cv[:, 1].max()) + 2)
        mask = np.zeros((H, W), dtype=bool)
        if w_min >= w_max or h_min >= h_max:
            return mask
        ww, hh = np.meshgrid(np.arange(w_min, w_max) + 0.5,
                              np.arange(h_min, h_max) + 0.5)
        pts = np.stack([ww.ravel(), hh.ravel()], axis=1)
        mask[h_min:h_max, w_min:w_max] = polygon.contains_points(pts).reshape(hh.shape)
        return mask

    def _get_spatial_features(self, multi_vehicle_case, ego_id):
        """Run pillar_vfe + scatter to get spatial features."""
        batch = self.perception.preprocessors[self.perception.fusion_method](
            multi_vehicle_case, ego_id)
        batch_data = self.perception.dataset.collate_batch_test([batch])
        from opencood.tools import train_utils
        batch_data = train_utils.to_device(batch_data, self.perception.device)
        bd = {
            'voxel_features': batch_data['ego']['processed_lidar']['voxel_features'],
            'voxel_coords': batch_data['ego']['processed_lidar']['voxel_coords'],
            'voxel_num_points': batch_data['ego']['processed_lidar']['voxel_num_points'],
            'record_len': batch_data['ego']['record_len'],
        }
        self.perception.model.pillar_vfe(bd)
        self.perception.model.scatter(bd)
        return bd['spatial_features'].clone().detach(), batch_data

    def _compute_multiview_features(self, case_frame, victim_id, attacker_id,
                                     attacker_index, bases_info, mode="spoof"):
        """Compute multi-view F_attack: for each vehicle, load its modified pcd
        into the attacker's slot (transformed to attacker's frame), run through
        pillar_vfe+scatter, and take per-voxel strongest by L2 norm."""
        vehicle_ids = list(case_frame.keys())
        atk_pose = case_frame[attacker_id]["lidar_pose"]
        candidates = []

        for vid in vehicle_ids:
            upd = bases_info[vid][mode]
            if upd["ignore_indices"].shape[0] == 0 and upd["append_data"].shape[0] == 0:
                continue

            modified_pcd = self.apply_ray_tracing(
                case_frame[vid]["lidar"].copy(),
                ignore_indices=upd["ignore_indices"],
                append_data=upd["append_data"])

            # Transform to attacker's frame if from another vehicle
            if vid != attacker_id:
                vid_pose = case_frame[vid]["lidar_pose"]
                pcd_map = pcd_sensor_to_map(modified_pcd[:, :3], vid_pose, dataset_name=self.perception.dataset_name)
                pcd_atk = pcd_map_to_sensor(pcd_map, atk_pose, dataset_name=self.perception.dataset_name)
                modified_pcd = np.hstack([pcd_atk[:, :3],
                    np.ones((pcd_atk.shape[0], 1))]).astype(np.float32)

            case_mod = copy.deepcopy(case_frame)
            case_mod[attacker_id]["lidar"] = modified_pcd

            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_mod, _ = self._get_spatial_features(case_mod, victim_id)
            candidates.append(F_mod[attacker_index].clone())

        if not candidates:
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_orig, _ = self._get_spatial_features(case_frame, victim_id)
            return F_orig[attacker_index].clone()

        stacked = torch.stack(candidates, dim=0)
        norms = stacked.norm(dim=1)
        best = norms.argmax(dim=0)
        C = stacked.shape[1]
        idx = best.unsqueeze(0).unsqueeze(0).expand(1, C, *best.shape)
        return stacked.gather(0, idx).squeeze(0)

    def _get_post_backbone_features(self, multi_vehicle_case, ego_id):
        """Run pillar_vfe + scatter + backbone + shrink to get post-backbone features."""
        batch = self.perception.preprocessors[self.perception.fusion_method](
            multi_vehicle_case, ego_id)
        batch_data = self.perception.dataset.collate_batch_test([batch])
        from opencood.tools import train_utils
        batch_data = train_utils.to_device(batch_data, self.perception.device)
        bd = {
            'voxel_features': batch_data['ego']['processed_lidar']['voxel_features'],
            'voxel_coords': batch_data['ego']['processed_lidar']['voxel_coords'],
            'voxel_num_points': batch_data['ego']['processed_lidar']['voxel_num_points'],
            'record_len': batch_data['ego']['record_len'],
        }
        self.perception.model.pillar_vfe(bd)
        self.perception.model.scatter(bd)
        self.perception.model.backbone(bd)
        sf2d = bd['spatial_features_2d']
        if hasattr(self.perception.model, 'shrink_flag') and self.perception.model.shrink_flag:
            sf2d = self.perception.model.shrink_conv(sf2d)
        if hasattr(self.perception.model, 'compression') and self.perception.model.compression:
            sf2d = self.perception.model.naive_compressor(sf2d)
        return sf2d.clone().detach(), batch_data

    def _bbox_to_feature2d_mask(self, bbox_ego, H_out, W_out):
        """Convert a bbox (in ego frame) to a 2D mask at post-backbone resolution."""
        from matplotlib.path import Path
        lr = self.lidar_range
        vs = self.voxel_size
        H_in = int((lr[4] - lr[1]) / vs[1])
        W_in = int((lr[3] - lr[0]) / vs[0])
        x, y, z, l, w, h, yaw = bbox_ego
        ext = self.extension
        hl, hw = (l + ext) / 2, (w + ext) / 2
        corners = np.array([[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]])
        c, s = np.cos(yaw), np.sin(yaw)
        corners = corners @ np.array([[c, s], [-s, c]]) + np.array([x, y])
        # Scale to post-backbone resolution
        h_scale = H_out / H_in
        w_scale = W_out / W_in
        cv = np.zeros_like(corners)
        cv[:, 0] = (corners[:, 0] - lr[0]) / vs[0] * w_scale
        cv[:, 1] = (corners[:, 1] - lr[1]) / vs[1] * h_scale
        polygon = Path(cv)
        w_min = max(0, int(cv[:, 0].min()) - 1)
        w_max = min(W_out, int(cv[:, 0].max()) + 2)
        h_min = max(0, int(cv[:, 1].min()) - 1)
        h_max = min(H_out, int(cv[:, 1].max()) + 2)
        mask = np.zeros((H_out, W_out), dtype=bool)
        if w_min >= w_max or h_min >= h_max:
            return mask
        ww, hh = np.meshgrid(np.arange(w_min, w_max) + 0.5,
                              np.arange(h_min, h_max) + 0.5)
        pts = np.stack([ww.ravel(), hh.ravel()], axis=1)
        mask[h_min:h_max, w_min:w_max] = polygon.contains_points(pts).reshape(hh.shape)
        return mask

    def _run_with_post_backbone_features(self, batch_data, sf2d):
        """Run fusion + detection heads on post-backbone features."""
        from opencood.utils import box_utils

        with torch.no_grad():
            record_len = batch_data['ego']['record_len']
            fused = self.perception.model.fusion_net(sf2d, record_len)
            psm = self.perception.model.cls_head(fused)
            rm = self.perception.model.reg_head(fused)

            output_dict = OrderedDict()
            output_dict['ego'] = {'psm': psm, 'rm': rm}

            post_result = self.perception.dataset.post_process(batch_data, output_dict)
            pred_box_tensor, pred_score = post_result[0], post_result[1]

            if pred_box_tensor is None:
                return np.array([]).reshape(0, 7), np.array([])

            pred_bboxes = pred_box_tensor.cpu().numpy()
            pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
            pred_bboxes[:, 2] -= 0.5 * pred_bboxes[:, 5]
            pred_scores = pred_score.cpu().numpy()

            if pred_scores.ndim == 2 and pred_scores.shape[1] == 2:
                pred_scores, pred_classes = pred_scores[:, 0], pred_scores[:, 1].astype(np.int)
                mask = np.logical_and(pred_scores >= 0.1, pred_classes == 1)
            else:
                mask = pred_scores >= 0.1

            return pred_bboxes[mask], pred_scores[mask]

    def _run_with_features(self, batch_data, spatial_features):
        """Run backbone + fusion + detection with modified spatial features.

        Injects modified spatial features into the model's forward pass,
        skipping pillar_vfe and scatter (which have already been run).
        Works with any model architecture by monkey-patching scatter output.
        """
        from opencood.utils import box_utils

        with torch.no_grad():
            model = self.perception.model

            # Monkey-patch scatter to return our modified features
            original_scatter_forward = model.scatter.forward
            def patched_scatter(bd):
                bd['spatial_features'] = spatial_features
                return bd
            model.scatter.forward = patched_scatter

            # Also patch pillar_vfe to be a no-op (features already computed)
            original_vfe_forward = model.pillar_vfe.forward
            def patched_vfe(bd):
                bd['pillar_features'] = torch.zeros(1, device=spatial_features.device)
                return bd
            model.pillar_vfe.forward = patched_vfe

            # Run the full model forward
            try:
                output_dict = model(batch_data['ego'])
            finally:
                # Restore original methods
                model.scatter.forward = original_scatter_forward
                model.pillar_vfe.forward = original_vfe_forward

            output_wrapped = OrderedDict()
            output_wrapped['ego'] = output_dict

            post_result = self.perception.dataset.post_process(batch_data, output_wrapped)
            pred_box_tensor, pred_score = post_result[0], post_result[1]

            if pred_box_tensor is None:
                return np.array([]).reshape(0, 7), np.array([])

            pred_bboxes = pred_box_tensor.cpu().numpy()
            pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
            pred_bboxes[:, 2] -= 0.5 * pred_bboxes[:, 5]
            pred_scores = pred_score.cpu().numpy()

            if pred_scores.ndim == 2 and pred_scores.shape[1] == 2:
                pred_scores, pred_classes = pred_scores[:, 0], pred_scores[:, 1].astype(np.int)
                mask = np.logical_and(pred_scores >= 0.1, pred_classes == 1)
            else:
                mask = pred_scores >= 0.1

            return pred_bboxes[mask], pred_scores[mask]

    def run(self, multi_frame_case, attack_opts):
        """
        Run the voxel-wise feature manipulation attack.

        Args:
            multi_frame_case: list of dicts, one per frame. Each dict maps
                vehicle_id -> {"lidar": ..., "lidar_pose": ..., "gt_bboxes": ..., ...}
            attack_opts: dict with keys:
                - attacker_vehicle_id, victim_vehicle_id, object_id
                - shift_direction, shift_distance, rotation
                - frame_ids: list of frame indices to attack

        Returns:
            (case, info) — same format as LidarShiftIntermediateAttacker.run()
        """
        case = copy.deepcopy(multi_frame_case)
        info = [{} for _ in range(10)]

        attacker_id = attack_opts["attacker_vehicle_id"]
        ego_id = attack_opts["victim_vehicle_id"]

        for frame_id in attack_opts["frame_ids"]:
            info[frame_id][ego_id] = {}

            # Load precomputed bases
            attack_index = None
            for aid, attack in enumerate(self.attack_list):
                if (attack["attack_opts"]["attacker_vehicle_id"] == attacker_id and
                    attack["attack_opts"]["object_id"] == attack_opts["object_id"] and
                    attack["attack_meta"]["case_id"] == attack_opts.get("case_id",
                        attack["attack_meta"]["case_id"])):
                    attack_index = aid
                    break

            bases_path = os.path.join(self.bases_dir,
                f"{attack_opts.get('attack_id', attack_index):06d}", "attack_info.pkl")

            if not os.path.exists(bases_path):
                logging.warning(f"Bases not found at {bases_path}, running without attack")
                pred_bboxes, pred_scores = self.perception.run(
                    case[frame_id], ego_id)
                info[frame_id][ego_id] = {
                    "pred_bboxes": pred_bboxes,
                    "pred_scores": pred_scores,
                }
                info[frame_id][attacker_id] = {}
                continue

            bases_info = pickle.load(open(bases_path, "rb"))

            # Get bboxes
            bbox_original, bbox_target, _ = self._get_bboxes(
                multi_frame_case, attack_opts, frame_id)
            atk_pose = multi_frame_case[frame_id][attacker_id]["lidar_pose"]
            vic_pose = multi_frame_case[frame_id][ego_id]["lidar_pose"]

            # Compute zone masks in ego's BEV frame
            _ds = self.perception.dataset_name
            bbox_orig_vic = bbox_map_to_sensor(
                bbox_sensor_to_map(bbox_original, atk_pose, dataset_name=_ds), vic_pose, dataset_name=_ds)
            bbox_tgt_vic = bbox_map_to_sensor(
                bbox_sensor_to_map(bbox_target, atk_pose, dataset_name=_ds), vic_pose, dataset_name=_ds)

            mask_orig = self._bbox_to_voxel_mask(bbox_orig_vic)
            mask_tgt = self._bbox_to_voxel_mask(bbox_tgt_vic)
            zone_A = mask_orig & ~mask_tgt
            zone_active = mask_orig | mask_tgt

            # Get vehicle ordering
            base_data_dict = self.perception.retrieve_base_data(
                case[frame_id], ego_id)
            attacker_index = list(base_data_dict.keys()).index(attacker_id)

            dev = self.perception.device

            with torch.no_grad():
                # Compute base features
                set_seed(42, set_python=False, set_numpy=False, set_torch=True)
                F_orig, batch_data = self._get_spatial_features(
                    case[frame_id], ego_id)

                # Compute attack features
                if self.multiview:
                    F_spoof = self._compute_multiview_features(
                        case[frame_id], ego_id, attacker_id,
                        attacker_index, bases_info, mode="spoof")
                    F_remove = self._compute_multiview_features(
                        case[frame_id], ego_id, attacker_id,
                        attacker_index, bases_info, mode="remove")
                else:
                    # Single-view: only attacker's own modified pcd
                    su = bases_info[attacker_id]["spoof"]
                    case_spoof = copy.deepcopy(case[frame_id])
                    case_spoof[attacker_id]["lidar"] = self.apply_ray_tracing(
                        case_spoof[attacker_id]["lidar"].copy(),
                        ignore_indices=su["ignore_indices"],
                        append_data=su["append_data"])
                    set_seed(42, set_python=False, set_numpy=False, set_torch=True)
                    F_spoof_full, _ = self._get_spatial_features(
                        case_spoof, ego_id)
                    F_spoof = F_spoof_full[attacker_index]

                    ru = bases_info[attacker_id]["remove"]
                    case_remove = copy.deepcopy(case[frame_id])
                    case_remove[attacker_id]["lidar"] = self.apply_ray_tracing(
                        case_remove[attacker_id]["lidar"].copy(),
                        ignore_indices=ru["ignore_indices"],
                        append_data=ru["append_data"])
                    set_seed(42, set_python=False, set_numpy=False, set_torch=True)
                    F_remove_full, _ = self._get_spatial_features(
                        case_remove, ego_id)
                    F_remove = F_remove_full[attacker_index]

                # Combine F_attack: spoof at target voxels, remove at original-only
                F_attack = F_spoof.clone()
                tA = torch.from_numpy(zone_A).to(dev)
                if tA.any():
                    F_attack[:, tA] = F_remove[:, tA]

                t_active = torch.from_numpy(zone_active).to(dev)

                # Apply voxel-wise feature manipulation
                features = F_orig.clone()
                if t_active.any():
                    if self.gamma > 0:
                        # Compute benign unit direction for projection
                        benign_indices = [i for i in range(F_orig.shape[0])
                                          if i != attacker_index]
                        F_benign_sum = sum(F_orig[i] for i in benign_indices)
                        benign_at_active = F_benign_sum[:, t_active]
                        benign_norm = benign_at_active.norm(
                            dim=0, keepdim=True).clamp(min=1e-8)
                        benign_unit = benign_at_active / benign_norm

                        features[attacker_index][:, t_active] = (
                            self.beta * F_attack[:, t_active] +
                            self.gamma * benign_unit
                        )
                    else:
                        features[attacker_index][:, t_active] = (
                            self.beta * F_attack[:, t_active]
                        )

                # Run perception with modified features
                pred_bboxes, pred_scores = self._run_with_features(
                    batch_data, features)

            info[frame_id][ego_id] = {
                "pred_bboxes": pred_bboxes,
                "pred_scores": pred_scores,
            }
            info[frame_id][attacker_id] = {
                "beta": self.beta,
                "n_active_voxels": int(zone_active.sum()),
            }

            if self.debug:
                info[frame_id][attacker_id].update({
                    "zone_A": zone_A,
                    "zone_active": zone_active,
                    "F_attack_norm": float(F_attack[:, t_active].norm().cpu()),
                })

        return case, info

    def _generate_spoof_pcd(self, pcd, bbox_target, bbox_original):
        """Generate spoofed point cloud: remove original object + add dense car at target.

        Uses 4 phantom LiDARs at uniform angles around the target for
        occlusion-free coverage. Also adds ground plane points near the
        target bbox for realistic ground features.
        """
        from mvp.tools.ray_tracing import get_model_mesh, ray_intersection
        from mvp.tools.ground_detection import get_ground_plane, get_ground_mesh
        from mvp.data.util import sort_lidar_points

        DEFAULT_CAR = "car_0200"
        N_PHANTOM = 4        # number of phantom LiDARs
        PHANTOM_DIST = 10.0  # distance from phantom to target center

        # Remove original object points
        bbox_del = np.copy(bbox_original)
        bbox_del[3:6] += 0.2
        del_indices = get_point_indices_in_bbox(bbox_del, pcd[:, :3])
        if len(del_indices) > 0:
            remain = np.ones(pcd.shape[0], dtype=bool)
            remain[del_indices] = False
            pcd_clean = pcd[remain]
        else:
            pcd_clean = pcd

        target_center = bbox_target[:3].copy()
        target_distance = np.sqrt(bbox_target[0] ** 2 + bbox_target[1] ** 2)

        # bbox z is the car bottom; mesh z starts at 0 (bottom=0, top=h),
        # so translate z to bbox_target[2] to align mesh bottom with ground
        bbox_for_mesh = np.copy(bbox_target)
        bbox_for_mesh[2] = bbox_target[2]
        car_mesh = get_model_mesh(DEFAULT_CAR, bbox_for_mesh)

        # LiDAR scan spec: 64 beams, elevation [-25, 2] deg, h-resolution 0.35 deg
        N_BEAMS = 64
        ELEV_MIN, ELEV_MAX = -25.0, 2.0
        H_RESOLUTION = 0.35  # degrees
        elevations = np.linspace(np.radians(ELEV_MIN), np.radians(ELEV_MAX), N_BEAMS)

        # Build scene with car mesh + optional ground mesh for single ray cast
        meshes = [car_mesh]
        # Skip ground plane for V2X-Real: real-world terrain is uneven,
        # flat ground plane produces physically implausible points
        ds_name = getattr(self.perception, 'dataset_name', 'OPV2V')
        use_ground = (ds_name == 'OPV2V')
        if use_ground:
            try:
                plane_model, _ = get_ground_plane(pcd, method="ransac")
                ground_mesh = get_ground_mesh(plane_model)
                meshes.append(ground_mesh)
            except Exception:
                pass

        # Place N_PHANTOM phantom LiDARs at uniform angles around target
        all_pts = []
        angles = np.linspace(0, 2 * np.pi, N_PHANTOM, endpoint=False)
        angles = angles + bbox_target[6]  # offset by target yaw

        # Build full 360-degree ray directions (shared across all phantoms)
        azimuths = np.arange(0, 360, H_RESOLUTION)
        az_rad = np.radians(azimuths)
        el_grid, az_grid = np.meshgrid(elevations, az_rad)
        direction = np.stack([
            np.cos(el_grid.flatten()) * np.cos(az_grid.flatten()),
            np.cos(el_grid.flatten()) * np.sin(az_grid.flatten()),
            np.sin(el_grid.flatten())
        ], axis=1)

        for angle in angles:
            phantom_pos = np.array([
                target_center[0] + PHANTOM_DIST * np.cos(angle),
                target_center[1] + PHANTOM_DIST * np.sin(angle),
                0.0
            ])

            rays = np.hstack([
                np.tile(phantom_pos, (direction.shape[0], 1)),
                direction
            ]).astype(np.float32)

            # Single ray cast against car + ground
            intersect = ray_intersection(meshes, rays)
            hit = intersect[:, 0] ** 2 < 10000
            if hit.any():
                all_pts.append(intersect[hit])

        if not all_pts:
            return pcd_clean

        append_data = np.vstack(all_pts)

        # Keep only points within 1m xy-distance from the boundary of
        # either original or target bbox (accounting for yaw rotation)
        KEEP_MARGIN = 1.0

        def rotated_bbox_dist(pts_xy, bbox):
            """Compute 2D distance from points to rotated bbox boundary."""
            # Rotate points into bbox-local frame
            cx, cy, yaw = bbox[0], bbox[1], bbox[6]
            cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
            dx = pts_xy[:, 0] - cx
            dy = pts_xy[:, 1] - cy
            lx = dx * cos_y - dy * sin_y
            ly = dx * sin_y + dy * cos_y
            # Distance to axis-aligned bbox boundary in local frame
            return np.maximum(np.abs(lx) - bbox[3] / 2, np.abs(ly) - bbox[4] / 2)

        d_orig = rotated_bbox_dist(append_data[:, :2], bbox_original)
        d_tgt = rotated_bbox_dist(append_data[:, :2], bbox_target)
        keep = (d_orig < KEEP_MARGIN) | (d_tgt < KEEP_MARGIN)
        append_data = append_data[keep]

        # Subsample for V2X-Real: ray casting produces too-dense points
        # that overwhelm the BatchNorm-based model. Target ~800-1200 points
        # (comparable to a real car at medium range in V2X-Real data).
        if ds_name != 'OPV2V' and len(append_data) > 1200:
            subsample_idx = np.random.choice(
                len(append_data), 1200, replace=False)
            append_data = append_data[subsample_idx]

        # Add Gaussian noise
        noise_std = 0.005 + 0.0008 * target_distance
        append_data = append_data + np.random.normal(0, noise_std, size=append_data.shape)

        combined = np.vstack([pcd_clean[:, :3], append_data])
        combined, _ = sort_lidar_points(combined)
        return np.hstack([combined, np.ones((combined.shape[0], 1))])

    def _generate_remove_pcd(self, pcd, bbox_original):
        """Generate removal point cloud: remove object + dense ground fill."""
        from mvp.tools.ground_detection import get_ground_plane, get_ground_mesh
        from mvp.tools.ray_tracing import ray_intersection
        from mvp.data.util import get_open3d_bbox, sort_lidar_points
        import open3d as o3d

        DENSE_DIST = 6

        bbox_ext = np.copy(bbox_original)
        bbox_ext[2] -= 5
        bbox_ext[3:5] += 0.6
        bbox_ext[5] = 10
        bbox_ext_o3d = get_open3d_bbox(bbox_ext)

        points = pcd[:, :3]
        point_indices = bbox_ext_o3d.get_point_indices_within_bounding_box(
            o3d.utility.Vector3dVector(points))
        point_indices = np.array(point_indices).reshape(-1).astype(np.int32)

        if len(point_indices) == 0:
            return pcd

        plane_model, _ = get_ground_plane(pcd, method="ransac")
        ground_mesh = get_ground_mesh(plane_model)

        distance = np.sqrt(np.sum(points ** 2, axis=1))
        direction = points / np.tile(distance.reshape(-1, 1), (1, 3))
        rays = np.hstack([np.zeros((direction.shape[0], 3)), direction])

        target_offset = bbox_original[:2]
        target_distance = np.sqrt(np.sum(target_offset ** 2))
        if target_distance > DENSE_DIST:
            lidar_offset = target_offset / target_distance * (target_distance - DENSE_DIST)
        else:
            lidar_offset = np.zeros(2)
        extra_rays = rays.copy()
        extra_rays[:, :2] = lidar_offset

        extra_intersect = ray_intersection([ground_mesh], extra_rays)
        hit = extra_intersect[:, 0] ** 2 < 10000
        extra_points = extra_intersect[hit]

        if extra_points.shape[0] > 0:
            inside = bbox_ext_o3d.get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(extra_points))
            if len(inside) > 0:
                extra_points = extra_points[np.array(inside)]
            else:
                extra_points = np.zeros((0, 3))

        remain = np.ones(pcd.shape[0], dtype=bool)
        remain[point_indices] = False
        remaining = pcd[remain, :3]

        if extra_points.shape[0] > 0:
            combined = np.vstack([remaining, extra_points])
        else:
            combined = remaining

        combined, _ = sort_lidar_points(combined)
        return np.hstack([combined, np.ones((combined.shape[0], 1))])

    def run_multi_vehicle(self, multi_vehicle_case, attack_opts):
        """
        Run voxel-wise feature attack on a single frame, compatible with
        ScenarioAttacker.perception_attack() interface.

        Args:
            multi_vehicle_case: dict mapping vehicle_id -> vehicle_data for ONE frame
            attack_opts: dict with keys:
                - attacker_vehicle_id, victim_vehicle_id
                - bbox_to_remove: (7,) bbox in attacker's sensor frame
                - bbox_to_spoof: (7,) bbox in attacker's sensor frame
                - post_backbone: (bool) if True, perturb post-backbone features
                  (after backbone+shrink, before fusion). Default False.

        Returns:
            dict with "pred_bboxes" and "pred_scores" in victim's sensor frame
        """
        attacker_id = attack_opts["attacker_vehicle_id"]
        victim_id = attack_opts["victim_vehicle_id"]
        bbox_to_remove = attack_opts["bbox_to_remove"]
        bbox_to_spoof = attack_opts["bbox_to_spoof"]
        post_backbone = attack_opts.get("post_backbone", False)

        atk_pose = multi_vehicle_case[attacker_id]["lidar_pose"]
        vic_pose = multi_vehicle_case[victim_id]["lidar_pose"]

        # Convert bboxes to victim (ego) frame for zone masks
        _ds = self.perception.dataset_name
        bbox_orig_vic = bbox_map_to_sensor(
            bbox_sensor_to_map(bbox_to_remove, atk_pose, dataset_name=_ds), vic_pose, dataset_name=_ds)
        bbox_tgt_vic = bbox_map_to_sensor(
            bbox_sensor_to_map(bbox_to_spoof, atk_pose, dataset_name=_ds), vic_pose, dataset_name=_ds)

        # Get vehicle ordering
        base_data_dict = self.perception.retrieve_base_data(
            multi_vehicle_case, victim_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)

        dev = self.perception.device

        if post_backbone:
            return self._run_post_backbone_attack(
                multi_vehicle_case, attack_opts,
                bbox_orig_vic, bbox_tgt_vic, attacker_index)

        zone_active = self._bbox_to_voxel_mask(bbox_orig_vic) | \
                      self._bbox_to_voxel_mask(bbox_tgt_vic)

        with torch.no_grad():
            # Original features
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_orig, batch_data = self._get_spatial_features(
                multi_vehicle_case, victim_id)

            # Generate spoofed point cloud (removes original + adds shifted car)
            atk_pcd = multi_vehicle_case[attacker_id]["lidar"]
            pcd_spoofed = self._generate_spoof_pcd(
                atk_pcd.copy(), bbox_to_spoof, bbox_to_remove)

            # Compute F_spoof
            case_spoof = copy.deepcopy(multi_vehicle_case)
            case_spoof[attacker_id]["lidar"] = pcd_spoofed
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_spoof_full, _ = self._get_spatial_features(case_spoof, victim_id)

            # Apply beta scaling in active zone
            t_active = torch.from_numpy(zone_active).to(dev)
            features = F_orig.clone()

            if self.pertnet is not None and t_active.any():
                # PertNet: learned per-voxel correction on top of beta scaling
                from mvp.attack.perturbation_network import (
                    build_geometric_encoding, get_active_zone_bounds)
                lr = np.array(self.lidar_range)
                vs = np.array(self.voxel_size)
                H = int((lr[4] - lr[1]) / vs[1])
                W = int((lr[3] - lr[0]) / vs[0])
                bounds = get_active_zone_bounds(
                    bbox_orig_vic, bbox_tgt_vic, lr, vs, H, W, padding=2)
                h_lo, h_hi, w_lo, w_hi = bounds

                if h_hi > h_lo and w_hi > w_lo:
                    vids = list(base_data_dict.keys())
                    ego_index = vids.index(victim_id)
                    vps = [(multi_vehicle_case[v]['lidar_pose'][0] - vic_pose[0],
                            multi_vehicle_case[v]['lidar_pose'][1] - vic_pose[1])
                           for v in vids]

                    F_diff = F_spoof_full[attacker_index] - F_orig[attacker_index]
                    foc = F_orig[attacker_index][:, h_lo:h_hi, w_lo:w_hi]
                    fdc = F_diff[:, h_lo:h_hi, w_lo:w_hi]
                    geo = build_geometric_encoding(
                        bbox_orig_vic, bbox_tgt_vic, vps, ego_index,
                        attacker_index, bounds, lr, vs, H, W,
                        max_vehicles=4).to(dev)

                    with torch.no_grad():
                        delta = self.pertnet(foc, fdc, geo)

                    # Apply: base_perturbation + PertNet correction
                    center = self.perception.point_to_voxel_index(bbox_orig_vic)
                    fs = 15
                    Hf, Wf = F_orig.shape[2], F_orig.shape[3]
                    center[0] = max(fs, min(Wf - fs, center[0]))
                    center[1] = max(fs, min(Hf - fs, center[1]))
                    cy, cx = center[1], center[0]

                    C = F_orig.shape[1]
                    fp = torch.zeros(C, 2*fs, 2*fs, device=dev)
                    hd, wd = delta.shape[1], delta.shape[2]
                    ho = max(0, (2*fs - hd) // 2)
                    wo = max(0, (2*fs - wd) // 2)
                    he = min(2*fs, ho + hd)
                    we = min(2*fs, wo + wd)
                    fp[:, ho:he, wo:we] = delta[:, :he-ho, :we-wo]

                    f_atk_crop = foc + fdc  # F_spoof in active zone
                    bp = torch.zeros_like(fp)
                    bp[:, ho:he, wo:we] = (
                        self.beta * f_atk_crop - foc)[:, :he-ho, :we-wo]
                    eps = self.pertnet_epsilon
                    correction = torch.clamp(fp, -eps, eps)
                    combined = bp + correction

                    features[attacker_index, :, cy-fs:cy+fs, cx-fs:cx+fs] = torch.clamp(
                        F_orig[attacker_index, :, cy-fs:cy+fs, cx-fs:cx+fs] + combined,
                        min=0, max=30)
                else:
                    # Fallback: beta-only if zone too small for PertNet
                    if t_active.any():
                        features[attacker_index][:, t_active] = (
                            self.beta * F_spoof_full[attacker_index][:, t_active])
            elif t_active.any():
                features[attacker_index][:, t_active] = (
                    self.beta * F_spoof_full[attacker_index][:, t_active]
                )

            # Run perception
            pred_bboxes, pred_scores = self._run_with_features(
                batch_data, features)

        return {
            "pred_bboxes": pred_bboxes,
            "pred_scores": pred_scores,
            "spatial_features": features.detach() if features is not None else None,
        }

    def _run_post_backbone_attack(self, multi_vehicle_case, attack_opts,
                                   bbox_orig_vic, bbox_tgt_vic, attacker_index):
        """
        Voxelwise attack on post-backbone features (after backbone+shrink,
        before fusion). For architectures like V2X-Real's BaseBEVBackbone
        where batch normalization absorbs pre-backbone perturbations.
        """
        attacker_id = attack_opts["attacker_vehicle_id"]
        victim_id = attack_opts["victim_vehicle_id"]
        bbox_to_remove = attack_opts["bbox_to_remove"]
        bbox_to_spoof = attack_opts["bbox_to_spoof"]
        dev = self.perception.device

        with torch.no_grad():
            # Get post-backbone features for normal, spoof, and remove cases
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F2d_orig, batch_data = self._get_post_backbone_features(
                multi_vehicle_case, victim_id)

            atk_pcd = multi_vehicle_case[attacker_id]["lidar"]
            pcd_spoofed = self._generate_spoof_pcd(
                atk_pcd.copy(), bbox_to_spoof, bbox_to_remove)
            pcd_removed = self._generate_remove_pcd(
                atk_pcd.copy(), bbox_to_remove)

            case_spoof = copy.deepcopy(multi_vehicle_case)
            case_spoof[attacker_id]["lidar"] = pcd_spoofed
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F2d_spoof, _ = self._get_post_backbone_features(case_spoof, victim_id)

            case_remove = copy.deepcopy(multi_vehicle_case)
            case_remove[attacker_id]["lidar"] = pcd_removed
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F2d_remove, _ = self._get_post_backbone_features(case_remove, victim_id)

            # Build zone masks at post-backbone resolution
            _, C_out, H_out, W_out = F2d_orig.shape
            n_agents = F2d_orig.shape[0] // 1  # total agents stacked
            record_len = batch_data['ego']['record_len']
            n_agents = int(record_len.sum().item())

            mask_orig = self._bbox_to_feature2d_mask(bbox_orig_vic, H_out, W_out)
            mask_tgt = self._bbox_to_feature2d_mask(bbox_tgt_vic, H_out, W_out)
            zone_A = mask_orig & ~mask_tgt
            zone_active = mask_orig | mask_tgt

            # Combine F_attack from spoof and remove
            F2d_attack = F2d_spoof[attacker_index].clone()
            tA = torch.from_numpy(zone_A).to(dev)
            if tA.any():
                F2d_attack[:, tA] = F2d_remove[attacker_index][:, tA]

            t_active = torch.from_numpy(zone_active).to(dev)

            # Apply beta amplification on post-backbone features
            features = F2d_orig.clone()
            if t_active.any():
                if self.gamma > 0:
                    benign_indices = [i for i in range(n_agents)
                                      if i != attacker_index]
                    F_benign_sum = sum(features[i] for i in benign_indices)
                    benign_at_active = F_benign_sum[:, t_active]
                    benign_norm = benign_at_active.norm(
                        dim=0, keepdim=True).clamp(min=1e-8)
                    benign_unit = benign_at_active / benign_norm
                    features[attacker_index][:, t_active] = (
                        self.beta * F2d_attack[:, t_active] +
                        self.gamma * benign_unit
                    )
                else:
                    features[attacker_index][:, t_active] = (
                        self.beta * F2d_attack[:, t_active]
                    )

            # Run fusion + heads
            pred_bboxes, pred_scores = self._run_with_post_backbone_features(
                batch_data, features)

        return {
            "pred_bboxes": pred_bboxes,
            "pred_scores": pred_scores,
            "spatial_features": features.detach() if features is not None else None,
        }
