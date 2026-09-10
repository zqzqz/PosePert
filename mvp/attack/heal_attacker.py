"""
HEAL-compatible attacker: overrides model-specific methods
from LidarShiftVoxelwiseAttacker to work with HEAL's encoder structure.

Key difference from base: HEAL voxelizes each vehicle's PCD in its OWN frame
(proj_first=false), so zone masks must be in the attacker's frame, not ego frame.
"""

import os
import copy
import torch
import numpy as np
from collections import OrderedDict, Counter

from .lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.util import set_seed
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from opencood.utils import box_utils


class HealLidarShiftVoxelwiseAttacker(LidarShiftVoxelwiseAttacker):
    def __init__(self, perception, dataset=None, beta=4.0, gamma=0.0,
                 extension=1.0, multiview=True, bases_dir=None, debug=False):
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
        self.pertnet = None
        self.pertnet_epsilon = 10.0

        if bases_dir is None:
            from mvp.config import data_root
            self.bases_dir = os.path.join(data_root,
                "OPV2V/multi_frame/attack/lidar_shift_bases")
        else:
            self.bases_dir = bases_dir

        self.lidar_range = perception.cav_lidar_range
        self.voxel_size = perception.voxel_size

    def _get_spatial_features(self, multi_vehicle_case, ego_id):
        return self.perception._get_spatial_features(multi_vehicle_case, ego_id)

    def run_multi_vehicle(self, multi_vehicle_case, attack_opts):
        """Attack with zone masks in attacker's own frame (HEAL uses per-vehicle features)."""
        attacker_id = attack_opts["attacker_vehicle_id"]
        victim_id = attack_opts["victim_vehicle_id"]
        bbox_to_remove = attack_opts["bbox_to_remove"]
        bbox_to_spoof = attack_opts["bbox_to_spoof"]

        base_data_dict = self.perception.retrieve_base_data(
            multi_vehicle_case, victim_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        dev = self.perception.device

        # Zone mask in ATTACKER's frame (bboxes are already in attacker frame)
        zone_active = (self._bbox_to_voxel_mask(bbox_to_remove) |
                       self._bbox_to_voxel_mask(bbox_to_spoof))

        with torch.no_grad():
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_orig, batch_data = self._get_spatial_features(
                multi_vehicle_case, victim_id)

            atk_pcd = multi_vehicle_case[attacker_id]["lidar"]
            pcd_spoofed = self._generate_spoof_pcd(
                atk_pcd.copy(), bbox_to_spoof, bbox_to_remove)

            case_spoof = copy.deepcopy(multi_vehicle_case)
            case_spoof[attacker_id]["lidar"] = pcd_spoofed
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_spoof_full, _ = self._get_spatial_features(case_spoof, victim_id)

            t_active = torch.from_numpy(zone_active).to(dev)
            features = [f.clone() for f in F_orig]

            if self.pertnet is not None and t_active.any():
                features = self._apply_pertnet_correction(
                    multi_vehicle_case, attack_opts, F_orig, F_spoof_full,
                    batch_data, attacker_index, t_active)
            elif t_active.any():
                features[attacker_index][:, t_active] = (
                    self.beta * F_spoof_full[attacker_index][:, t_active])

            pred_bboxes, pred_scores = self._run_with_features(
                batch_data, features)

        return {
            "pred_bboxes": pred_bboxes,
            "pred_scores": pred_scores,
            "spatial_features": [f.detach() for f in features] if features is not None else None,
        }

    def _apply_pertnet_correction(self, multi_vehicle_case, attack_opts,
                                  F_orig, F_spoof_full, batch_data,
                                  attacker_index, t_active):
        """Apply PertNet correction on top of beta scaling."""
        from mvp.attack.perturbation_network import (
            build_geometric_encoding, get_active_zone_bounds)

        attacker_id = attack_opts["attacker_vehicle_id"]
        victim_id = attack_opts["victim_vehicle_id"]
        bbox_to_remove = attack_opts["bbox_to_remove"]
        bbox_to_spoof = attack_opts["bbox_to_spoof"]
        dev = self.perception.device

        lr = np.array(self.lidar_range)
        vs = np.array(self.voxel_size)
        H = int((lr[4] - lr[1]) / vs[1])
        W = int((lr[3] - lr[0]) / vs[0])

        # Use bboxes in attacker frame for zone bounds
        bounds = get_active_zone_bounds(
            bbox_to_remove, bbox_to_spoof, lr, vs, H, W, padding=2)
        h_lo, h_hi, w_lo, w_hi = bounds

        features = [f.clone() for f in F_orig]

        if h_hi > h_lo and w_hi > w_lo:
            vic_pose = multi_vehicle_case[victim_id]['lidar_pose']
            base = self.perception.retrieve_base_data(
                multi_vehicle_case, victim_id)
            vids = list(base.keys())
            ego_index = vids.index(victim_id)
            vps = [(multi_vehicle_case[v]['lidar_pose'][0] - vic_pose[0],
                    multi_vehicle_case[v]['lidar_pose'][1] - vic_pose[1])
                   for v in vids]

            F_diff = F_spoof_full[attacker_index] - F_orig[attacker_index]
            foc = F_orig[attacker_index][:, h_lo:h_hi, w_lo:w_hi]
            fdc = F_diff[:, h_lo:h_hi, w_lo:w_hi]
            geo = build_geometric_encoding(
                bbox_to_remove, bbox_to_spoof, vps, ego_index,
                attacker_index, bounds, lr, vs, H, W,
                max_vehicles=4).to(dev)

            with torch.no_grad():
                delta = self.pertnet(foc, fdc, geo)

            center = self.perception.point_to_voxel_index(bbox_to_remove)
            fs = 15
            atk_feat = F_orig[attacker_index]
            Hf, Wf = atk_feat.shape[1], atk_feat.shape[2]
            center[0] = max(fs, min(Wf - fs, center[0]))
            center[1] = max(fs, min(Hf - fs, center[1]))
            cy, cx = center[1], center[0]

            C = atk_feat.shape[0]
            fp = torch.zeros(C, 2*fs, 2*fs, device=dev)
            hd, wd = delta.shape[1], delta.shape[2]
            ho = max(0, (2*fs - hd) // 2)
            wo = max(0, (2*fs - wd) // 2)
            he = min(2*fs, ho + hd)
            we = min(2*fs, wo + wd)
            fp[:, ho:he, wo:we] = delta[:, :he-ho, :we-wo]

            f_atk_crop = foc + fdc
            bp = torch.zeros_like(fp)
            bp[:, ho:he, wo:we] = (
                self.beta * f_atk_crop - foc)[:, :he-ho, :we-wo]
            eps = self.pertnet_epsilon
            correction = torch.clamp(fp, -eps, eps)
            combined = bp + correction

            features[attacker_index][:, cy-fs:cy+fs, cx-fs:cx+fs] = torch.clamp(
                F_orig[attacker_index][:, cy-fs:cy+fs, cx-fs:cx+fs] + combined,
                min=0, max=30)
        else:
            if t_active.any():
                features[attacker_index][:, t_active] = (
                    self.beta * F_spoof_full[attacker_index][:, t_active])

        return features

    def _run_with_features(self, batch_data, spatial_features):
        """Run backbone+fusion+heads on modified spatial features."""
        ego_data = batch_data['ego']
        agent_modality_list = ego_data['agent_modality_list']
        record_len = ego_data['record_len']
        model = self.perception.model

        from opencood.utils.transformation_utils import normalize_pairwise_tfm
        affine_matrix = normalize_pairwise_tfm(
            ego_data['pairwise_t_matrix'],
            model.H, model.W, model.fake_voxel_size)

        with torch.no_grad():
            per_modality_features = {}
            for i, mod_name in enumerate(agent_modality_list):
                if mod_name not in per_modality_features:
                    per_modality_features[mod_name] = []
                per_modality_features[mod_name].append(spatial_features[i])

            modality_feature_dict = {}
            for mod_name, feat_list in per_modality_features.items():
                stacked = torch.stack(feat_list)
                backbone = getattr(model, f"backbone_{mod_name}")
                feat_2d = backbone({"spatial_features": stacked})['spatial_features_2d']
                if self.perception.model_type == 'baseline':
                    feat_2d = getattr(model, f"shrinker_{mod_name}")(feat_2d)
                elif self.perception.model_type == 'pyramid':
                    feat_2d = getattr(model, f"aligner_{mod_name}")(feat_2d)
                modality_feature_dict[mod_name] = feat_2d

            counting = {m: 0 for m in model.modality_name_list}
            heter_features = []
            for mod_name in agent_modality_list:
                idx = counting[mod_name]
                heter_features.append(modality_feature_dict[mod_name][idx])
                counting[mod_name] += 1
            heter_feature_2d = torch.stack(heter_features)

            if self.perception.model_type == 'baseline':
                fused = model.fusion_net(heter_feature_2d, record_len, affine_matrix)
                if hasattr(model, 'shrink_flag') and model.shrink_flag:
                    fused = model.shrink_conv(fused)
            elif self.perception.model_type == 'pyramid':
                fused, _ = model.pyramid_backbone.forward_collab(
                    heter_feature_2d, record_len, affine_matrix,
                    agent_modality_list, model.cam_crop_info)
                if model.shrink_flag:
                    fused = model.shrink_conv(fused)

            psm = model.cls_head(fused)
            rm = model.reg_head(fused)

            output_dict = OrderedDict()
            output_dict['ego'] = {'cls_preds': psm, 'reg_preds': rm}
            if hasattr(model, 'dir_head'):
                output_dict['ego']['dir_preds'] = model.dir_head(fused)

            pred_box_tensor, pred_score = \
                self.perception.post_processor.post_process(batch_data, output_dict)

            if pred_box_tensor is None:
                return np.array([]).reshape(0, 7), np.array([])

            pred_bboxes = pred_box_tensor.cpu().numpy()
            pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
            pred_bboxes[:, 2] -= 0.5 * pred_bboxes[:, 5]
            pred_scores = pred_score.cpu().numpy()
            mask = pred_scores >= 0.1
            return pred_bboxes[mask], pred_scores[mask]

    def _get_post_backbone_features(self, multi_vehicle_case, ego_id):
        batch_data = self.perception._build_batch(multi_vehicle_case, ego_id)
        ego_data = batch_data['ego']
        agent_modality_list = ego_data['agent_modality_list']
        model = self.perception.model

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        with torch.no_grad():
            for mod_name in model.modality_name_list:
                if mod_name not in modality_count_dict:
                    continue
                encoder = getattr(model, f"encoder_{mod_name}")
                raw = encoder(ego_data, mod_name)
                backbone = getattr(model, f"backbone_{mod_name}")
                feat_2d = backbone({"spatial_features": raw})['spatial_features_2d']
                if self.perception.model_type == 'baseline':
                    feat_2d = getattr(model, f"shrinker_{mod_name}")(feat_2d)
                elif self.perception.model_type == 'pyramid':
                    feat_2d = getattr(model, f"aligner_{mod_name}")(feat_2d)
                modality_feature_dict[mod_name] = feat_2d

        counting = {m: 0 for m in model.modality_name_list}
        all_features = []
        for mod_name in agent_modality_list:
            idx = counting[mod_name]
            all_features.append(modality_feature_dict[mod_name][idx])
            counting[mod_name] += 1

        sf2d = torch.stack(all_features)
        return sf2d.clone().detach(), batch_data
