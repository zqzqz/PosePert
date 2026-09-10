"""
HEAL model perception wrapper for heterogeneous collaborative perception attacks.
Supports HeterModelBaseline (lidar_attfuse) and HeterPyramidCollab (HEAL) models.
"""
import os, sys
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import math
import copy

heal_root = os.path.join(os.path.dirname(__file__), "../../third_party/HEAL")
sys.path.insert(0, heal_root)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points
from opencood.utils.transformation_utils import get_pairwise_transformation
from opencood.utils.common_utils import merge_features_to_dict
from opencood.data_utils.pre_processor import build_preprocessor

from .perception import Perception

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from mvp.config import model_root, data_root
from mvp.data.util import pose_to_transformation


def simulate_32line_lidar(pcd, seed=None):
    """Simulate 32-line LiDAR from 64-line by beam-angle subsampling.
    Bins points by elevation angle into 64 beams, keeps every other beam."""
    if pcd.shape[0] == 0:
        return pcd
    r_xy = np.sqrt(pcd[:, 0]**2 + pcd[:, 1]**2)
    elevation = np.arctan2(pcd[:, 2], r_xy)
    elev_min, elev_max = -25.0 * np.pi / 180, 2.0 * np.pi / 180
    beam_idx = np.clip(
        ((elevation - elev_min) / (elev_max - elev_min) * 64).astype(np.int32),
        0, 63)
    keep = (beam_idx % 2 == 0)
    return pcd[keep]


class HealPerception(Perception):
    """Perception wrapper for HEAL heterogeneous models."""

    def __init__(self, model_dir, modality_config=None, device="cuda:0",
                 vehicle_modality_fn=None):
        """
        Args:
            model_dir: path to model directory (e.g., models/HEAL/lidar_attfuse)
            modality_config: dict overriding modality settings in hypes['heter']
            device: cuda device string
            vehicle_modality_fn: callable(vehicle_id, is_ego) -> modality_name.
                If None, all vehicles default to the ego modality.
        """
        super().__init__()
        self.devices = device
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.dataset_name = "OPV2V"
        self.fusion_method = "intermediate"

        class FakeOpt:
            pass
        opt = FakeOpt()
        opt.model_dir = model_dir

        hypes = yaml_utils.load_yaml(None, opt)

        hypes['root_dir'] = os.path.join(data_root, 'OPV2V/train')
        hypes['validate_dir'] = os.path.join(data_root, 'OPV2V/test')
        hypes['test_dir'] = os.path.join(data_root, 'OPV2V/test')

        assign_path = os.path.join(heal_root, 'opencood/modality_assign/opv2v_4modality.json')
        hypes['heter']['assignment_path'] = assign_path

        if modality_config:
            for k, v in modality_config.items():
                hypes['heter'][k] = v

        # Fix SECOND encoder's num_features_in (checkpoint bug: 64 → 4)
        if 'm3' in hypes.get('model', {}).get('args', {}):
            m3_args = hypes['model']['args']['m3'].get('encoder_args', {})
            spconv_cfg = m3_args.get('spconv', {})
            if spconv_cfg.get('num_features_in', 0) != 4:
                spconv_cfg['num_features_in'] = 4

        self.hypes = hypes
        self.ego_modality = hypes['heter'].get('ego_modality', 'm1')
        self.vehicle_modality_fn = vehicle_modality_fn
        self.lidar_channels_dict = hypes['heter'].get('lidar_channels_dict', {})

        self.model = train_utils.create_model(hypes)
        _, self.model = self._load_model_with_spconv_fix(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()

        hypes['validate_dir'] = hypes['test_dir']
        self.dataset = build_dataset(hypes, visualize=True, train=False)

        self._build_preprocessors(hypes)
        self._build_postprocessor(hypes)
        self._detect_model_type()

        self.preprocessors = {
            "intermediate": self._intermediate_preprocess,
        }

    def _load_model_with_spconv_fix(self, model_dir):
        """Load model with spconv 1.x → 2.x weight transpose fix."""
        import glob
        file_list = glob.glob(os.path.join(model_dir, 'net_epoch_bestval_at*.pth'))
        if file_list:
            ckpt_path = file_list[0]
        else:
            import re
            ckpt_files = glob.glob(os.path.join(model_dir, 'net_epoch*.pth'))
            epochs = []
            for f in ckpt_files:
                m = re.search(r'net_epoch(\d+)\.pth', f)
                if m:
                    epochs.append((int(m.group(1)), f))
            if not epochs:
                return 0, self.model
            ckpt_path = max(epochs, key=lambda x: x[0])[1]

        state_dict = torch.load(ckpt_path, map_location='cpu')
        model_state = self.model.state_dict()
        fixed = 0
        for k in list(state_dict.keys()):
            if k not in model_state:
                continue
            v = state_dict[k]
            expected = model_state[k].shape
            if v.shape == expected:
                continue
            if len(v.shape) == 5 and len(expected) == 5:
                # spconv 1.x → 2.x: [kD,kH,kW,in_C,out_C] → [out_C,kD,kH,kW,in_C]
                permuted = v.permute(4, 0, 1, 2, 3)
                if permuted.shape == expected:
                    state_dict[k] = permuted
                    fixed += 1
                elif permuted.shape[0] == expected[0] and permuted.shape[1:4] == expected[1:4]:
                    # Channel count mismatch (e.g., conv_input 64→4): truncate
                    state_dict[k] = permuted[:, :, :, :, :expected[4]]
                    fixed += 1
        if fixed > 0:
            print(f"Fixed {fixed} spconv weight shapes (1.x → 2.x)")
        self.model.load_state_dict(state_dict, strict=False)
        return 0, self.model

    def _build_preprocessors(self, hypes):
        """Build per-modality voxel preprocessors."""
        self.pre_processors = {}
        for mod_name, mod_setting in hypes['heter']['modality_setting'].items():
            if mod_setting['sensor_type'] == 'lidar':
                self.pre_processors[mod_name] = build_preprocessor(
                    mod_setting['preprocess'], train=False)

    def _build_postprocessor(self, hypes):
        """Build anchor box and post processor."""
        from opencood.data_utils.post_processor import build_postprocessor
        self.post_processor = build_postprocessor(hypes['postprocess'], train=False)
        self.anchor_box = self.post_processor.generate_anchor_box()
        self.cav_lidar_range = hypes['preprocess']['cav_lidar_range']
        self.voxel_size = hypes['preprocess']['args']['voxel_size']
        self.max_cav = hypes['train_params'].get('max_cav', 5)

    def _detect_model_type(self):
        cls_name = self.model.__class__.__name__
        if cls_name == 'HeterModelBaseline':
            self.model_type = 'baseline'
        elif cls_name == 'HeterPyramidCollab':
            self.model_type = 'pyramid'
        else:
            self.model_type = 'unknown'

        modality_list = self.model.modality_name_list
        self.has_m1 = 'm1' in modality_list
        self.has_m3 = 'm3' in modality_list
        self.model_name = "pointpillar" if self.has_m1 else "second"

    def run(self, multi_vehicle_case, ego_id):
        """Run detection, return (pred_bboxes, pred_scores) in ego frame."""
        from mvp.util import set_seed
        set_seed(42, set_python=False, set_numpy=False, set_torch=True)

        batch_data = self._build_batch(multi_vehicle_case, ego_id)
        with torch.no_grad():
            model_output = self.model(batch_data['ego'])
            output_dict = OrderedDict()
            output_dict['ego'] = model_output
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(batch_data, output_dict)

        if pred_box_tensor is None:
            return np.array([]).reshape(0, 7), np.array([])

        pred_bboxes = pred_box_tensor.cpu().numpy()
        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
        pred_bboxes[:, 2] -= 0.5 * pred_bboxes[:, 5]
        pred_scores = pred_score.cpu().numpy()
        mask = pred_scores >= 0.1
        return pred_bboxes[mask], pred_scores[mask]

    def run_multi_vehicle(self, multi_vehicle_case, ego_id):
        pred_bboxes, pred_scores = self.run(multi_vehicle_case, ego_id)
        if pred_bboxes.shape[0] == 0:
            multi_vehicle_case[ego_id]["pred_bboxes"] = np.array([])
            multi_vehicle_case[ego_id]["pred_scores"] = np.array([])
        else:
            multi_vehicle_case[ego_id]["pred_bboxes"] = pred_bboxes
            multi_vehicle_case[ego_id]["pred_scores"] = pred_scores
        return multi_vehicle_case

    def _build_batch(self, multi_vehicle_case, ego_id):
        """Build HEAL-format batch_data directly from our case format."""
        ego_pose = multi_vehicle_case[ego_id]["lidar_pose"]

        vehicle_ids = list(multi_vehicle_case.keys())
        if ego_id in vehicle_ids:
            vehicle_ids.remove(ego_id)
            vehicle_ids.insert(0, ego_id)

        base_data_dict = OrderedDict()
        agent_modality_list = []
        per_mod_features = {}

        for vehicle_id in vehicle_ids:
            vdata = multi_vehicle_case[vehicle_id]
            is_ego = (vehicle_id == ego_id)

            if self.vehicle_modality_fn is not None:
                mod_name = self.vehicle_modality_fn(vehicle_id, is_ego)
            else:
                mod_name = self.ego_modality
            agent_modality_list.append(mod_name)

            v_pose = vdata["lidar_pose"]
            t_matrix = np.dot(
                np.linalg.inv(pose_to_transformation(ego_pose)),
                pose_to_transformation(v_pose))

            base_data_dict[vehicle_id] = OrderedDict()
            base_data_dict[vehicle_id]['ego'] = is_ego
            base_data_dict[vehicle_id]['params'] = {
                'lidar_pose': v_pose,
                'lidar_pose_clean': v_pose,
                'transformation_matrix': t_matrix,
            }

            pcd = vdata["lidar"].astype(np.float32)
            if self.lidar_channels_dict.get(mod_name) == 32:
                pcd = simulate_32line_lidar(pcd)
            pcd = shuffle_points(pcd)
            pcd = mask_ego_points(pcd)
            pcd = mask_points_by_range(pcd, self.cav_lidar_range)

            if mod_name in self.pre_processors:
                processed = self.pre_processors[mod_name].preprocess(pcd)
                if mod_name not in per_mod_features:
                    per_mod_features[mod_name] = []
                per_mod_features[mod_name].append(processed)

        pairwise_t_matrix = get_pairwise_transformation(
            base_data_dict, self.max_cav, proj_first=False)

        n_vehicles = len(vehicle_ids)

        ego_data = {
            'agent_modality_list': agent_modality_list,
            'record_len': torch.tensor([n_vehicles], dtype=torch.int32).to(self.device),
            'pairwise_t_matrix': torch.from_numpy(
                pairwise_t_matrix).float().unsqueeze(0).to(self.device),
            'anchor_box': torch.from_numpy(self.anchor_box).float().to(self.device),
            'transformation_matrix': torch.eye(4).float().to(self.device),
        }

        for mod_name, feat_list in per_mod_features.items():
            merged = merge_features_to_dict(feat_list)
            collated = self.pre_processors[mod_name].collate_batch(merged)
            for k, v in collated.items():
                if isinstance(v, torch.Tensor):
                    collated[k] = v.to(self.device)
            ego_data[f'inputs_{mod_name}'] = collated

        gt_boxes = np.zeros((self.post_processor.params['max_num'], 7))
        gt_mask = np.zeros(self.post_processor.params['max_num'])
        if 'gt_bboxes' in multi_vehicle_case[ego_id]:
            gt = np.array(multi_vehicle_case[ego_id]['gt_bboxes'])
            if len(gt) > 0:
                n = min(len(gt), gt_boxes.shape[0])
                gt_boxes[:n] = gt[:n, :7]
                gt_mask[:n] = 1

        label_dict = self.post_processor.generate_label(
            gt_box_center=gt_boxes, anchors=self.anchor_box, mask=gt_mask)
        ego_data['label_dict'] = {
            k: torch.from_numpy(v).float().unsqueeze(0).to(self.device)
            if isinstance(v, np.ndarray) else v
            for k, v in label_dict.items()
        }
        ego_data['object_bbx_center'] = torch.from_numpy(
            gt_boxes).float().unsqueeze(0).to(self.device)
        ego_data['object_bbx_mask'] = torch.from_numpy(
            gt_mask).int().unsqueeze(0).to(self.device)

        batch_data = {'ego': ego_data}
        batch_data['ego']['_base_data_dict'] = base_data_dict
        batch_data['ego']['_vehicle_ids'] = vehicle_ids

        return batch_data

    def _intermediate_preprocess(self, multi_vehicle_case, ego_id):
        """For interface compat — returns the batch_data directly."""
        return self._build_batch(multi_vehicle_case, ego_id)

    def retrieve_base_data(self, multi_vehicle_case, ego_id):
        """Convert our case format to a base_data_dict ordered by ego first."""
        ego_pose = multi_vehicle_case[ego_id]["lidar_pose"]

        vehicle_ids = list(multi_vehicle_case.keys())
        if ego_id in vehicle_ids:
            vehicle_ids.remove(ego_id)
            vehicle_ids.insert(0, ego_id)

        data = OrderedDict()
        for vehicle_id in vehicle_ids:
            vehicle_data = multi_vehicle_case[vehicle_id]
            data[vehicle_id] = OrderedDict()
            data[vehicle_id]['ego'] = (vehicle_id == ego_id)
            data[vehicle_id]['cav_id'] = vehicle_id
            t_matrix = np.dot(
                np.linalg.inv(pose_to_transformation(ego_pose)),
                pose_to_transformation(vehicle_data["lidar_pose"]))
            if "params" in vehicle_data:
                data[vehicle_id]['params'] = copy.deepcopy(vehicle_data["params"])
                data[vehicle_id]['params']["lidar_pose"] = vehicle_data["lidar_pose"]
                data[vehicle_id]['params']["transformation_matrix"] = t_matrix
            else:
                data[vehicle_id]['params'] = {
                    "lidar_pose": vehicle_data["lidar_pose"],
                    "transformation_matrix": t_matrix,
                    "spatial_correction_matrix": np.eye(4),
                    "vehicles": {},
                }
            data[vehicle_id]['lidar_np'] = vehicle_data["lidar"].astype(np.float32)

            is_ego = (vehicle_id == ego_id)
            if self.vehicle_modality_fn is not None:
                mod = self.vehicle_modality_fn(vehicle_id, is_ego)
            else:
                mod = self.ego_modality
            data[vehicle_id]['modality_name'] = mod
        return data

    def attack_intermediate_forward(self, batch_data, attacker_index,
                                     perturbation=None, feature=None,
                                     max_perturb=1, center=None,
                                     feature_size=10, perturb_func=None):
        """Forward pass with perturbation at the spatial features level."""
        ego_data = batch_data['ego']
        agent_modality_list = ego_data['agent_modality_list']
        record_len = ego_data['record_len']

        from opencood.utils.transformation_utils import normalize_pairwise_tfm
        affine_matrix = normalize_pairwise_tfm(
            ego_data['pairwise_t_matrix'],
            self.model.H, self.model.W, self.model.fake_voxel_size)

        from collections import Counter
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.model.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            encoder = getattr(self.model, f"encoder_{modality_name}")
            feature_raw = encoder(ego_data, modality_name)
            backbone = getattr(self.model, f"backbone_{modality_name}")
            feature_2d = backbone({"spatial_features": feature_raw})['spatial_features_2d']

            if self.model_type == 'baseline':
                shrinker = getattr(self.model, f"shrinker_{modality_name}")
                feature_2d = shrinker(feature_2d)
            elif self.model_type == 'pyramid':
                aligner = getattr(self.model, f"aligner_{modality_name}")
                feature_2d = aligner(feature_2d)

            modality_feature_dict[modality_name] = feature_2d

        counting_dict = {m: 0 for m in self.model.modality_name_list}
        heter_feature_list = []
        for mod_name in agent_modality_list:
            idx = counting_dict[mod_name]
            heter_feature_list.append(modality_feature_dict[mod_name][idx])
            counting_dict[mod_name] += 1

        spatial_features = torch.stack(heter_feature_list)

        if perturb_func is not None:
            x = torch.clone(spatial_features).detach()
            spatial_features[attacker_index] = perturb_func(
                x[attacker_index].unsqueeze(0))[0]
        elif perturbation is not None:
            clipped = torch.clip(perturbation, min=-max_perturb, max=max_perturb)
            feature_map = torch.clone(spatial_features[attacker_index]).detach()
            aligned_center = center.astype(np.int32)
            C, H, W = feature_map.size()

            perturbation_features = torch.zeros_like(
                spatial_features[attacker_index]).to(self.device)
            perturbation_features[
                :,
                aligned_center[1]-feature_size:aligned_center[1]+feature_size,
                aligned_center[0]-feature_size:aligned_center[0]+feature_size
            ] = clipped

            theta = torch.tensor([[[1, 0, (center[1] - aligned_center[1]) * 2 / W],
                                   [0, 1, (center[0] - aligned_center[0]) * 2 / H]]],
                                 dtype=torch.float).to(self.device)
            grid = F.affine_grid(theta, (1, C, H, W))
            perturbation_features = F.grid_sample(
                perturbation_features.unsqueeze(0), grid)[0]
            spatial_features[attacker_index] = feature_map + perturbation_features
        elif feature is not None:
            aligned_center = center.astype(np.int32) if isinstance(center, np.ndarray) else center
            spatial_features[attacker_index][
                :,
                aligned_center[1]-feature_size:aligned_center[1]+feature_size,
                aligned_center[0]-feature_size:aligned_center[0]+feature_size
            ] = feature

        if self.model_type == 'baseline':
            fused_feature = self.model.fusion_net(
                spatial_features, record_len, affine_matrix)
            if hasattr(self.model, 'shrink_flag') and self.model.shrink_flag:
                fused_feature = self.model.shrink_conv(fused_feature)
        elif self.model_type == 'pyramid':
            fused_feature, _ = self.model.pyramid_backbone.forward_collab(
                spatial_features, record_len, affine_matrix,
                agent_modality_list, self.model.cam_crop_info)
            if self.model.shrink_flag:
                fused_feature = self.model.shrink_conv(fused_feature)

        psm = self.model.cls_head(fused_feature)
        rm = self.model.reg_head(fused_feature)

        output_dict = OrderedDict()
        output_dict['ego'] = {'psm': psm, 'rm': rm}

        return output_dict, perturbation, spatial_features

    def _get_spatial_features(self, multi_vehicle_case, ego_id):
        """Get VFE spatial features for all vehicles (before backbone)."""
        batch_data = self._build_batch(multi_vehicle_case, ego_id)
        ego_data = batch_data['ego']
        agent_modality_list = ego_data['agent_modality_list']

        from collections import Counter
        modality_count_dict = Counter(agent_modality_list)

        modality_features = {}
        for modality_name in self.model.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            encoder = getattr(self.model, f"encoder_{modality_name}")
            with torch.no_grad():
                feats = encoder(ego_data, modality_name)
            modality_features[modality_name] = feats

        counting = {m: 0 for m in self.model.modality_name_list}
        all_features = []
        for mod_name in agent_modality_list:
            idx = counting[mod_name]
            all_features.append(modality_features[mod_name][idx].clone().detach())
            counting[mod_name] += 1

        return all_features, batch_data

    def point_to_voxel_index(self, point, standard=True):
        lr = self.cav_lidar_range
        vs = self.voxel_size
        if standard:
            return ((point[:3] - np.floor(lr[:3])) / vs).astype(np.int32)
        else:
            return (point[:3] - np.floor(lr[:3])) / vs

    def forward_from_features(self, spatial_features, batch_data):
        """Run backbone+shrinker/aligner+fusion+heads from pre-backbone features.
        Returns raw model output dict (psm, rm, dm).
        Differentiable — no torch.no_grad wrapper."""
        ego_data = batch_data['ego']
        agent_modality_list = ego_data['agent_modality_list']
        record_len = ego_data['record_len']
        model = self.model

        from opencood.utils.transformation_utils import normalize_pairwise_tfm
        affine_matrix = normalize_pairwise_tfm(
            ego_data['pairwise_t_matrix'],
            model.H, model.W, model.fake_voxel_size)

        from collections import Counter
        counting = {m: 0 for m in model.modality_name_list}
        per_mod = {}
        for i, mod_name in enumerate(agent_modality_list):
            if mod_name not in per_mod:
                per_mod[mod_name] = []
            per_mod[mod_name].append(spatial_features[i])
            counting[mod_name] += 1

        modality_feature_dict = {}
        for mod_name, feat_list in per_mod.items():
            stacked = torch.stack(feat_list)
            backbone = getattr(model, f"backbone_{mod_name}")
            feat_2d = backbone({"spatial_features": stacked})['spatial_features_2d']
            if self.model_type == 'baseline':
                feat_2d = getattr(model, f"shrinker_{mod_name}")(feat_2d)
            elif self.model_type == 'pyramid':
                feat_2d = getattr(model, f"aligner_{mod_name}")(feat_2d)
            modality_feature_dict[mod_name] = feat_2d

        counting2 = {m: 0 for m in model.modality_name_list}
        heter_features = []
        for mod_name in agent_modality_list:
            idx = counting2[mod_name]
            heter_features.append(modality_feature_dict[mod_name][idx])
            counting2[mod_name] += 1
        heter_feature_2d = torch.stack(heter_features)

        if self.model_type == 'baseline':
            fused = model.fusion_net(heter_feature_2d, record_len, affine_matrix)
            if hasattr(model, 'shrink_flag') and model.shrink_flag:
                fused = model.shrink_conv(fused)
        elif self.model_type == 'pyramid':
            fused, _ = model.pyramid_backbone.forward_collab(
                heter_feature_2d, record_len, affine_matrix,
                agent_modality_list, model.cam_crop_info)
            if model.shrink_flag:
                fused = model.shrink_conv(fused)

        psm = model.cls_head(fused)
        rm = model.reg_head(fused)
        output = {'psm': psm, 'rm': rm}
        if hasattr(model, 'dir_head'):
            output['dm'] = model.dir_head(fused)
        return output

    def iou_torch(self, bboxes_a, bboxes_b):
        """Differentiable 3D IoU using oriented BEV intersection."""
        from opencood.utils.box_utils import boxes_to_corners2d
        from mvp.perception.iou_util import oriented_box_intersection_2d
        corners2d_a = torch.unsqueeze(boxes_to_corners2d(bboxes_a, order="lwh")[:,:,:2], 0)
        corners2d_b = torch.unsqueeze(boxes_to_corners2d(bboxes_b, order="lwh")[:,:,:2], 0)
        area_a = bboxes_a[:, 3] * bboxes_a[:, 4]
        area_b = bboxes_b[:, 3] * bboxes_b[:, 4]
        area_inter, _ = oriented_box_intersection_2d(corners2d_a, corners2d_b)
        area_inter = area_inter.squeeze()
        height_inter = torch.clip(
            torch.min(bboxes_a[:, 2] + 0.5 * bboxes_a[:, 5], bboxes_b[:, 2] + 0.5 * bboxes_b[:, 5]) - \
            torch.max(bboxes_a[:, 2] - 0.5 * bboxes_a[:, 5], bboxes_b[:, 2] - 0.5 * bboxes_b[:, 5]),
            min=0, max=5)
        iou = area_inter * height_inter / (
            area_a * bboxes_a[:, 5] + area_b * bboxes_b[:, 5] - area_inter * height_inter)
        return iou

    def detach_all(self, x):
        if isinstance(x, dict):
            return {k: self.detach_all(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self.detach_all(v) for v in x]
        elif isinstance(x, torch.Tensor):
            return x.detach()
        return x
