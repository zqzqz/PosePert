from concurrent.futures import process
import os, sys
from mvp.config import third_party_root, opencood_root
import numpy as np
from collections import OrderedDict
import torch
import math
import copy
import random
import logging
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .perception import Perception
from mvp.config import model_root, data_root, tmp_root
from mvp.tools.iou import iou3d
from mvp.evaluate.detection import iou3d_batch
from .iou_util import oriented_box_intersection_2d
from mvp.data.util import pcd_sensor_to_map, pcd_map_to_sensor, pose_to_transformation
from mvp.attack.attack_utils import get_adv_loss
from mvp.util import set_seed

sys.path.insert(0, opencood_root)
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import torch_tensor_to_numpy

class OpencoodPerception(Perception):
    def __init__(self, fusion_method="early", model_name="pointpillar", dataset_name=None):
        super().__init__()


        assert(model_name in ["pixor", "voxelnet", "second", "pointpillar", "v2vnet", "fpvrcnn"])
        assert(fusion_method in ["early", "intermediate", "late"])
        self.name = "{}_{}".format(model_name, fusion_method)
        self.devices = "cuda:0"
        self.model_name = model_name
        self.fusion_method = fusion_method
        if self.model_name == "v2vnet":
            self.model_dir = os.path.join(model_root, "OpenCOOD/v2vnet")
            self.fusion_method = "intermediate"
        else:
            self.model_dir = os.path.join(model_root, "OpenCOOD/{}_{}_fusion".format(self.model_name, self.fusion_method if self.fusion_method != "intermediate" else "attentive"))
        # Full dataset_name for model path (e.g., "V2X-Real-V2V" -> model dir suffix "v2xrealv2v")
        # Base dataset_name for feature/preprocessing checks (e.g., "V2X-Real-V2V" -> "V2X-Real")
        self.dataset_name_full = dataset_name
        self.dataset_name = dataset_name.split("-V2V")[0].split("-V2I")[0] if dataset_name else dataset_name
        if dataset_name is not None and dataset_name != "OPV2V":
            self.model_dir = self.model_dir + "_" + dataset_name.lower().replace('-', '').replace('_', '')
        self.config_file = os.path.join(self.model_dir, "config.yaml")
        self.preprocessors = {
            "early": self.early_preprocess,
            "intermediate": self.intermediate_preprocess,
            "late": self.late_preprocess,
        }
        self.inference_processors = {
            "early": inference_utils.inference_early_fusion,
            "intermediate": inference_utils.inference_intermediate_fusion,
            "late": inference_utils.inference_late_fusion,
        }

        hypes = yaml_utils.load_yaml(self.config_file, None)
        data_dataset_name = dataset_name.split("-V2V")[0].split("-V2I")[0] if dataset_name else dataset_name
        data_base = os.path.join(data_root, data_dataset_name)
        hypes["root_dir"] = os.path.join(data_base, "train")
        for vdir in ["validate", "val", "test"]:
            vpath = os.path.join(data_base, vdir)
            if os.path.isdir(vpath):
                hypes["validate_dir"] = vpath
                break
        else:
            hypes["validate_dir"] = os.path.join(data_base, "validate")
        self.dataset = build_dataset(hypes, visualize=False, train=False)
        self.model = train_utils.create_model(hypes)
        # we assume gpu is available
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = train_utils.load_saved_model(self.model_dir, self.model)
        self.model = ret[1]
        self.model.eval()

        # For intermediate attack
        if fusion_method == "intermediate":
            if self.dataset_name == "V2X-Real":
                fusion_module = self.model.fusion_net
            elif hasattr(self.model, 'fusion_net'):
                fusion_module = self.model.fusion_net if not isinstance(self.model.fusion_net, torch.nn.ModuleList) else self.model.fusion_net[0]
            elif hasattr(self.model.backbone, 'fuse_modules'):
                fusion_module = self.model.backbone.fuse_modules[0]
            else:
                fusion_module = None
            if fusion_module is not None:
                cls_name = fusion_module.__class__.__name__
                try:
                    self.attn_loss_fn = get_adv_loss(cls_name)
                except (AssertionError, KeyError):
                    self.attn_loss_fn = None
            else:
                self.attn_loss_fn = None

    def run(self, multi_vehicle_case, ego_id):
        set_seed(42, set_python=False, set_numpy=False, set_torch=True)
        batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        batch_data = self.dataset.collate_batch_test([batch])
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self.device)
            if self.dataset_name in ["OPV2V", "V2X4Real"]:
                pred_box_tensor, pred_score, gt_box_tensor = \
                    self.inference_processors[self.fusion_method](batch_data,
                                                                self.model,
                                                                self.dataset)
            else:
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    self.inference_processors[self.fusion_method](batch_data,
                                                                self.model,
                                                                self.dataset)
        pred_bboxes = pred_box_tensor.cpu().numpy()
        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
        pred_scores = pred_score.cpu().numpy()
        if pred_scores.ndim == 2 and pred_scores.shape[1] == 2:
            pred_scores, pred_classes = pred_scores[:, 0], pred_scores[:, 1].astype(np.int)
            mask = np.logical_and(pred_scores >= 0.1, pred_classes == 1)
        else:
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

    def attack_early(self, multi_vehicle_case, ego_id, attacker_id, bbox=None, bbox2=None, mode="spoof"):
        batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        batch_data = self.dataset.collate_batch_test([batch])
        if bbox is not None:
            bbox = np.copy(bbox)
            bbox[3:6] = bbox[[5,4,3]]
            bbox[2] += 0.5 * bbox[3]
            bbox = torch.from_numpy(bbox).type(torch.float32).to(self.device)
        if bbox2 is not None:
            bbox2 = np.copy(bbox2)
            bbox2[3:6] = bbox2[[5,4,3]]
            bbox2[2] += 0.5 * bbox2[3]
            bbox2 = torch.from_numpy(bbox2).type(torch.float32).to(self.device)

        with torch.no_grad():
            data_dict = train_utils.to_device(batch_data, self.device)
            output_dict = OrderedDict()
            for cav_id, cav_content in data_dict.items():
                output_dict[cav_id] = self.model(cav_content)

            pred_box_tensor, pred_score, gt_box_tensor = \
                self.dataset.post_process(data_dict,
                                          output_dict)

            anchor_box = data_dict['ego']['anchor_box']
            prob = F.sigmoid(output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1).detach()
            proposals = self.dataset.post_processor.delta_to_boxes3d(output_dict['ego']['rm'], anchor_box)[0].detach()

        pred_box = pred_box_tensor.cpu().numpy()
        pred_box = box_utils.corner_to_center(pred_box, order="lwh")
        pred_box[:,2] -= 0.5 * pred_box[:,5]
        return {
            "pred_bboxes": pred_box,
            "pred_scores": pred_score.cpu().numpy(),
            "prob": prob.cpu().detach().numpy(),
            "proposals": proposals[:,[0,1,2,5,4,3,6]].cpu().detach().numpy(),
        }

    def attack_late(self, multi_vehicle_case, ego_id, attacker_id, bbox=None, bbox2=None, mode="spoof"):
        batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        batch_data = self.dataset.collate_batch_test([batch])
        if bbox is not None:
            bbox = np.copy(bbox)
            bbox[3:6] = bbox[[5,4,3]]
            bbox[2] += 0.5 * bbox[3]
            bbox = torch.from_numpy(bbox).type(torch.float32).to(self.device)
        if bbox2 is not None:
            bbox2 = np.copy(bbox2)
            bbox2[3:6] = bbox2[[5,4,3]]
            bbox2[2] += 0.5 * bbox2[3]
            bbox2 = torch.from_numpy(bbox2).type(torch.float32).to(self.device)

        with torch.no_grad():
            data_dict = train_utils.to_device(batch_data, self.device)
            output_dict = OrderedDict()
            for cav_id, cav_content in data_dict.items():
                output_dict[cav_id] = self.model(cav_content)

            # the final bounding box list
            pred_box3d_list = []
            pred_box2d_list = []

            for cav_id, cav_content in data_dict.items():
                transformation_matrix = cav_content['transformation_matrix']
                anchor_box = cav_content['anchor_box']
                prob = output_dict[cav_id]['psm']
                prob = F.sigmoid(prob.permute(0, 2, 3, 1))
                prob = prob.reshape(1, -1)
                reg = output_dict[cav_id]['rm']
                batch_box3d = self.dataset.post_processor.delta_to_boxes3d(reg, anchor_box)
                mask = \
                    torch.gt(prob, self.dataset.post_processor.params['target_args']['score_threshold'])
                mask = mask.view(1, -1)
                mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

                boxes3d = torch.masked_select(batch_box3d[0],
                                            mask_reg[0]).view(-1, 7)
                scores = torch.masked_select(prob[0], mask[0])

                # convert output to bounding box
                if len(boxes3d) != 0:
                    if cav_id == attacker_id:
                        if mode == "spoof":
                            boxes3d = torch.vstack([boxes3d, torch.reshape(bbox, (1, 7))])
                            scores = torch.hstack([scores, torch.tensor([1.0]).type(scores.dtype).to(self.device)])
                        elif mode == "remove":
                            keep_index = torch.sum((boxes3d[:, :2] - bbox[:2]) ** 2, dim=1) > 4
                            boxes3d = boxes3d[keep_index]
                            scores = scores[keep_index]
                        elif mode == "shift":
                            keep_index = torch.sum((boxes3d[:, :2] - bbox[:2]) ** 2, dim=1) > 4
                            boxes3d = boxes3d[keep_index]
                            scores = scores[keep_index]
                            boxes3d = torch.vstack([boxes3d, torch.reshape(bbox2, (1, 7))])
                            scores = torch.hstack([scores, torch.tensor([1.0]).type(scores.dtype).to(self.device)])

                    # (N, 8, 3)
                    boxes3d_corner = \
                        box_utils.boxes_to_corners_3d(boxes3d,
                                                    order=self.dataset.post_processor.params['order'])
                    # (N, 8, 3)
                    projected_boxes3d = \
                        box_utils.project_box3d(boxes3d_corner,
                                                transformation_matrix)
                    # convert 3d bbx to 2d, (N,4)
                    projected_boxes2d = \
                        box_utils.corner_to_standup_box_torch(projected_boxes3d)
                    # (N, 5)
                    boxes2d_score = \
                        torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                    pred_box2d_list.append(boxes2d_score)
                    pred_box3d_list.append(projected_boxes3d)

            if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
                raise Exception("no detection result")
            # shape: (N, 5)
            pred_box2d_list = torch.vstack(pred_box2d_list)
            # scores
            scores = pred_box2d_list[:, -1]
            # predicted 3d bbx
            pred_box3d_tensor = torch.vstack(pred_box3d_list)
            # remove large bbx
            keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
            keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
            keep_index = torch.logical_and(keep_index_1, keep_index_2)
            pred_box3d_tensor = pred_box3d_tensor[keep_index]
            scores = scores[keep_index]

            # nms
            keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                            scores,
                                            self.dataset.post_processor.params['nms_thresh']
                                            )
            pred_box3d_tensor = pred_box3d_tensor[keep_index]

            # select cooresponding score
            scores = scores[keep_index]

            # filter out the prediction out of the range.
            mask = \
                box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
            pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
            scores = scores[mask]
            assert scores.shape[0] == pred_box3d_tensor.shape[0]

        pred_box = pred_box3d_tensor.cpu().numpy()
        pred_box = box_utils.corner_to_center(pred_box, order="lwh")
        pred_box[:,2] -= 0.5 * pred_box[:,5]
        return {
            "pred_bboxes": pred_box,
            "pred_scores": scores.cpu().numpy()
        }

    def attack_intermediate_forward(self, batch_data, attacker_index, perturbation=None, feature=None, max_perturb=1, center=None, feature_size=10, perturb_func=None):
        if perturbation is not None:
            clipped_perturbation = torch.clip(perturbation, min=-max_perturb, max=max_perturb)
        else:
            clipped_perturbation = None

        voxel_features = batch_data['ego']['processed_lidar']['voxel_features']
        voxel_coords = batch_data['ego']['processed_lidar']['voxel_coords']
        voxel_num_points = batch_data['ego']['processed_lidar']['voxel_num_points']
        record_len = batch_data['ego']['record_len']

        pairwise_t_matrix = batch_data['ego']['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points,
                    'record_len': record_len}

        if self.model_name == "v2vnet":
            batch_dict['voxel_features'] = batch_dict['voxel_features'].float()
        
        if self.model_name in ["pointpillar", "v2vnet"]:
            # n, 4 -> n, c
            self.model.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            self.model.scatter(batch_dict)

            spatial_features = batch_dict['spatial_features']
        elif self.model_name == "voxelnet":
            if voxel_coords.is_cuda:
                record_len_tmp = record_len.cpu()

            record_len_tmp = list(record_len_tmp.numpy())

            self.model.N = sum(record_len_tmp)

            # feature learning network
            vwfs = self.model.svfe(batch_dict)['pillar_features']

            voxel_coords = torch_tensor_to_numpy(voxel_coords)
            vwfs = self.model.voxel_indexing(vwfs, voxel_coords)

            # convolutional middle network
            vwfs = self.model.cml(vwfs)
            # convert from 3d to 2d N C H W
            vmfs = vwfs.view(self.model.N, -1, self.model.H, self.model.W)

            # compression layer
            if self.model.compression:
                vmfs = self.model.compression_layer(vmfs)
            
            spatial_features = vmfs
        else:
            raise NotImplementedError()

        if perturb_func is not None:
            x = torch.clone(spatial_features).detach()
            spatial_features[attacker_index] = perturb_func(x[attacker_index].unsqueeze(0))[0]
        elif feature is not None:
            # Or directly set the feature.
            spatial_features[attacker_index][:, center[1]-feature_size:center[1]+feature_size, center[0]-feature_size:center[0]+feature_size] = feature
            clipped_perturbation = None
        elif perturbation is not None:
            # Appends the perturbation.
            feature_map = torch.clone(spatial_features[attacker_index]).detach()
            # Interpolation of center indices
            aligned_center = center.astype(np.int32)
            C, H, W = feature_map.size()

            perturbation_features = torch.zeros_like(spatial_features[attacker_index]).to(self.device)
            perturbation_features[:, aligned_center[1]-feature_size:aligned_center[1]+feature_size,
                                    aligned_center[0]-feature_size:aligned_center[0]+feature_size] = clipped_perturbation
            theta = torch.tensor([[[1, 0, (center[1] - aligned_center[1]) * 2 / W],
                                [0, 1, (center[0] - aligned_center[0]) * 2 / H]]], dtype=torch.float).repeat(1, 1, 1).to(self.device)
            grid = torch.nn.functional.affine_grid(theta, (1, C, H, W))
            perturbation_features = torch.nn.functional.grid_sample(perturbation_features.unsqueeze(0), grid)[0]
            spatial_features[attacker_index] = feature_map[attacker_index] + perturbation_features

        if self.model_name in ["pointpillar", "v2vnet"]:
            batch_dict["spatial_features"] = spatial_features
            self.model.backbone(batch_dict)
            spatial_features_2d = batch_dict['spatial_features_2d']

            # Check if model has post-backbone fusion (v2vnet, cobevt, coalign, etc.)
            has_fusion = hasattr(self.model, 'fusion_net')

            if has_fusion:
                # downsample feature to reduce memory
                if hasattr(self.model, 'shrink_flag') and self.model.shrink_flag:
                    spatial_features_2d = self.model.shrink_conv(spatial_features_2d)
                # compressor
                if hasattr(self.model, 'compression') and self.model.compression:
                    spatial_features_2d = self.model.naive_compressor(spatial_features_2d)

                # Route by fusion module type
                fusion_cls = self.model.fusion_net.__class__.__name__
                if fusion_cls == 'SwapFusionEncoder':
                    # CoBEVT: needs regroup + com_mask
                    from opencood.models.fuse_modules.fuse_utils import regroup
                    from einops import repeat
                    regroup_feature, mask = regroup(spatial_features_2d,
                                                     record_len,
                                                     self.model.max_cav)
                    com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    com_mask = repeat(com_mask,
                                      'b h w c l -> b (h new_h) (w new_w) c l',
                                      new_h=regroup_feature.shape[3],
                                      new_w=regroup_feature.shape[4])
                    fused_feature = self.model.fusion_net(regroup_feature, com_mask)
                else:
                    # V2VNet and others: fusion_net takes features + record_len + pairwise_t_matrix
                    fused_feature = self.model.fusion_net(spatial_features_2d,
                                                    record_len,
                                                    pairwise_t_matrix)
                psm = self.model.cls_head(fused_feature)
                rm = self.model.reg_head(fused_feature)
            else:
                # PointPillar with AttBEVBackbone: fusion inside backbone
                psm = self.model.cls_head(spatial_features_2d)
                rm = self.model.reg_head(spatial_features_2d)

        elif self.model_name == "voxelnet":
            # information naive fusion
            vmfs_fusion = self.model.fusion_net(spatial_features, record_len)
            # map and regression map
            psm, rm = self.model.rpn(vmfs_fusion)
        else:
            raise NotImplementedError()

        output_dict = OrderedDict()
        output_dict['ego'] = {'psm': psm,
                            'rm': rm}

        return output_dict, clipped_perturbation, spatial_features

    def attack_intermediate_get_loss(self, case, ego_id, attacker_id, perturbation, max_perturb=1, bbox=None, bbox2=None, mode="spoof", feature_size=10, bbox_weight=1.0, attn_weight=0.0, const_weight=0.0):
        base_data_dict = self.retrieve_base_data(case, ego_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        assert(attacker_index >= 0)
        score_threshold = self.dataset.post_processor.params['target_args']['score_threshold']
        cav_lidar_range = self.dataset.params['preprocess']['cav_lidar_range']

        optimize_batch = self.preprocessors[self.fusion_method](case, ego_id)
        optimize_batch_data = train_utils.to_device(self.dataset.collate_batch_test([optimize_batch]), self.device)
        anchor_box = optimize_batch_data['ego']['anchor_box']

        with torch.no_grad():
            optimize_output_dict, _, optimize_feature = self.attack_intermediate_forward(optimize_batch_data, attacker_index)
            original_prob = F.sigmoid(optimize_output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1).detach()
            original_proposals = self.dataset.post_processor.delta_to_boxes3d(optimize_output_dict['ego']['rm'], anchor_box)[0].detach()

        if bbox is not None:
            ego_bbox_tensor = torch.from_numpy(bbox).to(self.device).type(torch.float32)
            bbox_tensor = torch.from_numpy(bbox).to(self.device).type(torch.float32)
            bbox_tensor[2] += 0.5 * bbox_tensor[5]
            center = self.point_to_voxel_index(bbox, standard=False)

        if bbox2 is not None:
            bbox2_tensor = torch.from_numpy(bbox2).to(self.device).type(torch.float32)
            bbox2_tensor[2] += 0.5 * bbox2_tensor[5]

        batch_data = self.detach_all(optimize_batch_data)
        output_dict, clipped_perturbation, spatial_features = self.attack_intermediate_forward(batch_data, attacker_index, perturbation=perturbation, max_perturb=max_perturb, center=center, feature_size=feature_size)
        prob = F.sigmoid(output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
        proposals = self.dataset.post_processor.delta_to_boxes3d(output_dict['ego']['rm'], anchor_box)[0]

        original_iou = torch.clip(self.iou_torch(
            original_proposals[:,[0,1,2,5,4,3,6]], 
            bbox_tensor.tile((original_proposals.shape[0],1))
        ), min=0, max=1)
        original_bbox_mask = (original_iou >= 0.01)

        if bbox is not None:
            iou = torch.clip(self.iou_torch(
                proposals[:,[0,1,2,5,4,3,6]], 
                bbox_tensor.tile((proposals.shape[0],1))
            ), min=0, max=1)
        else:
            iou = torch.ones(proposals.shape[0], dtype=torch.bool).to(self.device).detach()
        bbox_mask = (iou >= 0.01)

        if bbox2 is not None:
            iou2 = torch.clip(self.iou_torch(
                proposals[:,[0,1,2,5,4,3,6]], 
                bbox2_tensor.tile((proposals.shape[0],1))
            ), min=0, max=1)
        else:
            iou2 = torch.ones(proposals.shape[0], dtype=torch.bool).to(self.device).detach()
        box2_mask = (iou2 >= 0.01)

        prob_thres = 0.1
        prob_mask = (prob >= prob_thres)
        consistency_penalty = 0

        if bbox_weight > 0:
            if mode == "spoof":
                bbox_loss = (1 * iou[bbox_mask] * torch.log(1 - prob[bbox_mask])).sum() + consistency_penalty
            elif mode == "remove":
                bbox_loss = (-1 * iou[bbox_mask] * torch.log(1 - prob[bbox_mask])).sum() + consistency_penalty
            elif mode == "shift":
                mask_pre = original_bbox_mask
                while mask_pre.sum() == 0 and prob_thres > 0:
                    prob_thres -= 0.02
                    prob_mask = (prob >= prob_thres)
                    mask_pre = torch.logical_and(torch.logical_and(bbox_mask, box2_mask), prob_mask)
                if mask_pre.sum() == 0:
                    bbox_loss = 0xffffffff
                else:
                    mask_pre_indices = torch.nonzero(mask_pre).squeeze()
                    if mask_pre_indices.dim() == 0:
                        mask_pre_indices = mask_pre_indices.unsqueeze(0)
                    sort_data = prob.detach()
                    _, mask_indices = torch.topk(sort_data[mask_pre], k=min(mask_pre.sum().item(), 3))
                    mask = torch.zeros_like(prob_mask).to(self.device).detach()
                    mask[mask_pre_indices[mask_indices]] = 1
                    bbox_loss = (1 * torch.log(1 - iou2[mask]) + 0.5 * torch.clip(original_prob[mask] - prob[mask], 0, 1)).sum() / mask.sum()
            else:
                raise NotImplementedError("Attack mode not supported.")
        else:
            bbox_loss = 0

        if attn_weight > 0:
            attn_loss = self.attn_loss_fn(model=self.model, spatial_features=spatial_features, attacker_index=attacker_index, center=center, feature_size=feature_size, device=self.device)
        else:
            attn_loss = 0

        if const_weight > 0:
            mask = torch.norm(original_proposals[:, :2] - bbox_tensor[:2], dim=1) > 0.5
            const_loss = 1 * torch.clip(original_prob[mask] - prob[mask], 0, 1).mean() + \
                1 * torch.clip(torch.absolute(original_proposals[mask,:3] - proposals[mask,:3]), 0, 2).sum(dim=1).mean() + \
                1 * torch.clip(torch.absolute(original_proposals[mask,3:6] - proposals[mask,3:6]), 0, 1).sum(dim=1).mean() + \
                1 * torch.clip(torch.absolute(original_proposals[mask,6] - proposals[mask,6]), 0, 1.6).mean()
        else:
            const_loss = 0

        total_loss = bbox_weight * bbox_loss + attn_weight * attn_loss + const_weight * const_loss

        return total_loss

    def attack_intermediate_ensemble_hybrid_locations(self, cases, ego_id, attacker_id, max_perturb=1, max_iteration=25, bboxes=None, bboxes2=None, mode="spoof", feature_size=10, real_case=None, real_bbox=None, bbox_weight=1.0, attn_weight=1.0, const_weight=1.0, perturbation_configs={}):
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

        case, bbox, bbox2 = cases[-1], bboxes[-1], bboxes2[-1]

        # Sets up perturbation configurations.
        if self.model_name in ["pointpillar", "v2vnet"]:
            feature_dim = 64
        elif self.model_name == "voxelnet":
            feature_dim = 128
        else:
            raise NotImplementedError()

        # Basic data structure for the attack.
        base_data_dict = self.retrieve_base_data(cases[-1], ego_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        assert attacker_index >= 0

        # Optimize case processing.
        with torch.no_grad():
            optimize_batch = self.preprocessors[self.fusion_method](cases[-1], ego_id)
            optimize_batch_data = train_utils.to_device(
                self.dataset.collate_batch_test([optimize_batch]),
                self.device
            )
            _, _, optimize_feature = self.attack_intermediate_forward(
                optimize_batch_data, attacker_index
            )

        real_center = None
        if real_bbox is not None:
            real_center = self.point_to_voxel_index(real_bbox, standard=False)

        real_batch_data = None
        real_feature = None
        with torch.no_grad():
            if real_case is not None:
                real_batch = self.preprocessors[self.fusion_method](real_case, ego_id)
                real_batch_data = train_utils.to_device(
                    self.dataset.collate_batch_test([real_batch]),
                    self.device
                )
                _, _, real_feature = self.attack_intermediate_forward(
                    real_batch_data, attacker_index
                )

        pred_bboxes = np.array([])
        pred_scores = np.array([])
        result_prob = None
        result_proposals = None

        with torch.no_grad():
            if real_case is not None and real_center is not None:
                real_aligned_center = real_center.astype(np.int32)
                real_victim_cropped_feature = real_feature[0][:,
                        real_aligned_center[1]-feature_size:real_aligned_center[1]+feature_size,
                        real_aligned_center[0]-feature_size:real_aligned_center[0]+feature_size]
                real_attacker_cropped_feature = real_feature[attacker_index][:,
                        real_aligned_center[1]-feature_size:real_aligned_center[1]+feature_size,
                        real_aligned_center[0]-feature_size:real_aligned_center[0]+feature_size]

                total_perturbation = torch.relu(real_victim_cropped_feature - real_attacker_cropped_feature)

                victim_feat = real_victim_cropped_feature.detach().float().cpu()
                attacker_feat = real_attacker_cropped_feature.detach().float().cpu()
                diff = total_perturbation.detach().float().cpu()   # [C, H, W]

                # Aggregate across channels
                victim_map = victim_feat.norm(dim=0)     # [H, W]
                attacker_map = attacker_feat.norm(dim=0) # [H, W]
                diff_map = diff.norm(dim=0)              # [H, W]
                # alternatively:
                # diff_map = diff.mean(dim=0)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                im0 = axes[0].imshow(victim_map.numpy(), cmap="viridis")
                axes[0].set_title("Victim cropped feature")
                axes[0].axis("off")
                plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                im1 = axes[1].imshow(attacker_map.numpy(), cmap="viridis")
                axes[1].set_title("Attacker cropped feature")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                im2 = axes[2].imshow(diff_map.numpy(), cmap="hot")
                axes[2].set_title("|Victim - Attacker|")
                axes[2].axis("off")
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                plt.tight_layout()
                # Debug dump. tmp_root may not exist in a fresh checkout, and this used to
                # raise inside the attack, which run_pgd_baseline.py caught per case: every
                # case "failed" and the run reported NaN with no visible cause.
                os.makedirs(tmp_root, exist_ok=True)
                plt.savefig(os.path.join(tmp_root, "shift_attack_features.png"))

                real_batch_data = self.detach_all(real_batch_data)

                real_output_dict, _, _ = self.attack_intermediate_forward(
                    real_batch_data,
                    attacker_index,
                    perturbation=total_perturbation,
                    feature=None,
                    max_perturb=20,
                    center=real_aligned_center,
                    feature_size=feature_size
                )

                pred_box_tensor, pred_score_tensor, gt_box_tensor = self.dataset.post_process(
                    real_batch_data,
                    real_output_dict
                )

                result_anchor_box = real_batch_data['ego']['anchor_box']
                result_prob = torch.sigmoid(
                    real_output_dict['ego']['psm'].permute(0, 2, 3, 1)
                ).reshape(-1)

                result_proposals = self.dataset.post_processor.delta_to_boxes3d(
                    real_output_dict['ego']['rm'],
                    result_anchor_box
                )[0]

                if pred_box_tensor is not None:
                    pred_bboxes = pred_box_tensor.detach().cpu().numpy()
                    pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                    pred_bboxes[:, 2] -= 0.5 * pred_bboxes[:, 5]

                    pred_scores = pred_score_tensor.detach().cpu().numpy()

                # convert returned tensors to CPU / numpy
                result_prob = result_prob.detach().cpu().numpy()
                result_proposals = result_proposals[:, [0, 1, 2, 5, 4, 3, 6]].detach().cpu().numpy()

        return {
            "perturbation": None,
            "loss": None,
            "pred_bboxes": pred_bboxes,
            "pred_scores": pred_scores,
            "proposals": result_proposals,
            "prob": result_prob,
        }

    def attack_intermediate_ensemble_hybrid_locations_todo(self, cases, ego_id, attacker_id, max_perturb=1, max_iteration=25, bboxes=None, bboxes2=None, mode="spoof", feature_size=10, real_case=None, real_bbox=None, bbox_weight=1.0, attn_weight=1.0, const_weight=1.0, perturbation_configs={}):
        """
        perturbation_configs = [
            {
                "name": "perturbation_name",
                "loc" (dict): The filter to select locations (voxels) to apply perturbation; attacker/victim=True/False means that in the location the attacker/victim has object/ground features.
                "method" (string): The name of the perturbation method to use.
            }, ...
        ]
        """
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

        # Sets up perturbation configurations.
        if self.model_name in ["pointpillar", "v2vnet"]:
            feature_dim = 64
        elif self.model_name == "voxelnet":
            feature_dim = 128
        else:
            raise NotImplementedError()

        default_perturbation_config = {
            "mode": mode, "bbox_weight": bbox_weight, "attn_weight": attn_weight, "const_weight": attn_weight, "max_perturb": max_perturb
            # Currently we use the same feature_size for all perturbations.
        }
        perturbation_configs = copy.deepcopy(perturbation_configs)

        for cfg in perturbation_configs:
            tmp_cfg = copy.deepcopy(default_perturbation_config)
            tmp_cfg.update(cfg["loss_config"])
            cfg["loss_config"] = tmp_cfg
            if cfg["init_perturbation"] is not None:
                cfg["perturbation"] = torch.from_numpy(cfg["init_perturbation"]).to(self.device)
            else:
                cfg["perturbation"] = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device)
            cfg["perturbation"].requires_grad = True
            cfg["optimizer"] = torch.optim.Adam([cfg["perturbation"]], lr=cfg["learn_rate"])
            cfg["record"] = {"best_loss": 0xffffffff, "best_perturbation": None}

        # Basic data structure for the attack.
        base_data_dict = self.retrieve_base_data(cases[-1], ego_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        assert(attacker_index >= 0)

        # Optimize case processing.
        with torch.no_grad():
            optimize_batch = self.preprocessors[self.fusion_method](cases[-1], ego_id)
            optimize_batch_data = train_utils.to_device(self.dataset.collate_batch_test([optimize_batch]), self.device)
            _, _, optimize_feature = self.attack_intermediate_forward(optimize_batch_data, attacker_index)

        # Real attack case processing.
        if real_bbox is not None:
            real_center = self.point_to_voxel_index(real_bbox, standard=False)
        
        with torch.no_grad():
            if real_case is not None:
                real_batch = self.preprocessors[self.fusion_method](real_case, ego_id)
                real_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_batch]), self.device)
                _, _, real_feature = self.attack_intermediate_forward(real_batch_data, attacker_index)

        best_pred_bboxes = None
        best_pred_scores = None
        best_proposals = None
        best_prob = Nonecenter
        no_progress_iters = 0
        for it in range(max_iteration):
            # Optimizes the perturbation items one by one in a sequence.
            progress_flag = False
            total_perturbation = cfg["perturbation"]
            losses = []
            
            # TODO
            for cfg in perturbation_configs:
                pass
            aligned_center = center.astype(np.int32)


            for case_id in range(len(cases)):
                loss = self.attack_intermediate_get_loss(
                    cases[case_id], ego_id, attacker_id, 
                    total_perturbation, bbox=bboxes[case_id], bbox2=bboxes2[case_id],
                    feature_size=feature_size,
                    **cfg["loss_config"])
                losses.append(loss)

            total_loss = sum(losses) / len(losses)
            if total_loss.item() < cfg["record"]["best_loss"] or max_iteration <= 2:
                cfg["record"]["best_loss"] = total_loss.item()
                cfg["record"]["best_perturbation"] = cfg["perturbation"].cpu().detach().numpy()
                progress_flag = True
            
            total_loss.backward()
            cfg["optimizer"].step()
            cfg["optimizer"].zero_grad()
            logging.warn("Iteration {} - [{}] loss: {}, best loss: {}".format(it, cfg["name"], total_loss.item(), cfg["record"]["best_loss"]))
            
            # Get real attack impacts if the perturbation is ever updated.
            if progress_flag:
                with torch.no_grad():
                    if real_case is not None:
                        real_batch_data = self.detach_all(real_batch_data)
                        real_output_dict, _, _ = self.attack_intermediate_forward(real_batch_data, attacker_index, perturbation=total_perturbation, feature=None, max_perturb=max_perturb, center=real_center, feature_size=feature_size)

                        pred_box_tensor, pred_score_tensor, gt_box_tensor = \
                            self.dataset.post_process(real_batch_data,
                                                    real_output_dict)
                        result_anchor_box = real_batch_data['ego']['anchor_box']
                        result_prob = F.sigmoid(real_output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                        result_proposals = self.dataset.post_processor.delta_to_boxes3d(real_output_dict['ego']['rm'], result_anchor_box)[0]
                    else:
                        pred_box_tensor, pred_score_tensor = None, None
                        result_prob = None
                        result_proposals = None

                    if pred_box_tensor is None:
                        pred_bboxes = np.array([])
                        pred_scores = np.array([])
                    else:
                        pred_bboxes = pred_box_tensor.cpu().numpy()
                        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                        pred_scores = pred_score_tensor.cpu().numpy()

                best_pred_bboxes = pred_bboxes
                best_pred_scores = pred_scores
                best_proposals = result_proposals[:,[0,1,2,5,4,3,6]].cpu().detach().numpy()
                best_prob = result_prob.cpu().detach().numpy()
                no_progress_iters = 0
            else:
                no_progress_iters += 1
            
            # Early quit.
            # if no_progress_iters >= 30:
            #     break

        return {
            "perturbations": [cfg["record"]["best_perturbation"] for cfg in perturbation_configs],
            "losses": [cfg["record"]["best_loss"] for cfg in perturbation_configs],
            "pred_bboxes": best_pred_bboxes,
            "pred_scores": best_pred_scores,
            "proposals": best_proposals,
            "prob": best_prob,
        }

    def attack_intermediate_ensemble_multi_stages(self, cases, ego_id, attacker_id, max_perturb=1, max_iteration=25, bboxes=None, bboxes2=None, mode="spoof", feature_size=10, real_case=None, real_bbox=None, bbox_weight=1.0, attn_weight=1.0, const_weight=1.0, perturbation_configs={}):
        """
        perturbation_configs = [
            {
                "name": "perturbation_name",
                "init_perturbation" (np.ndarray): Initial perturbation, shape (feature_dim, 2 * feature_size, 2 * feature_size),
                "loss_config" (dict): Other arguments for attack_intermediate_get_loss. If not provided, using the default values.
            }, ...
        ]
        """
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

        # Sets up perturbation configurations.
        if self.model_name in ["pointpillar", "v2vnet"]:
            feature_dim = 64
        elif self.model_name == "voxelnet":
            feature_dim = 128
        else:
            raise NotImplementedError()

        default_perturbation_config = {
            "mode": mode, "bbox_weight": bbox_weight, "attn_weight": attn_weight, "const_weight": attn_weight, "max_perturb": max_perturb / len(perturbation_configs)
            # Currently we use the same feature_size for all perturbations.
        }
        perturbation_configs = copy.deepcopy(perturbation_configs)

        for cfg in perturbation_configs:
            tmp_cfg = copy.deepcopy(default_perturbation_config)
            tmp_cfg.update(cfg["loss_config"])
            cfg["loss_config"] = tmp_cfg
            if cfg["init_perturbation"] is not None:
                cfg["perturbation"] = torch.from_numpy(cfg["init_perturbation"]).to(self.device)
            else:
                cfg["perturbation"] = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device)
            cfg["perturbation"].requires_grad = True
            cfg["optimizer"] = torch.optim.Adam([cfg["perturbation"]], lr=cfg["learn_rate"])
            cfg["record"] = {"best_loss": 0xffffffff, "best_perturbation": None}

        # Basic data structure for the attack.
        base_data_dict = self.retrieve_base_data(cases[-1], ego_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        assert(attacker_index >= 0)
        
        # Real attack case processing.
        if real_bbox is not None:
            real_center = self.point_to_voxel_index(real_bbox, standard=False)
        
        with torch.no_grad():
            if real_case is not None:
                real_batch = self.preprocessors[self.fusion_method](real_case, ego_id)
                real_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_batch]), self.device)
                _, _, real_feature = self.attack_intermediate_forward(real_batch_data, attacker_index)

        best_pred_bboxes = None
        best_pred_scores = None
        best_proposals = None
        best_prob = None
        no_progress_iters = 0
        for it in range(max_iteration):
            # Optimizes the perturbation items one by one in a sequence.
            progress_flag = False
            total_perturbation = None
            
            for cfg in perturbation_configs:
                losses = []
                if total_perturbation is None:
                    total_perturbation = cfg["perturbation"]
                else:
                    total_perturbation = total_perturbation.detach() + cfg["perturbation"]

                for case_id in range(len(cases)):
                    loss = self.attack_intermediate_get_loss(
                        cases[case_id], ego_id, attacker_id, 
                        total_perturbation, bbox=bboxes[case_id], bbox2=bboxes2[case_id],
                        feature_size=feature_size,
                        **cfg["loss_config"])
                    losses.append(loss)

                total_loss = sum(losses) / len(losses)
                if total_loss.item() < cfg["record"]["best_loss"] or max_iteration <= 2:
                    cfg["record"]["best_loss"] = total_loss.item()
                    cfg["record"]["best_perturbation"] = cfg["perturbation"].cpu().detach().numpy()
                    progress_flag = True
                
                total_loss.backward()
                cfg["optimizer"].step()
                cfg["optimizer"].zero_grad()
                logging.warn("Iteration {} - [{}] loss: {}, best loss: {}".format(it, cfg["name"], total_loss.item(), cfg["record"]["best_loss"]))
            
            # Get real attack impacts if the perturbation is ever updated.
            if progress_flag:
                with torch.no_grad():
                    if real_case is not None:
                        real_batch_data = self.detach_all(real_batch_data)
                        real_output_dict, _, _ = self.attack_intermediate_forward(real_batch_data, attacker_index, perturbation=total_perturbation, feature=None, max_perturb=max_perturb, center=real_center, feature_size=feature_size)

                        pred_box_tensor, pred_score_tensor, gt_box_tensor = \
                            self.dataset.post_process(real_batch_data,
                                                    real_output_dict)
                        result_anchor_box = real_batch_data['ego']['anchor_box']
                        result_prob = F.sigmoid(real_output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                        result_proposals = self.dataset.post_processor.delta_to_boxes3d(real_output_dict['ego']['rm'], result_anchor_box)[0]
                    else:
                        pred_box_tensor, pred_score_tensor = None, None
                        result_prob = None
                        result_proposals = None

                    if pred_box_tensor is None:
                        pred_bboxes = np.array([])
                        pred_scores = np.array([])
                    else:
                        pred_bboxes = pred_box_tensor.cpu().numpy()
                        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                        pred_scores = pred_score_tensor.cpu().numpy()

                best_pred_bboxes = pred_bboxes
                best_pred_scores = pred_scores
                best_proposals = result_proposals[:,[0,1,2,5,4,3,6]].cpu().detach().numpy()
                best_prob = result_prob.cpu().detach().numpy()
                no_progress_iters = 0
            else:
                no_progress_iters += 1
            
            # Early quit.
            # if no_progress_iters >= 30:
            #     break

        return {
            "perturbations": [cfg["record"]["best_perturbation"] for cfg in perturbation_configs],
            "losses": [cfg["record"]["best_loss"] for cfg in perturbation_configs],
            "pred_bboxes": best_pred_bboxes,
            "pred_scores": best_pred_scores,
            "proposals": best_proposals,
            "prob": best_prob,
        }

    def attack_intermediate_ensemble(self, cases, ego_id, attacker_id, max_perturb=1, lr=0.2, max_iteration=25, bboxes=None, bboxes2=None, mode="spoof", feature_size=10, init_perturbation=None, real_case=None, real_bbox=None, bbox_weight=1.0, attn_weight=0.0, const_weight=0.0):
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

        base_data_dict = self.retrieve_base_data(cases[-1], ego_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        assert(attacker_index >= 0)

        if self.model_name in ["pointpillar", "v2vnet"]:
            feature_dim = 64
        elif self.model_name == "voxelnet":
            feature_dim = 128
        else:
            raise NotImplementedError()
        
        if real_bbox is not None:
            real_center = self.point_to_voxel_index(real_bbox, standard=False)
        
        with torch.no_grad():
            if real_case is not None:
                real_batch = self.preprocessors[self.fusion_method](real_case, ego_id)
                real_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_batch]), self.device)
                _, _, real_feature = self.attack_intermediate_forward(real_batch_data, attacker_index)

        if init_perturbation is not None:
            perturbation = torch.from_numpy(init_perturbation).to(self.device)
        else:
            perturbation = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device)
        perturbation.requires_grad = True
        optimizer = torch.optim.Adam([perturbation], lr=lr)

        best_loss = 0xffffffff
        best_loss_np = 0xffffffff
        best_perturbation = None
        best_pred_bboxes = None
        best_pred_scores = None
        no_progress_iters = 0

        for it in range(max_iteration):
            losses = []
            for case_id in range(len(cases)):
                case = cases[case_id]
                bbox = bboxes[case_id]
                bbox2 = bboxes2[case_id]
                loss = self.attack_intermediate_get_loss(case, ego_id, attacker_id, perturbation, max_perturb=max_perturb, bbox=bbox, bbox2=bbox2, mode=mode, feature_size=feature_size, bbox_weight=bbox_weight, attn_weight=attn_weight, const_weight=const_weight)
                losses.append(loss)

            with torch.no_grad():
                if real_case is not None:
                    real_batch_data = self.detach_all(real_batch_data)
                    real_output_dict, _, real_spatial_features = self.attack_intermediate_forward(real_batch_data, attacker_index, perturbation=perturbation, feature=None, max_perturb=max_perturb, center=real_center, feature_size=feature_size)

                    pred_box_tensor, pred_score_tensor, gt_box_tensor = \
                        self.dataset.post_process(real_batch_data,
                                                  real_output_dict)
                    result_anchor_box = real_batch_data['ego']['anchor_box']
                    result_prob = F.sigmoid(real_output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                    result_proposals = self.dataset.post_processor.delta_to_boxes3d(real_output_dict['ego']['rm'], result_anchor_box)[0]
                else:
                    pred_box_tensor, pred_score_tensor = None, None
                    result_prob = None
                    result_proposals = None

                if pred_box_tensor is None:
                    pred_bboxes = np.array([])
                    pred_scores = np.array([])
                else:
                    pred_bboxes = pred_box_tensor.cpu().numpy()
                    pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                    pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                    pred_scores = pred_score_tensor.cpu().numpy()
    
            total_loss = sum(losses) / len(losses)

            if total_loss.item() < -0xffff or total_loss.item() > 0xffff:
                break

            if total_loss.item() < best_loss or max_iteration <= 2:
                best_loss = total_loss.item()
                best_perturbation = perturbation.cpu().detach().numpy()
                best_pred_bboxes = pred_bboxes
                best_pred_scores = pred_scores
                best_proposals = result_proposals[:,[0,1,2,5,4,3,6]].cpu().detach().numpy()
                best_prob = result_prob.cpu().detach().numpy()
                no_progress_iters = 0
            else:
                no_progress_iters += 1
            
            # if no_progress_iters >= 30:
            #     break

            # optimization
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logging.warn("Iteration {} - loss: {}, best loss: {}".format(it, total_loss.item(), best_loss))

        # Get final spatial features with best perturbation applied
        final_spatial_features = None
        with torch.no_grad():
            if real_case is not None and best_perturbation is not None:
                real_batch_data = self.detach_all(real_batch_data)
                best_pert_tensor = torch.from_numpy(best_perturbation).to(self.device)
                _, _, final_spatial_features = self.attack_intermediate_forward(
                    real_batch_data, attacker_index, perturbation=best_pert_tensor,
                    feature=None, max_perturb=max_perturb, center=real_center,
                    feature_size=feature_size)

        return {
            "perturbation": best_perturbation,
            "loss": best_loss,
            "pred_bboxes": best_pred_bboxes,
            "pred_scores": best_pred_scores,
            "proposals": best_proposals,
            "prob": best_prob,
            "spatial_features": final_spatial_features.detach() if final_spatial_features is not None else None,
        }

    # Depracated for shift attack
    def attack_intermediate(self, multi_vehicle_case, ego_id, attacker_id, max_perturb=10, lr=0.2, max_iteration=25, bbox=None, bbox2=None, mode="spoof", real_case=None, original_case=None, real_original_case=None, real_bbox=None, init_perturbation=None, feature_size=10, loss_fn=None):
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        assert(attacker_index >= 0)
    
        optimize_batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        optimize_batch_data = train_utils.to_device(self.dataset.collate_batch_test([optimize_batch]), self.device)
        anchor_box = optimize_batch_data['ego']['anchor_box']

        if self.model_name in ["pointpillar", "v2vnet"]:
            feature_dim = 64
        elif self.model_name == "voxelnet":
            feature_dim = 128
        else:
            raise NotImplementedError()

        if bbox is not None:
            bbox_tensor = torch.from_numpy(bbox).to(self.device).type(torch.float32)
            bbox_tensor[2] += 0.5 * bbox_tensor[5]
            center = self.point_to_voxel_index(bbox)
        
        if bbox2 is not None:
            bbox2_tensor = torch.from_numpy(bbox2).to(self.device).type(torch.float32)
            bbox2_tensor[2] += 0.5 * bbox2_tensor[5]

        if real_bbox is not None:
            real_center = self.point_to_voxel_index(real_bbox)

        with torch.no_grad():
            optimize_output_dict, _, optimize_feature = self.attack_intermediate_forward(optimize_batch_data, attacker_index)
            original_prob = F.sigmoid(optimize_output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1).detach()
            original_proposals = self.dataset.post_processor.delta_to_boxes3d(optimize_output_dict['ego']['rm'], anchor_box)[0].detach()

            if real_case is not None:
                real_batch = self.preprocessors[self.fusion_method](real_case, ego_id)
                real_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_batch]), self.device)
                _, _, real_feature = self.attack_intermediate_forward(real_batch_data, attacker_index)

            if original_case is not None:
                original_batch = self.preprocessors[self.fusion_method](original_case, ego_id)
                original_batch_data = train_utils.to_device(self.dataset.collate_batch_test([original_batch]), self.device)
                _, _, original_feature = self.attack_intermediate_forward(original_batch_data, attacker_index)
                # TODO: interpolation of center indices
                base_perturbation = ((optimize_feature[attacker_index] - original_feature[attacker_index])[:, center[1]-feature_size:center[1]+feature_size, center[0]-feature_size:center[0]+feature_size]).detach()
            else:
                base_perturbation = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device).detach()

            if real_original_case is not None:
                real_original_batch = self.preprocessors[self.fusion_method](real_original_case, ego_id)
                real_original_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_original_batch]), self.device)
                _, _, real_original_feature = self.attack_intermediate_forward(real_original_batch_data, attacker_index)
                # TODO: interpolation of center indices
                real_base_perturbation = ((real_feature[attacker_index] - real_original_feature[attacker_index])[:, real_center[1]-feature_size:real_center[1]+feature_size, real_center[0]-feature_size:real_center[0]+feature_size]).detach()
            else:
                real_base_perturbation = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device).detach()

        if init_perturbation is not None:
            perturbation = torch.from_numpy(init_perturbation).to(self.device)
        else:
            perturbation = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device)
        perturbation.requires_grad = True
        optimizer = torch.optim.Adam([perturbation], lr=lr)

        best_loss = 0xffffffff
        best_perturbation = None
        best_pred_bboxes = None
        best_pred_scores = None
        no_progress_iters = 0

        for it in range(max_iteration):
            batch_data = self.detach_all(optimize_batch_data if original_case is None else original_batch_data)

            output_dict, clipped_perturbation, spatial_features = self.attack_intermediate_forward(batch_data, attacker_index, perturbation=(base_perturbation + perturbation), max_perturb=max_perturb, center=center, feature_size=feature_size)
            prob = F.sigmoid(output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
            proposals = self.dataset.post_processor.delta_to_boxes3d(output_dict['ego']['rm'], anchor_box)[0]

            if bbox is not None:
                iou = torch.clip(self.iou_torch(
                    proposals[:,[0,1,2,5,4,3,6]], 
                    bbox_tensor.tile((proposals.shape[0],1))
                ), min=0, max=1)
            else:
                iou = torch.ones(proposals.shape[0], dtype=torch.bool).to(self.device).detach()
            bbox_mask = (iou >= 0.01)

            if bbox2 is not None:
                iou2 = torch.clip(self.iou_torch(
                    proposals[:,[0,1,2,5,4,3,6]], 
                    bbox2_tensor.tile((proposals.shape[0],1))
                ), min=0, max=1)
            else:
                iou2 = torch.ones(proposals.shape[0], dtype=torch.bool).to(self.device).detach()
            box2_mask = (iou2 >= 0.01)

            prob_mask = (prob >= 0.1)

            with torch.no_grad():
                if real_case is not None:
                    real_batch_data = self.detach_all(real_batch_data)
                    real_output_dict, _, _ = self.attack_intermediate_forward(real_original_batch_data if real_original_case is not None else real_batch_data, attacker_index, perturbation=real_base_perturbation + perturbation, feature=None, max_perturb=max_perturb, center=real_center, feature_size=feature_size)

                    pred_box_tensor, pred_score_tensor, gt_box_tensor = \
                        self.dataset.post_process(real_batch_data,
                                                  real_output_dict)
                    result_anchor_box = real_batch_data['ego']['anchor_box']
                    result_prob = F.sigmoid(real_output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                    result_proposals = self.dataset.post_processor.delta_to_boxes3d(real_output_dict['ego']['rm'], result_anchor_box)[0]
                else:
                    pred_box_tensor, pred_score_tensor, gt_box_tensor = \
                        self.dataset.post_process(batch_data,
                                                  output_dict)
                    result_prob = prob
                    result_proposals = proposals

                if pred_box_tensor is None:
                    pred_bboxes = np.array([])
                    pred_scores = np.array([])
                else:
                    pred_bboxes = pred_box_tensor.cpu().numpy()
                    pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                    pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                    pred_scores = pred_score_tensor.cpu().numpy()

            if mode == "spoof":
                loss = (1 * iou[bbox_mask] * torch.log(1 - prob[bbox_mask])).sum()
            elif mode == "remove":
                loss = (-1 * iou[bbox_mask] * torch.log(1 - prob[bbox_mask])).sum()
            elif mode == "shift":
                mask = torch.logical_and(torch.logical_and(bbox_mask, box2_mask), prob_mask)
                loss = (torch.log(1 - iou2[mask])).sum() + 1 * torch.clip(original_prob[mask] - prob[mask], 0, 1).sum()
            else:
                raise NotImplementedError("Attack mode not supported.")

            if loss_fn is not None:
                attn_batch = {
                    'spatial_features': spatial_features,       
                    'record_len':       batch_data["ego"]['record_len']     
                }
                attn_loss = loss_fn(
                    model=self.model,
                    batch_dict=attn_batch
                )
            else:
                attn_loss = 0

            alpha=1.0
            total_loss = alpha*loss + 1.0*attn_loss

            if total_loss.item() < -0xffff:
                break

            if total_loss.item() < best_loss or max_iteration <= 2:
                best_loss = total_loss.item()
                best_perturbation = clipped_perturbation.cpu().detach().numpy()
                best_pred_bboxes = pred_bboxes
                best_pred_scores = pred_scores
                best_proposals = result_proposals[:,[0,1,2,5,4,3,6]].cpu().detach().numpy()
                best_prob = result_prob.cpu().detach().numpy()
                no_progress_iters = 0
            else:
                no_progress_iters += 1

            # optimization
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logging.warn("Iteration {} - loss: {}, best loss: {}".format(it, total_loss.item(), best_loss))

        return {
            "perturbation": best_perturbation,
            "loss": best_loss,
            "pred_bboxes": best_pred_bboxes,
            "pred_scores": best_pred_scores,
            "proposals": best_proposals,
            "prob": best_prob,
        }

    def retrieve_base_data(self, multi_vehicle_case, ego_id):
        data = OrderedDict()
        ego_pose = multi_vehicle_case[ego_id]["lidar_pose"]
        # Ensure ego is always first in the ordered dict (required by fusion modules)
        vehicle_ids = list(multi_vehicle_case.keys())
        if ego_id in vehicle_ids:
            vehicle_ids.remove(ego_id)
            vehicle_ids.insert(0, ego_id)
        for vehicle_id in vehicle_ids:
            vehicle_data = multi_vehicle_case[vehicle_id]
            data[vehicle_id] = OrderedDict()
            data[vehicle_id]['ego'] = (vehicle_id == ego_id)
            data[vehicle_id]["cav_id"] = vehicle_id
            data[vehicle_id]['time_delay'] = 0
            if "params" in vehicle_data:
                import copy as _copy
                data[vehicle_id]['params'] = _copy.deepcopy(vehicle_data["params"])
                data[vehicle_id]['params']["lidar_pose"] = vehicle_data["lidar_pose"]
                if self.dataset_name == "V2X-Real":
                    data[vehicle_id]['params']["transformation_matrix"] = x1_to_x2(vehicle_data["lidar_pose"], ego_pose)
                else:
                    data[vehicle_id]['params']["transformation_matrix"] = np.dot(np.linalg.inv(pose_to_transformation(ego_pose)), pose_to_transformation(vehicle_data["lidar_pose"]))
                if self.dataset_name == "V2X-Real" and hasattr(self.dataset, 'map_class_name_to_super_class_name'):
                    data[vehicle_id]['params']['vehicles'] = self.dataset.map_class_name_to_super_class_name(data[vehicle_id]['params']['vehicles'])
                    data[vehicle_id]['params']['vehicles'] = self.dataset.filter_boxes_by_class(data[vehicle_id]['params']['vehicles'])
                if self.dataset_name in ["V2V4Real", "V2X-Real"]:
                    data[vehicle_id]['params']['spatial_correction_matrix'] = np.eye(4)
                    data[vehicle_id]['params']["gt_transformation_matrix"] = data[vehicle_id]['params']["transformation_matrix"]
            else:
                if self.dataset_name == "V2X-Real":
                    t_matrix = x1_to_x2(vehicle_data["lidar_pose"], ego_pose)
                else:
                    t_matrix = np.dot(np.linalg.inv(pose_to_transformation(ego_pose)), pose_to_transformation(vehicle_data["lidar_pose"]))
                data[vehicle_id]['params'] = {
                    "lidar_pose": vehicle_data["lidar_pose"],
                    "transformation_matrix": t_matrix,
                    "spatial_correction_matrix": np.eye(4),
                    "vehicles": {},
                }
            if self.model_name in ["pointpillar"]:
                data[vehicle_id]['lidar_np'] = vehicle_data["lidar"].astype(np.float32)
                # V2X-Real model was trained with zero_intensity=True
                if self.dataset_name == "V2X-Real":
                    data[vehicle_id]['lidar_np'][:, 3] = 0
            else:
                data[vehicle_id]['lidar_np'] = vehicle_data["lidar"][:,:4].astype(np.float32)
            # V2X-Real requires augmentation keys (disabled for inference)
            if self.dataset_name == "V2X-Real":
                data[vehicle_id]['flip'] = [None, None, None]
                data[vehicle_id]['noise_rotation'] = None
                data[vehicle_id]['noise_scale'] = None
        return data

    def early_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_lidar_pose = base_data_dict[ego_id]["params"]['lidar_pose']

        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            # if distance > opencood.data_utils.datasets.COM_RANGE:
            #     continue

            selected_cav_processed = self.dataset.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            # all these lidar and object coordinates are projected to ego
            # already.
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.dataset.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.dataset.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack[:, :7]
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)

        # we do lidar filtering in the stacked lidar
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.dataset.params['preprocess'][
                                                         'cav_lidar_range'])
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid = \
            box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                     self.dataset.params['preprocess'][
                                                         'cav_lidar_range'],
                                                     self.dataset.params['postprocess'][
                                                         'order']
                                                     )
        # Two versions of OpenCOOD!
        if isinstance(object_bbx_center_valid, tuple):
            object_bbx_center_valid = object_bbx_center_valid[0]

        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.dataset.pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes and targets label
        if self.dataset_name == "V2X-Real":
            anchor_box, num_anchors_per_location = self.dataset.post_processor.generate_anchor_box()
            label_dict = \
                self.dataset.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    num_anchors_per_location=num_anchors_per_location,
                    mask=mask)
        else:
            anchor_box = self.dataset.post_processor.generate_anchor_box()
            label_dict = \
                self.dataset.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': lidar_dict,
             'label_dict': label_dict})
        if self.dataset_name == "V2X-Real":
            processed_data_dict['ego'].update(
            {'all_anchors': anchor_box,
             'num_anchors_per_location': num_anchors_per_location})

        return processed_data_dict

    def intermediate_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = None
        ego_lidar_pose = None

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_lidar_pose is not None, "Ego vehicle not found in base_data_dict"
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.dataset.get_pairwise_transformation(base_data_dict,
                                             self.dataset.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            # if distance > opencood.data_utils.datasets.COM_RANGE:
            #     continue

            selected_cav_processed = self.dataset.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            if self.dataset_name == "V2X-Real":
                selected_cav_processed = selected_cav_processed[0]

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(
                    selected_cav_processed['processed_features'])

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        # V2X-Real uses 8 columns (7 bbox + class_id), OPV2V uses 7
        n_cols = object_stack.shape[1] if len(object_stack) > 0 else 7
        object_bbx_center = \
            np.zeros((self.dataset.params['postprocess']['max_num'], n_cols))
        mask = np.zeros(self.dataset.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack[:, :n_cols]
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.dataset.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.dataset.post_processor.generate_anchor_box()

        # generate the anchor boxes and targets label
        if self.dataset_name == "V2X-Real":
            anchor_box, num_anchors_per_location = self.dataset.post_processor.generate_anchor_box()
            label_dict = \
                self.dataset.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    num_anchors_per_location=num_anchors_per_location,
                    mask=mask)
        else:
            anchor_box = self.dataset.post_processor.generate_anchor_box()
            label_dict = \
                self.dataset.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'velocity': [0 for i in range(len(multi_vehicle_case))],
             'time_delay': [0 for i in range(len(multi_vehicle_case))],
             'infra': [0 for i in range(len(multi_vehicle_case))],
             'spatial_correction_matrix': [np.eye(4) for i in range(len(multi_vehicle_case))],
            })
        if self.dataset_name == "V2X-Real":
            processed_data_dict['ego'].update(
            {'all_anchors': anchor_box,
             'num_anchors_per_location': num_anchors_per_location})

        return processed_data_dict

    def late_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)
        reformat_data_dict = self.dataset.get_item_test(base_data_dict)

        return reformat_data_dict

    def points_to_voxel_torch(self, pcd):
        # https://github.com/DerrickXuNu/OpenCOOD/blob/main/opencood/data_utils/pre_processor/voxel_preprocessor.py
        # full_mean = False
        # block_filtering = False
        data_dict = {}
        lidar_range = self.dataset.pre_processor.params["cav_lidar_range"]
        voxel_size = self.dataset.pre_processor.params["args"]["voxel_size"]
        max_points_per_voxel = self.dataset.pre_processor.params["args"]["max_points_per_voxel"]

        voxel_coords = torch.floor((pcd[:, :3] - 
                torch.tensor(lidar_range[:3]).to(self.device)
            ) / torch.tensor(voxel_size).to(self.device)).int()

        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = torch.unique(voxel_coords, dim=0,
                                                           return_inverse=True,
                                                           return_counts=True)
        
        voxel_features = torch.zeros((len(voxel_coords), max_points_per_voxel, 4), dtype=torch.float32).to(self.device)

        for i in range(len(voxel_coords)):
            pts = pcd[inv_ind == i]
            if voxel_counts[i] > max_points_per_voxel:
                pts = pts[:max_points_per_voxel, :]
                voxel_counts[i] = max_points_per_voxel

            voxel_features[i, :pts.shape[0], :] = pts

        data_dict['voxel_features'] = voxel_features
        data_dict['voxel_coords'] = voxel_coords
        data_dict['voxel_num_points'] = voxel_counts

        return data_dict

    def point_to_voxel_index(self, point, standard=True):
        lidar_range = self.dataset.pre_processor.params["cav_lidar_range"]
        voxel_size = self.dataset.pre_processor.params["args"]["voxel_size"]
        if standard:
            voxel_index = ((point[:3] - np.floor(lidar_range[:3])) / voxel_size).astype(np.int32)
        else:
            voxel_index = (point[:3] - np.floor(lidar_range[:3])) / voxel_size
        return voxel_index

    def iou_bev_torch(self, bboxes_a, bboxes_b):
        """BEV-only IoU (ignores z dimension). Differentiable."""
        corners2d_a = torch.unsqueeze(box_utils.boxes_to_corners2d(bboxes_a, order="lwh")[:,:,:2], 0)
        corners2d_b = torch.unsqueeze(box_utils.boxes_to_corners2d(bboxes_b, order="lwh")[:,:,:2], 0)
        area_a = bboxes_a[:, 3] * bboxes_a[:, 4]
        area_b = bboxes_b[:, 3] * bboxes_b[:, 4]
        area_inter, _ = oriented_box_intersection_2d(corners2d_a, corners2d_b)
        area_inter = area_inter.squeeze()
        iou = area_inter / (area_a + area_b - area_inter + 1e-8)
        return iou

    def iou_torch(self, bboxes_a, bboxes_b):
        corners2d_a = torch.unsqueeze(box_utils.boxes_to_corners2d(bboxes_a, order="lwh")[:,:,:2], 0)
        corners2d_b = torch.unsqueeze(box_utils.boxes_to_corners2d(bboxes_b, order="lwh")[:,:,:2], 0)
        area_a = bboxes_a[:, 3] * bboxes_a[:, 4]
        area_b = bboxes_b[:, 3] * bboxes_b[:, 4]
        area_inter, _ = oriented_box_intersection_2d(corners2d_a, corners2d_b)
        area_inter = area_inter.squeeze()
        height_inter = torch.clip(
            torch.min(bboxes_a[:, 2] + 0.5 * bboxes_a[:, 5], bboxes_b[:, 2] + 0.5 * bboxes_b[:, 5]) - \
            torch.max(bboxes_a[:, 2] - 0.5 * bboxes_a[:, 5], bboxes_b[:, 2] - 0.5 * bboxes_b[:, 5]),
            min=0, max=5)
        iou = area_inter * height_inter / (area_a * bboxes_a[:, 5] + area_b * bboxes_b[:, 5] - area_inter * height_inter)
        return iou

    def pose_to_transformation_torch(self, pose, dim=2):
        x, y, z, roll, yaw, pitch = pose[0], pose[1], pose[2], torch.deg2rad(pose[3]), torch.deg2rad(pose[4]), torch.deg2rad(pose[5])
        if dim == 2:
            T = torch.zeros((3, 3)).to(torch.float32).to(self.device)
            T[0, 0] = torch.cos(yaw)
            T[0, 1] = 0 - torch.sin(yaw)
            T[0, 2] = x
            T[1, 0] = torch.sin(yaw)
            T[1, 1] = torch.cos(yaw)
            T[1, 2] = y
            T[2, 2] = 1
        elif dim == 3:
            T = torch.tensor([[torch.cos(yaw)*torch.cos(pitch), 
                        torch.cos(yaw)*torch.sin(pitch)*torch.sin(roll)-torch.sin(yaw)*torch.cos(roll), 
                        torch.cos(yaw)*torch.sin(pitch)*torch.cos(roll)+torch.sin(yaw)*torch.sin(roll),
                        x],
                        [torch.sin(yaw)*torch.cos(pitch), 
                        torch.sin(yaw)*torch.sin(pitch)*torch.sin(roll)+torch.cos(yaw)*torch.cos(roll), 
                        torch.sin(yaw)*torch.sin(pitch)*torch.cos(roll)-torch.cos(yaw)*torch.sin(roll),
                        y],
                        [-torch.sin(pitch), 
                        torch.cos(pitch)*torch.sin(roll), 
                        torch.cos(pitch)*torch.cos(roll),
                        z],
                        [0, 0, 0, 1]]).to(self.device)
        return T

    def attacker_to_origin_transformation(self, T, attacker_pose, origin_pose, dim=2):
        attacker_T = self.pose_to_transformation_torch(attacker_pose, dim=dim)
        origin_T = self.pose_to_transformation_torch(origin_pose, dim=dim)
        return torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.inverse(origin_T), attacker_T), T), torch.inverse(attacker_T)), origin_T)

    def detach_all(self, x):
        if isinstance(x, dict):
            y = {}
            for key, value in x.items():
                y[key] = self.detach_all(value)
        elif isinstance(x, list):
            y = []
            for value in x:
                y.append(self.detach_all(value))
        elif isinstance(x, torch.Tensor):
            y = x.detach()
        else:
            y = x
        return y

    def pose_to_transformation_torch(self, pose, dim=2):
        x, y, z, roll, yaw, pitch = pose[0], pose[1], pose[2], torch.deg2rad(pose[3]), torch.deg2rad(pose[4]), torch.deg2rad(pose[5])
        if dim == 2:
            T = torch.zeros((3, 3)).to(torch.float32).to(self.device)
            T[0, 0] = torch.cos(yaw)
            T[0, 1] = 0 - torch.sin(yaw)
            T[0, 2] = x
            T[1, 0] = torch.sin(yaw)
            T[1, 1] = torch.cos(yaw)
            T[1, 2] = y
            T[2, 2] = 1
        elif dim == 3:
            T = torch.tensor([[torch.cos(yaw)*torch.cos(pitch), 
                        torch.cos(yaw)*torch.sin(pitch)*torch.sin(roll)-torch.sin(yaw)*torch.cos(roll), 
                        torch.cos(yaw)*torch.sin(pitch)*torch.cos(roll)+torch.sin(yaw)*torch.sin(roll),
                        x],
                        [torch.sin(yaw)*torch.cos(pitch), 
                        torch.sin(yaw)*torch.sin(pitch)*torch.sin(roll)+torch.cos(yaw)*torch.cos(roll), 
                        torch.sin(yaw)*torch.sin(pitch)*torch.cos(roll)-torch.cos(yaw)*torch.sin(roll),
                        y],
                        [-torch.sin(pitch), 
                        torch.cos(pitch)*torch.sin(roll), 
                        torch.cos(pitch)*torch.cos(roll),
                        z],
                        [0, 0, 0, 1]]).to(self.device)
        return T
