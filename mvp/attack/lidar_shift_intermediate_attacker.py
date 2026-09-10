import random
import pickle
import numpy as np
import copy
from collections import OrderedDict

from .attacker import Attacker
from mvp.data.util import bbox_map_to_sensor, bbox_sensor_to_map, pcd_sensor_to_map
from mvp.attack.shift_rotation import apply_shift
from .attack_utils import get_adv_loss


class LidarShiftIntermediateAttacker(Attacker):
    def __init__ (self, perception, dataset=None, step=40, learn_rate=None, sync=0, init=True, online=False, aggr=1, attn="none", loc="hybrid", debug=False):
        super().__init__()
        self.dataset = dataset
        self.name = "lidar_shift"
        self.load_benchmark_meta()
        self.name = "lidar_shift_intermediate"

        self.name += "_Step{}".format(step - 1)
        if sync > 0:
            self.name += "_Async"
        if init:
            self.name += "_Init"
        if online:
            self.name += "_Online"
        if aggr > 1:
            self.name += "_Aggr{}".format(aggr)
        # Using attn token to control how attn is considered in the optimization.
        assert(attn in ["none", "add", "only", "two-stage"])
        if attn is not None and attn != "none":
            self.name += "_Attn({})".format(attn)
        # Using loc token to control how voxels at different locations are managed differently.
        assert(loc in ["none", "hybrid"])
        if loc is not None and loc != "none":
            self.name += "_Loc({})".format(loc)
        if debug:
            self.name += "_debug"
        self.step = step
        self.sync = sync
        self.init = init
        self.online = online
        self.aggr = aggr
        self.attn = attn
        self.loc = loc
        self.debug = debug

        if perception.model_name != "pointpillar":
            self.name += "_{}".format(perception.model_name)

        self.perception = perception

        # TODO: tune the learn rate, max_perturb, and feature_size.
        if learn_rate is None:
            if step <=  2:
                self.learn_rate = 1
            else:
                self.learn_rate = 0.1
        else:
            self.learn_rate = learn_rate
        self.max_perturb = 10
        self.feature_size = 10

    def run(self, multi_frame_case, attack_opts):
        case = copy.deepcopy(multi_frame_case)
        info = [{} for i in range(10)]
        init_perturbation = None

        for frame_index, frame_id in enumerate(attack_opts["frame_ids"]):
            attacker_id = attack_opts["attacker_vehicle_id"]
            ego_id = attack_opts["victim_vehicle_id"]
            info[frame_id][ego_id] = {}

            real_frame_id = frame_id
            if self.sync == 0:
                optimize_frame_id = frame_id
            else:
                optimize_frame_id = frame_id - 1
            real_case = case[real_frame_id]

            if self.init:
                for fid in range(optimize_frame_id - self.aggr + 1, optimize_frame_id + 1):
                    case[fid][attacker_id]["lidar"] = self.apply_ray_tracing(case[fid][attacker_id]["lidar"], **attack_opts["attack_info"][fid])
                if real_case is not None:
                    real_case[attacker_id]["lidar"] = self.apply_ray_tracing(real_case[attacker_id]["lidar"], **attack_opts["attack_info"][real_frame_id])

            cases = []
            bboxes = []
            bboxes2 = []
            for fid in range(optimize_frame_id - self.aggr + 1, optimize_frame_id + 1):
                cases.append(multi_frame_case[fid])

                object_index = multi_frame_case[fid][attacker_id]["object_ids"].index(attack_opts["object_id"])
                bbox_to_remove = multi_frame_case[fid][attacker_id]["gt_bboxes"][object_index]
                bbox_to_remove_ego = bbox_map_to_sensor(
                    bbox_sensor_to_map(bbox_to_remove, multi_frame_case[fid][attacker_id]["lidar_pose"]),
                    multi_frame_case[fid][ego_id]["lidar_pose"])
                bboxes.append(bbox_to_remove_ego)

                bbox_to_spoof = apply_shift(bbox_to_remove, attack_opts)
                bbox_to_spoof_ego = bbox_map_to_sensor(
                    bbox_sensor_to_map(bbox_to_spoof, multi_frame_case[fid][attacker_id]["lidar_pose"]),
                    multi_frame_case[fid][ego_id]["lidar_pose"])
                bboxes2.append(bbox_to_spoof_ego)

            real_object_index = multi_frame_case[real_frame_id][attacker_id]["object_ids"].index(attack_opts["object_id"])
            real_bbox_to_remove = multi_frame_case[real_frame_id][attacker_id]["gt_bboxes"][real_object_index]
            real_bbox_to_remove_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(real_bbox_to_remove, multi_frame_case[real_frame_id][attacker_id]["lidar_pose"]),
                multi_frame_case[real_frame_id][ego_id]["lidar_pose"])

            if self.loc == "hybrid":
                perturbation_configs = [
                    {"name": "AnV", "loc": {"attacker": True, "victim": False, "threshold": 0.2}, "method": "object_enhance"},
                    {"name": "nAV", "loc": {"attacker": False, "victim": True, "threshold": 0.2}, "method": "attention_enhance"},
                ]

                result = self.perception.attack_intermediate_ensemble_hybrid_locations(cases, ego_id, attacker_id, max_perturb=self.max_perturb, mode="shift", bboxes=bboxes, bboxes2=bboxes2, max_iteration=self.step, real_case=real_case, real_bbox=real_bbox_to_remove_ego, feature_size=self.feature_size, perturbation_configs=perturbation_configs)
            elif self.attn == "two-stage":
                perturbation_configs = [
                    {"name": "attn", "init_perturbation": init_perturbation, "learn_rate": 1.0 * self.learn_rate, "loss_config": {"bbox_weight": 0, "attn_weight": 10, "const_weight": 1, "max_perturb": 0.2 * self.max_perturb}},
                    {"name": "bbox", "init_perturbation": None, "learn_rate": 1.0 * self.learn_rate, "loss_config": {"bbox_weight": 1, "attn_weight": 0, "const_weight": 0, "max_perturb": 0.8 * self.max_perturb}},
                ]

                result = self.perception.attack_intermediate_ensemble_multi_stages(cases, ego_id, attacker_id, max_perturb=self.max_perturb, mode="shift", bboxes=bboxes, bboxes2=bboxes2, max_iteration=self.step, real_case=real_case, real_bbox=real_bbox_to_remove_ego, feature_size=self.feature_size, perturbation_configs=perturbation_configs)
            else:
                if self.attn is None or self.attn == "none":
                    bbox_weight, attn_weight, const_weight = 1, 0, 0
                elif self.attn == "add":
                    bbox_weight, attn_weight, const_weight = 1, 10, 1
                elif self.attn == "only":
                    bbox_weight, attn_weight, const_weight = 0, 10, 1

                result = self.perception.attack_intermediate_ensemble(cases, ego_id, attacker_id, max_perturb=self.max_perturb, mode="shift", bboxes=bboxes, bboxes2=bboxes2, max_iteration=self.step, lr=self.learn_rate, real_case=real_case, real_bbox=real_bbox_to_remove_ego, feature_size=self.feature_size, init_perturbation=init_perturbation, bbox_weight=bbox_weight, attn_weight=attn_weight, const_weight=const_weight)

            if self.online:
                if self.attn == "two-stage":
                    init_perturbation = result["perturbations"][0]
                else:
                    init_perturbation = result["perturbation"]

            case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
            case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
            info[frame_id][ego_id] = {
                "pred_bboxes": result["pred_bboxes"],
                "pred_scores": result["pred_scores"],
                "proposals": result["proposals"],
                "prob": result["prob"],
                "spatial_features": result.get("spatial_features"),
            }
            if self.attn == "two-stage":
                info[frame_id][attacker_id] = {"perturbations": result["perturbations"]}
            else:
                info[frame_id][attacker_id] = {"perturbation": result["perturbation"]}

        return case, info
    
