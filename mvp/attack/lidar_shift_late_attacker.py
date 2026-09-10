import random
import pickle
import numpy as np
import copy

from .attacker import Attacker
from mvp.data.util import bbox_map_to_sensor, bbox_sensor_to_map, pcd_sensor_to_map


class LidarShiftLateAttacker(Attacker):
    def __init__ (self, perception, dataset=None):
        super().__init__()
        self.name = "lidar_shift"
        self.dataset = dataset
        self.load_benchmark_meta()
        self.perception = perception
        self.name = "lidar_shift_late"

    def run(self, multi_frame_case, attack_opts):
        case = copy.deepcopy(multi_frame_case)
        attack_results = []
        for frame_id in range(10):
            attack_results.append({})
            if frame_id == 9:
                multi_vehicle_case = multi_frame_case[frame_id]
                attacker_id = attack_opts["attacker_vehicle_id"]
                ego_id = attack_opts["victim_vehicle_id"]
                object_index = multi_vehicle_case[attacker_id]["object_ids"].index(attack_opts["object_id"])
                bbox_to_remove = multi_vehicle_case[attacker_id]["gt_bboxes"][object_index]
                bbox_to_spoof = copy.deepcopy(bbox_to_remove)
                bbox_to_spoof[0] += np.cos(attack_opts["shift_direction"]) * attack_opts["shift_distance"]
                bbox_to_spoof[1] += np.sin(attack_opts["shift_direction"]) * attack_opts["shift_distance"]
                result = self.perception.attack_late(multi_vehicle_case, ego_id, attacker_id, mode="shift", bbox=bbox_to_remove, bbox2=bbox_to_spoof)
                case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
                case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
                attack_results[-1][ego_id] = {
                    "pred_bboxes": result["pred_bboxes"],
                    "pred_scores": result["pred_scores"]
                }
        return case, attack_results
    
    def run_multi_vehicle(self, multi_vehicle_case, attack_opts):
        attacker_id, victim_id = attack_opts["attacker_vehicle_id"], attack_opts["victim_vehicle_id"]
        bbox_to_remove, bbox_to_spoof = attack_opts["bbox_to_remove"], attack_opts["bbox_to_spoof"]
        # bbox_to_remove/spoof are in attacker's sensor frame.
        # attack_late compares them against attacker's model output (also in attacker frame)
        # then projects everything to ego frame via transformation_matrix.
        result = self.perception.attack_late(multi_vehicle_case, victim_id, attacker_id,
                                             mode="shift", bbox=bbox_to_remove, bbox2=bbox_to_spoof)
        return result

    def build_benchmark_meta(self, write=False, max_cnt=500):
        raise NotImplementedError("Should use the same benchmark as LidarShiftAttacker.")