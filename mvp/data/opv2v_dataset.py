import pickle
import time
import os
import numpy as np
import yaml
import open3d as o3d
import copy
import cv2
from collections import OrderedDict

from .dataset import Dataset
from .util import read_pcd, read_bin, rotation_matrix, transformation_to_pose, bbox_map_to_sensor, bbox_sensor_to_map, pcd_map_to_sensor
from mvp.tools.sensor_calib import lidar_to_camera
from mvp.config import scenario_maps


class OPV2VDataset(Dataset):
    NumPointFeatures = 4

    def __init__(self, root_path, mode, dataset_name="OPV2V"):
        self.root_path = root_path
        self.mode = mode
        self.name = dataset_name

        with open(os.path.join(root_path, "{}.pkl".format(mode)), 'rb') as f:
            self.meta = pickle.load(f)

        cases_path = os.path.join(root_path, "{}_cases.pkl".format(mode))
        if os.path.exists(cases_path):
            with open(cases_path, 'rb') as f:
                self.cases = pickle.load(f)
        else:
            self.cases = {
                "single_vehicle": [],
                "multi_vehicle": [],
                "multi_frame": [],
                "scenario": []
            }
            self._build_cases()
            with open(cases_path, 'wb') as f:
                pickle.dump(self.cases, f)

        attacks_path = os.path.join(root_path, "{}_attacks.pkl".format(mode))
        if os.path.exists(attacks_path):
            with open(attacks_path, 'rb') as f:
                self.attacks = pickle.load(f)
        else:
            self.attacks = []
            self._build_attacks()
            with open(attacks_path, 'wb') as f:
                pickle.dump(self.attacks, f)

        scenario_attacks_path = os.path.join(root_path, "{}_scenario_attacks.pkl".format(mode))
        if os.path.exists(scenario_attacks_path):
            with open(scenario_attacks_path, 'rb') as f:
                self.scenario_attacks = pickle.load(f)
        else:
            self.scenario_attacks = []
            self._build_scenario_attacks()
            with open(scenario_attacks_path, 'wb') as f:
                pickle.dump(self.scenario_attacks, f)

        # self.cache = OrderedDict()
        # self.cache_size = 1

    def _build_cases(self):
        frame_num_per_case = 10
        frame_ids = []
        for scenario_id, scenario_data in self.meta.items():
            self.cases["scenario"].append({
                "scenario_id": scenario_id, 
                "frame_ids": list(scenario_data["data"].keys()),
                "vehicle_ids": list(list(scenario_data["data"].values())[0].keys())
            })
            for frame_id, frame_data in scenario_data["data"].items():
                self.cases["multi_vehicle"].append({
                    "scenario_id": scenario_id, 
                    "frame_id": frame_id
                })
                frame_ids.append(frame_id)
                if len(frame_ids) == frame_num_per_case:
                    self.cases["multi_frame"].append({
                        "scenario_id": scenario_id,
                        "frame_ids": frame_ids
                    })
                    frame_ids = []
                for vehicle_id, vehicle_data in frame_data.items():
                    self.cases["single_vehicle"].append({
                        "scenario_id": scenario_id, 
                        "frame_id": frame_id,
                        "vehicle_id": vehicle_id
                    })
            frame_ids = []

    def _build_attacks(self):
        for idx, case in enumerate(self.cases["multi_frame"]):
            scenario_id, frame_ids = case["scenario_id"], case["frame_ids"]
            for attacker_vehicle_id in self.meta[scenario_id]["vehicle_ids"]:
                for victim_vehicle_id in self.meta[scenario_id]["vehicle_ids"]:
                    if victim_vehicle_id == attacker_vehicle_id:
                        continue
                    self.attacks.append({
                        "case_id": idx,
                        "scenario_id": scenario_id,
                        "frame_ids": frame_ids,
                        "attacker_vehicle_id": attacker_vehicle_id,
                        "victim_vehicle_id": victim_vehicle_id,
                        "vehicle_ids": self.meta[scenario_id]["vehicle_ids"],
                    })

    def _build_scenario_attacks(self):
        for idx, case in enumerate(self.cases["scenario"]):
            scenario_id = case["scenario_id"]
            for attacker_vehicle_id in self.meta[scenario_id]["vehicle_ids"]:
                for victim_vehicle_id in self.meta[scenario_id]["vehicle_ids"]:
                    if victim_vehicle_id == attacker_vehicle_id:
                        continue
                    for frame_sections in range(80, len(case["frame_ids"]), 40):
                        self.scenario_attacks.append({
                            "case_id": idx,
                            "scenario_id": scenario_id,
                            "frame_ids": case["frame_ids"][frame_sections - 80: frame_sections],
                            "vehicle_ids": case["vehicle_ids"],
                            "attacker_vehicle_id": attacker_vehicle_id,
                            "victim_vehicle_id": victim_vehicle_id,
                        })

    def _get_lidar(self, scenario_id, frame_id, vehicle_id):
        file_path = os.path.join(self.root_path, self.meta[scenario_id]["data"][frame_id][vehicle_id]["lidar"])
        if self.name in ["OPV2V", "V2V4Real"]:
            pcd = read_pcd(file_path)
        elif self.name in ["V2X-Real"]:
            pcd = read_bin(file_path)
        pcd = self.preprocess_lidar(pcd)
        return pcd

    def _get_camera(self, scenario_id, frame_id, vehicle_id, camera_id=0):
        return cv2.imread(os.path.join(self.root_path, self.meta[scenario_id]["data"][frame_id][vehicle_id]["camera{}".format(camera_id)]), cv2.IMREAD_UNCHANGED)

    def _get_calib(self, scenario_id, frame_id, vehicle_id):
        calib = self.meta[scenario_id]["data"][frame_id][vehicle_id]["calib"]
        if isinstance(calib, str):
            with open(os.path.join(self.root_path, calib), 'r') as f:
                calib = yaml.load(f, Loader=yaml.Loader)
        return calib

    def _get_scenario_meta(self, scenario_id):
        meta = {}
        for k, v in self.meta[scenario_id].items():
            if k != "data" and k != "label":
                meta[k] = v
        return meta

    def preprocess_lidar(self, pcd_data):
        # chop size
        center = np.zeros(2)
        distance = np.sqrt(np.sum((pcd_data[:,:2] - center) ** 2, axis=1))
        mask_index = np.argwhere(distance ** 2 < 100 ** 2).reshape(-1)
        pcd_data = pcd_data[mask_index,:]

        # drop the points on the roof of the ego vehicle
        distance = distance[mask_index]
        rot = np.pi / 2
        projected_distance = np.absolute(np.dot(pcd_data[:,:2] - center, np.array([np.cos(rot), np.sin(rot)]).T))
        ego_mask = (projected_distance < 1.5) * (distance < 4.5)
        mask_index = np.argwhere(ego_mask == 0).reshape(-1)
        pcd_data = pcd_data[mask_index,:]

        return pcd_data

    def case_number(self, tag="multi_frame"):
        return len(self.cases[tag])

    def attack_number(self):
        return len(self.attacks)

    def get_case(self, idx, tag="multi_frame", use_lidar=True, use_camera=False, vehicle_ids=None, frame_ids=None):
        # if use_lidar or use_camera:
        #     case_key = "{}_{:06d}_{}_{}".format(tag, idx, use_lidar, use_camera)
        #     if case_key in self.cache:
        #         return self.cache[case_key]
        
        case_meta = self.cases[tag][idx]
        case = self.get_case_by_meta(case_meta, tag, use_lidar=use_lidar, use_camera=use_camera, vehicle_ids=vehicle_ids, frame_ids=frame_ids)

        # if use_lidar or use_camera:
        #     if len(self.cache) >= self.cache_size:
        #         self.cache.popitem(last=False)
        #     self.cache[case_key] = case

        return case
    
    def get_case_by_meta(self, case_meta, tag="multi_frame", use_lidar=True, use_camera=False, vehicle_ids=None, frame_ids=None, classes=['Car']):
        if tag == "single_vehicle":
            calib = self._get_calib(**case_meta)
            lidar = self._get_lidar(**case_meta) if use_lidar else None
            cameras = []
            if use_camera:
                # TODO: Only works on OPV2V
                for i in range(4):
                    cameras.append({
                        "image": self._get_camera(**case_meta, camera_id=i),
                        "cords": np.array(calib["camera{}".format(i)]["cords"]).reshape(-1),
                        "calib": {
                            "extrinsic": np.array(calib["camera{}".format(i)]["extrinsic"]).reshape(4,4),
                            "intrinsic": np.array(calib["camera{}".format(i)]["intrinsic"]).reshape(3,3)
                        }
                    })

            gt_bboxes = np.array([[*object_data["location"][:3], 
                          *((np.array(object_data["extent"])*2).tolist()), 
                          object_data["angle"][1]*np.pi/180] for object_id, object_data in calib["vehicles"].items()])
            
            lidar_pose = np.array(calib["lidar_pose"])
            if lidar_pose.shape == (4, 4):
                lidar_pose = transformation_to_pose(lidar_pose)

            if self.name == "V2X-Real":
                # V2X-Real: LiDAR bin files are in sensor frame.
                # Bboxes need world→sensor transform using V2X-Real's own convention.
                # Import V2X-Real's transformation utilities directly.
                from opencood.utils.transformation_utils import x1_to_x2
                from opencood.utils.box_utils import create_bbx, corner_to_center

                new_gt_bboxes = []
                for object_id, object_data in calib["vehicles"].items():
                    loc = object_data["location"]
                    center = object_data.get("center", [0, 0, 0])
                    rot = object_data["angle"]
                    ext = object_data["extent"]  # half-extents
                    obj_pose = [loc[0] + center[0], loc[1] + center[1],
                                loc[2] + center[2],
                                rot[0], rot[1], rot[2]]
                    obj2lidar = x1_to_x2(obj_pose, lidar_pose)
                    bbx = create_bbx(ext).T  # (3, 8)
                    bbx = np.r_[bbx, [np.ones(8)]]  # (4, 8)
                    bbx_lidar = np.dot(obj2lidar, bbx).T  # (8, 4)
                    bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)  # (1, 8, 3)
                    bbx_center = corner_to_center(bbx_lidar, order='lwh')  # (1, 7)
                    # Use world-frame dimensions to avoid rotation-inflated sizes
                    bbx_center[0, 3] = ext[0] * 2  # l
                    bbx_center[0, 4] = ext[1] * 2  # w
                    bbx_center[0, 5] = ext[2] * 2  # h
                    # Adjust z to be bottom (center_z - h/2)
                    bbx_center[0, 2] = bbx_center[0, 2] - ext[2]
                    new_gt_bboxes.append(bbx_center[0])

                gt_bboxes = np.array(new_gt_bboxes) if new_gt_bboxes else np.zeros((0, 7))

            elif "V2V4Real" not in self.root_path:
                gt_bboxes = bbox_map_to_sensor(gt_bboxes, lidar_pose)
            gt_bboxes_9d = np.zeros((gt_bboxes.shape[0], 9))
            gt_bboxes_9d[:,[0,1,2,3,4,5,7]] = gt_bboxes[:,[0,1,2,3,4,5,6]]
            gt_bboxes_9d[:,[6,8]] = np.array([[object_data["angle"][0], object_data["angle"][2]] for object_id, object_data in calib["vehicles"].items()])

            object_ids = [object_id for object_id in calib["vehicles"]]
            object_classes = ['Car' if self.name in ["OPV2V", "V2V4Real"] else calib["vehicles"][object_id]["obj_type"] for object_id in calib["vehicles"]]

            # Filtering object types.
            if classes is not None:
                select_indices =  np.asarray([idx for idx, c in enumerate(object_classes) if c in classes]).astype(np.int)
                gt_bboxes, gt_bboxes_9d = gt_bboxes[select_indices], gt_bboxes_9d[select_indices]
                object_ids = [x for idx, x in enumerate(object_ids) if idx in select_indices]
                object_classes = [x for idx, x in enumerate(object_classes) if idx in select_indices]

            result = {
                "lidar": lidar,
                "lidar_pose": lidar_pose,
                "cameras": cameras,
                "gt_bboxes": gt_bboxes,
                "gt_bboxes_9d": gt_bboxes_9d,
                "vehicle_id": case_meta["vehicle_id"],
                "object_ids": object_ids,
                "object_classes": object_classes,
                "map": scenario_maps[case_meta["scenario_id"]] if case_meta["scenario_id"] in scenario_maps else None,
                "params": calib,
                "loader": OPV2VDataset,
            }

        elif tag == "multi_vehicle":
            result = {}
            scenario_meta = self._get_scenario_meta(case_meta["scenario_id"])
            
            for vehicle_id in scenario_meta["vehicle_ids"]:
                if vehicle_ids is not None and vehicle_id not in vehicle_ids:
                    result[vehicle_id] = {}
                    continue
                new_case_meta = copy.deepcopy(case_meta)
                new_case_meta["vehicle_id"] = vehicle_id
                sub_case = self.get_case_by_meta(new_case_meta, tag="single_vehicle", use_lidar=use_lidar, use_camera=use_camera, vehicle_ids=vehicle_ids, frame_ids=frame_ids)
                result[vehicle_id] = sub_case
            
            for vehicle_id in result:
                if len(result[vehicle_id]) == 0:
                    continue
                for _, vehicle_data in result.items():
                    if len(vehicle_data) == 0:
                        continue
                    if vehicle_id in vehicle_data["object_ids"]:
                        object_index = vehicle_data["object_ids"].index(vehicle_id)
                        object_bbox = vehicle_data["gt_bboxes"][object_index]
                        object_bbox = bbox_sensor_to_map(object_bbox, vehicle_data["lidar_pose"], dataset_name=self.name)
                        result[vehicle_id]["ego_bbox"] = object_bbox
                        break
                if "ego_bbox" not in result[vehicle_id]:
                    result[vehicle_id]["ego_bbox"] = np.array([0,0,0,5,3,1.7,0])
                    result[vehicle_id]["ego_bbox"][:2] = result[vehicle_id]["lidar_pose"][:2]
                    result[vehicle_id]["ego_bbox"][6] = np.radians(result[vehicle_id]["lidar_pose"][4])
        elif tag == "multi_frame":
            result = []
            for frame_index, frame_id in enumerate(case_meta["frame_ids"]):
                if frame_ids is not None and frame_index not in frame_ids:
                    result.append({})
                new_case_meta = {
                    "scenario_id": case_meta["scenario_id"],
                    "frame_id": frame_id
                }
                sub_case = self.get_case_by_meta(new_case_meta, tag="multi_vehicle", use_lidar=use_lidar, use_camera=use_camera, vehicle_ids=vehicle_ids, frame_ids=frame_ids)
                result.append(sub_case)
        elif tag == "scenario":
            result = []
            for frame_index, frame_id in enumerate(case_meta["frame_ids"]):
                if frame_ids is not None and frame_index not in frame_ids:
                    result.append({})
                new_case_meta = {
                    "scenario_id": case_meta["scenario_id"],
                    "frame_id": frame_id
                }
                sub_case = self.get_case_by_meta(new_case_meta, tag="multi_vehicle", use_lidar=use_lidar, use_camera=use_camera, vehicle_ids=vehicle_ids, frame_ids=frame_ids)
                result.append(sub_case)
        else:
            raise NotImplementedError
        return result

    def case_generator(self, tag="multi_frame", index=False, use_lidar=True, use_camera=False):
        for idx in range(self.case_number(tag)):
            if index:
                yield idx, self.get_case(idx, tag, use_lidar=use_lidar, use_camera=use_camera)
            else:
                yield self.get_case(idx, tag, use_lidar=use_lidar, use_camera=use_camera)

    @staticmethod
    def lidar_to_camera(lidar_data, calib):
        return lidar_to_camera(lidar_data, calib["extrinsic"], calib["intrinsic"])
