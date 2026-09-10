import numpy as np
import logging
from shapely.ops import unary_union

from .scenario_attacker_util import *
from mvp.attack.lidar_shift_late_attacker import LidarShiftLateAttacker
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor, get_distance
from mvp.config import model_root, third_party_root

import os, sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../third_party/AdvTrajectoryPrediction")
sys.path.append(root)


class ScenarioAttacker:
    def __init__(self,
                 dataset_name = "OPV2V",
                 attack_num_frames = 3,
                 history_num_frames = 20,
                 predict_num_frames = 20,
                 detection_func=perception_opencood,
                 tracking_func=tracking_ab3dmot,
                 prediction_func=prediction_grip,
                 perception_attacker=None,
                 occupancy_func=None,
                 perturbation_type="location",
                 optimization_type="grad",
                 attack_type="blackbox"):
        super().__init__()
        self.attack_num_frames = attack_num_frames
        self.history_num_frames = history_num_frames
        self.predict_num_frames = predict_num_frames
        self.total_num_frames = self.history_num_frames + self.attack_num_frames + self.predict_num_frames
        self.attack_start_frame_id = self.history_num_frames
        self.attack_end_frame_id = self.attack_start_frame_id + self.attack_num_frames - 1
        self.attack_frame_ids = [i for i in range(self.attack_start_frame_id, self.attack_end_frame_id + 1)]

        self.detection_func = detection_func
        self.tracking_func = tracking_func
        self.prediction_func = prediction_func
        self.occupancy_func = occupancy_func
        self.perturbation_type = perturbation_type
        self.optimization_type = optimization_type
        self.attack_type = attack_type

        from mvp.perception.opencood_perception import OpencoodPerception
        self.perception_model_api = OpencoodPerception(fusion_method="late", model_name="pointpillar", dataset_name=dataset_name)
        self.perception_attacker = perception_attacker if perception_attacker is not None else LidarShiftLateAttacker(self.perception_model_api)
        from prediction.model.GRIP import GRIPInterface
        self.prediction_model_api = GRIPInterface(
            history_num_frames, predict_num_frames,
            pre_load_model=os.path.join(model_root, f"GRIP/{dataset_name}/checkpoint.pt"))
        from mvp.tools.object_tracking import Ab3dmotTracker
        self.init_tracks_func = Ab3dmotTracker

        self.device = "cuda:0"

    def get_gt_trajectories(self, scenario_case, observer_id, frame_ids=None):
        if frame_ids is None:
            frame_ids = [i for i in range(len(scenario_case))]

        num_frames = len(frame_ids)
        unique_object_ids = list(set(sum([scenario_case[frame_id][observer_id]["object_ids"] for frame_id in frame_ids], [])))
        result = {object_id: np.zeros((num_frames, 7)) for object_id in unique_object_ids + [observer_id]}

        for frame_id in frame_ids:
            vehicle_data = scenario_case[frame_id][observer_id]
            result[observer_id][frame_id] = vehicle_data["ego_bbox"]
            for object_index, object_id in enumerate(vehicle_data["object_ids"]):
                result[object_id][frame_id] = bbox_sensor_to_map(vehicle_data["gt_bboxes"][object_index], scenario_case[frame_id][observer_id]["lidar_pose"])

        return result

    def get_gt_visibility(self, scenario_case, observer_ids, frame_ids=None):
        if frame_ids is None:
            frame_ids = [i for i in range(len(scenario_case))]

        result = []

        for frame_id in frame_ids:
            frame_data = scenario_case[frame_id]
            observable_areas = []
            for vehicle_id in observer_ids:
                observable_areas += frame_data[vehicle_id]["occupied_areas"]
                observable_areas += frame_data[vehicle_id]["free_areas"]
            observable_area = unary_union(observable_areas)
            result.append(observable_area)
        
        return result

    def detection_gt(self, case, frame_id, vehicle_id):
        bboxes = bbox_sensor_to_map(case[frame_id][vehicle_id]["gt_bboxes"], case[frame_id][vehicle_id]["lidar_pose"])
        case[frame_id][vehicle_id]["detections"] = bboxes

    def detection(self, case, frame_id, vehicle_id):
        pred_bboxes = self.detection_func(case[frame_id], vehicle_id, {"model_api": self.perception_model_api})
        detections = bbox_sensor_to_map(pred_bboxes, case[frame_id][vehicle_id]["lidar_pose"])
        case[frame_id][vehicle_id]["detections"] = detections

    def tracking_gt(self, case, frame_id, vehicle_id):
        if frame_id > 0:
            tracks = copy.deepcopy(case[frame_id - 1][vehicle_id]["tracks"])
        else:
            tracks = {}

        detections = case[frame_id][vehicle_id]["detections"]
        object_ids = case[frame_id][vehicle_id]["object_ids"]

        for object_index, object_id in enumerate(object_ids):
            if object_id not in tracks:
                tracks[object_id] = [detections[object_index]]
            else:
                tracks[object_id].append(detections[object_index])
        for object_id in tracks:
            if object_id not in object_ids:
                tracks[object_id].append(np.zeros(7))

        trajectories = {}
        all_trajectories = {}
        for object_id, track in tracks.items():
            traj = np.asarray(track)
            if traj.shape[0] < frame_id + 1:
                full_traj = traj = np.vstack([np.zeros((frame_id + 1 - traj.shape[0], 7)), traj])
            else:
                full_traj = traj.copy()
            trajectories[object_id] = traj
            all_trajectories[object_id] = full_traj

        case[frame_id][vehicle_id]["tracks"] = tracks
        case[frame_id][vehicle_id]["observed_trajectories"] = all_trajectories
        case[frame_id][vehicle_id]["trajectories"] = all_trajectories

    def tracking(self, case, frame_id, vehicle_id):
        if frame_id == 0 or "tracks" not in case[frame_id - 1][vehicle_id]:
            tracks = self.init_tracks_func()
        else:
            tracks = copy.deepcopy(case[frame_id - 1][vehicle_id]["tracks"])
        detections = case[frame_id][vehicle_id]["detections"]
        tracks, indexed_detections = self.tracking_func(tracks, frame_id * 0.1, detections)
        case[frame_id][vehicle_id]["tracks"] = tracks
        trajectories = {}
        for object_id, bbox in indexed_detections.items():
            trajectories[object_id] = bbox[np.newaxis, :]
        trajectories[vehicle_id] = case[frame_id][vehicle_id]["ego_bbox"][np.newaxis, :]
        if frame_id == 0 or "observed_trajectories" not in case[frame_id - 1][vehicle_id]:
            pass
        else:
            prev_trajectories = case[frame_id - 1][vehicle_id]["observed_trajectories"]
            for object_id, trajectory in prev_trajectories.items():
                if object_id in trajectories:
                    trajectories[object_id] = np.vstack([trajectory, trajectories[object_id]])
                else:
                    trajectories[object_id] = np.vstack([trajectory, np.zeros((1, trajectory.shape[1]))])
            for object_id, trajectory in trajectories.items():
                if object_id not in prev_trajectories and frame_id > 1:
                    trajectories[object_id] = np.vstack([np.zeros((frame_id - 1, trajectory.shape[1])), trajectory])
        case[frame_id][vehicle_id]["observed_trajectories"] = trajectories
        case[frame_id][vehicle_id]["trajectories"] = trajectories

    def prediction_gt(self, case, frame_id, vehicle_id):
        observed_trajectories = case[frame_id][vehicle_id]["observed_trajectories"]
        predicted_trajectories = {object_id: [] for object_id in observed_trajectories}
        
        for i in range(self.predict_num_frames):
            fid = frame_id + 1 + i
            if fid >= len(case):
                break
            object_ids = case[fid][vehicle_id]["object_ids"]
            detections = bbox_sensor_to_map(case[fid][vehicle_id]["gt_bboxes"], case[fid][vehicle_id]["lidar_pose"])
            for object_index, object_id in enumerate(object_ids):
                if object_id in predicted_trajectories:
                    predicted_trajectories[object_id].append(detections[object_index])
            for object_id in predicted_trajectories:
                if object_id not in object_ids:
                    predicted_trajectories[object_id].append(np.zeros(7))
        
        for object_id in predicted_trajectories:
            predicted_trajectories[object_id] = np.asarray(predicted_trajectories[object_id])
        case[frame_id][vehicle_id]["predicted_trajectories"] = predicted_trajectories

    def prediction(self, case, frame_id, vehicle_id):
        observed_trajectories = case[frame_id][vehicle_id]["observed_trajectories"]
        predicted_trajectories = self.prediction_func(
            observed_trajectories,
            model_args={"model_api": self.prediction_model_api,
                        "obs_length": self.history_num_frames,
                        "pred_length": self.predict_num_frames},
            num_frames=self.predict_num_frames)
        case[frame_id][vehicle_id]["predicted_trajectories"] = predicted_trajectories
        return case

    def occupancy_gt(self, case, frame_id, vehicle_id):
        # Assume the offline generated occupancy maps are loaded in keys "occupied_areas", "free_areas", "ego_area".
        pass

    def occupancy(self, case, frame_id, vehicle_id):
        result = self.occupancy_func(case[frame_id][vehicle_id])
        case[frame_id][vehicle_id].update(result)

    def attack_fitness(self, case, attack_opts, frame_id=None):
        return NotImplementedError()
    
    def attack_fitness_simplified(self, case, attack_opts, frame_id=None):
        return NotImplementedError()

    def attack_constraint(self, case, attack_opts, frame_id=None):
        return NotImplementedError()
    
    def attack_projection(self, case, attack_opts, frame_id=None):
        initial_perturbation = attack_opts["perturbation"].copy()
        p = 1.0
        if frame_id is None:
            frame_index = 0
        else:
            frame_index = frame_id - self.attack_start_frame_id
        while not self.attack_constraint(case, attack_opts) and p > 0:
            p -= 0.02
            attack_opts["perturbation"][frame_index:] = initial_perturbation[frame_index:] * p
    
    def attack_projected_fitness(self, case, attack_opts, frame_id=None, simple=False):
        if not simple:
            self.attack_projection(case, attack_opts, frame_id)
            return self.attack_fitness(case, attack_opts, frame_id)
        else:
            return self.attack_fitness_simplified(case, attack_opts, frame_id)
    
    def step(self, case, frame_id, vehicle_id, gt=True):
        if gt:
            self.detection_gt(case, frame_id, vehicle_id)
            self.tracking_gt(case, frame_id, vehicle_id)
            if frame_id <= self.attack_end_frame_id:
                self.prediction_gt(case, frame_id, vehicle_id)
            # self.occupancy_gt(case, frame_id, vehicle_id)
        else:
            if "detections" not in case[frame_id][vehicle_id]:
                self.detection(case, frame_id, vehicle_id)
            if "observed_trajectories" not in case[frame_id][vehicle_id]:
                self.tracking(case, frame_id, vehicle_id)
            else:
                case[frame_id][vehicle_id]["trajectories"] = case[frame_id][vehicle_id]["observed_trajectories"].copy()
            if "predicted_trajectories" not in case[frame_id][vehicle_id]:
                self.prediction(case, frame_id, vehicle_id)
            # self.occupancy(case, frame_id, vehicle_id)

    def preprocess(self, case, attack_opts):
        attacker_vehicle_id = attack_opts["attacker_vehicle_id"]
        victim_vehicle_id = attack_opts["victim_vehicle_id"]
        gt = attack_opts["gt"]

        # Preparation phase — only iterate over frames that exist in the case
        n_available = min(self.total_num_frames, len(case))
        for frame_id in range(n_available):
            self.step(case, frame_id, attacker_vehicle_id, gt)
            self.step(case, frame_id, victim_vehicle_id, gt)

        return case

    def init_seeds(self, case, attack_opts):
        return NotImplementedError()

    def select_seeds(self, case, attack_opts, seeds, k=1):
        fitness_scores = np.zeros(len(seeds))
        logging.info("Total seed count {}".format(len(seeds)))
        for idx, seed in enumerate(seeds):
            fitness_scores[idx] = self.attack_projected_fitness(case, seed, simple=True)
        sorted_indices = np.argsort(-fitness_scores)
        result = []
        for idx in sorted_indices[:k]:
            result.append(seeds[idx])
        return result
    
    def get_track_id(self, case, vehicle_id, target_id):
        perception_frame = case[self.attack_end_frame_id][vehicle_id]
        if target_id in perception_frame.get("object_ids", []):
            gt_location = perception_frame["gt_bboxes"][perception_frame["object_ids"].index(target_id)]
            gt_location = bbox_sensor_to_map(gt_location, perception_frame["lidar_pose"])[:2]
        elif target_id in case[self.attack_end_frame_id]:
            gt_location = np.array(case[self.attack_end_frame_id][target_id]["lidar_pose"][:2])
        else:
            raise ValueError(f"target_id {target_id} not found as object or vehicle")
        trajectories = case[self.attack_end_frame_id][vehicle_id]["observed_trajectories"]
        track_ids = list(trajectories.keys())
        real_locations = np.asarray([trajectories[i][-1, :2] for i in track_ids])
        track_index = int(np.argmin(get_distance(real_locations, gt_location)))
        return track_ids[track_index]

    def init_seed_default(self, case, attack_opts):
        if self.perturbation_type == "location":
            init_perturbation = np.zeros((self.attack_num_frames, 2))
        elif self.perturbation_type == "control":
            init_perturbation = np.hstack((np.random.random((self.attack_num_frames, 1)) * 1 - 0.5,
                                        np.random.random((self.attack_num_frames, 1)) * 0.04 - 0.02))
        else:
            raise NotImplementedError()

        if "gt" in attack_opts and attack_opts["gt"]:
            victim_target_track_id = attack_opts["target_id"]
            attacker_target_track_id = attack_opts["target_id"]
            attacker_victim_track_id = attack_opts["victim_vehicle_id"]
        else:
            victim_target_track_id = self.get_track_id(case, attack_opts["victim_vehicle_id"], attack_opts["target_id"])
            attacker_target_track_id = self.get_track_id(case, attack_opts["attacker_vehicle_id"], attack_opts["target_id"])
            attacker_victim_track_id = self.get_track_id(case, attack_opts["attacker_vehicle_id"], attack_opts["victim_vehicle_id"])

        seed = copy.deepcopy(attack_opts)
        seed.update({
            "victim_target_track_id": victim_target_track_id,
            "attacker_target_track_id": attacker_target_track_id,
            "attacker_victim_track_id": attacker_victim_track_id,
            "perturbation": init_perturbation,
            "ideal_predicted_trajectories": {},
            "real_predicted_trajectories": {},
            "target_trajectory": None,
            "real_target_trajectory": np.zeros((self.attack_num_frames, 7)),
        })
        return seed

    def apply_perturbation(self, case, attack_opts, requires_grad=False):
        attacker_vehicle_id, victim_vehicle_id = attack_opts["attacker_vehicle_id"], attack_opts["victim_vehicle_id"]
        target_id, target_perturbation = attack_opts["attacker_target_track_id"], attack_opts["perturbation"]

        if self.perturbation_type == "location":
            target_trajectory = case[self.attack_end_frame_id][attacker_vehicle_id]["observed_trajectories"][target_id].copy()
            target_trajectory = get_complete_trajectory(target_trajectory)
            target_trajectory = target_trajectory[-self.attack_num_frames:]
            if requires_grad:
                target_trajectory = torch.from_numpy(target_trajectory).to(self.device)
            target_trajectory[:, :2] = target_trajectory[:, :2] + target_perturbation
        elif self.perturbation_type == "control":
            target_trajectory = case[self.attack_end_frame_id + 3][attacker_vehicle_id]["observed_trajectories"][target_id].copy()
            target_trajectory = get_complete_trajectory(target_trajectory)
            target_trajectory_length = target_trajectory.shape[0]
            assert(target_trajectory_length >= self.attack_num_frames + 6)

            ref_bbox = target_trajectory[-self.attack_num_frames - 4]
            z, l, w, h = ref_bbox[2], ref_bbox[3], ref_bbox[4], ref_bbox[5]
            x, y, yaw = ref_bbox[0], ref_bbox[1], ref_bbox[6]
            v_list, t_list, s_list = self.dynamic_model.fit(target_trajectory)
            v, t_list, s_list = v_list[-self.attack_num_frames - 4], t_list[-self.attack_num_frames - 4:], s_list[-self.attack_num_frames - 4:]
            v_list =  v_list[-self.attack_num_frames - 4:]
            traj = target_trajectory[-self.attack_num_frames - 4:]
            if requires_grad:
                t_list, s_list = torch.from_numpy(t_list).to(self.device), torch.from_numpy(s_list).to(self.device)
                tmp = torch.from_numpy(np.asarray([x, y, z, l, w, h, yaw, v])).to(self.device)
                x, y, z, l, w, h, yaw, v = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]

            target_trajectory = []
            for i in range(self.attack_num_frames):
                if requires_grad:
                    new_x, new_y, new_yaw, new_v, _, _, _ = self.dynamic_model.step_torch(
                        x, y, yaw, v, t_list[i] + target_perturbation[i, 0], s_list[i] + target_perturbation[i, 1]
                    )
                else:
                    new_x, new_y, new_yaw, new_v, _, _, _ = self.dynamic_model.step(
                        x, y, yaw, v, t_list[i] + target_perturbation[i, 0], s_list[i] + target_perturbation[i, 1]
                    )
                x, y, yaw, v = new_x, new_y, new_yaw, new_v
                target_trajectory.append([x, y, z, l, w, h, yaw])
            
            if requires_grad:
                target_trajectory = torch.stack([torch.stack(target_trajectory[ts]) for ts in range(len(target_trajectory))])
            else:
                target_trajectory = np.array(target_trajectory)
        else:
            raise NotImplementedError()

        attack_opts["target_trajectory"] = target_trajectory

    def perception_attack(self, case, attack_opts, frame_id):
        attacker_vehicle_id, victim_vehicle_id = attack_opts["attacker_vehicle_id"], attack_opts["victim_vehicle_id"]
        target_track_id, target_trajectory = attack_opts["attacker_target_track_id"], attack_opts["target_trajectory"]
        attacker_lidar_pose = case[frame_id][attacker_vehicle_id]["lidar_pose"]
        bbox_to_remove = bbox_map_to_sensor(case[frame_id][attacker_vehicle_id]["observed_trajectories"][target_track_id][-1], attacker_lidar_pose)
        bbox_to_spoof = bbox_map_to_sensor(target_trajectory[frame_id - self.attack_start_frame_id], attacker_lidar_pose)
        result = self.perception_attacker.run_multi_vehicle(case[frame_id], {
            "attacker_vehicle_id": attacker_vehicle_id,
            "victim_vehicle_id": victim_vehicle_id,
            "bbox_to_remove": bbox_to_remove,
            "bbox_to_spoof": bbox_to_spoof,
        })
        detections = bbox_sensor_to_map(result["pred_bboxes"], case[frame_id][victim_vehicle_id]["lidar_pose"])
        case[frame_id][victim_vehicle_id]["detections"] = detections
        return detections

    def optimize(self, case, attack_opts, frame_id, **kwargs):
        raise NotImplementedError()

    def attack(self, case, attack_opts):
        # Initialize the seeds.
        if "init_seeds" in attack_opts and attack_opts["init_seeds"]:
            seeds = self.init_seeds(case, attack_opts)
            seeds = self.select_seeds(case, attack_opts, seeds)
            self.seeds = seeds
            seed = seeds[0]
        else:
            seed = self.init_seed_default(case, attack_opts)

        logging.info(f"Initial attack_opts: {seed}")
        attack_opts_best = copy.deepcopy(seed)
        victim_vehicle_id, victim_target_track_id = attack_opts_best["victim_vehicle_id"], attack_opts_best["victim_target_track_id"]
        case_update = [{key: {} for key, _ in case[frame_id].items()} for frame_id in range(len(case))]
        case_copy = copy.deepcopy(case)
        for frame_id in self.attack_frame_ids:
            attack_opts_best = self.optimize(case_copy, attack_opts_best, frame_id)
            self.perception_attack(case_copy, attack_opts_best, frame_id)
            self.tracking(case_copy, frame_id, victim_vehicle_id)
            self.prediction(case_copy, frame_id, victim_vehicle_id)

            case_update[frame_id][victim_vehicle_id]["detections"] = case_copy[frame_id][victim_vehicle_id]["detections"]
            case_update[frame_id][victim_vehicle_id]["observed_trajectories"] = case_copy[frame_id][victim_vehicle_id]["observed_trajectories"]
            case_update[frame_id][victim_vehicle_id]["predicted_trajectories"] = case_copy[frame_id][victim_vehicle_id]["predicted_trajectories"]

        # Populate real_target_trajectory and real_predicted_trajectories
        try:
            obs_traj = case_copy[self.attack_end_frame_id][victim_vehicle_id]["observed_trajectories"]
            if victim_target_track_id in obs_traj:
                traj = obs_traj[victim_target_track_id]
                attack_opts_best["real_target_trajectory"] = traj[-self.attack_num_frames:]

            for frame_id in self.attack_frame_ids:
                pred_traj = case_copy[frame_id][victim_vehicle_id].get("predicted_trajectories", {})
                if victim_target_track_id in pred_traj:
                    attack_opts_best["real_predicted_trajectories"][frame_id] = pred_traj[victim_target_track_id]
        except (KeyError, IndexError):
            pass

        return {
            "update": case_update,
            "attack_opts": attack_opts_best,
        }

    def visualize(self, case, case_update, attack_opts, show=False, save=None):
        attacker_vehicle_id, victim_vehicle_id = attack_opts["attacker_vehicle_id"], attack_opts["victim_vehicle_id"]
        target_id, target_trajectory = attack_opts["target_id"], attack_opts["target_trajectory"]
        victim_target_track_id = attack_opts["victim_target_track_id"]
        gt_trajectories = self.get_gt_trajectories(case, attacker_vehicle_id, frame_ids=[i for i in range(self.total_num_frames)])
        observed_trajectories = case[self.attack_end_frame_id][victim_vehicle_id]["observed_trajectories"]

        import matplotlib.pyplot as plt
        from mvp.visualize.general import draw_trajectories, set_equal_axis_scale, show_or_save
        fig, ax = plt.subplots(figsize=(40,40))
        set_equal_axis_scale(ax)

        # Original prediction
        predicted_trajectory = case[self.attack_end_frame_id][victim_vehicle_id]["predicted_trajectories"][victim_target_track_id]
        draw_trajectories(ax, predicted_trajectory, color='y', alpha=0.5)

        # Ideal prediction
        predicted_trajectory = attack_opts["ideal_predicted_trajectories"][self.attack_end_frame_id]
        draw_trajectories(ax, predicted_trajectory, color='y', alpha=0.7)

        # Final malicious prediction
        predicted_trajectory = case_update[self.attack_end_frame_id][victim_vehicle_id]["predicted_trajectories"][victim_target_track_id]
        draw_trajectories(ax, predicted_trajectory, color='y', alpha=1.0)

        # Temporary: assuming the length of target trajectory is not larger than observation length
        target_trajectory = case_update[self.attack_end_frame_id][victim_vehicle_id]["observed_trajectories"][victim_target_track_id][-20:,:]

        for object_id, traj in observed_trajectories.items():
            filter_indices = np.argwhere(traj[:, 3] > 0).reshape(-1)
            if len(filter_indices) == 0:
                continue
            traj = traj[filter_indices]
            traj_id = "object{}, frame {}".format(object_id, filter_indices[0])
            if object_id == victim_target_track_id:
                draw_trajectories(ax, np.vstack((traj[:self.history_num_frames], target_trajectory[0][np.newaxis, :])), traj_id, color='b', alpha=0.3)
                draw_trajectories(ax, np.vstack((target_trajectory[-1][np.newaxis, :], traj[self.history_num_frames + self.attack_num_frames - 1:])), color='b', alpha=0.3)
                draw_trajectories(ax, target_trajectory, color='b')
            else:
                pass

        for object_id, traj in gt_trajectories.items():
            filter_indices = np.argwhere(traj[:, 3] > 0).reshape(-1)
            if len(filter_indices) == 0:
                continue
            traj = traj[filter_indices]
            traj_id = "object{}, frame {}".format(object_id, filter_indices[0])
            if object_id == victim_vehicle_id:
                draw_trajectories(ax, traj, traj_id, color='g', alpha=0.3)
                draw_trajectories(ax, traj[self.attack_frame_ids], color='g')
            elif object_id == attacker_vehicle_id:
                draw_trajectories(ax, traj, traj_id, color='r', alpha=0.3)
                draw_trajectories(ax, traj[self.attack_frame_ids], color='r')
            else:
                pass

        show_or_save(show=show, save=save)