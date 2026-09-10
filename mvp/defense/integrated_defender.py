"""
Integrated defense: safety-aware anomaly detection with ego fallback.

Pipeline:
  1. Safety Estimator — identify critical objects using GT ego future trajectory
  2. Fused vs Ego filter — check if fused and ego detections disagree
  3. Anomaly Detection — run LUCIA or MADE on disagreeing critical objects
  4. Fallback — if anomalous, replace fused detection with ego detection
"""

import numpy as np
import copy
import logging
import torch

from mvp.defense.lucia import LocalLuciaDefender
from mvp.defense.made import LocalMadeDefender
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d


class TrajectoryPredictor:
    """Base class for trajectory prediction."""
    def predict(self, bboxes, observed_trajectories=None):
        raise NotImplementedError

    def predict_ego(self, ego_bbox, ego_velocity=None):
        raise NotImplementedError


class LinearPredictor(TrajectoryPredictor):
    """Simple constant-velocity linear prediction."""
    def __init__(self, predict_frames=20, time_step=0.5):
        self.predict_frames = predict_frames
        self.time_step = time_step

    def predict(self, bboxes, observed_trajectories=None):
        M = len(bboxes)
        trajs = np.zeros((M, self.predict_frames, 2))
        for i in range(M):
            vel = np.zeros(2)
            if observed_trajectories is not None and i in observed_trajectories:
                obs = observed_trajectories[i]
                if len(obs) >= 2:
                    vel = (obs[-1, :2] - obs[-2, :2]) / self.time_step
            for t in range(self.predict_frames):
                trajs[i, t] = bboxes[i, :2] + vel * (t + 1) * self.time_step
        return trajs

    def predict_ego(self, ego_bbox, ego_velocity=None):
        traj = np.zeros((self.predict_frames, 2))
        vel = ego_velocity if ego_velocity is not None else np.zeros(2)
        for t in range(self.predict_frames):
            traj[t] = ego_bbox[:2] + vel * (t + 1) * self.time_step
        return traj


class GRIPPredictor(TrajectoryPredictor):
    """GRIP trajectory prediction model wrapper."""
    def __init__(self, predict_frames=20, observe_frames=20, time_step=0.1,
                 model_path=None):
        self.predict_frames = predict_frames
        self.observe_frames = observe_frames
        self.time_step = time_step
        self.model = None

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path):
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/AdvTrajectoryPrediction"))
        try:
            from prediction.model.GRIP import GRIPInterface
            self.model = GRIPInterface(
                self.observe_frames, self.predict_frames,
                pre_load_model=model_path)
        except Exception as e:
            logging.warning(f"Failed to load GRIP model: {e}. Falling back to linear.")
            self.model = None

    def predict(self, bboxes, observed_trajectories=None):
        if self.model is None or observed_trajectories is None:
            linear = LinearPredictor(self.predict_frames, self.time_step)
            return linear.predict(bboxes, observed_trajectories)

        input_data = {
            "observe_length": self.observe_frames,
            "predict_length": self.predict_frames,
            "time_step": self.time_step,
            "feature_dimension": 5,
            "objects": {},
        }

        for i in range(len(bboxes)):
            if observed_trajectories is not None and i in observed_trajectories:
                obs = observed_trajectories[i]
            else:
                obs = bboxes[i:i+1]

            traj_len = obs.shape[0]
            if traj_len < 2:
                continue

            if traj_len > self.observe_frames:
                obs = obs[-self.observe_frames:]
                traj_len = self.observe_frames

            vdata = {
                "type": 1, "complete": True, "visible": True,
                "observe_trace": np.zeros((self.observe_frames, 2)),
                "observe_feature": np.zeros((self.observe_frames, 5)),
                "observe_mask": np.zeros(self.observe_frames),
                "future_trace": np.zeros((self.predict_frames, 2)),
                "future_feature": np.zeros((self.predict_frames, 5)),
                "predict_trace": np.zeros((self.predict_frames, 2)),
                "future_mask": np.zeros(self.predict_frames),
            }
            vdata["observe_trace"][-traj_len:] = obs[:, :2]
            if obs.shape[1] > 2:
                vdata["observe_feature"][-traj_len:] = obs[:, 2:7] if obs.shape[1] >= 7 else obs[:, 2:]
            vdata["observe_mask"] = np.sum(vdata["observe_trace"] ** 2, axis=1) > 0
            input_data["objects"][str(i)] = vdata

        if len(input_data["objects"]) == 0:
            linear = LinearPredictor(self.predict_frames, self.time_step)
            return linear.predict(bboxes, observed_trajectories)

        try:
            output_data = self.model.run(input_data)
            M = len(bboxes)
            trajs = np.zeros((M, self.predict_frames, 2))
            for obj_id, vdata in output_data["objects"].items():
                idx = int(obj_id)
                if idx < M:
                    trajs[idx] = vdata["predict_trace"][:self.predict_frames, :2]
            return trajs
        except Exception:
            linear = LinearPredictor(self.predict_frames, self.time_step)
            return linear.predict(bboxes, observed_trajectories)

    def predict_ego(self, ego_bbox, ego_velocity=None):
        linear = LinearPredictor(self.predict_frames, self.time_step)
        return linear.predict_ego(ego_bbox, ego_velocity)


class SafetyEstimator:
    """
    Module 1: Identify safety-critical objects.

    An object is critical if its predicted future trajectory comes close
    to the ego's future path. Ego future can be GT (known to ego).
    """
    def __init__(self, predictor=None, safety_threshold=5.0):
        self.predictor = predictor or LinearPredictor()
        self.safety_threshold = safety_threshold

    def min_dist(self, traj1, traj2):
        """Minimum distance between two trajectories over time."""
        n = min(len(traj1), len(traj2))
        if n == 0:
            return float('inf')
        return np.sqrt(np.min(np.sum((traj1[:n] - traj2[:n]) ** 2, axis=1)))

    def get_critical_objects(self, bboxes, ego_future_traj,
                             observed_trajectories=None):
        """
        Identify safety-critical objects.

        Args:
            bboxes: (M, 7) detected bboxes in ego frame
            ego_future_traj: (T, 2) GT ego future positions (known to ego)
            observed_trajectories: dict {obj_idx: (T, 7)} past trajectories

        Returns:
            critical_indices: list of indices into bboxes
            safety_scores: (M,) score per object (higher = more critical)
        """
        M = len(bboxes)
        if M == 0:
            return [], np.array([])

        obj_trajs = self.predictor.predict(bboxes, observed_trajectories)

        safety_scores = np.zeros(M)
        for i in range(M):
            md = self.min_dist(obj_trajs[i], ego_future_traj)
            safety_scores[i] = 1.0 / (md + 0.1)

        critical_indices = [i for i in range(M)
                           if safety_scores[i] > 1.0 / self.safety_threshold]

        return critical_indices, safety_scores


class IntegratedDefender:
    """
    Combined defense pipeline:
      1. Fused + ego perception
      2. Identify safety-critical objects (using GT ego future)
      3. Filter: check if fused and ego detections disagree on critical objects
      4. Anomaly detection on disagreeing critical objects only
      5. Fallback to ego detection if anomalous
    """
    def __init__(self, perception,
                 safety_threshold=5.0,
                 anomaly_threshold=0.5,
                 disagree_iou_threshold=0.5,
                 disagree_dist_threshold=1.0,
                 anomaly_method="local_lucia",
                 predictor=None):
        """
        Args:
            perception: OpencoodPerception instance
            safety_threshold: objects with MinDist < this are critical (meters)
            anomaly_threshold: local trust below this triggers anomaly
            disagree_iou_threshold: IoU below this = fused/ego disagree
            disagree_dist_threshold: position diff above this = disagree (meters)
            anomaly_method: "local_lucia" or "local_made"
            predictor: TrajectoryPredictor instance (default: LinearPredictor)
        """
        self.perception = perception
        self.safety_est = SafetyEstimator(predictor=predictor,
                                           safety_threshold=safety_threshold)

        if anomaly_method == "local_lucia":
            self.anomaly_detector = LocalLuciaDefender(perception, padding=2)
        elif anomaly_method == "local_made":
            self.anomaly_detector = LocalMadeDefender(perception)
        self.anomaly_method = anomaly_method
        self.anomaly_threshold = anomaly_threshold
        self.disagree_iou_threshold = disagree_iou_threshold
        self.disagree_dist_threshold = disagree_dist_threshold

    def _find_ego_match(self, bbox_fused, bboxes_ego):
        """Find closest ego detection to a fused detection."""
        best_iou = 0
        best_idx = -1
        best_dist = float('inf')
        for ei in range(len(bboxes_ego)):
            iou = iou3d(bbox_fused, bboxes_ego[ei])
            dist = np.linalg.norm(bbox_fused[:2] - bboxes_ego[ei, :2])
            if iou > best_iou:
                best_iou = iou
                best_idx = ei
                best_dist = dist
            elif best_idx < 0 and dist < best_dist:
                best_dist = dist
                best_idx = ei
        return best_idx, best_iou, best_dist

    def _check_disagreement(self, bbox_fused, bbox_ego):
        """Check if fused and ego detections disagree."""
        iou = iou3d(bbox_fused, bbox_ego)
        dist = np.linalg.norm(bbox_fused[:2] - bbox_ego[:2])
        return (iou < self.disagree_iou_threshold or
                dist > self.disagree_dist_threshold)

    def defend(self, multi_vehicle_case, ego_id, ego_future_traj=None,
               ego_velocity=None):
        """
        Run the full integrated defense.

        Args:
            multi_vehicle_case: dict of vehicle data for one frame
            ego_id: ego vehicle ID
            ego_future_traj: (T, 2) GT ego future positions. If None,
                falls back to linear prediction from current state.
            ego_velocity: (2,) ego velocity (used only if ego_future_traj is None)

        Returns:
            final_bboxes: (M, 7) defended detections
            final_scores: (M,) detection scores
            defense_info: dict with defense details
        """
        # Step 1: Fused perception
        bboxes_fused, scores_fused = self.perception.run(
            multi_vehicle_case, ego_id)

        if len(bboxes_fused) == 0:
            return bboxes_fused, scores_fused, {
                "n_critical": 0, "n_disagree": 0, "n_anomalous": 0,
                "n_fallback": 0}

        # Step 2: Ego-only perception
        ego_only_case = {ego_id: multi_vehicle_case[ego_id]}
        bboxes_ego, scores_ego = self.perception.run(ego_only_case, ego_id)

        # Ego future trajectory: use GT if provided, otherwise linear prediction
        if ego_future_traj is None:
            ego_bbox = np.array([0, 0, 0, 4.5, 1.9, 1.5, 0])
            predictor = self.safety_est.predictor
            ego_future_traj = predictor.predict_ego(ego_bbox, ego_velocity)

        # Step 3: Identify safety-critical objects
        critical_indices, safety_scores = \
            self.safety_est.get_critical_objects(
                bboxes_fused, ego_future_traj)

        # Step 4: Filter — check fused vs ego disagreement on critical objects
        # Only objects where fused and ego disagree need anomaly checking
        disagree_indices = []   # critical objects with fused/ego disagreement
        ego_matches = {}        # obj_idx -> (ego_idx, iou, dist)

        for obj_idx in critical_indices:
            ego_idx, iou, dist = self._find_ego_match(
                bboxes_fused[obj_idx], bboxes_ego)

            if ego_idx < 0:
                # Ego can't see this object — fused-only detection, suspicious
                disagree_indices.append(obj_idx)
                ego_matches[obj_idx] = (ego_idx, 0, float('inf'))
                continue

            ego_matches[obj_idx] = (ego_idx, iou, dist)

            if self._check_disagreement(bboxes_fused[obj_idx],
                                         bboxes_ego[ego_idx]):
                disagree_indices.append(obj_idx)

        # Step 5: Anomaly detection on disagreeing critical objects only
        anomaly_scores = {}
        anomalous_indices = []

        if len(disagree_indices) > 0:
            from mvp.util import set_seed
            from opencood.tools import train_utils

            if self.anomaly_method == "local_lucia":
                set_seed(42, set_python=False, set_numpy=False, set_torch=True)
                batch = self.perception.preprocessors[self.perception.fusion_method](
                    multi_vehicle_case, ego_id)
                batch_data = self.perception.dataset.collate_batch_test([batch])
                batch_data = train_utils.to_device(batch_data, self.perception.device)
                bd = {k: batch_data['ego']['processed_lidar'][k]
                      for k in ['voxel_features', 'voxel_coords', 'voxel_num_points']}
                bd['record_len'] = batch_data['ego']['record_len']
                self.perception.model.pillar_vfe(bd)
                self.perception.model.scatter(bd)
                spatial_features = bd['spatial_features'].detach()

                disagree_bboxes = bboxes_fused[disagree_indices]

                # Use magnitude-based anomaly detection (catches β-amplified attacks)
                per_obj_anomaly, per_obj_trust = \
                    self.anomaly_detector.compute_magnitude_anomaly(
                        spatial_features, disagree_bboxes, ego_index=0)

                for di_idx, obj_idx in enumerate(disagree_indices):
                    # Max anomaly across non-ego agents
                    max_anomaly = max(per_obj_anomaly[di_idx, j]
                                      for j in range(per_obj_anomaly.shape[1])
                                      if j != 0)
                    anomaly_scores[obj_idx] = max_anomaly

                    if max_anomaly > self.anomaly_threshold:
                        anomalous_indices.append(obj_idx)

                del spatial_features, bd, batch_data
                torch.cuda.empty_cache()

            elif self.anomaly_method == "local_made":
                # MADE: autoencoder reconstruction loss
                set_seed(42, set_python=False, set_numpy=False, set_torch=True)
                batch = self.perception.preprocessors[self.perception.fusion_method](
                    multi_vehicle_case, ego_id)
                batch_data = self.perception.dataset.collate_batch_test([batch])
                batch_data = train_utils.to_device(batch_data, self.perception.device)
                bd = {k: batch_data['ego']['processed_lidar'][k]
                      for k in ['voxel_features', 'voxel_coords', 'voxel_num_points']}
                bd['record_len'] = batch_data['ego']['record_len']
                self.perception.model.pillar_vfe(bd)
                self.perception.model.scatter(bd)
                spatial_features = bd['spatial_features'].detach()

                disagree_bboxes = bboxes_fused[disagree_indices]
                per_obj_scores = self.anomaly_detector.compute_local_anomaly(
                    spatial_features, disagree_bboxes, ego_index=0)

                for di_idx, obj_idx in enumerate(disagree_indices):
                    anomaly_scores[obj_idx] = per_obj_scores[di_idx]
                    if per_obj_scores[di_idx] > self.anomaly_threshold:
                        anomalous_indices.append(obj_idx)

                del spatial_features, bd, batch_data
                torch.cuda.empty_cache()

        # Step 6: Fallback — replace anomalous detections with ego detections
        final_bboxes = bboxes_fused.copy()
        final_scores = scores_fused.copy()
        fallback_indices = []

        for obj_idx in anomalous_indices:
            ego_idx, _, _ = ego_matches.get(obj_idx, (-1, 0, float('inf')))

            if ego_idx >= 0:
                # Replace with ego detection
                final_bboxes[obj_idx] = bboxes_ego[ego_idx]
                final_scores[obj_idx] = scores_ego[ego_idx]
                fallback_indices.append(obj_idx)
            else:
                # Ego can't see this object — remove it (set score to 0)
                final_scores[obj_idx] = 0
                fallback_indices.append(obj_idx)

        return final_bboxes, final_scores, {
            "n_objects": len(bboxes_fused),
            "n_critical": len(critical_indices),
            "critical_indices": critical_indices,
            "safety_scores": safety_scores,
            "n_disagree": len(disagree_indices),
            "disagree_indices": disagree_indices,
            "anomaly_scores": anomaly_scores,
            "n_anomalous": len(anomalous_indices),
            "anomalous_indices": anomalous_indices,
            "n_fallback": len(fallback_indices),
            "fallback_indices": fallback_indices,
            "bboxes_fused": bboxes_fused,
            "bboxes_ego": bboxes_ego,
            "ego_matches": ego_matches,
        }
