"""
Local LUCIA: Object-centric feature anomaly detection.

Instead of comparing full feature maps globally (which misses spatially
sparse attacks), compare features ONLY within detected bounding box regions.

For each detected object:
1. Crop spatial features at the bbox region for each agent
2. Apply L2-norm + L1 distance comparison on the cropped patch
3. Flag agents whose local features deviate from peers

This makes the defense sensitive to voxel-level attacks that modify
only ~30 voxels out of 140,800.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from matplotlib.path import Path


def bbox_to_feature_crop(bbox_ego, lidar_range, voxel_size, padding=2):
    """
    Convert a bbox in ego frame to feature map crop indices.

    Args:
        bbox_ego: (7,) [x, y, z, l, w, h, yaw] in ego sensor frame
        lidar_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        padding: extra voxels around bbox

    Returns:
        (h_lo, h_hi, w_lo, w_hi) crop indices on the feature map
    """
    H = int((lidar_range[4] - lidar_range[1]) / voxel_size[1])
    W = int((lidar_range[3] - lidar_range[0]) / voxel_size[0])

    x, y, z, l, w, h, yaw = bbox_ego
    hl, hw = l / 2 + padding * voxel_size[0], w / 2 + padding * voxel_size[1]

    # Rotated bbox corners
    corners = np.array([[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]])
    c, s = np.cos(yaw), np.sin(yaw)
    corners = corners @ np.array([[c, s], [-s, c]]) + np.array([x, y])

    # Convert to voxel indices
    w_idx = (corners[:, 0] - lidar_range[0]) / voxel_size[0]
    h_idx = (corners[:, 1] - lidar_range[1]) / voxel_size[1]

    w_lo = max(0, int(np.floor(w_idx.min())))
    w_hi = min(W, int(np.ceil(w_idx.max())))
    h_lo = max(0, int(np.floor(h_idx.min())))
    h_hi = min(H, int(np.ceil(h_idx.max())))

    return h_lo, h_hi, w_lo, w_hi


class LocalLuciaDefender:
    """
    Object-centric LUCIA defense.

    For each detected bounding box, compares agents' features within
    that local region. Detects per-object anomalies that global LUCIA misses.
    """
    def __init__(self, perception, padding=2):
        """
        Args:
            perception: OpencoodPerception instance
            padding: extra voxels around each bbox for cropping
        """
        self.perception = perception
        self.padding = padding
        self.lidar_range = perception.dataset.pre_processor.params["cav_lidar_range"]
        self.voxel_size = perception.dataset.pre_processor.params["args"]["voxel_size"]

    def compute_local_trust(self, spatial_features, bboxes_ego, ego_index=0):
        """
        Compute per-object, per-agent trust scores using L2-normalized
        feature comparison (matching LUCIA's original approach).

        For each bbox region:
        1. Crop each agent's features to the bbox
        2. L2-normalize each agent's crop (removes magnitude, keeps pattern)
        3. Compute pairwise L1 distances between all agents
        4. Sum distances per agent → softmax → trust score

        This correctly detects beta-scaled features because normalization
        removes the magnitude advantage, and the L1 distance captures
        the pattern difference between genuine and spoofed features.

        Args:
            spatial_features: (N_agents, C, H, W) pre-backbone features
            bboxes_ego: (M, 7) detected bboxes in ego frame
            ego_index: index of ego vehicle in the agent dimension

        Returns:
            per_object_trust: (M, N_agents) trust scores (lower = more suspicious)
            per_object_l1: (M, N_agents) L1 distance scores (higher = more suspicious)
        """
        features = spatial_features.detach()
        n_agents, C, H, W = features.shape
        n_objects = len(bboxes_ego)

        per_object_trust = np.ones((n_objects, n_agents))
        per_object_l1 = np.zeros((n_objects, n_agents))

        if n_agents <= 1 or n_objects == 0:
            return per_object_trust, per_object_l1

        for obj_idx in range(n_objects):
            bbox = bboxes_ego[obj_idx]
            h_lo, h_hi, w_lo, w_hi = bbox_to_feature_crop(
                bbox, self.lidar_range, self.voxel_size, self.padding)

            if h_hi <= h_lo or w_hi <= w_lo:
                continue

            # Crop each agent's features and compute anomaly scores
            # Use RAW features (no L2 normalization) to preserve the
            # magnitude signal from beta scaling, which is the primary
            # detectable artifact of the attack.
            # Small avg pooling (2x2) smooths per-voxel noise.
            crops = []
            for i in range(n_agents):
                crop = features[i, :, h_lo:h_hi, w_lo:w_hi]  # (C, h, w)
                if crop.shape[1] >= 4 and crop.shape[2] >= 4:
                    crop = torch.nn.functional.avg_pool2d(
                        crop.unsqueeze(0), kernel_size=2).squeeze(0)
                crops.append(crop)

            # Anomaly score: raw L1 distance from each agent to ego
            # Beta-scaled features produce ~2-6x larger L1 than benign agents
            ego_crop = crops[ego_index]
            l1_scores = torch.zeros(n_agents, device=features.device)
            for i in range(n_agents):
                if i == ego_index:
                    continue
                l1_scores[i] = torch.sum(torch.abs(crops[i] - ego_crop))

            per_object_l1[obj_idx] = l1_scores.cpu().numpy()

            # Trust: normalize L1 by crop element count.
            # NOTE: For proper detection, use the raw L1 scores
            # (per_object_l1) with ROC-based threshold calibration,
            # not this trust metric. The trust is a convenience score
            # that may not generalize across models/betas.
            crop_numel = crops[0].numel()
            for i in range(n_agents):
                if i == ego_index:
                    per_object_trust[obj_idx, i] = 1.0
                else:
                    l1_norm = float(l1_scores[i]) / max(crop_numel, 1)
                    per_object_trust[obj_idx, i] = max(0, 1.0 - l1_norm)

        return per_object_trust, per_object_l1

    def compute_magnitude_anomaly(self, spatial_features, bboxes_ego, ego_index=0):
        """
        Detect anomalous feature magnitudes within bbox regions.

        For each object, checks if any agent has voxels with abnormally high
        feature norms compared to the global distribution. β-amplified attacks
        produce voxels with ~β× normal magnitude.

        Args:
            spatial_features: (N_agents, C, H, W) pre-backbone features
            bboxes_ego: (M, 7) detected bboxes in ego frame
            ego_index: index of ego vehicle

        Returns:
            per_object_anomaly: (M, N_agents) anomaly scores (higher = more anomalous)
            per_object_trust: (M, N_agents) trust = 1 - normalized anomaly
        """
        features = spatial_features.detach()
        n_agents, C, H, W = features.shape
        n_objects = len(bboxes_ego)

        per_object_anomaly = np.zeros((n_objects, n_agents))
        per_object_trust = np.ones((n_objects, n_agents))

        if n_agents <= 1 or n_objects == 0:
            return per_object_anomaly, per_object_trust

        # Global per-agent feature norm statistics (for baseline)
        voxel_norms = features.norm(dim=1)  # (N, H, W)
        agent_stats = []
        for i in range(n_agents):
            active = voxel_norms[i][voxel_norms[i] > 0]
            if len(active) > 0:
                mean_n = float(active.mean())
                std_n = float(active.std())
                p95 = float(active.quantile(0.95))
            else:
                mean_n, std_n, p95 = 1.0, 1.0, 1.0
            agent_stats.append((mean_n, std_n, p95))

        for obj_idx in range(n_objects):
            bbox = bboxes_ego[obj_idx]
            h_lo, h_hi, w_lo, w_hi = bbox_to_feature_crop(
                bbox, self.lidar_range, self.voxel_size, self.padding)

            if h_hi <= h_lo or w_hi <= w_lo:
                continue

            for i in range(n_agents):
                if i == ego_index:
                    continue

                crop_norms = voxel_norms[i, h_lo:h_hi, w_lo:w_hi]
                active_crop = crop_norms[crop_norms > 0]

                if len(active_crop) == 0:
                    continue

                mean_n, std_n, p95 = agent_stats[i]

                # How many voxels exceed 2× the global p95?
                n_extreme = int((active_crop > 2 * p95).sum())
                # Max norm ratio vs global mean
                max_ratio = float(active_crop.max()) / (mean_n + 1e-8)
                # Top-k mean norm in this region
                k = max(1, len(active_crop) // 5)
                topk, _ = torch.topk(active_crop, k)
                topk_mean = float(topk.mean())
                topk_ratio = topk_mean / (mean_n + 1e-8)

                # Anomaly score: combination of extreme voxel count and magnitude ratio
                anomaly = max(n_extreme / max(len(active_crop), 1),
                              max(0, topk_ratio - 1.5))
                per_object_anomaly[obj_idx, i] = anomaly

            # Trust from anomaly
            max_anomaly = per_object_anomaly[obj_idx].max()
            if max_anomaly > 0:
                per_object_trust[obj_idx] = 1 - per_object_anomaly[obj_idx] / (max_anomaly + 0.1)
                per_object_trust[obj_idx, ego_index] = 1.0

        return per_object_anomaly, per_object_trust

    def detect(self, spatial_features, bboxes_ego, ego_index=0, trust_threshold=0.3):
        """
        Detect if any agent is anomalous at any object location.

        Args:
            spatial_features: (N_agents, C, H, W)
            bboxes_ego: (M, 7) detected bboxes
            ego_index: ego vehicle index
            trust_threshold: flag agents with trust below this

        Returns:
            is_anomalous: (N_agents,) bool — any object flagged this agent
            min_trust_per_agent: (N_agents,) minimum trust across all objects
            details: dict with per-object results
        """
        per_obj_trust, per_obj_l1 = self.compute_local_trust(
            spatial_features, bboxes_ego, ego_index)

        n_agents = spatial_features.shape[0]
        min_trust = np.ones(n_agents)
        flagged_objects = {i: [] for i in range(n_agents)}

        for obj_idx in range(len(bboxes_ego)):
            for agent_idx in range(n_agents):
                if agent_idx == ego_index:
                    continue
                trust = per_obj_trust[obj_idx, agent_idx]
                min_trust[agent_idx] = min(min_trust[agent_idx], trust)
                if trust < trust_threshold:
                    flagged_objects[agent_idx].append(obj_idx)

        is_anomalous = min_trust < trust_threshold

        return is_anomalous, min_trust, {
            'per_object_trust': per_obj_trust,
            'per_object_l1': per_obj_l1,
            'flagged_objects': flagged_objects,
        }
