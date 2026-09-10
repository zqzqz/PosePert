"""
Local MADE: Object-centric match cost defense.

Instead of comparing all detections globally, compares how each
agent affects detection of SPECIFIC objects. For each detected object:
1. Run perception with all agents → get object's bbox
2. Run perception without agent i → get object's bbox
3. Compare IoU: low IoU = agent i strongly influences this object

If an agent disproportionately affects a specific object's detection,
it may be manipulating that object's features.
"""

import numpy as np
import copy
import logging

from mvp.tools.iou import iou3d


class LocalMadeDefender:
    """
    Object-centric MADE defense.

    For each detected object, measures how much each agent influences
    that specific detection via IoU comparison.
    """
    def __init__(self, perception):
        self.perception = perception

    def compute_local_influence(self, multi_vehicle_case, ego_id):
        """
        Compute per-object, per-agent influence scores.

        For each agent i and each detection d:
          influence[d, i] = 1 - IoU(bbox_d_with_all, bbox_d_without_i)

        High influence = agent strongly changes this detection.

        Args:
            multi_vehicle_case: dict of vehicle data for one frame
            ego_id: ego vehicle ID

        Returns:
            influence: dict mapping agent_id → list of (obj_iou_change, bbox_all, bbox_without)
            detections_all: (M, 7) all detections
        """
        # Full perception
        pb_all, ps_all = self.perception.run(multi_vehicle_case, ego_id)

        vehicle_ids = list(multi_vehicle_case.keys())
        influence = {}

        for vid in vehicle_ids:
            if vid == ego_id:
                influence[vid] = [(0.0, None, None)] * len(pb_all)
                continue

            # Perception without this agent
            case_without = {k: v for k, v in multi_vehicle_case.items() if k != vid}
            if len(case_without) < 1:
                influence[vid] = [(0.0, None, None)] * len(pb_all)
                continue

            try:
                pb_without, ps_without = self.perception.run(case_without, ego_id)
            except Exception:
                influence[vid] = [(0.0, None, None)] * len(pb_all)
                continue

            # For each detection in pb_all, find best match in pb_without
            per_obj = []
            for d_idx in range(len(pb_all)):
                best_iou = 0.0
                best_bbox = None
                for w_idx in range(len(pb_without)):
                    iou = iou3d(pb_all[d_idx], pb_without[w_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_bbox = pb_without[w_idx]

                # Influence = 1 - best_iou (how much the detection changes)
                change = 1.0 - best_iou
                per_obj.append((change, pb_all[d_idx], best_bbox))

            influence[vid] = per_obj

        return influence, pb_all

    def detect_for_target(self, multi_vehicle_case, ego_id, target_bbox,
                           iou_threshold=0.3):
        """
        Check if any agent suspiciously influences the detection nearest
        to target_bbox.

        Args:
            multi_vehicle_case: vehicle data
            ego_id: ego vehicle ID
            target_bbox: (7,) bbox to check (in ego's sensor frame)
            iou_threshold: minimum IoU to match a detection to target

        Returns:
            agent_influence: dict mapping agent_id → influence score for the target object
            target_det_idx: index of the detection matching the target (-1 if no match)
        """
        influence, pb_all = self.compute_local_influence(multi_vehicle_case, ego_id)

        # Find detection closest to target_bbox
        target_det_idx = -1
        best_iou = 0.0
        for i in range(len(pb_all)):
            iou = iou3d(pb_all[i], target_bbox)
            if iou > best_iou:
                best_iou = iou
                target_det_idx = i

        if best_iou < iou_threshold:
            target_det_idx = -1

        # Extract influence for target detection
        agent_influence = {}
        for vid, per_obj in influence.items():
            if target_det_idx >= 0 and target_det_idx < len(per_obj):
                agent_influence[vid] = per_obj[target_det_idx][0]
            else:
                agent_influence[vid] = 0.0

        return agent_influence, target_det_idx, pb_all
