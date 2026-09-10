"""
SORT-3D: Simple Online Realtime 3D Tracking.

3D adaptation of the SORT algorithm (Bewley et al., 2016).
Uses 3D Kalman filter with IoU-based association.

Compatible with the Ab3dmotTracker interface used in the scenario attacker.
"""

import numpy as np
from .kalman3d import KalmanBox3D
from .matching import associate_iou


class Sort3DTracker:
    """
    SORT-based 3D multi-object tracker.

    Interface matches Ab3dmotTracker:
        tracker = Sort3DTracker()
        bboxes, bbox_ids, bbox_features = tracker.update(t, detections, info)
    """

    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.1):
        """
        Args:
            max_age: max frames to keep track alive without detection
            min_hits: min hits before track is reported
            iou_threshold: minimum IoU for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.n_tracks_total = 0
        self.n_tracks_last = 0
        self.iframe = 0
        self.reference = None
        KalmanBox3D.count = 0

    @property
    def n_tracks(self):
        return self.n_tracks_total

    @property
    def n_tracks_active(self):
        return self.n_tracks_last

    @property
    def tracks(self):
        return self.trackers

    def update(self, t, detections, info):
        """
        Update tracker with new detections.

        Args:
            t: timestamp (unused, kept for interface compatibility)
            detections: (N, 7) array [x, y, z, l, w, h, theta]
            info: (N, 1) array of confidence scores (unused)

        Returns:
            bboxes: (M, 7) tracked bounding boxes
            bbox_ids: (M,) track IDs
            bbox_features: (M, 3) velocity features [vx, vy, vz]
        """
        self.iframe += 1

        # Predict existing tracks
        predicted = np.zeros((len(self.trackers), 7))
        for i, trk in enumerate(self.trackers):
            predicted[i] = trk.predict()

        # Associate detections to trackers
        matches, unmatched_dets, unmatched_trks = associate_iou(
            detections, predicted, self.iou_threshold)

        # Update matched tracks
        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBox3D(detections[det_idx])
            self.trackers.append(trk)
            self.n_tracks_total += 1

        # Collect results and remove dead tracks
        results_bbox = []
        results_id = []
        results_vel = []

        trackers_to_keep = []
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                continue
            trackers_to_keep.append(trk)

            if trk.time_since_update < 1 and \
               (trk.hit_streak >= self.min_hits or self.iframe <= self.min_hits):
                results_bbox.append(trk.get_state())
                results_id.append(trk.id)
                results_vel.append(trk.get_velocity())

        self.trackers = trackers_to_keep
        self.n_tracks_last = len(results_bbox)

        if len(results_bbox) == 0:
            return (np.empty((0, 7)), np.array([], dtype=int),
                    np.empty((0, 3)))

        return (np.array(results_bbox),
                np.array(results_id, dtype=int),
                np.array(results_vel))
