"""
DeepSORT-3D: 3D Multi-Object Tracking with Deep Association.

Extends SORT-3D with:
  1. Cascaded matching: prioritizes recently-seen tracks
  2. Appearance features (optional): uses BEV feature embeddings for re-ID
  3. Distance-based fallback matching for occluded objects
  4. Track state management: Tentative → Confirmed → Deleted

Compatible with the Ab3dmotTracker interface.
"""

import numpy as np
from .kalman3d import KalmanBox3D
from .matching import (compute_iou_matrix, compute_distance_matrix,
                       associate_iou, associate_distance)
from scipy.optimize import linear_sum_assignment


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """Single object track with state management."""

    def __init__(self, bbox, track_id, n_init=3, max_age=30, feature=None):
        self.kalman = KalmanBox3D(bbox)
        self.kalman.id = track_id
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.n_init = n_init
        self.max_age = max_age

        # Appearance feature gallery (for re-ID)
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self.max_features = 100

    def predict(self):
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kalman.get_state()

    def update(self, bbox, feature=None):
        self.kalman.update(bbox)
        self.hits += 1
        self.time_since_update = 0

        if feature is not None:
            self.features.append(feature)
            if len(self.features) > self.max_features:
                self.features.pop(0)

        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative


class DeepSort3DTracker:
    """
    DeepSORT-style 3D multi-object tracker.

    Features over basic SORT:
    - Cascaded matching prioritizing recently-updated tracks
    - Optional appearance feature matching for re-identification
    - Track state machine (Tentative → Confirmed → Deleted)
    - Distance-based fallback for low-IoU scenarios

    Interface matches Ab3dmotTracker:
        tracker = DeepSort3DTracker()
        bboxes, bbox_ids, bbox_features = tracker.update(t, detections, info)

    For appearance-enhanced tracking:
        bboxes, bbox_ids, bbox_features = tracker.update(
            t, detections, info, features=det_features)
    """

    def __init__(self, max_age=30, n_init=3, iou_threshold=0.1,
                 dist_threshold=5.0, use_appearance=False,
                 appearance_threshold=0.3):
        """
        Args:
            max_age: max frames before deleting unmatched track
            n_init: hits needed to confirm tentative track
            iou_threshold: minimum IoU for primary matching
            dist_threshold: max distance for fallback matching
            use_appearance: whether to use appearance features
            appearance_threshold: max cosine distance for appearance matching
        """
        self.max_age = max_age
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.use_appearance = use_appearance
        self.appearance_threshold = appearance_threshold

        self._tracks = []
        self._next_id = 0
        self.iframe = 0
        self.n_tracks_total = 0
        self.reference = None

    @property
    def n_tracks(self):
        return self.n_tracks_total

    @property
    def n_tracks_active(self):
        return sum(1 for t in self._tracks if t.is_confirmed())

    @property
    def tracks(self):
        return self._tracks

    def _create_track(self, bbox, feature=None):
        track = Track(bbox, self._next_id, self.n_init, self.max_age, feature)
        self._next_id += 1
        self.n_tracks_total += 1
        return track

    def update(self, t, detections, info, features=None):
        """
        Update tracker with new detections.

        Args:
            t: timestamp (for interface compatibility)
            detections: (N, 7) array [x, y, z, l, w, h, theta]
            info: (N, 1) confidence scores
            features: optional (N, D) appearance feature vectors

        Returns:
            bboxes: (M, 7) tracked bounding boxes
            bbox_ids: (M,) track IDs
            bbox_features: (M, 3) velocity [vx, vy, vz]
        """
        self.iframe += 1

        # Predict all tracks
        for track in self._tracks:
            track.predict()

        # Split tracks into confirmed and tentative
        confirmed_tracks = [t for t in self._tracks if t.is_confirmed()]
        tentative_tracks = [t for t in self._tracks if t.is_tentative()]

        # --- Stage 1: Match detections to confirmed tracks (cascaded) ---
        matches_1, unmatched_dets, unmatched_trks_confirmed = \
            self._cascaded_match(detections, confirmed_tracks, features)

        # --- Stage 2: Match remaining dets to tentative tracks ---
        if tentative_tracks and unmatched_dets:
            tent_predicted = np.array([t.kalman.get_state() for t in tentative_tracks])
            det_remaining = detections[unmatched_dets]

            iou_mat = compute_iou_matrix(det_remaining, tent_predicted)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)

            matches_2 = []
            matched_det_local = set()
            matched_trk_local = set()
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    matches_2.append((unmatched_dets[r], c))
                    matched_det_local.add(r)
                    matched_trk_local.add(c)

            unmatched_dets = [unmatched_dets[i] for i in range(len(unmatched_dets))
                              if i not in matched_det_local]
            unmatched_trks_tentative = [i for i in range(len(tentative_tracks))
                                        if i not in matched_trk_local]
        else:
            matches_2 = []
            unmatched_trks_tentative = list(range(len(tentative_tracks)))

        # --- Apply updates ---
        for det_idx, trk_idx in matches_1:
            feat = features[det_idx] if features is not None else None
            confirmed_tracks[trk_idx].update(detections[det_idx], feat)

        for det_idx, trk_idx in matches_2:
            feat = features[det_idx] if features is not None else None
            tentative_tracks[trk_idx].update(detections[det_idx], feat)

        # Mark unmatched tracks as missed
        for trk_idx in unmatched_trks_confirmed:
            confirmed_tracks[trk_idx].mark_missed()
        for trk_idx in unmatched_trks_tentative:
            tentative_tracks[trk_idx].mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            feat = features[det_idx] if features is not None else None
            self._tracks.append(self._create_track(detections[det_idx], feat))

        # Remove deleted tracks
        self._tracks = [t for t in self._tracks if not t.is_deleted()]

        # Collect output (confirmed tracks only)
        results_bbox = []
        results_id = []
        results_vel = []

        for track in self._tracks:
            if not track.is_confirmed():
                continue
            if track.time_since_update > 0:
                continue  # Only report tracks updated this frame
            results_bbox.append(track.kalman.get_state())
            results_id.append(track.track_id)
            results_vel.append(track.kalman.get_velocity())

        if len(results_bbox) == 0:
            return (np.empty((0, 7)), np.array([], dtype=int),
                    np.empty((0, 3)))

        return (np.array(results_bbox),
                np.array(results_id, dtype=int),
                np.array(results_vel))

    def _cascaded_match(self, detections, confirmed_tracks, features=None):
        """
        Cascaded matching: match by age priority, with optional appearance.
        """
        if not confirmed_tracks or len(detections) == 0:
            return ([], list(range(len(detections))),
                    list(range(len(confirmed_tracks))))

        predicted = np.array([t.kalman.get_state() for t in confirmed_tracks])
        ages = np.array([t.time_since_update for t in confirmed_tracks])

        all_matches = []
        remaining_dets = list(range(len(detections)))
        remaining_trks = list(range(len(confirmed_tracks)))

        # Match by age level (0 = just updated, 1 = 1 frame old, etc.)
        max_age_level = int(ages.max()) if len(ages) > 0 else 0
        for age_level in range(max_age_level + 1):
            if not remaining_dets:
                break

            age_trks = [t for t in remaining_trks if ages[t] == age_level]
            if not age_trks:
                continue

            det_arr = detections[remaining_dets]
            trk_arr = predicted[age_trks]

            # Compute cost matrix
            cost_matrix = self._compute_cost(
                det_arr, trk_arr, remaining_dets, age_trks,
                confirmed_tracks, features)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            new_matches = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] > 0.9:  # reject high-cost matches
                    continue
                new_matches.append((remaining_dets[r], age_trks[c]))

            for d, t in new_matches:
                all_matches.append((d, t))
                remaining_dets.remove(d)
                remaining_trks.remove(t)

        # Distance fallback for remaining
        if remaining_dets and remaining_trks:
            det_arr = detections[remaining_dets]
            trk_arr = predicted[remaining_trks]
            dist_mat = compute_distance_matrix(det_arr, trk_arr)
            row_ind, col_ind = linear_sum_assignment(dist_mat)

            for r, c in zip(row_ind, col_ind):
                if dist_mat[r, c] <= self.dist_threshold:
                    all_matches.append((remaining_dets[r], remaining_trks[c]))
                    remaining_dets.remove(remaining_dets[r])
                    remaining_trks.remove(remaining_trks[c])
                    break  # re-index safety

        matched_dets = set(m[0] for m in all_matches)
        matched_trks = set(m[1] for m in all_matches)
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        unmatched_trks = [t for t in range(len(confirmed_tracks)) if t not in matched_trks]

        return all_matches, unmatched_dets, unmatched_trks

    def _compute_cost(self, det_arr, trk_arr, det_indices, trk_indices,
                      confirmed_tracks, features):
        """
        Compute cost matrix combining IoU and optional appearance distance.
        """
        N, M = len(det_arr), len(trk_arr)

        # IoU cost (1 - IoU)
        iou_mat = compute_iou_matrix(det_arr, trk_arr)
        iou_cost = 1.0 - iou_mat

        # Gate by IoU threshold
        iou_cost[iou_mat < self.iou_threshold] = 1.0

        if not self.use_appearance or features is None:
            return iou_cost

        # Appearance cost (cosine distance)
        app_cost = np.ones((N, M))
        for i, di in enumerate(det_indices):
            if features[di] is None:
                continue
            det_feat = features[di]
            det_feat = det_feat / (np.linalg.norm(det_feat) + 1e-8)

            for j, tj in enumerate(trk_indices):
                track = confirmed_tracks[tj]
                if not track.features:
                    continue
                # Min cosine distance to any stored feature
                min_dist = 1.0
                for tf in track.features[-10:]:  # last 10 features
                    tf_norm = tf / (np.linalg.norm(tf) + 1e-8)
                    cos_dist = 1.0 - np.dot(det_feat, tf_norm)
                    min_dist = min(min_dist, cos_dist)
                app_cost[i, j] = min_dist

        # Gate by appearance threshold
        app_cost[app_cost > self.appearance_threshold] = 1.0

        # Combined cost: lambda * appearance + (1-lambda) * IoU
        lambda_app = 0.3
        cost = lambda_app * app_cost + (1 - lambda_app) * iou_cost

        return cost
