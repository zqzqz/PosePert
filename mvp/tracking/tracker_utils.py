"""
Tracking utility functions compatible with scenario_attacker_util interface.

Usage:
    from mvp.tracking import Sort3DTracker, DeepSort3DTracker
    from mvp.tracking.tracker_utils import tracking_sort3d, tracking_deepsort3d

    # Drop-in replacement for tracking_ab3dmot:
    tracks = Sort3DTracker()
    tracks, indexed_detections = tracking_sort3d(tracks, t, detections)
"""

import numpy as np


def tracking_sort3d(tracks, t, detections):
    """
    SORT-3D tracking wrapper matching tracking_ab3dmot interface.

    Args:
        tracks: Sort3DTracker instance
        t: timestamp
        detections: (N, 7) array [x, y, z, l, w, h, theta]

    Returns:
        tracks: updated tracker
        indexed_detections: {track_id: bbox_7d}
    """
    info = np.ones((detections.shape[0], 1))
    bboxes, bbox_ids, _ = tracks.update(t, detections, info)
    indexed_detections = {}
    for i in range(len(bboxes)):
        indexed_detections[bbox_ids[i]] = bboxes[i]
    return tracks, indexed_detections


def tracking_deepsort3d(tracks, t, detections, features=None):
    """
    DeepSORT-3D tracking wrapper matching tracking_ab3dmot interface.

    Args:
        tracks: DeepSort3DTracker instance
        t: timestamp
        detections: (N, 7) array [x, y, z, l, w, h, theta]
        features: optional (N, D) appearance features

    Returns:
        tracks: updated tracker
        indexed_detections: {track_id: bbox_7d}
    """
    info = np.ones((detections.shape[0], 1))
    bboxes, bbox_ids, _ = tracks.update(t, detections, info, features=features)
    indexed_detections = {}
    for i in range(len(bboxes)):
        indexed_detections[bbox_ids[i]] = bboxes[i]
    return tracks, indexed_detections
