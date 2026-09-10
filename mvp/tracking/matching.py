"""
Association methods for 3D multi-object tracking.

Provides IoU-based and distance-based matching using the Hungarian algorithm.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_3d_axis_aligned(bbox_a, bbox_b):
    """
    Compute axis-aligned 3D IoU between two bounding boxes.

    Args:
        bbox_a, bbox_b: [x, y, z, l, w, h, theta] (theta ignored for AABB)

    Returns:
        IoU value
    """
    # Axis-aligned approximation: use 2D BEV IoU * height overlap
    xa, ya, za, la, wa, ha = bbox_a[:6]
    xb, yb, zb, lb, wb, hb = bbox_b[:6]

    # BEV overlap (axis-aligned)
    x_overlap = max(0, min(xa + la / 2, xb + lb / 2) - max(xa - la / 2, xb - lb / 2))
    y_overlap = max(0, min(ya + wa / 2, yb + wb / 2) - max(ya - wa / 2, yb - wb / 2))
    z_overlap = max(0, min(za + ha / 2, zb + hb / 2) - max(za - ha / 2, zb - hb / 2))

    intersection = x_overlap * y_overlap * z_overlap
    vol_a = la * wa * ha
    vol_b = lb * wb * hb
    union = vol_a + vol_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def center_distance(bbox_a, bbox_b):
    """Euclidean distance between centers of two 3D bboxes."""
    return np.linalg.norm(bbox_a[:3] - bbox_b[:3])


def compute_iou_matrix(detections, trackers):
    """
    Compute pairwise IoU matrix.

    Args:
        detections: (N, 7) array
        trackers: (M, 7) array

    Returns:
        (N, M) IoU matrix
    """
    N, M = len(detections), len(trackers)
    iou_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            iou_mat[i, j] = iou_3d_axis_aligned(detections[i], trackers[j])
    return iou_mat


def compute_distance_matrix(detections, trackers):
    """
    Compute pairwise center distance matrix.

    Args:
        detections: (N, 7) array
        trackers: (M, 7) array

    Returns:
        (N, M) distance matrix
    """
    N, M = len(detections), len(trackers)
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = center_distance(detections[i], trackers[j])
    return dist_mat


def associate_iou(detections, trackers, iou_threshold=0.1):
    """
    Associate detections to trackers using IoU and Hungarian algorithm.

    Returns:
        matches: list of (det_idx, trk_idx) pairs
        unmatched_dets: list of detection indices
        unmatched_trks: list of tracker indices
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))

    iou_mat = compute_iou_matrix(detections, trackers)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)  # maximize IoU

    matches = []
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(trackers)))

    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= iou_threshold:
            matches.append((r, c))
            unmatched_dets.remove(r)
            unmatched_trks.remove(c)

    return matches, unmatched_dets, unmatched_trks


def associate_distance(detections, trackers, dist_threshold=5.0):
    """
    Associate detections to trackers using center distance and Hungarian algorithm.

    Returns:
        matches, unmatched_dets, unmatched_trks
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))

    dist_mat = compute_distance_matrix(detections, trackers)
    row_ind, col_ind = linear_sum_assignment(dist_mat)

    matches = []
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(trackers)))

    for r, c in zip(row_ind, col_ind):
        if dist_mat[r, c] <= dist_threshold:
            matches.append((r, c))
            unmatched_dets.remove(r)
            unmatched_trks.remove(c)

    return matches, unmatched_dets, unmatched_trks


def cascaded_associate(detections, trackers, track_ages,
                       iou_threshold=0.1, dist_threshold=5.0, max_age_cascade=30):
    """
    DeepSORT-style cascaded matching: prioritize recently-seen tracks.

    First matches detections to tracks seen recently (low time_since_update),
    then uses remaining detections on older tracks with distance-based fallback.

    Args:
        detections: (N, 7) array
        trackers: (M, 7) predicted tracker states
        track_ages: (M,) time_since_update for each tracker
        iou_threshold: IoU threshold for matching
        dist_threshold: distance threshold for fallback matching
        max_age_cascade: max age bins for cascade

    Returns:
        matches, unmatched_dets, unmatched_trks
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))

    all_matches = []
    remaining_dets = list(range(len(detections)))
    remaining_trks = list(range(len(trackers)))

    # Cascade: match by age groups (0, 1, 2, ...)
    for age in range(max_age_cascade + 1):
        if not remaining_dets:
            break

        # Tracks at this age level
        age_trks = [t for t in remaining_trks if track_ages[t] == age]
        if not age_trks:
            continue

        det_arr = detections[remaining_dets]
        trk_arr = trackers[age_trks]

        iou_mat = compute_iou_matrix(det_arr, trk_arr)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_threshold:
                all_matches.append((remaining_dets[r], age_trks[c]))
                remaining_dets.remove(remaining_dets[r])
                remaining_trks.remove(age_trks[c])
                # Indices shifted — restart to avoid index issues
                break
        else:
            continue
        # Redo this age level with updated remaining
        # (simpler than careful index management for small N)

    # Distance-based fallback for remaining
    if remaining_dets and remaining_trks:
        det_arr = detections[remaining_dets]
        trk_arr = trackers[remaining_trks]
        dist_mat = compute_distance_matrix(det_arr, trk_arr)
        row_ind, col_ind = linear_sum_assignment(dist_mat)

        for r, c in zip(row_ind, col_ind):
            if dist_mat[r, c] <= dist_threshold:
                all_matches.append((remaining_dets[r], remaining_trks[c]))

    matched_dets = set(m[0] for m in all_matches)
    matched_trks = set(m[1] for m in all_matches)
    unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
    unmatched_trks = [t for t in range(len(trackers)) if t not in matched_trks]

    return all_matches, unmatched_dets, unmatched_trks
