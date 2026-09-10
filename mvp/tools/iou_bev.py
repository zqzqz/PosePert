"""BEV (Bird's Eye View) IoU for rotated bounding boxes."""
import numpy as np


def iou_bev(bbox1, bbox2):
    """
    Compute 2D BEV IoU between two bboxes [x, y, z, l, w, h, yaw].
    Rotation is around Z-axis (yaw).

    Returns:
        float: IoU value in [0, 1]
    """
    try:
        from shapely.geometry import Polygon

        def bbox_to_polygon(bbox):
            x, y, z, l, w, h, yaw = bbox[:7]
            corners = np.array([[-l/2, -w/2], [l/2, -w/2],
                                [l/2, w/2], [-l/2, w/2]])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            corners = corners @ R.T + np.array([x, y])
            return Polygon(corners)

        p1 = bbox_to_polygon(bbox1)
        p2 = bbox_to_polygon(bbox2)
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter / (union + 1e-8)
    except Exception:
        return 0.0
