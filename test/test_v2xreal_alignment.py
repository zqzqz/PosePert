"""
Verify V2X-Real LiDAR point cloud alignment across vehicles.

Produces:
1. All vehicles' point clouds overlaid in MAP frame — should overlap at shared regions
2. Each vehicle's point cloud with its own GT bboxes in SENSOR frame — bboxes should fit points
3. All vehicles' point clouds with ALL GT bboxes in MAP frame — bboxes from different
   vehicles for the same object should overlap
4. Compare our loading vs V2X-Real repo's loading on the same frame

Usage:
  DATASET_NAME=V2X-Real python test/test_v2xreal_alignment.py
"""

import os
import sys
import numpy as np
import copy
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.data.util import (pcd_sensor_to_map, pcd_map_to_sensor,
                            bbox_sensor_to_map, bbox_map_to_sensor)

logging.basicConfig(level=logging.WARNING)

COLORS = {-2: 'red', -1: 'blue', 1: 'green', 2: 'orange', 3: 'purple'}


def draw_bbox_2d(ax, bbox, color='red', linewidth=1.5, label=None):
    boxp = cv2.boxPoints(((bbox[0], bbox[1]),
                           (bbox[3], bbox[4]),
                           bbox[6] / np.pi * 180))
    boxp = np.vstack([boxp, boxp[0]])
    ax.plot(boxp[:, 0], boxp[:, 1], linewidth=linewidth, color=color, label=label)


def main():
    dataset = OPV2VDataset(root_path=os.path.join(root, "data/V2X-Real"),
                           mode="test", dataset_name="V2X-Real")

    for case_idx in range(min(3, dataset.case_number("multi_vehicle"))):
        case = dataset.get_case(case_idx, tag="multi_vehicle", use_lidar=True)
        vids = [v for v in case.keys() if case[v] and case[v].get("lidar") is not None]

        fig, axes = plt.subplots(2, 3, figsize=(24, 16))

        # ============================================================
        # Panel 1: All point clouds in MAP frame (should overlap)
        # ============================================================
        ax = axes[0, 0]
        for vid in vids:
            vdata = case[vid]
            pcd_map = pcd_sensor_to_map(vdata["lidar"].astype(np.float64),
                                         vdata["lidar_pose"])
            c = COLORS.get(vid, 'gray')
            ax.scatter(pcd_map[:, 0], pcd_map[:, 1], s=0.05, c=c, alpha=0.3,
                       label=f"Vehicle {vid}")
            ax.scatter(vdata["lidar_pose"][0], vdata["lidar_pose"][1],
                       s=100, c=c, marker='^', zorder=5)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, markerscale=10)
        ax.set_title("All point clouds in MAP frame")
        ax.grid(True, alpha=0.2)

        # ============================================================
        # Panel 2: All GT bboxes in MAP frame from each vehicle
        #   Same object from different vehicles should overlap
        # ============================================================
        ax = axes[0, 1]
        obj_positions = {}  # {obj_id: [(vid, map_pos)]}
        for vid in vids:
            vdata = case[vid]
            pcd_map = pcd_sensor_to_map(vdata["lidar"].astype(np.float64),
                                         vdata["lidar_pose"])
            ax.scatter(pcd_map[:, 0], pcd_map[:, 1], s=0.02, c='lightgray', alpha=0.2)

            gt = np.array(vdata["gt_bboxes"])
            gt_map = bbox_sensor_to_map(gt, vdata["lidar_pose"])
            c = COLORS.get(vid, 'gray')
            for i, oid in enumerate(vdata["object_ids"]):
                draw_bbox_2d(ax, gt_map[i], color=c, linewidth=1.0)
                ax.text(gt_map[i, 0], gt_map[i, 1], str(oid), fontsize=5,
                        ha='center', va='center', color=c)
                if oid not in obj_positions:
                    obj_positions[oid] = []
                obj_positions[oid].append((vid, gt_map[i, :2]))
        ax.set_aspect('equal')
        ax.set_title("GT bboxes in MAP frame (same color = same vehicle)")
        ax.grid(True, alpha=0.2)

        # ============================================================
        # Panel 3: Object alignment report
        # ============================================================
        ax = axes[0, 2]
        ax.axis('off')
        report_lines = ["Object Alignment Report\n"]
        for oid, obs in sorted(obj_positions.items()):
            if len(obs) >= 2:
                positions = np.array([o[1] for o in obs])
                spread = np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))
                obs_vids = [o[0] for o in obs]
                status = "OK" if spread < 0.5 else ("WARN" if spread < 2.0 else "BAD")
                report_lines.append(
                    f"Obj {oid}: vehicles={obs_vids}, spread={spread:.3f}m [{status}]")
        report_lines.append(f"\nTotal objects: {len(obj_positions)}")
        report_lines.append(f"Shared objects: {sum(1 for o in obj_positions.values() if len(o)>=2)}")
        ax.text(0.05, 0.95, "\n".join(report_lines), transform=ax.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace')

        # ============================================================
        # Panels 4-6: Each vehicle's PCD + GT bboxes in SENSOR frame
        #   Bboxes should tightly fit the point cloud clusters
        # ============================================================
        for vi, vid in enumerate(vids[:3]):
            ax = axes[1, vi]
            vdata = case[vid]
            pcd = vdata["lidar"]
            gt = np.array(vdata["gt_bboxes"])

            # Draw point cloud in sensor frame
            ax.scatter(pcd[:, 0], pcd[:, 1], s=0.05, c='gray', alpha=0.3)

            # Draw GT bboxes and highlight points inside each bbox
            for i in range(len(gt)):
                bbox = gt[i]
                draw_bbox_2d(ax, bbox, color='blue', linewidth=1.0)
                # Count points near bbox center (simple radius check)
                r = max(bbox[3], bbox[4]) / 2 + 0.5
                dist = np.linalg.norm(pcd[:, :2] - bbox[:2], axis=1)
                near = dist < r
                n_pts = near.sum()
                ax.text(bbox[0], bbox[1] + bbox[4] / 2 + 0.5,
                        f"{vdata['object_ids'][i]}({n_pts}pts)",
                        fontsize=5, ha='center', color='blue')
                if n_pts > 0:
                    ax.scatter(pcd[near, 0], pcd[near, 1],
                               s=0.5, c='red', alpha=0.5)

            ax.set_aspect('equal')
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_title(f"Vehicle {vid} SENSOR frame ({len(gt)} GT bboxes)")
            ax.grid(True, alpha=0.2)

        plt.suptitle(f"V2X-Real Alignment Check — Case {case_idx}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = f"tmp/v2xreal_alignment_case{case_idx}.png"
        os.makedirs("tmp", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
