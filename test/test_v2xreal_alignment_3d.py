"""
3D visualization of V2X-Real LiDAR alignment using matplotlib 3D.

Renders point clouds from all vehicles in MAP frame with different colors,
from multiple viewpoints (BEV + perspective). Also shows GT bboxes as wireframes.

Usage:
  DATASET_NAME=V2X-Real python test/test_v2xreal_alignment_3d.py
  DATASET_NAME=V2X-Real python test/test_v2xreal_alignment_3d.py --case 5
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.join(os.path.dirname(__file__), "..")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.data.util import pcd_sensor_to_map, bbox_sensor_to_map


VEHICLE_COLORS = {
    -2: 'red', -1: 'blue', 1: 'green', 2: 'orange', 3: 'purple',
}


def bbox_wireframe(bbox, color='blue', alpha=0.3):
    """Return 3D wireframe lines for a 7-DOF bbox [x,y,z,l,w,h,yaw]."""
    cx, cy, cz, l, w, h, yaw = bbox
    corners = np.array([
        [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
    ])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0, 0, 1]])
    corners = corners @ R.T + np.array([cx, cy, cz])

    edges = [[0,1],[1,2],[2,3],[3,0],
             [4,5],[5,6],[6,7],[7,4],
             [0,4],[1,5],[2,6],[3,7]]
    lines = []
    for i, j in edges:
        lines.append([corners[i], corners[j]])
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="V2X-Real")
    parser.add_argument("--downsample", type=int, default=10,
                        help="Point cloud downsample factor for rendering speed")
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset = OPV2VDataset(
        root_path=os.path.join(root, f"data/{dataset_name}"),
        mode="test", dataset_name=dataset_name)

    frame = dataset.get_case(args.case, tag="multi_vehicle", use_lidar=True)
    vids = [v for v in frame.keys() if frame[v] and frame[v].get("lidar") is not None]
    print(f"Vehicles: {vids}")

    # Collect data
    all_pcds = {}
    all_gt_map = {}
    all_poses = {}
    for vid in vids:
        vdata = frame[vid]
        pcd_map = pcd_sensor_to_map(vdata["lidar"].astype(np.float64),
                                     vdata["lidar_pose"])
        # Downsample for rendering
        idx = np.random.choice(len(pcd_map),
                                min(len(pcd_map), len(pcd_map) // args.downsample + 1),
                                replace=False)
        all_pcds[vid] = pcd_map[idx]
        gt = np.array(vdata["gt_bboxes"])
        if gt.ndim == 2 and gt.shape[0] > 0:
            all_gt_map[vid] = bbox_sensor_to_map(gt, vdata["lidar_pose"])
        else:
            all_gt_map[vid] = np.empty((0, 7))
        all_poses[vid] = vdata["lidar_pose"]

    # Compute center and extent
    all_pts = np.vstack(list(all_pcds.values()))
    center = all_pts[:, :2].mean(axis=0)
    extent = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
                 all_pts[:, 1].max() - all_pts[:, 1].min()) / 2 + 10

    # Create figure with 3 viewpoints
    fig = plt.figure(figsize=(24, 8))

    views = [
        ("BEV (top-down)", 90, -90),
        ("Perspective view 1", 45, -60),
        ("Perspective view 2", 30, -120),
    ]

    for vi, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, vi + 1, projection='3d')

        # Draw point clouds
        for vid in vids:
            pcd = all_pcds[vid]
            c = VEHICLE_COLORS.get(vid, 'gray')
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
                       s=0.1, c=c, alpha=0.4, label=f"V{vid}")

        # Draw GT bboxes as wireframes
        for vid in vids:
            c = VEHICLE_COLORS.get(vid, 'gray')
            for i in range(len(all_gt_map[vid])):
                lines = bbox_wireframe(all_gt_map[vid][i])
                for line in lines:
                    pts = np.array(line)
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                            color=c, linewidth=0.5, alpha=0.6)

        # Draw vehicle positions
        for vid in vids:
            pose = all_poses[vid]
            c = VEHICLE_COLORS.get(vid, 'gray')
            ax.scatter([pose[0]], [pose[1]], [pose[2]],
                       s=100, c=c, marker='^', zorder=5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

        # Set limits
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)

        if vi == 0:
            ax.legend(fontsize=8, markerscale=20, loc='upper left')

    plt.suptitle(
        f"{dataset_name} LiDAR Alignment — Case {args.case}\n"
        f"Vehicles: {vids} | Each color = one vehicle's point cloud + GT boxes",
        fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs("tmp", exist_ok=True)
    save_path = f"tmp/v2xreal_alignment_3d_case{args.case}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
