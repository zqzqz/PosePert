"""
Generate occupancy maps for V2X-Real dataset (CAD defense).

V2X-Real lacks CARLA map data, so we use RANSAC ground plane and
DBSCAN clustering (instead of map-based ground + SqueezeSeg used for OPV2V).

Output: data/V2X-Real/normal/{case_id:06d}.pkl
  Format: {vehicle_id: {occupied_areas, free_areas, ego_area}}

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n advCP python test/gen_v2xreal_occupancy.py
"""
import os, sys, pickle, logging, numpy as np

root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "third_party/OpenCOOD"))

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.data.util import pcd_sensor_to_map
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.lidar_seg import lidar_segmentation_dbscan
from mvp.tools.ground_detection import get_ground_plane_ransac
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_occupancy_ransac(vehicle_data):
    lidar, lidar_pose = vehicle_data["lidar"], vehicle_data["lidar_pose"]
    pcd = pcd_sensor_to_map(lidar, lidar_pose)

    plane_model, ground_indices = get_ground_plane_ransac(pcd, err_thres=0.2)
    [a, b, c, d] = plane_model
    norm = np.sqrt(a**2 + b**2 + c**2)
    point_height = (np.sum(pcd[:, :3] * np.array([a, b, c]), axis=1) + d) / norm
    in_lane_mask = np.ones(pcd.shape[0], dtype=bool)

    lidar_seg = lidar_segmentation_dbscan(lidar, ground_indices,
                                          cluster_thres=0.5, min_point_num=8)
    object_segments = filter_segmentation(
        lidar, lidar_seg, lidar_pose,
        in_lane_mask=in_lane_mask, point_height=point_height, max_range=50
    )
    object_mask = np.zeros(pcd.shape[0], dtype=bool)
    if len(object_segments) > 0:
        object_mask[np.hstack(object_segments)] = True

    ego_bbox = vehicle_data["ego_bbox"]
    ego_area = bbox_to_polygon(ego_bbox)

    height_thres = 0
    occupied_areas, occupied_areas_height = get_occupied_space(
        pcd, object_segments, point_height=point_height, height_thres=height_thres
    )
    free_areas = get_free_space(
        lidar, lidar_pose, object_mask,
        in_lane_mask=in_lane_mask, point_height=point_height,
        max_range=50, height_thres=height_thres, height_tolerance=0.2
    )

    return {
        "occupied_areas": occupied_areas,
        "occupied_areas_height": occupied_areas_height,
        "free_areas": free_areas,
        "ego_area": ego_area,
    }


if __name__ == "__main__":
    dataset = OPV2VDataset(
        root_path=os.path.join(root, "data/V2X-Real"),
        mode="test", dataset_name="V2X-Real"
    )

    out_dir = os.path.join(root, "data/V2X-Real/normal")
    os.makedirs(out_dir, exist_ok=True)

    test_pkl = pickle.load(open(os.path.join(root, "data/V2X-Real/attack/lidar_shift.pkl"), "rb"))
    case_ids = sorted(set(a["attack_meta"]["case_id"] for a in test_pkl))
    logger.info(f"Generating occupancy maps for {len(case_ids)} V2X-Real cases")

    for i, case_id in enumerate(case_ids):
        save_path = os.path.join(out_dir, f"{case_id:06d}.pkl")
        if os.path.exists(save_path):
            logger.info(f"  [{i+1}/{len(case_ids)}] case {case_id}: cached")
            continue

        try:
            case = dataset.get_case(case_id, tag="multi_vehicle", use_lidar=True)
            vehicle_ids = list(case.keys())
            occ_data = {}
            for vid in vehicle_ids:
                occ_data[vid] = compute_occupancy_ransac(case[vid])
            pickle.dump(occ_data, open(save_path, "wb"))
            logger.info(f"  [{i+1}/{len(case_ids)}] case {case_id}: {len(vehicle_ids)} vehicles done")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(case_ids)}] case {case_id}: FAILED - {e}")
            import traceback; traceback.print_exc()

    logger.info("Done.")
