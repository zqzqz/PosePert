import os, sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)
import numpy as np
import matplotlib.pyplot as plt
import pickle
import traceback

from test_base import *
from mvp.config import data_root
from mvp.data.util import pcd_sensor_to_map
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon
from mvp.tools.squeezeseg.interface import SqueezeSegInterface


def test_occupancy_map(case, lidar_seg_api):
    lidar, lidar_pose = case["lidar"], case["lidar_pose"]
    pcd = pcd_sensor_to_map(lidar, lidar_pose)

    lane_info = pickle.load(open(os.path.join(data_root, "carla/{}_lane_info.pkl".format(case["map"])), "rb"))
    lane_areas = pickle.load(open(os.path.join(data_root, "carla/{}_lane_areas.pkl".format(case["map"])), "rb"))
    lane_planes = pickle.load(open(os.path.join(data_root, "carla/{}_ground_planes.pkl".format(case["map"])), "rb"))

    ground_indices, in_lane_mask, point_height = get_ground_plane(pcd, lane_info=lane_info, lane_areas=lane_areas, lane_planes=lane_planes, method="map")
    lidar_seg = lidar_segmentation(lidar, method="squeezeseq", interface=lidar_seg_api)
    
    object_segments = filter_segmentation(lidar, lidar_seg, lidar_pose, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50)
    object_mask = np.zeros(pcd.shape[0]).astype(bool)
    if len(object_segments) > 0:
        object_indices = np.hstack(object_segments)
        object_mask[object_indices] = True

    ego_bbox = case["ego_bbox"]
    ego_area = bbox_to_polygon(ego_bbox)
    ego_area_height = ego_bbox[5]

    ret = {
        "ego_area": ego_area,
        "ego_area_height": ego_area_height,
        "plane": None,
        "ground_indices": ground_indices,
        "point_height": point_height,
        "object_segments": object_segments,
    }

    height_thres = 0
    occupied_areas, occupied_areas_height = get_occupied_space(pcd, object_segments, point_height=point_height, height_thres=height_thres)
    free_areas = get_free_space(lidar, lidar_pose, object_mask, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50, height_thres=height_thres, height_tolerance=0.2)
    ret["occupied_areas"] = occupied_areas
    ret["occupied_areas_height"] = occupied_areas_height
    ret["free_areas"] = free_areas

    return ret


def build_case_occupancy(dataset, case_id, lidar_seg_api, frame_id=9):
    """Occupancy maps for every vehicle in one case, in the format CAD consumes.

    Computed on the same frame the defense scores (frame 9 of the multi_frame case,
    frame 0 for V2X-Real), so the maps line up with the detections they validate.
    Returns {vehicle_id: {occupied_areas, free_areas, ego_area}} -- the three fields
    mvp/defense/perception_defender.py reads; the extra diagnostics returned by
    test_occupancy_map() are dropped.
    """
    case = dataset.get_case(case_id, tag="multi_frame", use_lidar=True)
    frame = case[frame_id]
    out = {}
    for vehicle_id, vehicle_data in frame.items():
        omap = test_occupancy_map(vehicle_data, lidar_seg_api)
        out[vehicle_id] = {
            "occupied_areas": omap["occupied_areas"],
            "free_areas": omap["free_areas"],
            "ego_area": omap["ego_area"],
        }
    return out


def main():
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Generate the occupancy maps the CAD defense validates against.")
    parser.add_argument("--dataset", default=os.getenv("DATASET_NAME", "OPV2V"),
                        choices=["OPV2V", "V2X-Real"])
    parser.add_argument("--attacks", default=None,
                        help="test case pkl (default data/<dataset>/attack/lidar_shift.pkl)")
    parser.add_argument("--out", default=None,
                        help="output directory (default data/<dataset>/normal)")
    parser.add_argument("--frame", type=int, default=None,
                        help="frame index to build maps on (default 9 for OPV2V, 0 for V2X-Real)")
    parser.add_argument("--n_cases", type=int, default=None, help="limit for a smoke run")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute cases that already have an output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("occupancy")

    data_dir = os.path.join(root, "data", args.dataset)
    attacks_path = args.attacks or os.path.join(data_dir, "attack/lidar_shift.pkl")
    out_dir = args.out or os.path.join(data_dir, "normal")
    frame_id = args.frame if args.frame is not None else (0 if args.dataset == "V2X-Real" else 9)
    os.makedirs(out_dir, exist_ok=True)

    with open(attacks_path, "rb") as f:
        attacks = pickle.load(f)
    # Several test cases share a case_id (they differ in attacker/victim/target), and the
    # maps depend only on the case, so build each one once.
    case_ids = sorted({a["attack_meta"]["case_id"] for a in attacks})
    if args.n_cases:
        case_ids = case_ids[:args.n_cases]
    logger.info("%d test cases -> %d distinct case_ids, frame %d, out %s",
                len(attacks), len(case_ids), frame_id, out_dir)

    dataset = OPV2VDataset(root_path=data_dir, mode="test", dataset_name=args.dataset)
    lidar_seg_api = SqueezeSegInterface()

    written = skipped = failed = 0
    for i, case_id in enumerate(case_ids):
        out_path = os.path.join(out_dir, "{:06d}.pkl".format(case_id))
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            occ = build_case_occupancy(dataset, case_id, lidar_seg_api, frame_id=frame_id)
            with open(out_path, "wb") as f:
                pickle.dump(occ, f)
            written += 1
            logger.info("[%d/%d] case %d: %d vehicles -> %s",
                        i + 1, len(case_ids), case_id, len(occ), os.path.basename(out_path))
        except Exception:
            failed += 1
            logger.warning("case %d failed:\n%s", case_id, traceback.format_exc())

    logger.info("done: %d written, %d skipped, %d failed", written, skipped, failed)


if __name__ == "__main__":
    main()
