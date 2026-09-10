"""
Run CAD defense on V2X-Real with precomputed occupancy maps.

Loads spoofed PCD from attack cache, runs perception, then evaluates
CAD (occupancy consistency) on both normal and attack frames.

Prerequisites: run gen_v2xreal_occupancy.py first.

Usage:
  CUDA_VISIBLE_DEVICES=1 python test/run_cad_v2xreal.py
"""
import os, sys, pickle, copy, numpy as np, torch, time, logging, traceback

os.environ.setdefault("DATASET_NAME", "V2X-Real")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "V2X-Real"))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_train import _apply_warp_patches
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.defense.perception_defender import PerceptionDefender
from mvp.data.util import pcd_sensor_to_map, pcd_map_to_sensor, bbox_sensor_to_map
from mvp.tools.ground_detection import get_ground_plane_ransac
from mvp.tools.lidar_seg import lidar_segmentation_dbscan
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recompute_occupancy_for_spoofed(vehicle_data, spoof_pcd):
    """Recompute occupancy for the attacker vehicle with spoofed PCD."""
    lidar_pose = vehicle_data["lidar_pose"]
    pcd = pcd_sensor_to_map(spoof_pcd, lidar_pose)

    plane_model, ground_indices = get_ground_plane_ransac(pcd, err_thres=0.2)
    [a, b, c, d] = plane_model
    norm = np.sqrt(a**2 + b**2 + c**2)
    point_height = (np.sum(pcd[:, :3] * np.array([a, b, c]), axis=1) + d) / norm
    in_lane_mask = np.ones(pcd.shape[0], dtype=bool)

    lidar_seg = lidar_segmentation_dbscan(spoof_pcd, ground_indices,
                                          cluster_thres=0.5, min_point_num=8)
    object_segments = filter_segmentation(
        spoof_pcd, lidar_seg, lidar_pose,
        in_lane_mask=in_lane_mask, point_height=point_height, max_range=50
    )
    object_mask = np.zeros(pcd.shape[0], dtype=bool)
    if len(object_segments) > 0:
        object_mask[np.hstack(object_segments)] = True

    ego_bbox = vehicle_data["ego_bbox"]
    ego_area = bbox_to_polygon(ego_bbox)

    occupied_areas, occupied_areas_height = get_occupied_space(
        pcd, object_segments, point_height=point_height, height_thres=0)
    free_areas = get_free_space(
        spoof_pcd, lidar_pose, object_mask,
        in_lane_mask=in_lane_mask, point_height=point_height,
        max_range=50, height_thres=0, height_tolerance=0.2)

    return {
        "occupied_areas": occupied_areas,
        "occupied_areas_height": occupied_areas_height,
        "free_areas": free_areas,
        "ego_area": ego_area,
    }


if __name__ == '__main__':
    data_path = os.path.join(root, 'data/V2X-Real')
    cache_dir = os.path.join(data_path, 'attack_cache_paper')
    test_pkl_path = os.path.join(data_path, 'attack/lidar_shift.pkl')
    occ_dir = os.path.join(data_path, 'normal')
    result_dir = os.path.join(root, 'results_paper/D_v2xreal')
    os.makedirs(result_dir, exist_ok=True)

    warp_patches = _apply_warp_patches()
    perception = OpencoodPerception(fusion_method='intermediate', model_name='pointpillar',
                                    dataset_name='V2X-Real')
    perception.model.eval()
    dataset = OPV2VDataset(root_path=data_path, mode='test', dataset_name='V2X-Real')

    cad = PerceptionDefender()
    v2v_filter_fn = lambda frame: {v: d for v, d in frame.items() if isinstance(v, int) and v < 0}

    with open(test_pkl_path, 'rb') as f:
        attacks = pickle.load(f)

    results = []
    t0 = time.time()

    for ci in range(len(attacks)):
        cache_path = os.path.join(cache_dir, f'{ci:06d}.pkl')
        if not os.path.exists(cache_path):
            continue

        meta = attacks[ci]['attack_meta']
        case_id = meta['case_id']
        ai = meta['attacker_vehicle_id']
        vi = meta['victim_vehicle_id']

        occ_path = os.path.join(occ_dir, f'{case_id:06d}.pkl')
        if not os.path.exists(occ_path):
            results.append({'case_idx': ci, 'cad_max_spoof_normal': -1,
                            'cad_max_spoof_attack': -1, 'cad_detected': None})
            continue

        try:
            cached = pickle.load(open(cache_path, 'rb'))
            case = dataset.get_case(case_id, tag='multi_frame', use_lidar=True)
            frame_full = case[min(9, len(case) - 1)]
            occ_data = pickle.load(open(occ_path, 'rb'))

            if ai not in frame_full or vi not in frame_full:
                continue

            frame = v2v_filter_fn(frame_full)
            if ai not in frame or vi not in frame:
                continue

            # Normal frame with occupancy
            frame_normal = copy.deepcopy(frame)
            for vid in list(frame_normal.keys()):
                if vid in occ_data:
                    frame_normal[vid].update(occ_data[vid])

            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            pred_n, _ = perception.run(frame, vi)
            frame_normal[vi]['pred_bboxes'] = pred_n

            spoof_n = []
            if len(pred_n) > 0:
                try:
                    _, _, metrics_n = cad.run({9: frame_normal}, {'frame_ids': [9],
                        'vehicle_ids': [v for v in frame_normal if isinstance(v, int)]})
                    spoof_n = [m[1] for m in metrics_n[9].get(vi, {}).get('spoof', [])]
                except Exception as e_cad:
                    logger.warning(f"  Case {ci} CAD normal error: {e_cad}")

            # Attack frame: replace attacker's LiDAR with spoofed PCD
            frame_atk = copy.deepcopy(frame)
            frame_atk[ai]['lidar'] = cached['spoof_pcd']
            for vid in list(frame_atk.keys()):
                if vid in occ_data:
                    frame_atk[vid].update(occ_data[vid])

            # Recompute occupancy for attacker with spoofed PCD
            try:
                occ_atk = recompute_occupancy_for_spoofed(frame_atk[ai], cached['spoof_pcd'])
                frame_atk[ai].update(occ_atk)
            except Exception as e_occ:
                logger.warning(f"  Case {ci} occ recompute error: {e_occ}")

            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            pred_a, _ = perception.run(frame_atk, vi)
            frame_atk[vi]['pred_bboxes'] = pred_a

            spoof_a = []
            if len(pred_a) > 0:
                try:
                    _, _, metrics_a = cad.run({9: frame_atk}, {'frame_ids': [9],
                        'vehicle_ids': [v for v in frame_atk if isinstance(v, int)]})
                    spoof_a = [m[1] for m in metrics_a[9].get(vi, {}).get('spoof', [])]
                except Exception as e_cad:
                    logger.warning(f"  Case {ci} CAD attack error: {e_cad}")

            entry = {
                'case_idx': ci, 'case_id': case_id,
                'cad_max_spoof_normal': max(spoof_n) if spoof_n else 0,
                'cad_max_spoof_attack': max(spoof_a) if spoof_a else 0,
                'cad_detected': (max(spoof_a) if spoof_a else 0) > cad.thres,
            }
            results.append(entry)
            torch.cuda.empty_cache()

            if len(results) % 10 == 0:
                det = sum(1 for r in results if r.get('cad_detected') == True)
                avail = sum(1 for r in results if r.get('cad_detected') is not None)
                logger.info(f"  [{len(results)}] CAD detected: {det}/{avail}")

        except Exception as e:
            logger.warning(f"Case {ci}: {e}")
            traceback.print_exc()
            results.append({'case_idx': ci, 'cad_max_spoof_normal': -1,
                            'cad_max_spoof_attack': -1, 'cad_detected': None})
            continue

    # Save
    with open(os.path.join(result_dir, 'cad_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Summary
    valid = [r for r in results if r.get('cad_detected') is not None]
    det = sum(1 for r in valid if r['cad_detected'])
    normal_mean = np.mean([r['cad_max_spoof_normal'] for r in valid])
    attack_mean = np.mean([r['cad_max_spoof_attack'] for r in valid])

    logger.info(f"\n{'='*60}")
    logger.info(f"CAD V2X-Real: {len(valid)} valid cases")
    logger.info(f"Normal mean max_spoof: {normal_mean:.3f}")
    logger.info(f"Attack mean max_spoof: {attack_mean:.3f}")
    logger.info(f"CAD detected (spoof > {cad.thres}): {det}/{len(valid)} ({100*det/len(valid):.1f}%)")
    logger.info(f"Time: {time.time()-t0:.0f}s")

    # Update defense_summary.json
    import json
    summary_path = os.path.join(result_dir, 'defense_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        summary['cad'] = {
            'detected': det,
            'detection_rate': 100*det/len(valid) if valid else 0,
            'threshold': f'spoof > {cad.thres}',
            'normal_mean': float(normal_mean),
            'attack_mean': float(attack_mean),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Updated {summary_path}")
