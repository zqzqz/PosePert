"""
Refine test cases:
1. OPV2V scenario: 333 → ~100 quality cases
2. V2X-Real perception: generate ~300 quality cases
3. V2X-Real scenario: generate ~100 quality cases

Quality criteria:
- Target object is consistently detected across frames (stable detection)
- Reasonable distance between attacker/victim/target
- At least 2 vehicles (V2V mode)

Old files are backed up as *_backup.pkl.

Usage:
  CUDA_VISIBLE_DEVICES=1 python test/refine_test_cases.py
  CUDA_VISIBLE_DEVICES=1 DATASET_NAME=V2X-Real python test/refine_test_cases.py --dataset V2X-Real
"""
import os
import sys
import pickle
import random
import shutil
import numpy as np
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.data.util import bbox_sensor_to_map
from mvp.tools.iou import iou3d
from mvp.attack.shift_rotation import sample_shift_rotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


def backup_file(path):
    """Backup file to *_backup.pkl if it exists."""
    if os.path.exists(path):
        backup = path.replace('.pkl', '_backup.pkl')
        shutil.copy2(path, backup)
        logger.info(f"Backed up {path} → {backup}")


def check_detection(perception, frame, victim_id, bbox_orig_map, threshold_iou=0.2):
    """Check if target is detected by normal perception."""
    try:
        pred, scores = perception.run(frame, victim_id)
        if len(pred) == 0:
            return False, 0.0, 0.0
        vic_pose = frame[victim_id]['lidar_pose']
        pred_map = bbox_sensor_to_map(pred, vic_pose)
        d = np.linalg.norm(pred_map[:, :2] - bbox_orig_map[:2], axis=1)
        idx = d.argmin()
        iou = iou3d(pred_map[idx], bbox_orig_map)
        conf = float(scores[idx]) if scores is not None else 0.0
        return iou > threshold_iou, iou, conf
    except:
        return False, 0.0, 0.0


def refine_opv2v_scenario(dataset, perception, max_cases=100):
    """Refine OPV2V scenario test cases from 333 → ~100."""
    pkl_path = os.path.join(root, 'data/OPV2V/test_scenario_attacks.pkl')
    backup_file(pkl_path)

    with open(pkl_path, 'rb') as f:
        scenarios = pickle.load(f)
    logger.info(f"Loaded {len(scenarios)} OPV2V scenario cases")

    quality_scored = []
    for si, s in enumerate(scenarios):
        try:
            case = dataset.get_case(s['case_id'], tag='multi_frame', use_lidar=True)
            ai = s['attacker_vehicle_id']
            vi = s['victim_vehicle_id']
            target_id = s['target_id']

            # Check multiple frames for stable detection
            detect_count = 0
            total_iou = 0
            n_check = min(5, len(case))
            check_frames = [case[i] for i in np.linspace(0, len(case)-1, n_check, dtype=int)]

            for frame in check_frames:
                if ai not in frame or vi not in frame:
                    continue
                gt = np.array(frame[ai]['gt_bboxes'])
                oids = frame[ai]['object_ids']
                if target_id not in oids:
                    continue
                tidx = oids.index(target_id)
                bbox_orig = gt[tidx]
                atk_pose = frame[ai]['lidar_pose']
                bbox_orig_map = bbox_sensor_to_map(bbox_orig, atk_pose)

                detected, iou, conf = check_detection(perception, frame, vi, bbox_orig_map)
                if detected:
                    detect_count += 1
                    total_iou += iou

            # Score: detection stability × avg IoU
            if detect_count >= 2:
                stability = detect_count / n_check
                avg_iou = total_iou / detect_count
                score = stability * avg_iou
                quality_scored.append((si, score, stability, avg_iou))

        except Exception as e:
            continue

        if (si + 1) % 50 == 0:
            logger.info(f"  Scored {si+1}/{len(scenarios)}, {len(quality_scored)} valid")

    # Sort by quality score, take top max_cases
    quality_scored.sort(key=lambda x: -x[1])
    selected_indices = [q[0] for q in quality_scored[:max_cases]]
    selected = [scenarios[i] for i in selected_indices]

    logger.info(f"Selected {len(selected)} scenario cases (from {len(quality_scored)} valid)")
    if quality_scored:
        scores = [q[1] for q in quality_scored[:max_cases]]
        logger.info(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")

    with open(pkl_path, 'wb') as f:
        pickle.dump(selected, f)
    logger.info(f"Saved {len(selected)} cases to {pkl_path}")
    return selected


def generate_v2xreal_perception(dataset, perception, max_cases=300):
    """Generate V2X-Real perception test cases (~300)."""
    pkl_path = os.path.join(root, 'data/V2X-Real/attack/lidar_shift.pkl')
    backup_file(pkl_path)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    attacks = dataset.attacks
    logger.info(f"V2X-Real has {len(attacks)} raw attack cases")

    # V2V filter helper
    def is_vehicle(vid):
        return isinstance(vid, int) and vid < 0

    attack_list = []
    for ci in range(len(attacks)):
        if len(attack_list) >= max_cases:
            break
        a = attacks[ci]
        try:
            case = dataset.get_case(a['case_id'], tag='multi_frame', use_lidar=True)
            if len(case) < 10:
                continue
            frame = case[9]
            ai, vi = a['attacker_vehicle_id'], a['victim_vehicle_id']
            if not is_vehicle(ai) or not is_vehicle(vi):
                continue
            if ai not in frame or vi not in frame:
                continue

            # V2V filter
            v2v_frame = {vid: vdata for vid, vdata in frame.items() if is_vehicle(vid)}

            gt = np.array(v2v_frame[ai]['gt_bboxes'])
            oids = v2v_frame[ai]['object_ids']
            candidates = [(i, oid) for i, oid in enumerate(oids) if oid not in [ai, vi]]
            if not candidates:
                continue

            # Check each candidate for detection quality
            atk_pose = v2v_frame[ai]['lidar_pose']
            best_candidate = None
            best_score = -1

            for tidx, toid in candidates:
                bbox_orig = gt[tidx]
                bbox_orig_map = bbox_sensor_to_map(bbox_orig, atk_pose)
                detected, iou, conf = check_detection(perception, v2v_frame, vi, bbox_orig_map)
                if detected and conf > 0.3:
                    score = iou * conf
                    if score > best_score:
                        best_score = score
                        best_candidate = (tidx, toid, bbox_orig)

            if best_candidate is None:
                continue

            tidx, toid, bbox_orig = best_candidate

            # Get bboxes across 10 frames
            bboxes_all = []
            valid = True
            for fi in range(min(10, len(case))):
                f = case[fi]
                if ai not in f:
                    valid = False; break
                gt_fi = np.array(f[ai]['gt_bboxes'])
                oids_fi = f[ai]['object_ids']
                if toid in oids_fi:
                    bboxes_all.append(gt_fi[oids_fi.index(toid)])
                else:
                    valid = False; break
            if not valid or len(bboxes_all) < 10:
                continue

            bboxes = np.array(bboxes_all[:10])

            # Random shift + rotation
            shift_distance = np.random.uniform(0.5, 1.5)
            shift_direction = np.random.uniform(0, 2 * np.pi)
            rotation = sample_shift_rotation()   # +/-10 deg, see mvp.attack.shift_rotation

            new_bboxes = np.copy(bboxes)
            new_bboxes[:, 0] += shift_distance * np.cos(shift_direction)
            new_bboxes[:, 1] += shift_distance * np.sin(shift_direction)
            new_bboxes[:, 6] += rotation
            new_bboxes[:, 6] = new_bboxes[:, 6] % (2 * np.pi)

            entry = {
                'attack_opts': {
                    'frame_ids': list(range(10)),
                    'attacker_vehicle_id': ai,
                    'victim_vehicle_id': vi,
                    'object_id': toid,
                    'shift_direction': float(shift_direction),
                    'shift_distance': float(shift_distance),
                    'rotation': float(rotation),
                },
                'attack_meta': {
                    'case_id': a['case_id'],
                    'scenario_id': str(a['case_id']),
                    'frame_ids': list(range(10)),
                    'attacker_vehicle_id': ai,
                    'victim_vehicle_id': vi,
                    'vehicle_ids': [vid for vid in v2v_frame.keys()],
                    'attack_frame_ids': list(range(10)),
                    'new_bbox': new_bboxes[-1],
                    'bbox': bboxes[-1],
                    'num_points': 0,
                    'difficulty': 0,
                    'bboxes': bboxes,
                    'new_bboxes': new_bboxes,
                    'new_num_points': 0,
                },
            }
            attack_list.append(entry)

            if len(attack_list) % 50 == 0:
                logger.info(f"  Generated {len(attack_list)}/{max_cases} perception cases")

        except Exception as e:
            continue

    with open(pkl_path, 'wb') as f:
        pickle.dump(attack_list, f)
    logger.info(f"Saved {len(attack_list)} V2X-Real perception cases to {pkl_path}")
    return attack_list


def generate_v2xreal_scenario(dataset, perception, max_cases=100):
    """Generate V2X-Real scenario test cases (~100)."""
    pkl_path = os.path.join(root, 'data/V2X-Real/test_scenario_attacks.pkl')
    backup_file(pkl_path)

    attacks = dataset.attacks

    def is_vehicle(vid):
        return isinstance(vid, int) and vid < 0

    scenario_list = []
    seen = set()

    for ci in range(len(attacks)):
        if len(scenario_list) >= max_cases:
            break
        a = attacks[ci]
        try:
            case = dataset.get_case(a['case_id'], tag='multi_frame', use_lidar=True)
            if len(case) < 10:
                continue
            ai, vi = a['attacker_vehicle_id'], a['victim_vehicle_id']
            if not is_vehicle(ai) or not is_vehicle(vi):
                continue

            key = (a['case_id'], ai, vi)
            if key in seen:
                continue

            frame = case[min(9, len(case) - 1)]
            if ai not in frame or vi not in frame:
                continue

            v2v_frame = {vid: vdata for vid, vdata in frame.items() if is_vehicle(vid)}
            gt = np.array(v2v_frame[ai]['gt_bboxes'])
            oids = v2v_frame[ai]['object_ids']
            candidates = [oid for oid in oids if oid not in [ai, vi]]
            if not candidates:
                continue

            # Pick target with best detection
            atk_pose = v2v_frame[ai]['lidar_pose']
            best_tgt = None
            best_score = -1
            for toid in candidates:
                tidx = oids.index(toid)
                bbox_map = bbox_sensor_to_map(gt[tidx], atk_pose)
                detected, iou, conf = check_detection(perception, v2v_frame, vi, bbox_map)
                if detected:
                    score = iou * conf
                    if score > best_score:
                        best_score = score
                        best_tgt = toid

            if best_tgt is None:
                continue

            frame_ids = list(range(min(len(case), 60)))
            vehicle_ids = list(v2v_frame.keys())

            entry = {
                'case_id': a['case_id'],
                'scenario_id': str(a['case_id']),
                'frame_ids': frame_ids,
                'vehicle_ids': vehicle_ids,
                'attacker_vehicle_id': ai,
                'victim_vehicle_id': vi,
                'target_id': best_tgt,
            }
            scenario_list.append(entry)
            seen.add(key)

            if len(scenario_list) % 20 == 0:
                logger.info(f"  Generated {len(scenario_list)}/{max_cases} scenario cases")

        except Exception as e:
            continue

    with open(pkl_path, 'wb') as f:
        pickle.dump(scenario_list, f)
    logger.info(f"Saved {len(scenario_list)} V2X-Real scenario cases to {pkl_path}")
    return scenario_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["OPV2V", "V2X-Real", "all"])
    args = parser.parse_args()

    if args.dataset in ["OPV2V", "all"]:
        logger.info("=== OPV2V Scenario Refinement ===")
        dataset = OPV2VDataset(root_path=os.path.join(root, "data/OPV2V"),
                                mode="test", dataset_name="OPV2V")
        perception = OpencoodPerception(fusion_method="intermediate",
                                         model_name="pointpillar", dataset_name="OPV2V")
        refine_opv2v_scenario(dataset, perception, max_cases=100)
        del perception
        import torch; torch.cuda.empty_cache()

    if args.dataset in ["V2X-Real", "all"]:
        logger.info("\n=== V2X-Real Test Case Generation ===")
        os.environ['DATASET_NAME'] = 'V2X-Real'
        # Need to reimport for V2X-Real
        from importlib import reload
        import mvp.config; reload(mvp.config)

        dataset_vr = OPV2VDataset(root_path=os.path.join(root, "data/V2X-Real"),
                                    mode="test", dataset_name="V2X-Real")
        perception_vr = OpencoodPerception(fusion_method="intermediate",
                                            model_name="pointpillar", dataset_name="V2X-Real")
        generate_v2xreal_perception(dataset_vr, perception_vr, max_cases=300)
        generate_v2xreal_scenario(dataset_vr, perception_vr, max_cases=100)

    logger.info("\nDone!")
