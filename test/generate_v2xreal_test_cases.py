"""
Generate V2X-Real test cases for perception-level and scenario-level attacks.
Produces files in the same format as OPV2V:
  - data/V2X-Real/attack/lidar_shift.pkl  (perception attack cases)
  - data/V2X-Real/test_scenario_attacks.pkl (scenario attack cases)

Usage:
  DATASET_NAME=V2X-Real python test/generate_v2xreal_test_cases.py
"""
import os
import sys
import pickle
import random
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.shift_rotation import sample_shift_rotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


def generate_perception_cases(dataset, max_cases=300):
    """Generate perception attack test cases in OPV2V lidar_shift.pkl format."""
    attack_list = []

    for ci in range(dataset.attacks.__len__()):
        if len(attack_list) >= max_cases:
            break

        a = dataset.attacks[ci]
        try:
            case = dataset.get_case(a['case_id'], tag='multi_frame', use_lidar=False)
            n_frames = len(case)
            if n_frames < 10:
                continue

            frame = case[min(9, n_frames - 1)]
            ai = a['attacker_vehicle_id']
            vi = a['victim_vehicle_id']

            # V2V filter: keep only vehicles (negative IDs)
            if ai >= 0 or vi >= 0:
                continue
            if ai not in frame or vi not in frame:
                continue

            gt_bboxes = np.array(frame[ai]['gt_bboxes'])
            object_ids = frame[ai]['object_ids']

            # Find target candidates: non-ego, non-participant objects
            candidates = [(i, oid) for i, oid in enumerate(object_ids)
                          if oid not in [ai, vi]]
            if not candidates:
                continue

            # Pick a random target
            target_idx, target_oid = random.choice(candidates)

            # Get bboxes across all frames
            bboxes_all = []
            valid = True
            for fi in range(min(10, n_frames)):
                f = case[fi]
                if ai not in f:
                    valid = False
                    break
                gt_fi = np.array(f[ai]['gt_bboxes'])
                oids_fi = f[ai]['object_ids']
                if target_oid in oids_fi:
                    tidx = oids_fi.index(target_oid)
                    bboxes_all.append(gt_fi[tidx])
                else:
                    valid = False
                    break
            if not valid or len(bboxes_all) < 10:
                continue

            bboxes = np.array(bboxes_all[:10])

            # Generate random shift
            shift_distance = random.random() * 0.5 + 0.5  # 0.5 to 1.0m
            shift_direction = random.random() * 2 * np.pi
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
                    'object_id': target_oid,
                    'shift_direction': shift_direction,
                    'shift_distance': shift_distance,
                    'rotation': rotation,
                },
                'attack_meta': {
                    'case_id': a['case_id'],
                    'scenario_id': str(a['case_id']),
                    'frame_ids': list(range(10)),
                    'attacker_vehicle_id': ai,
                    'victim_vehicle_id': vi,
                    'vehicle_ids': [vid for vid in frame.keys() if isinstance(vid, int)],
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
                logger.info(f"Perception cases: {len(attack_list)}/{max_cases}")

        except Exception as e:
            logger.warning(f"Skip case {ci}: {e}")
            continue

    return attack_list


def generate_scenario_cases(dataset, max_cases=300):
    """Generate scenario attack test cases in OPV2V test_scenario_attacks.pkl format."""
    scenario_list = []
    seen = set()

    for ci in range(dataset.attacks.__len__()):
        if len(scenario_list) >= max_cases:
            break

        a = dataset.attacks[ci]
        try:
            case = dataset.get_case(a['case_id'], tag='multi_frame', use_lidar=False)
            n_frames = len(case)
            if n_frames < 20:
                continue

            ai = a['attacker_vehicle_id']
            vi = a['victim_vehicle_id']
            if ai >= 0 or vi >= 0:
                continue

            # Deduplicate by (case_id, attacker, victim)
            key = (a['case_id'], ai, vi)
            if key in seen:
                continue

            frame = case[min(9, n_frames - 1)]
            if ai not in frame or vi not in frame:
                continue

            # Find target: non-ego, non-participant objects
            gt_bboxes = np.array(frame[ai]['gt_bboxes'])
            object_ids = frame[ai]['object_ids']
            candidates = [oid for oid in object_ids if oid not in [ai, vi]]
            if not candidates:
                continue

            target_id = random.choice(candidates)

            # Use all available frames (up to 60)
            frame_ids = list(range(min(n_frames, 60)))

            vehicle_ids = [vid for vid in frame.keys() if isinstance(vid, int) and vid < 0]

            entry = {
                'case_id': a['case_id'],
                'scenario_id': str(a['case_id']),
                'frame_ids': frame_ids,
                'vehicle_ids': vehicle_ids,
                'attacker_vehicle_id': ai,
                'victim_vehicle_id': vi,
                'target_id': target_id,
            }
            scenario_list.append(entry)
            seen.add(key)

            if len(scenario_list) % 50 == 0:
                logger.info(f"Scenario cases: {len(scenario_list)}/{max_cases}")

        except Exception as e:
            logger.warning(f"Skip scenario {ci}: {e}")
            continue

    return scenario_list


if __name__ == "__main__":
    dataset = OPV2VDataset(root_path=os.path.join(root, "data/V2X-Real"),
                            mode="test", dataset_name="V2X-Real")
    logger.info(f"V2X-Real dataset: {len(dataset.attacks)} existing attacks")

    # Generate perception test cases
    logger.info("Generating perception attack test cases...")
    perception_cases = generate_perception_cases(dataset, max_cases=300)
    save_dir = os.path.join(root, "data/V2X-Real/attack")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lidar_shift.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(perception_cases, f)
    logger.info(f"Saved {len(perception_cases)} perception cases to {save_path}")

    # Generate scenario test cases
    logger.info("Generating scenario attack test cases...")
    scenario_cases = generate_scenario_cases(dataset, max_cases=300)
    save_path = os.path.join(root, "data/V2X-Real/test_scenario_attacks.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(scenario_cases, f)
    logger.info(f"Saved {len(scenario_cases)} scenario cases to {save_path}")

    logger.info("Done!")
