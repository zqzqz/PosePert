"""
Data collection for perturbation network training.

Collects (F_orig, F_diff, geo_encoding, target_info) tuples from OPV2V
attack scenarios. Cached to disk for efficient training.
"""

import os
import sys
import numpy as np
import torch
import copy
import pickle
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_network import (
    build_geometric_encoding, get_active_zone_bounds)
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.util import set_seed
from opencood.tools import train_utils


def collect_training_data(dataset, perception, n_cases=100,
                          shifts=None, save_dir='data/perturbation_train'):
    """
    Collect training data for the perturbation network.

    For each attack case and shift distance, computes:
    - F_orig: spatial features for all agents (pre-backbone)
    - F_diff: ray-cast feature difference at attacker's slot
    - geo_encoding: geometric context
    - batch_data: collated batch for running the model
    - target_info: bbox locations, anchor indices, vehicle poses

    Args:
        dataset: OPV2VDataset
        perception: OpencoodPerception (intermediate fusion)
        n_cases: number of attack cases to process
        shifts: list of (dx, dy) shift vectors in attacker sensor frame
        save_dir: directory to save cached data
    """
    if shifts is None:
        # 32 randomized shifts: random direction and distance, no rotation
        n_shifts_per_case = 32
        shifts = []
        for _ in range(n_shifts_per_case):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.5, 2.0)
            shifts.append((dist * np.cos(angle), dist * np.sin(angle)))

    os.makedirs(save_dir, exist_ok=True)
    attacks = dataset.attacks

    lidar_range = perception.dataset.pre_processor.params["cav_lidar_range"]
    voxel_size = perception.dataset.pre_processor.params["args"]["voxel_size"]
    H = int((lidar_range[4] - lidar_range[1]) / voxel_size[1])
    W = int((lidar_range[3] - lidar_range[0]) / voxel_size[0])

    attacker_obj = LidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)

    samples = []
    sample_idx = 0

    for ci in range(min(n_cases, len(attacks))):
        try:
            attack = attacks[ci]
            case = dataset.get_case(attack['case_id'], tag='multi_frame',
                                     use_lidar=True)
            frame = case[min(9, len(case) - 1)]
            atk_id = attack['attacker_vehicle_id']
            vic_id = attack['victim_vehicle_id']

            # Find target object
            gt = np.array(frame[atk_id]['gt_bboxes'])
            obj_ids = frame[atk_id]['object_ids']
            ti = next((i for i, o in enumerate(obj_ids)
                        if o not in [atk_id, vic_id]), None)
            if ti is None:
                continue
            bbox_orig = gt[ti]

            # Get vehicle ordering and poses
            base_data_dict = perception.retrieve_base_data(frame, vic_id)
            attacker_index = list(base_data_dict.keys()).index(atk_id)
            ego_index = list(base_data_dict.keys()).index(vic_id)

            # Vehicle positions in ego frame
            atk_pose = frame[atk_id]['lidar_pose']
            vic_pose = frame[vic_id]['lidar_pose']
            vehicle_poses = []
            for vid in base_data_dict.keys():
                vpose = frame[vid]['lidar_pose']
                # Transform to ego frame (x, y only)
                dx = vpose[0] - vic_pose[0]
                dy = vpose[1] - vic_pose[1]
                vehicle_poses.append((dx, dy))

            # Get normal spatial features
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_orig, batch_data = attacker_obj._get_spatial_features(
                frame, vic_id)

            # Convert bbox to ego frame
            bbox_orig_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(bbox_orig, atk_pose), vic_pose)

            for shift_dx, shift_dy in shifts:
                bbox_tgt = bbox_orig.copy()
                bbox_tgt[0] += shift_dx
                bbox_tgt[1] += shift_dy

                bbox_tgt_ego = bbox_map_to_sensor(
                    bbox_sensor_to_map(bbox_tgt, atk_pose), vic_pose)

                # Compute active zone
                bounds = get_active_zone_bounds(
                    bbox_orig_ego, bbox_tgt_ego, lidar_range, voxel_size,
                    H, W, padding=2)
                h_lo, h_hi, w_lo, w_hi = bounds
                if h_hi <= h_lo or w_hi <= w_lo:
                    continue

                # Ray-cast feature difference
                atk_pcd = frame[atk_id]['lidar']
                pcd_spoofed = attacker_obj._generate_spoof_pcd(
                    atk_pcd.copy(), bbox_tgt, bbox_orig)
                case_spoof = copy.deepcopy(frame)
                case_spoof[atk_id]['lidar'] = pcd_spoofed
                set_seed(42, set_python=False, set_numpy=False,
                          set_torch=True)
                F_spoof_full, _ = attacker_obj._get_spatial_features(
                    case_spoof, vic_id)
                F_diff = F_spoof_full[attacker_index] - F_orig[attacker_index]

                # Crop to active zone
                f_orig_crop = F_orig[attacker_index][
                    :, h_lo:h_hi, w_lo:w_hi].cpu()
                f_diff_crop = F_diff[:, h_lo:h_hi, w_lo:w_hi].cpu()

                # Geometric encoding
                geo_enc = build_geometric_encoding(
                    bbox_orig_ego, bbox_tgt_ego, vehicle_poses,
                    ego_index, attacker_index, bounds,
                    lidar_range, voxel_size, H, W,
                    max_vehicles=4)

                sample = {
                    'f_orig_crop': f_orig_crop,
                    'f_diff_crop': f_diff_crop,
                    'geo_encoding': geo_enc,
                    'bbox_orig_ego': bbox_orig_ego,
                    'bbox_tgt_ego': bbox_tgt_ego,
                    'bounds': bounds,
                    'attacker_index': attacker_index,
                    'ego_index': ego_index,
                    'n_agents': F_orig.shape[0],
                    # Save full features for training forward pass
                    'F_orig': F_orig.cpu(),
                    'record_len': batch_data['ego']['record_len'].cpu(),
                    'batch_data_keys': {
                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in batch_data['ego'].items()
                        if k in ['record_len', 'anchor_box', 'all_anchors',
                                 'num_anchors_per_location',
                                 'pairwise_t_matrix', 'label_dict']
                    },
                }

                # Save individual sample
                save_path = os.path.join(save_dir, f'sample_{sample_idx:04d}.pt')
                torch.save(sample, save_path)
                samples.append(save_path)
                sample_idx += 1

            del F_orig, F_spoof_full, batch_data
            torch.cuda.empty_cache()

            if (ci + 1) % 10 == 0:
                print(f'[{ci+1}/{n_cases}] {sample_idx} samples collected')

        except Exception:
            traceback.print_exc()
            continue

    # Save index
    index_path = os.path.join(save_dir, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump({'sample_paths': samples, 'n_samples': len(samples)}, f)

    print(f'\nCollected {len(samples)} samples, saved to {save_dir}')
    return samples


class PerturbationDataset(torch.utils.data.Dataset):
    """Dataset for loading cached perturbation training samples."""

    def __init__(self, data_dir='data/perturbation_train'):
        index_path = os.path.join(data_dir, 'index.pkl')
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        self.sample_paths = index['sample_paths']

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        return torch.load(self.sample_paths[idx], map_location='cpu')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases', type=int, default=100)
    parser.add_argument('--save_dir', type=str,
                        default='data/perturbation_train')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Dataset split to collect from')
    args = parser.parse_args()

    dataset = OPV2VDataset(root_path='data/OPV2V', mode=args.mode,
                           dataset_name='OPV2V')
    perception = OpencoodPerception(fusion_method='intermediate',
                                    model_name='pointpillar',
                                    dataset_name='OPV2V')
    collect_training_data(dataset, perception, n_cases=args.n_cases,
                          save_dir=args.save_dir)
