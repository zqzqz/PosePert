"""
Example: Run PosePert perception attack on a single frame.

Demonstrates the full attack pipeline:
  1. Ray-cast initialization (beta=1)
  2. Beta-scaled attack (beta=2.0)
  3. Full attack with PertNet correction

Usage:
  CUDA_VISIBLE_DEVICES=0 python test/test_posepert_attack.py
"""
import os, sys, pickle, copy, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.attack.perturbation_network import PerturbationNetwork
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d
from mvp.util import set_seed


def main():
    # --- Setup ---
    warp_patches = _apply_warp_patches()
    perception = build_perception('pointpillar')
    perception.model.eval()
    dataset = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')

    # Load test cases
    with open('data/OPV2V/attack/lidar_shift.pkl', 'rb') as f:
        test_cases = pickle.load(f)

    # Pick a test case
    tc = test_cases[0]
    meta = tc['attack_meta']
    ao = tc['attack_opts']
    case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
    frame = case[9]
    ai = meta['attacker_vehicle_id']
    vi = meta['victim_vehicle_id']

    atk_pose = frame[ai]['lidar_pose']
    vic_pose = frame[vi]['lidar_pose']

    # Target bbox in victim frame
    cache_file = f'data/OPV2V/attack_cache_paper/{0:06d}.pkl'
    if os.path.exists(cache_file):
        cached = pickle.load(open(cache_file, 'rb'))
        bbox_orig = cached['bbox_orig']
        bbox_tgt = cached['bbox_tgt']
    else:
        # Generate shift manually
        obj_idx = frame[ai]['object_ids'].index(ao['object_id'])
        bbox_orig = frame[ai]['gt_bboxes'][obj_idx].copy()
        bbox_tgt = bbox_orig.copy()
        bbox_tgt[0] += ao['shift_distance'] * np.cos(ao['shift_direction'])
        bbox_tgt[1] += ao['shift_distance'] * np.sin(ao['shift_direction'])

    bbox_tgt_ego = bbox_map_to_sensor(
        bbox_sensor_to_map(np.array([bbox_tgt]), atk_pose), vic_pose)[0]

    # --- Stage 1: Ray-cast only (beta=1) ---
    print("\n=== Stage 1: Ray-cast (beta=1) ===")
    atk1 = LidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)
    result1 = atk1.run_multi_vehicle(frame, {
        'attacker_vehicle_id': ai, 'victim_vehicle_id': vi,
        'bbox_to_remove': bbox_orig, 'bbox_to_spoof': bbox_tgt,
    })
    pred1 = result1['pred_bboxes']
    if len(pred1) > 0:
        d = np.linalg.norm(pred1[:, :2] - bbox_tgt_ego[:2], axis=1)
        iou1 = float(iou3d(pred1[d.argmin()], bbox_tgt_ego))
    else:
        iou1 = 0.0
    print(f"  IoU with target: {iou1:.3f} ({len(pred1)} detections)")

    # --- Stage 2: Beta-scaled (beta=2.0) ---
    print("\n=== Stage 2: +Beta scaling (beta=2.0) ===")
    atk2 = LidarShiftVoxelwiseAttacker(perception, dataset, beta=2.0)
    result2 = atk2.run_multi_vehicle(frame, {
        'attacker_vehicle_id': ai, 'victim_vehicle_id': vi,
        'bbox_to_remove': bbox_orig, 'bbox_to_spoof': bbox_tgt,
    })
    pred2 = result2['pred_bboxes']
    if len(pred2) > 0:
        d = np.linalg.norm(pred2[:, :2] - bbox_tgt_ego[:2], axis=1)
        iou2 = float(iou3d(pred2[d.argmin()], bbox_tgt_ego))
    else:
        iou2 = 0.0
    print(f"  IoU with target: {iou2:.3f} ({len(pred2)} detections)")

    # --- Stage 3: Full attack with PertNet ---
    print("\n=== Stage 3: +PertNet ===")
    atk3 = LidarShiftVoxelwiseAttacker(perception, dataset, beta=2.0)
    pertnet_path = 'models/perturbation_net_paper_pointpillar/perturbation_net_ep35.pt'
    if os.path.exists(pertnet_path):
        ckpt = torch.load(pertnet_path, map_location='cpu')
        pertnet = PerturbationNetwork(
            feature_channels=ckpt['feature_channels'],
            geo_channels=ckpt['geo_channels']).to(perception.device)
        pertnet.load_state_dict(ckpt['model_state'])
        pertnet.eval()
        atk3.pertnet = pertnet
        atk3.pertnet_epsilon = 10.0
        print(f"  Loaded PertNet from {pertnet_path}")
    else:
        print(f"  PertNet checkpoint not found at {pertnet_path}, skipping")

    result3 = atk3.run_multi_vehicle(frame, {
        'attacker_vehicle_id': ai, 'victim_vehicle_id': vi,
        'bbox_to_remove': bbox_orig, 'bbox_to_spoof': bbox_tgt,
    })
    pred3 = result3['pred_bboxes']
    if len(pred3) > 0:
        d = np.linalg.norm(pred3[:, :2] - bbox_tgt_ego[:2], axis=1)
        iou3 = float(iou3d(pred3[d.argmin()], bbox_tgt_ego))
    else:
        iou3 = 0.0
    print(f"  IoU with target: {iou3:.3f} ({len(pred3)} detections)")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Attack Summary:")
    print(f"  Ray-cast (beta=1):   IoU = {iou1:.3f}")
    print(f"  +Beta (beta=2.0):    IoU = {iou2:.3f}")
    print(f"  +PertNet (full):     IoU = {iou3:.3f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
