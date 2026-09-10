"""
Example: Run the integrated PoseGuard defense pipeline.

Demonstrates the full defense:
  1. Run fused perception (collaborative) and ego-only perception
  2. Identify safety-critical objects via trajectory prediction
  3. Filter: check fused vs ego disagreement on critical objects
  4. Anomaly detection (local L1) on disagreeing objects only
  5. Fallback to ego-only detection if anomalous

Usage:
  CUDA_VISIBLE_DEVICES=0 python test/test_integrated_defense.py
"""
import os, sys, pickle, copy, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.attack.perturbation_network import PerturbationNetwork
from mvp.defense.integrated_defender import IntegratedDefender, LinearPredictor
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d
from mvp.util import set_seed


def main():
    # --- Setup ---
    warp_patches = _apply_warp_patches()
    perception = build_perception('pointpillar')
    perception.model.eval()
    dataset = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')

    with open('data/OPV2V/attack/lidar_shift.pkl', 'rb') as f:
        test_cases = pickle.load(f)

    # Setup attacker with PertNet
    atk = LidarShiftVoxelwiseAttacker(perception, dataset, beta=2.0)
    pertnet_path = 'models/perturbation_net_paper_pointpillar/perturbation_net_ep35.pt'
    if os.path.exists(pertnet_path):
        ckpt = torch.load(pertnet_path, map_location='cpu')
        pertnet = PerturbationNetwork(
            feature_channels=ckpt['feature_channels'],
            geo_channels=ckpt['geo_channels']).to(perception.device)
        pertnet.load_state_dict(ckpt['model_state'])
        pertnet.eval()
        atk.pertnet = pertnet
        atk.pertnet_epsilon = 10.0

    # Setup integrated defender
    defender = IntegratedDefender(
        perception,
        safety_threshold=5.0,        # objects within 5m are critical
        anomaly_threshold=0.5,        # local anomaly above this triggers alert
        disagree_iou_threshold=0.5,   # fused/ego IoU below this = disagreement
        disagree_dist_threshold=1.0,  # center dist above this = disagreement
        anomaly_method="local_lucia", # use local L1 comparison
        predictor=LinearPredictor(predict_frames=20, time_step=0.5),
    )

    # --- Run on test cases ---
    n_cases = min(10, len(test_cases))
    n_detected = 0
    n_fallback = 0
    n_total = 0

    for ci in range(n_cases):
        tc = test_cases[ci]
        meta = tc['attack_meta']

        cache_file = f'data/OPV2V/attack_cache_paper/{ci:06d}.pkl'
        if not os.path.exists(cache_file):
            continue

        cached = pickle.load(open(cache_file, 'rb'))
        case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
        frame = case[9]
        ai = meta['attacker_vehicle_id']
        vi = meta['victim_vehicle_id']
        if ai not in frame or vi not in frame:
            continue

        atk_pose = frame[ai]['lidar_pose']
        vic_pose = frame[vi]['lidar_pose']
        bbox_tgt_ego = bbox_map_to_sensor(
            bbox_sensor_to_map(np.array([cached['bbox_tgt']]), atk_pose), vic_pose)[0]

        # --- Attack without defense ---
        result_atk = atk.run_multi_vehicle(frame, {
            'attacker_vehicle_id': ai, 'victim_vehicle_id': vi,
            'bbox_to_remove': cached['bbox_orig'], 'bbox_to_spoof': cached['bbox_tgt'],
        })
        pred_atk = result_atk['pred_bboxes']
        if len(pred_atk) > 0:
            d = np.linalg.norm(pred_atk[:, :2] - bbox_tgt_ego[:2], axis=1)
            iou_atk = float(iou3d(pred_atk[d.argmin()], bbox_tgt_ego))
        else:
            iou_atk = 0.0

        # --- Attack with integrated defense ---
        # Create attacked frame (inject spoofed point cloud)
        frame_attacked = copy.deepcopy(frame)
        frame_attacked[ai]['lidar'] = cached['spoof_pcd']

        # Ego future trajectory (simplified: straight line forward)
        ego_future = np.zeros((20, 2))
        ego_vel = np.array([2.0, 0.0])  # assume 2 m/s forward
        for t in range(20):
            ego_future[t] = ego_vel * (t + 1) * 0.5

        try:
            final_bboxes, final_scores, defense_info = defender.defend(
                frame_attacked, vi,
                ego_future_traj=ego_future)

            if len(final_bboxes) > 0:
                d = np.linalg.norm(final_bboxes[:, :2] - bbox_tgt_ego[:2], axis=1)
                iou_def = float(iou3d(final_bboxes[d.argmin()], bbox_tgt_ego))
            else:
                iou_def = 0.0

            n_total += 1
            if defense_info['n_anomalous'] > 0:
                n_detected += 1
            if defense_info['n_fallback'] > 0:
                n_fallback += 1

            print(f"  case {ci}: atk_IoU={iou_atk:.3f} → def_IoU={iou_def:.3f} | "
                  f"critical={defense_info['n_critical']} "
                  f"disagree={defense_info['n_disagree']} "
                  f"anomalous={defense_info['n_anomalous']} "
                  f"fallback={defense_info['n_fallback']}")

        except Exception as e:
            print(f"  case {ci}: defense error: {e}")
            n_total += 1

        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Integrated Defense Summary ({n_total} cases)")
    print(f"{'='*60}")
    print(f"  Attacks detected:   {n_detected}/{n_total} ({100*n_detected/max(n_total,1):.0f}%)")
    print(f"  Fallbacks applied:  {n_fallback}/{n_total} ({100*n_fallback/max(n_total,1):.0f}%)")
    print()
    print("Pipeline stages:")
    print("  1. Safety estimator: identify objects near ego's future path")
    print("  2. Disagreement filter: compare fused vs ego-only detections")
    print("  3. Anomaly detection: local L1 on disagreeing critical objects")
    print("  4. Fallback: replace with ego detection if anomalous")


if __name__ == '__main__':
    main()
