"""
Example: Run PoseGuard defense on attacked features.

Demonstrates the defense pipeline:
  1. Run full attack (beta + PertNet)
  2. Compute global LUCIA trust scores (baseline)
  3. Compute local L1 anomaly scores (PoseGuard)
  4. Compare detection performance

Usage:
  CUDA_VISIBLE_DEVICES=0 python test/test_poseguard_defense.py
"""
import os, sys, pickle, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.attack.perturbation_network import PerturbationNetwork
from mvp.defense.lucia.lucia import LuciaDefender
from mvp.defense.lucia.local_lucia import LocalLuciaDefender
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.util import set_seed


def main():
    # --- Setup ---
    warp_patches = _apply_warp_patches()
    perception = build_perception('pointpillar')
    perception.model.eval()
    dataset = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')

    with open('data/OPV2V/attack/lidar_shift.pkl', 'rb') as f:
        test_cases = pickle.load(f)

    # Initialize defenses
    lucia_global = LuciaDefender(compression_ratio=32)
    lucia_local = LocalLuciaDefender(perception)

    # Load PertNet for full attack
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

    # Run on multiple cases
    n_cases = min(20, len(test_cases))
    l1_normals, l1_attacks = [], []
    trust_normals, trust_attacks = [], []

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

        # Get attacker index
        base = perception.retrieve_base_data(frame, vi)
        vids = list(base.keys())
        aidx = vids.index(ai)

        # Normal features
        set_seed(42, set_python=False, set_numpy=False, set_torch=True)
        F_n, _ = atk._get_spatial_features(frame, vi)

        # Attack features (full: beta + PertNet)
        result = atk.run_multi_vehicle(frame, {
            'attacker_vehicle_id': ai, 'victim_vehicle_id': vi,
            'bbox_to_remove': cached['bbox_orig'],
            'bbox_to_spoof': cached['bbox_tgt'],
        })
        F_a = result['spatial_features']
        pred_a = result['pred_bboxes']

        # --- Global LUCIA ---
        trust_n = lucia_global.compute_trust(F_n)
        trust_a = lucia_global.compute_trust(F_a)
        trust_normals.append(float(trust_n[aidx]))
        trust_attacks.append(float(trust_a[aidx]))

        # --- Local LUCIA (PoseGuard) ---
        if len(pred_a) > 0:
            _, l1_n = lucia_local.compute_local_trust(F_n, pred_a, ego_index=0)
            _, l1_a = lucia_local.compute_local_trust(F_a, pred_a, ego_index=0)
            l1_n_val = float(l1_n[:, aidx].max()) if l1_n.shape[1] > aidx else 0
            l1_a_val = float(l1_a[:, aidx].max()) if l1_a.shape[1] > aidx else 0
        else:
            l1_n_val, l1_a_val = 0, 0

        l1_normals.append(l1_n_val)
        l1_attacks.append(l1_a_val)

        print(f"  case {ci}: Global trust n={trust_n[aidx]:.3f} a={trust_a[aidx]:.3f} | "
              f"Local L1 n={l1_n_val:.0f} a={l1_a_val:.0f} "
              f"(ratio={l1_a_val/(l1_n_val+1e-8):.1f}x)")

        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Defense Summary ({len(l1_normals)} cases)")
    print(f"{'='*60}")

    # Global LUCIA: trust scores (lower = more suspicious)
    trust_n_arr = np.array(trust_normals)
    trust_a_arr = np.array(trust_attacks)
    print(f"\nGlobal LUCIA trust:")
    print(f"  Normal: mean={trust_n_arr.mean():.3f}")
    print(f"  Attack: mean={trust_a_arr.mean():.3f}")
    print(f"  Delta:  {trust_n_arr.mean()-trust_a_arr.mean():.4f} (small = undetectable)")

    # Local L1 (PoseGuard): higher = more suspicious
    l1_n_arr = np.array(l1_normals)
    l1_a_arr = np.array(l1_attacks)
    print(f"\nPoseGuard (Local L1):")
    print(f"  Normal: mean={l1_n_arr.mean():.0f}, max={l1_n_arr.max():.0f}")
    print(f"  Attack: mean={l1_a_arr.mean():.0f}, min={l1_a_arr.min():.0f}")
    print(f"  Ratio:  {l1_a_arr.mean()/(l1_n_arr.mean()+1e-8):.1f}x (large = detectable)")

    # ROC-based detection
    try:
        from sklearn.metrics import roc_curve, auc
        n = len(l1_normals)
        labels = np.array([0]*n + [1]*n)

        # Global LUCIA ROC
        fpr_g, tpr_g, _ = roc_curve(labels, np.concatenate([-trust_n_arr, -trust_a_arr]))
        auc_g = auc(fpr_g, tpr_g)
        idx_g = np.searchsorted(fpr_g, 0.05)
        tpr5_g = tpr_g[min(idx_g, len(tpr_g)-1)]

        # PoseGuard ROC
        fpr_l, tpr_l, _ = roc_curve(labels, np.concatenate([l1_n_arr, l1_a_arr]))
        auc_l = auc(fpr_l, tpr_l)
        idx_l = np.searchsorted(fpr_l, 0.05)
        tpr5_l = tpr_l[min(idx_l, len(tpr_l)-1)]

        print(f"\nROC Analysis:")
        print(f"  Global LUCIA: AUC={auc_g:.3f}, TPR@5%FPR={tpr5_g*100:.1f}%")
        print(f"  PoseGuard:    AUC={auc_l:.3f}, TPR@5%FPR={tpr5_l*100:.1f}%")
    except ImportError:
        print("\n  (Install scikit-learn for ROC analysis)")

    print(f"\nConclusion: PoseGuard detects the attack by localizing L1")
    print(f"comparison to object regions, achieving ~{l1_a_arr.mean()/(l1_n_arr.mean()+1e-8):.0f}x signal ratio.")


if __name__ == '__main__':
    main()
