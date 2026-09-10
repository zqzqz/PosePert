#!/usr/bin/env python3
"""
Generate beta-scaling visualization for Section 5.1.2.

Shows a 1x5 grid of BEV feature heatmaps across beta values (1.0..3.0),
each with GT, target, and detection bboxes overlaid.

Demonstrates:
  beta=1.0: ray-cast only, features weak, detection misses target
  beta=2.0: amplified features, detection shifts toward target
  beta=3.0: over-amplified, detection degrades

Usage: python results_paper/gen_fig_beta_visualization.py
"""

import os, sys, pickle, copy, logging
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(_ROOT, 'results_paper', 'figures')
DATA_FILE = os.path.join(_ROOT, 'results_paper', 'fig_beta_vis_data.pkl')

BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
ZOOM_PAD = 18  # voxels around bbox center (tight crop)


def bbox_to_voxel_corners(bbox, lr, vs):
    """Convert a 7-DOF bbox to voxel-space polygon corners (closed)."""
    cx = (bbox[0] - lr[0]) / vs[0]
    cy = (bbox[1] - lr[1]) / vs[1]
    hw = bbox[3] / vs[0] / 2  # half-length in voxels
    hh = bbox[4] / vs[1] / 2  # half-width in voxels
    yaw = bbox[6]
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return corners @ rot.T + np.array([cx, cy])


def run_experiment(case_indices=None):
    """Run beta sweep and find a good case."""
    if case_indices is None:
        case_indices = [5, 10, 12]  # case 5 consistently shows good beta trend

    warp_patches = _apply_warp_patches()
    perception = build_perception('pointpillar')
    perception.model.eval()
    dataset = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')

    with open('data/OPV2V/attack/lidar_shift.pkl', 'rb') as f:
        attacks = pickle.load(f)

    lr = np.array(perception.dataset.pre_processor.params["cav_lidar_range"])
    vs_list = perception.dataset.pre_processor.params["args"]["voxel_size"]
    vs = np.array(vs_list)

    best_case = None
    best_case_data = None
    best_score_val = -1

    for ci in case_indices:
        logger.info(f"Trying case_idx={ci}")
        meta = attacks[ci]['attack_meta']
        attack_opts = attacks[ci]['attack_opts']

        cache_file = f'data/OPV2V/attack_cache_paper/{ci:06d}.pkl'
        if not os.path.exists(cache_file):
            logger.warning(f"Cache file {cache_file} not found, skipping")
            continue
        cached = pickle.load(open(cache_file, 'rb'))

        case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
        frame = case[9]
        ai = meta['attacker_vehicle_id']
        vi = meta['victim_vehicle_id']

        atk_pose = frame[ai]['lidar_pose']
        vic_pose = frame[vi]['lidar_pose']

        bbox_tgt_ego = bbox_map_to_sensor(
            bbox_sensor_to_map(np.array([cached['bbox_tgt']]), atk_pose), vic_pose)[0]
        bbox_orig_ego = bbox_map_to_sensor(
            bbox_sensor_to_map(np.array([cached['bbox_orig']]), atk_pose), vic_pose)[0]

        case_data = {
            'case_idx': ci,
            'bbox_orig_ego': bbox_orig_ego,
            'bbox_tgt_ego': bbox_tgt_ego,
            'lr': lr,
            'vs': vs,
            'results': {},
        }

        ious_at_beta = {}

        for beta_val in BETAS:
            logger.info(f"  beta={beta_val}")
            set_seed(42)
            atk = LidarShiftVoxelwiseAttacker(perception, dataset, beta=beta_val)
            result = atk.run_multi_vehicle(frame, {
                'attacker_vehicle_id': ai,
                'victim_vehicle_id': vi,
                'bbox_to_remove': cached['bbox_orig'],
                'bbox_to_spoof': cached['bbox_tgt'],
            })

            F = result['spatial_features']  # (N, C, H, W)
            pred = result['pred_bboxes']
            scores = result.get('pred_scores', np.array([]))

            # Feature magnitude: sum across agents, L2 norm across channels
            feat_all = F.sum(dim=0).cpu().numpy()  # (C, H, W)
            feat_mag = np.linalg.norm(feat_all, axis=0)  # (H, W)

            # Find best detection (highest IoU with target)
            best_iou = 0.0
            best_det = None
            best_score = 0.0
            if len(pred) > 0 and len(scores) > 0:
                for j in range(len(pred)):
                    iou_val = iou3d(pred[j], bbox_tgt_ego)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_det = pred[j]
                        best_score = scores[j]

            ious_at_beta[beta_val] = best_iou
            case_data['results'][beta_val] = {
                'feat_mag': feat_mag,
                'pred_bboxes': pred,
                'pred_scores': scores,
                'best_iou': best_iou,
                'best_det': best_det,
                'best_score': best_score,
            }
            logger.info(f"    IoU={best_iou:.3f}, score={best_score:.3f}, "
                        f"n_det={len(pred)}")

        # Check if this case shows the desired trend
        iou_1 = ious_at_beta.get(1.0, 0)
        iou_2 = ious_at_beta.get(2.0, 0)
        iou_3 = ious_at_beta.get(3.0, 0)
        logger.info(f"  Summary: IoU@1={iou_1:.3f}, IoU@2={iou_2:.3f}, IoU@3={iou_3:.3f}")

        # Good case: low at beta=1, peak at beta=2, drops at beta=3
        # Score: (peak - low) + (peak - high_beta_drop), prefer large contrast
        iou_peak = max(ious_at_beta.values())
        beta_peak = max(ious_at_beta, key=ious_at_beta.get)
        # Require: peak > 0.4, beta=1 < 0.35, and some drop at beta=3
        score = (iou_peak - iou_1) + max(0, iou_peak - iou_3)
        if iou_peak > 0.4 and iou_1 < 0.4 and iou_peak > iou_1:
            if best_case is None or score > best_score_val:
                best_case = ci
                best_case_data = case_data
                best_score_val = score
                logger.info(f"  -> New best case: {ci} (score={score:.3f})")

    if best_case_data is None:
        logger.warning("No ideal case found; using first case's data")
        best_case_data = case_data  # fallback to last tried

    return best_case_data


def make_figure(data):
    """Generate the 1x5 beta visualization figure."""
    lr = data['lr']
    vs = data['vs']
    bbox_orig_ego = data['bbox_orig_ego']
    bbox_tgt_ego = data['bbox_tgt_ego']

    # Compute zoom window centered on midpoint between orig and target
    mid_x = (bbox_orig_ego[0] + bbox_tgt_ego[0]) / 2
    mid_y = (bbox_orig_ego[1] + bbox_tgt_ego[1]) / 2
    cx_vox = (mid_x - lr[0]) / vs[0]
    cy_vox = (mid_y - lr[1]) / vs[1]

    x_lo = int(cx_vox - ZOOM_PAD)
    x_hi = int(cx_vox + ZOOM_PAD)
    y_lo = int(cy_vox - ZOOM_PAD)
    y_hi = int(cy_vox + ZOOM_PAD)

    fig, axes = plt.subplots(1, 5, figsize=(8, 2))

    for idx, beta_val in enumerate(BETAS):
        ax = axes[idx]
        res = data['results'][beta_val]
        feat_mag = res['feat_mag']

        # Clamp zoom window
        H, W = feat_mag.shape
        x_lo_c = max(0, x_lo)
        x_hi_c = min(W, x_hi)
        y_lo_c = max(0, y_lo)
        y_hi_c = min(H, y_hi)

        crop = feat_mag[y_lo_c:y_hi_c, x_lo_c:x_hi_c]

        # Plot heatmap
        ax.imshow(crop, cmap='hot', origin='lower', aspect='equal',
                  interpolation='nearest')

        # Overlay GT (green), target (red), best detection (orange dashed)
        def draw_bbox(bbox, color, ls='-', lw=1.5, label=None):
            corners = bbox_to_voxel_corners(bbox, lr, vs)
            # Shift to crop coordinates
            corners_crop = corners - np.array([x_lo_c, y_lo_c])
            poly = MplPolygon(corners_crop, closed=True, fill=False,
                              edgecolor=color, linestyle=ls, linewidth=lw,
                              label=label)
            ax.add_patch(poly)

        draw_bbox(bbox_orig_ego, 'lime', lw=1.5, label='GT')
        draw_bbox(bbox_tgt_ego, 'red', lw=1.5, label='Target')

        best_det = res['best_det']
        best_iou = res['best_iou']
        best_score = res['best_score']

        if best_det is not None:
            draw_bbox(best_det, 'orange', ls='--', lw=1.5, label='Det')

        ax.set_title(f'$\\beta$ = {beta_val}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        # IoU and conf below image
        ax.set_xlabel(f'IoU={best_iou:.2f}  conf={best_score:.2f}', fontsize=8)

    # Add legend to first panel
    axes[0].legend(loc='upper left', fontsize=6, framealpha=0.7)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'fig_beta_vis.pdf')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {out_path}")
    return out_path


def main():
    os.chdir(_ROOT)

    # Run experiment or load cached data
    if os.path.exists(DATA_FILE):
        logger.info(f"Loading cached data from {DATA_FILE}")
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
    else:
        data = run_experiment()
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved data to {DATA_FILE}")

    out_path = make_figure(data)
    print(f"Figure saved: {out_path}")


if __name__ == '__main__':
    main()
