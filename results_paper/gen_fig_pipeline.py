#!/usr/bin/env python3
"""
Generate perception attack pipeline visualization (2x3 grid).

Row 1: Attacker's feature map (single agent) at each step -- shows
       the raw effect of ray-casting, beta scaling, and PertNet on the
       compromised agent's contribution.
Row 2: Fused feature map (sum over all agents) at each step -- shows
       the detection-relevant features with bbox overlays.

Columns:
  1. Ray-cast (beta=1): After ray-casting spoofed points
  2. +beta scaling (beta=2.0): After amplifying features in active zone
  3. +PertNet: After PertNet correction on top of beta

Each panel overlays GT bbox (green), target bbox (red), best detection
(orange dashed) with IoU annotation.

Usage: python results_paper/gen_fig_pipeline.py
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
from matplotlib.lines import Line2D

from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_network import (
    PerturbationNetwork, build_geometric_encoding, get_active_zone_bounds)
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(_ROOT, 'results_paper', 'figures')
DATA_FILE = os.path.join(_ROOT, 'results_paper', 'fig_pipeline_vis_data_v2.pkl')
# This figure's loader needs a PertNet built with an explicit rank/hidden_dim, which the
# perturbation_net_paper_* checkpoints do not carry (they store beta instead). Keep the
# older low-rank checkpoint; link_local_data.sh stages it.
PERTNET_CKPT = os.path.join(_ROOT, 'models', 'perturbation_net', 'perturbation_net_best.pt')

ZOOM_PAD_FEAT = 22     # voxels around bbox center for feature heatmap


def bbox_to_voxel_corners(bbox, lr, vs):
    """Convert a 7-DOF bbox to voxel-space polygon corners (closed)."""
    cx = (bbox[0] - lr[0]) / vs[0]
    cy = (bbox[1] - lr[1]) / vs[1]
    hw = bbox[3] / vs[0] / 2
    hh = bbox[4] / vs[1] / 2
    yaw = bbox[6]
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return corners @ rot.T + np.array([cx, cy])


def find_best_detection(pred_bboxes, pred_scores, bbox_tgt):
    """Find detection with highest IoU to target."""
    best_iou = 0.0
    best_det = None
    best_score = 0.0
    if len(pred_bboxes) > 0 and len(pred_scores) > 0:
        for j in range(len(pred_bboxes)):
            iou_val = iou3d(pred_bboxes[j], bbox_tgt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_det = pred_bboxes[j]
                best_score = pred_scores[j]
    return best_det, best_iou, best_score


def load_pertnet(device):
    """Load trained PertNet from checkpoint."""
    ckpt = torch.load(PERTNET_CKPT, map_location='cpu')
    net = PerturbationNetwork(
        feature_channels=ckpt['feature_channels'],
        geo_channels=ckpt['geo_channels'],
        rank=ckpt['rank'],
        hidden_dim=ckpt['hidden_dim'])
    net.load_state_dict(ckpt['model_state'])
    net = net.to(device)
    net.eval()
    return net


def run_experiment(case_indices=None):
    """Run the 3 pipeline stages and collect data."""
    if case_indices is None:
        case_indices = [0, 5, 10, 15, 20, 25, 30, 35, 1, 2, 3, 4, 6, 7, 8, 9]

    warp_patches = _apply_warp_patches()
    perception = build_perception('pointpillar')
    perception.model.eval()
    dataset = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')
    device = perception.device

    pertnet = load_pertnet(device)

    with open('data/OPV2V/attack/lidar_shift.pkl', 'rb') as f:
        attacks = pickle.load(f)

    lr = np.array(perception.dataset.pre_processor.params["cav_lidar_range"])
    vs_list = perception.dataset.pre_processor.params["args"]["voxel_size"]
    vs = np.array(vs_list)

    best_data = None
    best_iou_full = -1

    for ci in case_indices:
        logger.info(f"Trying case_idx={ci}")
        if ci >= len(attacks):
            continue

        meta = attacks[ci]['attack_meta']
        attack_opts = attacks[ci]['attack_opts']

        cache_file = f'data/OPV2V/attack_cache_paper/{ci:06d}.pkl'
        if not os.path.exists(cache_file):
            logger.warning(f"Cache not found: {cache_file}, skipping")
            continue
        cached = pickle.load(open(cache_file, 'rb'))

        case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
        frame = case[9]
        ai = meta['attacker_vehicle_id']
        vi = meta['victim_vehicle_id']

        atk_pose = frame[ai]['lidar_pose']
        vic_pose = frame[vi]['lidar_pose']

        # Original and target bboxes in attacker frame
        bbox_orig_atk = np.array(cached['bbox_orig']).copy()
        shift_dir = attack_opts.get('shift_direction', 0.0)

        # Create target with 1m shift + 10 degree rotation
        bbox_tgt_atk = bbox_orig_atk.copy()
        bbox_tgt_atk[0] += 1.0 * np.cos(shift_dir)
        bbox_tgt_atk[1] += 1.0 * np.sin(shift_dir)
        bbox_tgt_atk[6] += np.radians(10)

        # Convert to ego (victim) frame
        bbox_orig_ego = bbox_map_to_sensor(
            bbox_sensor_to_map(np.array([bbox_orig_atk]), atk_pose), vic_pose)[0]
        bbox_tgt_ego = bbox_map_to_sensor(
            bbox_sensor_to_map(np.array([bbox_tgt_atk]), atk_pose), vic_pose)[0]

        atk_opts_common = {
            'attacker_vehicle_id': ai,
            'victim_vehicle_id': vi,
            'bbox_to_remove': bbox_orig_atk,
            'bbox_to_spoof': bbox_tgt_atk,
        }

        # Get attacker index
        base_data_dict = perception.retrieve_base_data(frame, vi)
        attacker_index = list(base_data_dict.keys()).index(ai)

        results = {}

        # --- Stage 1: Ray-cast only (beta=1, no PertNet) ---
        atk1 = LidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)
        res1 = atk1.run_multi_vehicle(frame, atk_opts_common)
        det1, iou1, sc1 = find_best_detection(
            res1['pred_bboxes'], res1['pred_scores'], bbox_tgt_ego)

        F1 = res1['spatial_features']
        feat1_atk = np.linalg.norm(F1[attacker_index].cpu().numpy(), axis=0)
        feat1_fused = np.linalg.norm(F1.sum(dim=0).cpu().numpy(), axis=0)

        results['raycast'] = {
            'feat_atk': feat1_atk,
            'feat_fused': feat1_fused,
            'best_det': det1, 'best_iou': iou1, 'best_score': sc1,
        }
        logger.info(f"  Ray-cast: IoU={iou1:.3f}")

        # --- Stage 2: beta scaling (beta=2.0, no PertNet) ---
        atk2 = LidarShiftVoxelwiseAttacker(perception, dataset, beta=2.0)
        res2 = atk2.run_multi_vehicle(frame, atk_opts_common)
        det2, iou2, sc2 = find_best_detection(
            res2['pred_bboxes'], res2['pred_scores'], bbox_tgt_ego)

        F2 = res2['spatial_features']
        feat2_atk = np.linalg.norm(F2[attacker_index].cpu().numpy(), axis=0)
        feat2_fused = np.linalg.norm(F2.sum(dim=0).cpu().numpy(), axis=0)

        results['beta'] = {
            'feat_atk': feat2_atk,
            'feat_fused': feat2_fused,
            'best_det': det2, 'best_iou': iou2, 'best_score': sc2,
        }
        logger.info(f"  Beta=2.0: IoU={iou2:.3f}")

        # --- Stage 3: beta=2.0 + PertNet ---
        atk3 = LidarShiftVoxelwiseAttacker(perception, dataset, beta=2.0)
        atk3.pertnet = pertnet
        res3 = atk3.run_multi_vehicle(frame, atk_opts_common)
        det3, iou3, sc3 = find_best_detection(
            res3['pred_bboxes'], res3['pred_scores'], bbox_tgt_ego)

        F3 = res3['spatial_features']
        feat3_atk = np.linalg.norm(F3[attacker_index].cpu().numpy(), axis=0)
        feat3_fused = np.linalg.norm(F3.sum(dim=0).cpu().numpy(), axis=0)

        results['pertnet'] = {
            'feat_atk': feat3_atk,
            'feat_fused': feat3_fused,
            'best_det': det3, 'best_iou': iou3, 'best_score': sc3,
        }
        logger.info(f"  PertNet: IoU={iou3:.3f}")

        # Select case where full attack achieves IoU > 0.5
        if iou3 > best_iou_full and iou3 > 0.3:
            best_iou_full = iou3
            best_data = {
                'case_idx': ci,
                'bbox_orig_ego': bbox_orig_ego,
                'bbox_tgt_ego': bbox_tgt_ego,
                'lr': lr,
                'vs': vs,
                'results': results,
            }
            logger.info(f"  -> New best case: {ci} (full IoU={iou3:.3f})")

            if iou3 > 0.5:
                break  # good enough

    if best_data is None:
        logger.warning("No good case found. Using last case data as fallback.")
        best_data = {
            'case_idx': ci,
            'bbox_orig_ego': bbox_orig_ego,
            'bbox_tgt_ego': bbox_tgt_ego,
            'lr': lr,
            'vs': vs,
            'results': results,
        }

    return best_data


def make_figure(data):
    """Generate the 2x3 pipeline visualization figure.

    Row 1: Attacker's feature map (single agent) -- shows beta effect
    Row 2: Fused feature map (all agents) -- shows detection result
    """
    lr = data['lr']
    vs = data['vs']
    bbox_orig = data['bbox_orig_ego']
    bbox_tgt = data['bbox_tgt_ego']

    # Zoom center: midpoint between GT and target
    mid_x = (bbox_orig[0] + bbox_tgt[0]) / 2
    mid_y = (bbox_orig[1] + bbox_tgt[1]) / 2

    # Voxel-space zoom
    cx_vox = (mid_x - lr[0]) / vs[0]
    cy_vox = (mid_y - lr[1]) / vs[1]
    vx_lo = int(cx_vox - ZOOM_PAD_FEAT)
    vx_hi = int(cx_vox + ZOOM_PAD_FEAT)
    vy_lo = int(cy_vox - ZOOM_PAD_FEAT)
    vy_hi = int(cy_vox + ZOOM_PAD_FEAT)

    stage_keys = ['raycast', 'beta', 'pertnet']
    stage_titles = [
        r'(a) Ray-cast ($\beta$=1)',
        r'(b) +$\beta$ scaling ($\beta$=2.0)',
        r'(c) +PertNet',
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))

    # Compute consistent vmax for fused row only.
    # For attacker row, use the ray-cast (beta=1) column's vmax
    # so that beta scaling visibly brightens the features.
    vmax_fused = 0
    for key in stage_keys:
        res = data['results'][key]
        fm_fused = res['feat_fused']
        H, W = fm_fused.shape
        vx_lo_c = max(0, vx_lo); vx_hi_c = min(W, vx_hi)
        vy_lo_c = max(0, vy_lo); vy_hi_c = min(H, vy_hi)
        crop_f = fm_fused[vy_lo_c:vy_hi_c, vx_lo_c:vx_hi_c]
        vmax_fused = max(vmax_fused, np.percentile(crop_f, 99.5))
    # Use ray-cast vmax for attacker row (so beta=2 looks visibly brighter)
    res_rc = data['results']['raycast']
    fm_atk_rc = res_rc['feat_atk']
    H, W = fm_atk_rc.shape
    vx_lo_c = max(0, vx_lo); vx_hi_c = min(W, vx_hi)
    vy_lo_c = max(0, vy_lo); vy_hi_c = min(H, vy_hi)
    crop_rc = fm_atk_rc[vy_lo_c:vy_hi_c, vx_lo_c:vx_hi_c]
    vmax_atk = np.percentile(crop_rc, 99.5)

    for col, (key, title) in enumerate(zip(stage_keys, stage_titles)):
        res = data['results'][key]
        best_det = res['best_det']
        best_iou = res['best_iou']

        for row, (feat_key, vmax, cmap, row_label) in enumerate([
            ('feat_atk', vmax_atk, 'inferno', "Attacker's Features"),
            ('feat_fused', vmax_fused, 'hot', 'Fused Features'),
        ]):
            ax = axes[row, col]
            feat_mag = res[feat_key]
            H, W = feat_mag.shape

            vx_lo_c = max(0, vx_lo); vx_hi_c = min(W, vx_hi)
            vy_lo_c = max(0, vy_lo); vy_hi_c = min(H, vy_hi)
            crop = feat_mag[vy_lo_c:vy_hi_c, vx_lo_c:vx_hi_c]

            ax.imshow(crop, cmap=cmap, origin='lower', aspect='equal',
                      interpolation='nearest', vmin=0, vmax=vmax)

            # Draw bboxes with high-contrast colors and thicker lines
            def draw_bbox(ax, bbox, color, ls='-', lw=2.0):
                corners = bbox_to_voxel_corners(bbox, lr, vs)
                corners_crop = corners - np.array([vx_lo_c, vy_lo_c])
                poly = MplPolygon(corners_crop, closed=True, fill=False,
                                  edgecolor=color, linestyle=ls, linewidth=lw)
                ax.add_patch(poly)

            draw_bbox(ax, bbox_orig, '#00ff00', lw=2.0)       # bright green
            draw_bbox(ax, bbox_tgt, '#ff3333', lw=2.0)        # bright red

            if best_det is not None:
                draw_bbox(ax, best_det, '#00ccff', ls='--', lw=2.0)  # cyan dashed

            # IoU annotation on both rows, placed at upper-right of crop
            if best_det is not None:
                ax.text(0.97, 0.97,
                        f'IoU={best_iou:.2f}',
                        color='white', fontsize=9, fontweight='bold',
                        ha='right', va='top',
                        transform=ax.transAxes,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                  alpha=0.8, edgecolor='none'))

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(title, fontsize=10, pad=4)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10)

    # Legend on bottom-left of first fused panel (row 1, col 0)
    legend_elements = [
        Line2D([0], [0], color='#00ff00', lw=2, label='GT'),
        Line2D([0], [0], color='#ff3333', lw=2, label='Target'),
        Line2D([0], [0], color='#00ccff', lw=2, ls='--', label='Detection'),
    ]
    axes[1, 0].legend(handles=legend_elements, loc='lower left', fontsize=7,
                      framealpha=0.85, facecolor='black', labelcolor='white',
                      edgecolor='gray')

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'fig_pipeline_vis.pdf')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {out_path}")
    return out_path


def main():
    os.chdir(_ROOT)

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
