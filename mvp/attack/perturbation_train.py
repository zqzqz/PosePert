"""
Train the perturbation network for voxelwise shift attacks.

End-to-end training through frozen perception model:
  perturbation_network(F_crop, F_diff, geo) → δ
  F_attack = F_orig + δ at active voxels
  pred = frozen_model(F_attack)
  loss = -score_at_target + score_at_original + λ*||δ||
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mvp.attack.perturbation_network import PerturbationNetwork
from mvp.attack.perturbation_data import PerturbationDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.data.util import bbox_sensor_to_map
from opencood.tools import train_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bbox_to_anchor_mask(bbox_ego, anchor_box, iou_threshold=0.1):
    """
    Find anchor indices that overlap with a bbox.

    Args:
        bbox_ego: (7,) bbox in ego frame [x,y,z,l,w,h,yaw]
        anchor_box: (H, W, num_anchors, 7) anchor boxes

    Returns:
        mask: (H*W*num_anchors,) boolean mask of matching anchors
    """
    # Flatten anchors
    orig_shape = anchor_box.shape
    anchors_flat = anchor_box.reshape(-1, 7)  # (N, 7)

    # Simple center-distance matching (fast approximation)
    dist = np.sqrt((anchors_flat[:, 0] - bbox_ego[0]) ** 2 +
                   (anchors_flat[:, 1] - bbox_ego[1]) ** 2)

    # Match anchors within bbox diagonal distance
    diag = np.sqrt(bbox_ego[3] ** 2 + bbox_ego[4] ** 2)
    mask = dist < diag * 0.8

    return mask


def compute_attack_loss(perception, sample, delta, device):
    """
    Compute attack loss using the proven distance-based approach.

    Uses attack_intermediate_forward to inject perturbation, then computes
    distance-based loss on decoded proposals (matching the PGD baseline).

    Args:
        perception: frozen OpencoodPerception model
        sample: training sample dict
        delta: (C, h, w) perturbation from the network
        device: torch device

    Returns:
        loss: scalar attack loss
        info: dict with loss components
    """
    import torch.nn.functional as F

    F_orig = sample['F_orig'].to(device)
    attacker_index = sample['attacker_index']
    bbox_orig_ego = sample['bbox_orig_ego']
    bbox_tgt_ego = sample['bbox_tgt_ego']

    # Compute center in voxel coords for the perturbation region
    center = perception.point_to_voxel_index(bbox_orig_ego)
    feature_size = 15  # half-width, matching the proven attacker
    # Clamp center so perturbation region stays within feature map
    H_feat, W_feat = F_orig.shape[2], F_orig.shape[3]
    center[0] = max(feature_size, min(W_feat - feature_size, center[0]))
    center[1] = max(feature_size, min(H_feat - feature_size, center[1]))

    # Perturbation budget
    base_beta = sample.get('beta', 2.0)  # from sample dict or default 2.0

    C = F_orig.shape[1]
    h_delta, w_delta = delta.shape[1], delta.shape[2]
    full_pert = torch.zeros(C, 2 * feature_size, 2 * feature_size, device=device)
    h_off = max(0, (2 * feature_size - h_delta) // 2)
    w_off = max(0, (2 * feature_size - w_delta) // 2)
    h_end = min(2 * feature_size, h_off + h_delta)
    w_end = min(2 * feature_size, w_off + w_delta)
    full_pert[:, h_off:h_end, w_off:w_end] = delta[:, :h_end-h_off, :w_end-w_off]

    # Base perturbation: makes F_orig + base_pert = beta * F_attack
    # where F_attack = F_orig_crop + F_diff_crop (natural spoofed feature)
    f_diff_crop = sample['f_diff_crop'].to(device)
    f_orig_crop = sample['f_orig_crop'].to(device)
    f_attack_crop = f_orig_crop + f_diff_crop

    # base_pert = beta * F_attack - F_orig, so F_orig + base_pert = beta * F_attack
    base_pert = torch.zeros(C, 2 * feature_size, 2 * feature_size, device=device)
    base_pert[:, h_off:h_end, w_off:w_end] = (
        base_beta * f_attack_crop - f_orig_crop)[:, :h_end-h_off, :w_end-w_off]

    # Correction bound: fixed Linf=10
    max_perturb = 10
    correction_clamped = torch.clamp(full_pert, -max_perturb, max_perturb)
    combined_pert = base_pert + correction_clamped

    record_len = sample['record_len'].to(device)

    # Build attacked feature map
    F_attack = F_orig.clone()
    cy, cx = center[1], center[0]
    F_attack[attacker_index, :, cy-feature_size:cy+feature_size,
             cx-feature_size:cx+feature_size] = torch.clamp(
        F_orig[attacker_index, :, cy-feature_size:cy+feature_size,
               cx-feature_size:cx+feature_size] + combined_pert,
        min=0.0, max=30.0)

    # Distance from original (for regularization)
    F_orig_region = F_orig[attacker_index, :, cy-feature_size:cy+feature_size,
                           cx-feature_size:cx+feature_size].detach()
    F_attacked_region = F_attack[attacker_index, :, cy-feature_size:cy+feature_size,
                                  cx-feature_size:cx+feature_size]
    dist_from_orig = (F_attacked_region - F_orig_region).pow(2).mean()  # available but not used in default loss

    record_len = sample['record_len'].to(device)
    model = perception.model

    # Build batch_data for the model's forward pass
    n_agents = sample['n_agents']
    batch_ego = {
        'processed_lidar': {
            'voxel_features': torch.zeros(1, device=device),
            'voxel_coords': torch.zeros(1, 4, dtype=torch.int32, device=device),
            'voxel_num_points': torch.zeros(1, dtype=torch.int32, device=device),
        },
        'record_len': record_len,
    }
    if 'batch_data_keys' in sample:
        for k, v in sample['batch_data_keys'].items():
            if k not in batch_ego:
                batch_ego[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    # Add missing keys needed by specific models
    max_cav = getattr(model, 'max_cav', 7)
    if 'spatial_correction_matrix' not in batch_ego:
        batch_ego['spatial_correction_matrix'] = torch.eye(4, device=device).unsqueeze(0).repeat(1, max_cav, 1, 1)
    # Override pairwise_t_matrix with identity to skip warping
    # (features are already in ego frame from preprocessing)
    if 'pairwise_t_matrix' in batch_ego:
        shape = batch_ego['pairwise_t_matrix'].shape
        batch_ego['pairwise_t_matrix'] = torch.eye(4, device=device).reshape(
            1, 1, 1, 4, 4).expand(shape).contiguous()

    # Monkey-patch to inject features and bypass warp
    original_scatter = model.scatter.forward
    original_vfe = model.pillar_vfe.forward

    # Warp/discretization patches are applied externally (in train() or pipeline)
    # to avoid patch/restore conflicts across calls

    try:
        # Get original prob
        def patched_scatter_orig(bd):
            bd['spatial_features'] = F_orig.clone()
            return bd
        def patched_vfe(bd):
            bd['pillar_features'] = torch.zeros(1, device=device)
            return bd
        model.scatter.forward = patched_scatter_orig
        model.pillar_vfe.forward = patched_vfe

        with torch.no_grad():
            orig_output = model(batch_ego)

        # Get attacked output
        def patched_scatter_atk(bd):
            bd['spatial_features'] = F_attack
            return bd
        model.scatter.forward = patched_scatter_atk

        output = model(batch_ego)
    finally:
        model.scatter.forward = original_scatter
        model.pillar_vfe.forward = original_vfe

    # Decode proposals
    # Try anchor_box from batch_data, then all_anchors, then generate
    batch_keys = sample.get('batch_data_keys', {})
    anchor_box = batch_keys.get('all_anchors', batch_keys.get('anchor_box', None))
    if anchor_box is None:
        anchor_box = perception.dataset.post_processor.generate_anchor_box()
        if isinstance(anchor_box, tuple):
            anchor_box = anchor_box[0]

    # Detect multi-class anchors (V2X-Real): list of arrays or 5D tensor
    is_multiclass = isinstance(anchor_box, (list, tuple))
    if isinstance(anchor_box, torch.Tensor) and anchor_box.dim() == 5:
        is_multiclass = True
    if isinstance(anchor_box, np.ndarray) and anchor_box.ndim == 5:
        is_multiclass = True

    if is_multiclass:
        # Multi-class path (V2X-Real): anchors are (num_class, H, W, anchor_num, 7)
        if isinstance(anchor_box, (list, tuple)):
            anchor_tensor = torch.from_numpy(np.array(anchor_box)).to(device).float()
        elif isinstance(anchor_box, np.ndarray):
            anchor_tensor = torch.from_numpy(anchor_box).to(device).float()
        else:
            anchor_tensor = anchor_box.to(device).float()
        # anchor_tensor: (num_class, H, W, anchor_num, 7)
        num_class = anchor_tensor.shape[0]
        # Permute to (H, W, num_class, anchor_num, 7) then flatten to (N_anchors, 7)
        all_anchors_flat = anchor_tensor.permute(1, 2, 0, 3, 4).contiguous().view(-1, 7)
        num_anchors = all_anchors_flat.shape[0]

        # PSM: (B, num_class*anchor_num*num_class, H, W) -> per-anchor max prob
        psm = output['psm']
        B = psm.shape[0]
        prob_full = torch.sigmoid(psm.permute(0, 2, 3, 1))  # (B, H, W, C_psm)
        prob_full = prob_full.reshape(B, num_anchors, num_class)  # (B, N_anchors, num_class)
        prob, _ = prob_full.max(dim=-1)  # (B, N_anchors)
        prob = prob.reshape(-1)

        # RM: (B, anchor_num*num_class*7, H, W) -> (B, N_anchors, 7)
        rm = output['rm']
        rm_perm = rm.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C_rm)
        rm_perm = rm_perm.view(B, num_anchors, 7)

        # Decode boxes (channel_swap=False since already reshaped)
        proposals = perception.dataset.post_processor.delta_to_boxes3d(
            rm_perm, all_anchors_flat, channel_swap=False)[0]

        # Also compute original_prob for multi-class
        orig_prob_full = torch.sigmoid(orig_output['psm'].permute(0, 2, 3, 1))
        orig_prob_full = orig_prob_full.reshape(B, num_anchors, num_class)
        original_prob, _ = orig_prob_full.max(dim=-1)
        original_prob = original_prob.reshape(-1).detach()
    else:
        # Single-class path (OPV2V): anchors are (H, W, anchor_num, 7)
        if isinstance(anchor_box, np.ndarray):
            anchor_box = torch.from_numpy(anchor_box).to(device)
        elif isinstance(anchor_box, torch.Tensor):
            anchor_box = anchor_box.to(device)

        prob = torch.sigmoid(
            output['psm'].permute(0, 2, 3, 1)).reshape(-1)
        original_prob = torch.sigmoid(
            orig_output['psm'].permute(0, 2, 3, 1)).reshape(-1).detach()
        proposals = perception.dataset.post_processor.delta_to_boxes3d(
            output['rm'], anchor_box)[0]

    # Target bbox tensors
    bbox_tensor = torch.from_numpy(bbox_orig_ego).to(device).float()
    bbox_tensor[2] += 0.5 * bbox_tensor[5]
    bbox2_tensor = torch.from_numpy(bbox_tgt_ego).to(device).float()
    bbox2_tensor[2] += 0.5 * bbox2_tensor[5]

    # IoU masks — use BEV IoU for V2X-Real (z convention mismatch)
    bev_only = sample.get('bev_only', False)
    iou_fn = perception.iou_bev_torch if bev_only else perception.iou_torch

    iou_orig = torch.clip(iou_fn(
        proposals[:, [0,1,2,5,4,3,6]],
        bbox_tensor.tile((proposals.shape[0], 1))
    ), min=0, max=1)
    bbox_mask = (iou_orig >= 0.01)

    iou_tgt = torch.clip(iou_fn(
        proposals[:, [0,1,2,5,4,3,6]],
        bbox2_tensor.tile((proposals.shape[0], 1))
    ), min=0, max=1)
    box2_mask = (iou_tgt >= 0.01)

    prob_mask = (prob >= 0.1)
    mask = torch.logical_and(torch.logical_and(bbox_mask, box2_mask), prob_mask)

    if mask.sum() == 0:
        # Fallback: use any proposal near original with prob
        mask = torch.logical_and(bbox_mask, prob_mask)

    if mask.sum() == 0:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        info = {'loss_iou': 0.0, 'loss_prob': 0.0, 'loss_norm': 0.0,
                'n_proposals': 0}
        return loss, info

    # IoU-based loss (proven from attack_intermediate):
    # Maximize IoU with target: log(1 - iou_tgt) → -inf as iou→1
    # Maintain detection confidence: penalize prob drop
    loss_iou = torch.log(1 - iou_tgt[mask] + 1e-8).sum()

    # Maintain detection confidence
    loss_prob = torch.clip(original_prob[mask] - prob[mask], 0, 1).sum()

    # Distance regularization: penalize large deviation from original
    loss = loss_iou + 1.0 * loss_prob + 0.01 * dist_from_orig

    info = {
        'loss_iou': float(loss_iou),
        'loss_prob': float(loss_prob),
        'loss_dist': float(dist_from_orig),
        'n_proposals': int(mask.sum()),
    }
    return loss, info


def build_perception(model_name='pointpillar'):
    """Build perception for the specified model."""
    if model_name == 'cobevt':
        import opencood.hypes_yaml.yaml_utils as yaml_utils
        from opencood.data_utils.datasets import build_dataset
        from opencood.tools import inference_utils
        from mvp.config import model_root, data_root
        p = OpencoodPerception.__new__(OpencoodPerception)
        p.model_name = 'pointpillar'; p.fusion_method = 'intermediate'
        p.dataset_name_full = 'CoBEVT'; p.dataset_name = 'OPV2V'
        p.model_dir = os.path.join(model_root, 'OpenCOOD/pointpillar_attentive_fusion_cobevt')
        p.config_file = os.path.join(p.model_dir, 'config.yaml')
        hypes = yaml_utils.load_yaml(p.config_file, None)
        hypes['root_dir'] = os.path.join(data_root, 'OPV2V/train')
        hypes['validate_dir'] = os.path.join(data_root, 'OPV2V/validate')
        p.dataset = build_dataset(hypes, visualize=False, train=False)
        p.model = train_utils.create_model(hypes)
        p.model.cuda()
        p.device = torch.device('cuda')
        _, p.model = train_utils.load_saved_model(p.model_dir, p.model)
        p.model.eval()
        p.preprocessors = {'intermediate': p.intermediate_preprocess,
                           'early': p.early_preprocess, 'late': p.late_preprocess}
        p.inference_processors = {'intermediate': inference_utils.inference_intermediate_fusion}
        p.name = 'pointpillar_intermediate'; p.devices = 'cuda:0'; p.attn_loss_fn = None
        return p
    elif model_name == 'v2vnet':
        return OpencoodPerception(fusion_method='intermediate', model_name='v2vnet',
                                  dataset_name='OPV2V')
    else:
        return OpencoodPerception(fusion_method='intermediate', model_name='pointpillar',
                                  dataset_name='OPV2V')


def _apply_warp_patches():
    """Patch warp/discretization to identity for V2VNet/CoAlign training.

    Patches both the source module AND the importing module's local namespace
    to handle Python's 'from X import Y' binding semantics.
    """
    patches = {}
    def identity_warp(src, M, dsize, **kwargs):
        return src
    def identity_discretize(matrix, discrete_ratio, downsample_rate):
        N, L = matrix.shape[:2]
        eye = torch.zeros(N, L, 2, 3, device=matrix.device, dtype=matrix.dtype)
        eye[:, :, 0, 0] = 1.0; eye[:, :, 1, 1] = 1.0
        return eye

    # Patch in the source module
    try:
        from opencood.models.sub_modules import torch_transformation_utils as ttu
        patches['ttu_warp'] = (ttu, 'warp_affine', ttu.warp_affine)
        ttu.warp_affine = identity_warp
        patches['ttu_disc'] = (ttu, 'get_discretized_transformation_matrix',
                                ttu.get_discretized_transformation_matrix)
        ttu.get_discretized_transformation_matrix = identity_discretize
    except ImportError: pass

    # Also patch in importing modules (v2v_fuse, coalign_fuse)
    # to override the local 'from ... import' bindings
    try:
        import opencood.models.fuse_modules.v2v_fuse as v2v_mod
        if hasattr(v2v_mod, 'warp_affine'):
            patches['v2v_warp'] = (v2v_mod, 'warp_affine', v2v_mod.warp_affine)
            v2v_mod.warp_affine = identity_warp
        if hasattr(v2v_mod, 'get_discretized_transformation_matrix'):
            patches['v2v_disc'] = (v2v_mod, 'get_discretized_transformation_matrix',
                                    v2v_mod.get_discretized_transformation_matrix)
            v2v_mod.get_discretized_transformation_matrix = identity_discretize
        if hasattr(v2v_mod, 'get_transformation_matrix'):
            from opencood.models.sub_modules.torch_transformation_utils import get_transformation_matrix as orig_gtm
            patches['v2v_gtm'] = (v2v_mod, 'get_transformation_matrix', v2v_mod.get_transformation_matrix)
            def identity_gtm(M, dsize):
                # Return (N, 2, 3) identity
                N = M.shape[0]
                eye = torch.zeros(N, 2, 3, device=M.device, dtype=M.dtype)
                eye[:, 0, 0] = 1.0; eye[:, 1, 1] = 1.0
                return eye
            v2v_mod.get_transformation_matrix = identity_gtm
        if hasattr(v2v_mod, 'get_rotated_roi'):
            patches['v2v_roi'] = (v2v_mod, 'get_rotated_roi', v2v_mod.get_rotated_roi)
            def identity_roi(shape, M):
                return torch.ones(shape, device=M.device)
            v2v_mod.get_rotated_roi = identity_roi
    except ImportError: pass

    try:
        import opencood.models.fuse_modules.coalign_fuse as coalign_mod
        if hasattr(coalign_mod, 'warp_affine'):
            patches['coalign_warp'] = (coalign_mod, 'warp_affine', coalign_mod.warp_affine)
            coalign_mod.warp_affine = identity_warp
    except ImportError: pass

    return patches

def _restore_warp_patches(patches):
    for key, (mod, attr, orig) in patches.items():
        setattr(mod, attr, orig)


def train(args):
    # Load perception model (frozen)
    perception = build_perception(args.model)
    perception.model.eval()
    for p in perception.model.parameters():
        p.requires_grad = False
    device = perception.device

    # Apply warp patches for the entire training
    warp_patches = _apply_warp_patches()

    # Load dataset
    train_dataset = PerturbationDataset(args.data_dir)
    logger.info(f"Training samples: {len(train_dataset)}")

    # Determine feature channels from first sample
    sample0 = train_dataset[0]
    C = sample0['f_orig_crop'].shape[0]
    G = sample0['geo_encoding'].shape[0]
    logger.info(f"Feature channels: {C}, Geo channels: {G}")

    # Create network
    net = PerturbationNetwork(
        feature_channels=C, geo_channels=G, rank=args.rank,
        hidden_dim=args.hidden_dim).to(device)
    logger.info(f"Network params: {sum(p.numel() for p in net.parameters())}")

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        net.train()
        epoch_stats = defaultdict(list)

        indices = np.random.permutation(len(train_dataset))
        for si in indices:
            sample = train_dataset[si]

            f_orig_crop = sample['f_orig_crop'].to(device)
            f_diff_crop = sample['f_diff_crop'].to(device)
            geo_enc = sample['geo_encoding'].to(device)

            # Forward through perturbation network
            delta = net(f_orig_crop, f_diff_crop, geo_enc)

            # Skip samples with mismatched record_len and F_orig
            if sample['F_orig'].shape[0] != int(sample['record_len'].sum()):
                continue

            # Compute attack loss through frozen perception model
            try:
                loss, info = compute_attack_loss(perception, sample, delta, device)
            except RuntimeError:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            for k, v in info.items():
                epoch_stats[k].append(v)

            torch.cuda.empty_cache()

        scheduler.step()

        # Log
        avg_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        total_loss = (avg_stats.get('loss_iou', 0) + avg_stats.get('loss_prob', 0)
                      + 0.01 * avg_stats.get('loss_dist', 0))
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"loss={total_loss:.4f} "
            f"iou={avg_stats['loss_iou']:.3f} "
            f"prob={avg_stats['loss_prob']:.3f} "
            f"dist={avg_stats['loss_dist']:.2f} "
            f"n_prop={avg_stats['n_proposals']:.1f}")

        # Save best
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save({
                'model_state': net.state_dict(),
                'rank': args.rank,
                'hidden_dim': args.hidden_dim,
                'feature_channels': C,
                'geo_channels': G,
                'epoch': epoch + 1,
                'loss': total_loss,
            }, os.path.join(args.save_dir, 'perturbation_net_best.pt'))

        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state': net.state_dict(),
                'rank': args.rank,
                'hidden_dim': args.hidden_dim,
                'feature_channels': C,
                'geo_channels': G,
                'epoch': epoch + 1,
            }, os.path.join(args.save_dir, f'perturbation_net_ep{epoch+1}.pt'))

    _restore_warp_patches(warp_patches)
    logger.info(f"Training done. Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/perturbation_train')
    parser.add_argument('--save_dir', type=str,
                        default='models/perturbation_net')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--model', type=str, default='pointpillar',
                        choices=['pointpillar', 'v2vnet', 'cobevt'])
    args = parser.parse_args()
    train(args)
