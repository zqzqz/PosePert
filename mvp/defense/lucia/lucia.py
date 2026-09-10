"""
LUCIA defense for collaborative perception.

Implements feature-level anomaly detection from SOMBRA_LUCIA:
https://github.com/WiSeR-Lab/SOMBRA_LUCIA

LUCIA detects adversarial features by:
1. Average-pooling spatial features to reduce dimensions
2. L2-normalizing each agent's features
3. Computing pairwise L1 distances between agents
4. Softmax to produce trust scores (dissimilar agents get low trust)

Trust scores are applied to the attention mechanism to downweight
contributions from suspicious agents.

Usage with OpencoodPerception:
    from mvp.defense.lucia import LuciaDefender

    defender = LuciaDefender(compression_ratio=32)
    trust_scores = defender.compute_trust(spatial_features)
    # Pass trust_scores to AttFusion during backbone forward
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging


class LuciaDefender:
    """
    LUCIA: Feature-level anomaly detection for cooperative perception.

    Computes trust scores for each agent based on feature similarity.
    Agents whose features deviate from peers receive lower trust.
    """
    def __init__(self, compression_ratio=32, temperature=1.0, ego_index=0):
        """
        Args:
            compression_ratio: factor to reduce spatial dims via avg pooling
            temperature: softmax temperature for trust score computation
            ego_index: index of the ego vehicle (always trusted)
        """
        self.compression_ratio = compression_ratio
        self.temperature = temperature
        self.ego_index = ego_index

    def compute_trust(self, features):
        """
        Compute trust scores for each agent.

        Args:
            features: (N, C, H, W) spatial features for N agents

        Returns:
            trust_scores: (N,) trust values in [0, 1], higher = more trusted
        """
        features = features.detach()
        n, C, H, W = features.shape

        if n <= 1:
            return torch.ones(n, device=features.device)

        # Step 1: Average pool to reduce spatial dimensions
        pool_size = min(self.compression_ratio, H, W)
        pooled = F.avg_pool2d(features, kernel_size=pool_size)

        # Step 2: L2 normalize each agent's features
        norms = torch.norm(pooled, p=2, dim=(1, 2, 3), keepdim=True)
        normalized = pooled / (norms + 1e-8)

        # Step 3: Compute pairwise L1 distances
        l1_scores = torch.zeros(n, device=features.device)

        if n == 2:
            l1_scores[1] = torch.sum(torch.abs(normalized[1] - normalized[0]))
            l1_scores[0] = 0.0
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    l1_dist = torch.sum(torch.abs(normalized[i] - normalized[j]))
                    l1_scores[i] += l1_dist
                    l1_scores[j] += l1_dist

        # Ego is always trusted (set to min score)
        l1_scores[self.ego_index] = torch.min(l1_scores)

        # Step 4: Softmax and invert (dissimilar = low trust)
        trust_scores = 1 - F.softmax(l1_scores / self.temperature, dim=0)

        return trust_scores

    def apply_trust_to_attention(self, attn_scores, trust_scores):
        """
        Apply trust scores to attention weights.

        Multiplies attention scores by trust before softmax,
        then multiplies attention weights by trust after softmax,
        then renormalizes.

        Args:
            attn_scores: (HW, N, N) raw attention scores
            trust_scores: (N,) trust values

        Returns:
            modified_attn: (HW, N, N) trust-weighted attention
        """
        # Pre-softmax: multiply scores by trust
        trust_expanded = trust_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        modified_scores = attn_scores * trust_expanded

        # Softmax
        attn_weights = F.softmax(modified_scores, dim=-1)

        # Post-softmax: multiply by trust and renormalize
        attn_weights = attn_weights * trust_expanded
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return attn_weights


def run_with_lucia_defense(perception, multi_vehicle_case, ego_id,
                            compression_ratio=32, temperature=1.0):
    """
    Run perception with LUCIA defense enabled.

    This is a drop-in replacement for perception.run() that adds
    LUCIA trust score computation and application.

    Args:
        perception: OpencoodPerception instance
        multi_vehicle_case: dict of vehicle data for one frame
        ego_id: ego/victim vehicle ID
        compression_ratio: LUCIA pooling factor
        temperature: softmax temperature

    Returns:
        pred_bboxes, pred_scores (same as perception.run())
        trust_scores: (N,) trust values for each agent
    """
    from opencood.tools import train_utils
    from opencood.utils import box_utils
    from collections import OrderedDict

    defender = LuciaDefender(compression_ratio=compression_ratio,
                              temperature=temperature)

    # Preprocess
    batch = perception.preprocessors[perception.fusion_method](
        multi_vehicle_case, ego_id)
    batch_data = perception.dataset.collate_batch_test([batch])
    batch_data = train_utils.to_device(batch_data, perception.device)

    # Get spatial features
    bd = {
        'voxel_features': batch_data['ego']['processed_lidar']['voxel_features'],
        'voxel_coords': batch_data['ego']['processed_lidar']['voxel_coords'],
        'voxel_num_points': batch_data['ego']['processed_lidar']['voxel_num_points'],
        'record_len': batch_data['ego']['record_len'],
    }
    perception.model.pillar_vfe(bd)
    perception.model.scatter(bd)
    spatial_features = bd['spatial_features']

    # Compute LUCIA trust scores on pre-backbone features
    with torch.no_grad():
        trust_scores = defender.compute_trust(spatial_features)
        logging.debug(f"LUCIA trust scores: {trust_scores.cpu().numpy()}")

    # Run backbone with trust-aware attention
    # We need to hook into the AttFusion modules to pass trust_scores
    hooks = []
    def make_trust_hook(trust):
        def hook(module, args, kwargs):
            # Inject trust_score into the forward call
            if 'trust_score' not in kwargs:
                kwargs['trust_score'] = trust
            return args, kwargs
        return hook

    # For PointPillarIntermediate with AttBEVBackbone:
    # The backbone's fuse_modules are AttFusion instances
    # We hook them to pass trust_scores
    for fuse_mod in perception.model.backbone.fuse_modules:
        # Use a wrapper approach since AttFusion.forward signature varies
        original_forward = fuse_mod.forward
        def wrapped_forward(x, record_len, _orig=original_forward, _trust=trust_scores):
            # Call with trust_score if supported
            try:
                return _orig(x, record_len, trust_score=_trust)
            except TypeError:
                # Original AttFusion doesn't accept trust_score
                return _orig(x, record_len)
        fuse_mod.forward = wrapped_forward
        hooks.append((fuse_mod, original_forward))

    try:
        with torch.no_grad():
            bd['spatial_features'] = spatial_features
            perception.model.backbone(bd)
            sf2d = bd['spatial_features_2d']
            psm = perception.model.cls_head(sf2d)
            rm = perception.model.reg_head(sf2d)

            output_dict = OrderedDict()
            output_dict['ego'] = {'psm': psm, 'rm': rm}

            pred_box_tensor, pred_score, _ = \
                perception.dataset.post_process(batch_data, output_dict)

            if pred_box_tensor is None:
                pred_bboxes = np.array([]).reshape(0, 7)
                pred_scores = np.array([])
            else:
                pred_bboxes = pred_box_tensor.cpu().numpy()
                pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                pred_bboxes[:, 2] -= 0.5 * pred_bboxes[:, 5]
                pred_scores = pred_score.cpu().numpy()

                if pred_scores.ndim == 2 and pred_scores.shape[1] == 2:
                    pred_scores, pred_classes = pred_scores[:, 0], pred_scores[:, 1].astype(np.int)
                    mask = np.logical_and(pred_scores >= 0.1, pred_classes == 1)
                else:
                    mask = pred_scores >= 0.1
                pred_bboxes = pred_bboxes[mask]
                pred_scores = pred_scores[mask]
    finally:
        # Restore original forward methods
        for fuse_mod, original_forward in hooks:
            fuse_mod.forward = original_forward

    return pred_bboxes, pred_scores, trust_scores.cpu().numpy()
