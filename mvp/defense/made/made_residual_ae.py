"""
MADE Residual Autoencoder — faithful implementation of the MADE paper.

MADE detects malicious agents by analyzing RESIDUAL features:
  residual_i = fused_with_all_agents - fused_without_agent_i

An autoencoder trained on clean residuals learns normal collaboration
patterns. Malicious agents produce abnormal residuals with high
reconstruction loss.

Both global and local (per-bbox) variants are supported.

Key fix: operates on pre-computed spatial features (including attacked
features), not re-derived from point clouds.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import OrderedDict

from mvp.defense.lucia.local_lucia import bbox_to_feature_crop


class ResidualAutoencoder(nn.Module):
    """
    Encoder-decoder AE for residual feature maps.

    Input: residual feature (C, H, W) — difference between fused-with-all
           and fused-without a specific agent.
    Output: reconstructed residual, same shape.
    """
    def __init__(self, in_channels=256, base_channels=32, latent_dim=256):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(2 * c * 16, latent_dim),
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * c * 16),
            nn.GELU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(2 * c, 2 * c, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * c, c, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c, in_channels, 3, stride=2, padding=1, output_padding=1),
        )
        self.base_channels = base_channels

    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder_linear(z)
        h = h.view(h.shape[0], 2 * self.base_channels, 4, 4)
        recon = self.decoder_conv(h)
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=x.shape[2:], mode='bilinear',
                                  align_corners=False)
        return recon, z


class MadeResidualDetector:
    """
    MADE residual-based anomaly detector.

    Operates on pre-computed spatial features (the actual features used
    in fusion, including any attack modifications). Computes residuals
    by running the backbone+fusion with/without each agent.

    Supports both global (full feature map) and local (per-bbox) scoring.
    """
    def __init__(self, perception, ae_checkpoint=None, device=None):
        self.perception = perception
        self.device = device or perception.device
        self.ae = None
        self.data_mean = None
        self.calibration_threshold = None

        if ae_checkpoint is not None:
            self.load_model(ae_checkpoint)

    def load_model(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        in_ch = ckpt.get('in_channels', 256)
        base_ch = ckpt.get('base_channels', 32)
        latent = ckpt.get('latent_dim', 256)
        self.ae = ResidualAutoencoder(in_ch, base_ch, latent)
        self.ae.load_state_dict(ckpt['model_state'])
        self.ae.to(self.device).eval()
        if 'data_mean' in ckpt and ckpt['data_mean'] is not None:
            self.data_mean = torch.tensor(ckpt['data_mean']).to(self.device)
        self.calibration_threshold = ckpt.get('threshold_95', None)
        logging.info(f"Loaded MADE residual AE from {checkpoint_path}")

    def _run_backbone_fusion(self, spatial_features, record_len):
        """
        Run backbone (conv blocks + attention fusion) on spatial features.

        Args:
            spatial_features: (N_total, C, H, W) — concatenated per-agent features
            record_len: (B,) — number of agents per batch item

        Returns:
            fused_2d: (B, C_out, H_out, W_out) — fused backbone output for ego
        """
        with torch.no_grad():
            bd = {'spatial_features': spatial_features, 'record_len': record_len}
            self.perception.model.backbone(bd)
            return bd['spatial_features_2d']

    def _run_first_block_fusion(self, spatial_features, record_len):
        """
        Run only the first conv block + attention fusion (pre-backbone level).
        This matches MADE's original design which operates on communication-
        level features, not post-backbone features.

        Args:
            spatial_features: (N_total, C, H, W)
            record_len: (B,)

        Returns:
            fused: (B, C_block, H', W') — fused after first block only
        """
        with torch.no_grad():
            x = spatial_features
            backbone = self.perception.model.backbone
            x = backbone.blocks[0](x)
            x_fuse = backbone.fuse_modules[0](x, record_len)
            return x_fuse

    def compute_residuals_from_features(self, spatial_features, record_len,
                                         use_pre_backbone=True):
        """
        Compute residual features for each non-ego agent.

        residual_i = fuse(all_agents) - fuse(all_except_agent_i)

        When use_pre_backbone=True (default, matching MADE's design),
        computes residuals at the first fusion block level (before backbone
        processes further). This preserves the attack signal which gets
        diluted by ~5x through the full backbone.

        Args:
            spatial_features: (N_agents, C, H, W) pre-backbone spatial features
            record_len: tensor([N_agents])
            use_pre_backbone: if True, compute at first-block fusion level;
                if False, compute at full post-backbone level (legacy)

        Returns:
            fused_all: fused output with all agents
            residuals: list of residual tensors, one per non-ego agent
            agent_indices: list of int
        """
        n_agents = spatial_features.shape[0]
        fuse_fn = self._run_first_block_fusion if use_pre_backbone else self._run_backbone_fusion

        fused_all = fuse_fn(spatial_features, record_len)
        if fused_all.dim() == 4:
            fused_all = fused_all[0].detach()
        else:
            fused_all = fused_all.detach()

        residuals = []
        agent_indices = []

        for i in range(1, n_agents):  # skip ego (index 0)
            mask = [j for j in range(n_agents) if j != i]
            sf_without = spatial_features[mask]
            rl_without = torch.tensor([len(mask)], device=spatial_features.device)

            fused_without = fuse_fn(sf_without, rl_without)
            if fused_without.dim() == 4:
                fused_without = fused_without[0].detach()
            else:
                fused_without = fused_without.detach()

            residual = fused_all - fused_without
            residuals.append(residual)
            agent_indices.append(i)

        return fused_all, residuals, agent_indices

    def compute_global_anomaly(self, spatial_features, record_len):
        """
        Global MADE: AE reconstruction loss on full residual feature maps.

        Args:
            spatial_features: (N_agents, C, H, W) — actual features used in fusion
            record_len: tensor([N_agents])

        Returns:
            scores: list of float — reconstruction loss per non-ego agent
            agent_indices: list of int — agent index for each score
        """
        _, residuals, agent_indices = self.compute_residuals_from_features(
            spatial_features, record_len)

        scores = []
        if self.ae is None:
            return [0.0] * len(residuals), agent_indices

        with torch.no_grad():
            for residual in residuals:
                inp = residual.unsqueeze(0)
                if self.data_mean is not None:
                    inp = inp - self.data_mean
                recon, _ = self.ae(inp)
                loss = float(F.mse_loss(recon, inp, reduction='sum'))
                scores.append(loss)

        return scores, agent_indices

    def compute_local_anomaly(self, spatial_features, record_len, bboxes_ego,
                               lidar_range=None, voxel_size=None, padding=2):
        """
        Local MADE: AE reconstruction loss within each bbox region.

        Args:
            spatial_features: (N_agents, C, H, W) — actual features
            record_len: tensor([N_agents])
            bboxes_ego: (M, 7) bboxes in ego sensor frame
            lidar_range, voxel_size: for coordinate conversion
            padding: extra voxels around bbox

        Returns:
            per_object_scores: (M, N_non_ego) anomaly scores
            agent_indices: list of non-ego agent indices
        """
        if lidar_range is None:
            lidar_range = self.perception.dataset.pre_processor.params["cav_lidar_range"]
        if voxel_size is None:
            voxel_size = self.perception.dataset.pre_processor.params["args"]["voxel_size"]

        _, residuals, agent_indices = self.compute_residuals_from_features(
            spatial_features, record_len)

        n_objects = len(bboxes_ego)
        per_object_scores = np.zeros((n_objects, len(agent_indices)))

        if self.ae is None:
            return per_object_scores, agent_indices

        with torch.no_grad():
            for ai, residual in enumerate(residuals):
                inp = residual.unsqueeze(0)
                if self.data_mean is not None:
                    inp = inp - self.data_mean
                recon, _ = self.ae(inp)

                # Per-voxel reconstruction error
                per_voxel_error = (recon[0] - inp[0]) ** 2  # (C, H, W)

                for obj_idx in range(n_objects):
                    # Note: bboxes are in ego sensor frame, but features_2d
                    # may have different spatial dimensions than pre-backbone.
                    # Use the output feature map dimensions.
                    C_out, H_out, W_out = residual.shape

                    # Scale bbox crop to match feature_2d resolution
                    # The backbone downsamples spatially — need to adjust
                    H_in = int((lidar_range[4] - lidar_range[1]) / voxel_size[1])
                    W_in = int((lidar_range[3] - lidar_range[0]) / voxel_size[0])

                    h_lo, h_hi, w_lo, w_hi = bbox_to_feature_crop(
                        bboxes_ego[obj_idx], lidar_range, voxel_size, padding)

                    # Scale to output resolution
                    h_scale = H_out / H_in
                    w_scale = W_out / W_in
                    h_lo_s = max(0, int(h_lo * h_scale))
                    h_hi_s = min(H_out, int(h_hi * h_scale))
                    w_lo_s = max(0, int(w_lo * w_scale))
                    w_hi_s = min(W_out, int(w_hi * w_scale))

                    if h_hi_s <= h_lo_s or w_hi_s <= w_lo_s:
                        continue

                    local_error = per_voxel_error[:, h_lo_s:h_hi_s, w_lo_s:w_hi_s].sum()
                    per_object_scores[obj_idx, ai] = float(local_error)

        return per_object_scores, agent_indices

    def compute_global_anomaly_no_ae(self, spatial_features, record_len):
        """
        Global MADE without AE: just use raw residual L2 norm as anomaly score.
        Useful as baseline or when AE is not trained.

        Returns:
            scores: list of float — residual L2 norm per non-ego agent
            agent_indices: list of int
        """
        _, residuals, agent_indices = self.compute_residuals_from_features(
            spatial_features, record_len)

        scores = [float(r.norm()) for r in residuals]
        return scores, agent_indices

    def compute_local_anomaly_no_ae(self, spatial_features, record_len,
                                     bboxes_ego, lidar_range=None,
                                     voxel_size=None, padding=2):
        """
        Local MADE without AE: raw residual L2 norm within each bbox.

        Returns:
            per_object_scores: (M, N_non_ego)
            agent_indices: list of int
        """
        if lidar_range is None:
            lidar_range = self.perception.dataset.pre_processor.params["cav_lidar_range"]
        if voxel_size is None:
            voxel_size = self.perception.dataset.pre_processor.params["args"]["voxel_size"]

        _, residuals, agent_indices = self.compute_residuals_from_features(
            spatial_features, record_len)

        n_objects = len(bboxes_ego)
        per_object_scores = np.zeros((n_objects, len(agent_indices)))

        for ai, residual in enumerate(residuals):
            C_out, H_out, W_out = residual.shape
            H_in = int((lidar_range[4] - lidar_range[1]) / voxel_size[1])
            W_in = int((lidar_range[3] - lidar_range[0]) / voxel_size[0])

            for obj_idx in range(n_objects):
                h_lo, h_hi, w_lo, w_hi = bbox_to_feature_crop(
                    bboxes_ego[obj_idx], lidar_range, voxel_size, padding)
                h_scale = H_out / H_in
                w_scale = W_out / W_in
                h_lo_s = max(0, int(h_lo * h_scale))
                h_hi_s = min(H_out, int(h_hi * h_scale))
                w_lo_s = max(0, int(w_lo * w_scale))
                w_hi_s = min(W_out, int(w_hi * w_scale))

                if h_hi_s <= h_lo_s or w_hi_s <= w_lo_s:
                    continue

                local_norm = residual[:, h_lo_s:h_hi_s, w_lo_s:w_hi_s].norm()
                per_object_scores[obj_idx, ai] = float(local_norm)

        return per_object_scores, agent_indices
