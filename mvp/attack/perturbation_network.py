"""
Geometry-aware low-rank perturbation network for voxelwise shift attacks.

The network takes current features, ray-cast feature differences, and
geometric encoding as input, and outputs a low-rank perturbation that
shifts object detections. Designed for real-time inference (<1ms).

Architecture:
    Input: F_orig_crop (C, h, w) + F_diff_crop (C, h, w) + geo_enc (G, h, w)
    Encoder: 2 conv layers
    Branch 1 → spatial weights α (k, h, w)
    Branch 2 → channel directions U (C, k)
    Output: δ = U @ α.reshape(k, h*w), reshaped to (C, h, w)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PerturbationNetwork(nn.Module):
    """
    Lightweight conv network that outputs per-voxel perturbation.

    Directly outputs δ(C, h, w) via conv layers — each voxel gets a
    different perturbation conditioned on its local feature context.
    No global pooling or low-rank constraint.
    """
    def __init__(self, feature_channels=64, geo_channels=10, rank=4,
                 hidden_dim=64):
        super().__init__()
        self.C = feature_channels
        self.geo_channels = geo_channels

        in_channels = 2 * feature_channels + geo_channels

        # Conv encoder → decoder producing per-voxel (C,) perturbation
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, feature_channels, 1),  # per-voxel output
        )

    def forward(self, f_orig_crop, f_diff_crop, geo_encoding):
        """
        Args:
            f_orig_crop: (C, h, w) attacker's features at active zone
            f_diff_crop: (C, h, w) ray-cast feature difference (spoof - orig)
            geo_encoding: (G, h, w) geometric context

        Returns:
            delta: (C, h, w) per-voxel perturbation
        """
        if f_orig_crop.dim() == 3:
            f_orig_crop = f_orig_crop.unsqueeze(0)
            f_diff_crop = f_diff_crop.unsqueeze(0)
            geo_encoding = geo_encoding.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        x = torch.cat([f_orig_crop, f_diff_crop, geo_encoding], dim=1)
        delta = self.net(x)

        if squeeze:
            delta = delta.squeeze(0)

        return delta


def build_geometric_encoding(bbox_orig, bbox_tgt, vehicle_poses, ego_index,
                              attacker_index, active_zone_bounds,
                              lidar_range, voxel_size, H, W,
                              max_vehicles=4):
    """
    Build geometric encoding tensor for the active zone crop.

    Args:
        bbox_orig: (7,) original bbox in ego frame [x,y,z,l,w,h,yaw]
        bbox_tgt: (7,) target bbox in ego frame
        vehicle_poses: list of (x, y) positions of all vehicles in ego frame
        ego_index: index of ego vehicle
        attacker_index: index of attacker vehicle
        active_zone_bounds: (h_lo, h_hi, w_lo, w_hi) crop bounds in feature map
        lidar_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        H, W: full feature map dimensions
        max_vehicles: max benign vehicles to encode

    Returns:
        geo_enc: (G, h, w) tensor where G = 4 + 2 + 3*max_vehicles
    """
    h_lo, h_hi, w_lo, w_hi = active_zone_bounds
    h = h_hi - h_lo
    w = w_hi - w_lo

    if h <= 0 or w <= 0:
        return torch.zeros(4 + 2 + 3 * max_vehicles, 1, 1)

    lr = lidar_range
    vs = voxel_size

    # Build coordinate grids for the crop (in world/ego meters)
    w_coords = (torch.arange(w_lo, w_hi).float() + 0.5) * vs[0] + lr[0]  # x
    h_coords = (torch.arange(h_lo, h_hi).float() + 0.5) * vs[1] + lr[1]  # y
    grid_x, grid_y = torch.meshgrid(w_coords, h_coords, indexing='xy')
    # grid_x: (h, w), grid_y: (h, w) — note meshgrid with xy indexing

    channels = []

    # 1. Offset from original bbox center (2 channels)
    dx_orig = grid_x - bbox_orig[0]
    dy_orig = grid_y - bbox_orig[1]
    channels.extend([dx_orig, dy_orig])

    # 2. Offset from target bbox center (2 channels)
    dx_tgt = grid_x - bbox_tgt[0]
    dy_tgt = grid_y - bbox_tgt[1]
    channels.extend([dx_tgt, dy_tgt])

    # 3. Binary masks (2 channels)
    # Original bbox mask
    cos_o, sin_o = np.cos(bbox_orig[6]), np.sin(bbox_orig[6])
    local_x_o = dx_orig * cos_o + dy_orig * sin_o
    local_y_o = -dx_orig * sin_o + dy_orig * cos_o
    mask_orig = ((local_x_o.abs() < bbox_orig[3] / 2) &
                 (local_y_o.abs() < bbox_orig[4] / 2)).float()

    cos_t, sin_t = np.cos(bbox_tgt[6]), np.sin(bbox_tgt[6])
    local_x_t = dx_tgt * cos_t + dy_tgt * sin_t
    local_y_t = -dx_tgt * sin_t + dy_tgt * cos_t
    mask_tgt = ((local_x_t.abs() < bbox_tgt[3] / 2) &
                (local_y_t.abs() < bbox_tgt[4] / 2)).float()
    channels.extend([mask_orig, mask_tgt])

    # 4. Benign vehicle geometry (3 channels per vehicle: cos θ, sin θ, 1/d)
    benign_indices = [i for i in range(len(vehicle_poses))
                      if i != attacker_index]
    for vi in range(max_vehicles):
        if vi < len(benign_indices):
            vx, vy = vehicle_poses[benign_indices[vi]]
            dx_v = grid_x - vx
            dy_v = grid_y - vy
            dist_v = (dx_v ** 2 + dy_v ** 2).sqrt().clamp(min=1.0)
            cos_v = dx_v / dist_v
            sin_v = dy_v / dist_v
            inv_d = 1.0 / dist_v
            channels.extend([cos_v, sin_v, inv_d])
        else:
            channels.extend([torch.zeros(h, w)] * 3)

    geo_enc = torch.stack(channels, dim=0)  # (G, h, w)
    return geo_enc


def get_active_zone_bounds(bbox_orig, bbox_tgt, lidar_range, voxel_size,
                            H, W, padding=2):
    """
    Compute the active zone bounds (union of original and target bboxes)
    in feature map coordinates.

    Returns:
        (h_lo, h_hi, w_lo, w_hi) in feature map indices
    """
    lr = lidar_range
    vs = voxel_size

    # Get extent of both bboxes in world coordinates
    all_x = []
    all_y = []
    for bbox in [bbox_orig, bbox_tgt]:
        x, y, z, l, w, h, yaw = bbox
        hl, hw = l / 2, w / 2
        corners = np.array([[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]])
        c, s = np.cos(yaw), np.sin(yaw)
        corners = corners @ np.array([[c, s], [-s, c]]) + np.array([x, y])
        all_x.extend(corners[:, 0].tolist())
        all_y.extend(corners[:, 1].tolist())

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Convert to feature map coordinates
    w_lo = max(0, int((x_min - lr[0]) / vs[0]) - padding)
    w_hi = min(W, int((x_max - lr[0]) / vs[0]) + padding + 1)
    h_lo = max(0, int((y_min - lr[1]) / vs[1]) - padding)
    h_hi = min(H, int((y_max - lr[1]) / vs[1]) + padding + 1)

    return h_lo, h_hi, w_lo, w_hi
