"""
Pure-PyTorch drop-in replacement for the CUDA `sort_vertices` extension used by
mvp/perception/cuda_op/cuda_ext.py (AdvCollaborativePerception). Avoids needing
nvcc / a Blackwell CUDA build.

Contract (matches the original op, see mvp/perception/iou_util.py sort_indices/calculate_area):
    sort_vertices_forward(vertices, mask, num_valid) -> idx (B, N, 9) long
      vertices: (B, N, 24, 2) float, already centred on the polygon centroid
      mask:     (B, N, 24) bool, valid vertices
      num_valid:(B, N) int, == mask.sum(-1)
    The returned 9 indices gather the intersection-polygon vertices in CCW order,
    closing back to the first vertex; positions >= num_valid repeat the first index
    (identical consecutive points contribute 0 to the shoelace area, so the area in
    calculate_area() is exact).
"""
import torch

_BIG = 1e6


def sort_vertices_forward(vertices, mask, num_valid):
    B, N, V, _ = vertices.shape
    # angle of each vertex around the (already-subtracted) centroid
    angle = torch.atan2(vertices[..., 1], vertices[..., 0])      # (B, N, 24)
    # push invalid vertices to the end of the ascending sort
    angle = torch.where(mask, angle, torch.full_like(angle, _BIG))
    order = torch.argsort(angle, dim=2)                          # (B, N, 24), valid (CCW) first

    idx9 = order[:, :, :9].clone()                               # first 9 (num_valid <= 8 always)
    first = order[:, :, 0:1]                                     # (B, N, 1) the first vertex index
    pos = torch.arange(9, device=vertices.device).view(1, 1, 9)  # (1, 1, 9)
    nv = num_valid.long().unsqueeze(-1)                          # (B, N, 1)
    # positions [0, num_valid) keep CCW order; positions [num_valid, 9) repeat the first index
    idx = torch.where(pos < nv, idx9, first)
    return idx.int()
