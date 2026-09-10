import os
"""
Generate Figure 8: Factors affecting attack success.
4 subfigures: (a) Benign distance, (b) Benign count, (c) Point density, (d) Attacker count.

Subfigure (d) uses multi-attacker v2 results (with PertNet).

Usage:
    python results_paper/gen_fig_factors.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.data.util import bbox_sensor_to_map

OUT = os.path.join(os.environ.get('FIG_OUT_DIR', 'results_paper/figures'), 'fig_factor_distance.pdf')

# ---- Load data ----
with open('results_paper/E1_pp_attentive/per_case_results.pkl', 'rb') as f:
    pp_results = pickle.load(f)

with open('data/OPV2V/attack/lidar_shift.pkl', 'rb') as f:
    attacks = pickle.load(f)

with open('results_paper/multi_attacker/results.pkl', 'rb') as f:
    ma_results = pickle.load(f)

dataset = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')

# ---- Precompute per-case metadata ----
print("Computing per-case metadata...")
case_meta = []
for ci in range(len(pp_results)):
    meta = attacks[ci]['attack_meta']
    try:
        case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
        frame = case[9]
        ai = meta['attacker_vehicle_id']
        vi = meta['victim_vehicle_id']
        if ai not in frame or vi not in frame:
            case_meta.append(None); continue

        bbox_tgt = meta['new_bbox']
        atk_pose = frame[ai]['lidar_pose']
        bbox_tgt_map = bbox_sensor_to_map(np.array(bbox_tgt), atk_pose)
        tgt_pos = bbox_tgt_map[:2]

        benign_ids = [v for v in frame if v != ai and v != vi]
        n_benign = len(benign_ids)
        dists = [np.linalg.norm(frame[v]['lidar_pose'][:2] - tgt_pos) for v in benign_ids]
        min_dist = min(dists) if dists else 999

        # Count benign LiDAR points on target: points from benign vehicles
        # whose projection falls within the target bbox footprint.
        # Use bbox corner polygon in map frame.
        from shapely.geometry import Point, box as shapely_box
        import math
        x, y, z, l, w, h, yaw = bbox_tgt_map
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        corners = np.array([[-l/2, -w/2], [-l/2, w/2], [l/2, w/2], [l/2, -w/2]])
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        corners_map = (rot @ corners.T).T + np.array([x, y])
        from shapely.geometry import Polygon as ShapelyPolygon
        bbox_poly = ShapelyPolygon(corners_map)
        # Expand by 2m for nearby points
        bbox_expanded = bbox_poly.buffer(2.0)

        n_pts = 0
        for bv in benign_ids:
            pcd = frame[bv].get('lidar', np.zeros((0, 4)))
            if len(pcd) > 0:
                bv_pose = frame[bv]['lidar_pose']
                # Transform points to map frame (approx: translate + rotate yaw)
                bv_yaw = np.radians(bv_pose[4]) if len(bv_pose) > 4 else 0
                c_bv, s_bv = np.cos(bv_yaw), np.sin(bv_yaw)
                pts_map = pcd[:, :2].copy()
                pts_rot = np.column_stack([
                    c_bv * pts_map[:, 0] - s_bv * pts_map[:, 1],
                    s_bv * pts_map[:, 0] + c_bv * pts_map[:, 1]])
                pts_rot[:, 0] += bv_pose[0]
                pts_rot[:, 1] += bv_pose[1]
                # Count points inside expanded bbox
                from matplotlib.path import Path as MplPath
                expanded_coords = np.array(bbox_expanded.exterior.coords)
                path = MplPath(expanded_coords)
                inside = path.contains_points(pts_rot)
                n_pts += int(inside.sum())

        case_meta.append({'n_benign': n_benign, 'min_dist': min_dist, 'n_pts': n_pts})
    except:
        case_meta.append(None)

valid = sum(1 for x in case_meta if x is not None)
print(f"Computed metadata for {valid}/{len(pp_results)} cases")

# ---- Create figure ----
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

# (a) Benign distance
ax = axes[0, 0]
bins = [(0, 10, '0-10'), (10, 20, '10-20'), (20, 30, '20-30')]
normal_vals, attack_vals = [], []
for lo, hi, _ in bins:
    cases = [(i, r) for i, r in enumerate(pp_results)
             if case_meta[i] is not None and lo <= case_meta[i]['min_dist'] < hi]
    if cases:
        normal_vals.append(np.mean([r['normal']['iou_tgt'] for _, r in cases]))
        attack_vals.append(np.mean([r['pertnet']['iou_tgt'] for _, r in cases]))
    else:
        normal_vals.append(0); attack_vals.append(0)

x = np.arange(len(bins))
w = 0.35
ax.bar(x - w/2, normal_vals, w, label='Normal', color='tab:blue')
ax.bar(x + w/2, attack_vals, w, label='PosePert', color='tab:red', hatch='//')
ax.set_xticks(x)
ax.set_xticklabels([b[2] for b in bins])
ax.set_xlabel('Closest benign dist (m)')
ax.set_ylabel('Avg IoU')
ax.set_title('(a) Benign distance', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)

# (b) Benign count
ax = axes[0, 1]
n_vals = sorted(set(m['n_benign'] for m in case_meta if m is not None))
normal_b, attack_b = [], []
valid_n = []
for n in n_vals:
    cases = [(i, r) for i, r in enumerate(pp_results)
             if case_meta[i] is not None and case_meta[i]['n_benign'] == n]
    if len(cases) >= 5:
        normal_b.append(np.mean([r['normal']['iou_tgt'] for _, r in cases]))
        attack_b.append(np.mean([r['pertnet']['iou_tgt'] for _, r in cases]))
        valid_n.append(n)

x = np.arange(len(valid_n))
ax.bar(x - w/2, normal_b, w, label='Normal', color='tab:blue')
ax.bar(x + w/2, attack_b, w, label='PosePert', color='tab:red', hatch='//')
ax.set_xticks(x)
ax.set_xticklabels(valid_n)
ax.set_xlabel('# benign vehicles')
ax.set_ylabel('Avg IoU')
ax.set_title('(b) Benign count', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)

# (c) Benign point density
ax = axes[1, 0]
pt_bins = [(0, 100, '<100'), (100, 1000, '100-1K'), (1000, 5000, '1K-5K')]
normal_p, attack_p = [], []
for lo, hi, _ in pt_bins:
    cases = [(i, r) for i, r in enumerate(pp_results)
             if case_meta[i] is not None and lo <= case_meta[i]['n_pts'] < hi]
    if cases:
        normal_p.append(np.mean([r['normal']['iou_tgt'] for _, r in cases]))
        attack_p.append(np.mean([r['pertnet']['iou_tgt'] for _, r in cases]))
    else:
        normal_p.append(0); attack_p.append(0)

x = np.arange(len(pt_bins))
ax.bar(x - w/2, normal_p, w, label='Normal', color='tab:blue')
ax.bar(x + w/2, attack_p, w, label='PosePert', color='tab:red', hatch='//')
ax.set_xticks(x)
ax.set_xticklabels([b[2] for b in pt_bins])
ax.set_xlabel('Benign points on target')
ax.set_ylabel('Avg IoU')
ax.set_title('(c) Benign point density', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)

# (d) Attacker count (from multi_attacker v2 with PertNet)
ax = axes[1, 1]
normal_iou = np.mean([r['normal']['iou_tgt'] for r in pp_results])
avg_ious = [normal_iou]
for n_atk in [1, 2, 3]:
    key = f"atk_{n_atk}"
    valid = [r for r in ma_results if r.get(key) is not None]
    avg_ious.append(np.mean([r[key]['iou_tgt'] for r in valid]))

xs = [0, 1, 2, 3]
ax.plot(xs, avg_ious, 'o-', color='tab:red', linewidth=2.5, markersize=10)
for xv, yv in zip(xs, avg_ious):
    ax.annotate(f'{yv:.2f}', (xv, yv), textcoords="offset points",
                xytext=(-5, 12), ha='center', fontsize=11)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='IoU=0.5')
ax.set_xticks(xs)
ax.set_xticklabels(['0\n(normal)', '1', '2', '3'])
ax.set_xlabel('# attackers')
ax.set_ylabel('Avg IoU with target')
ax.set_title('(d) Attacker count', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT, bbox_inches='tight', dpi=300)
print(f"Saved combined: {OUT}")
print(f"  (d) data: normal={avg_ious[0]:.2f}, 1atk={avg_ious[1]:.3f}, 2atk={avg_ious[2]:.3f}, 3atk={avg_ious[3]:.3f}")
plt.close()

# ---- Also save each subfigure independently ----
out_dir = os.environ.get('FIG_OUT_DIR', 'results_paper/figures')

def save_subfig(ax_func, fname):
    f, a = plt.subplots(1, 1, figsize=(4.5, 3.8))
    ax_func(a)
    f.tight_layout()
    path = os.path.join(out_dir, fname)
    f.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(f)
    print(f"  Saved {path}")

def draw_a(ax):
    ax.bar(np.arange(len(bins)) - w/2, normal_vals, w, label='Normal', color='tab:blue')
    ax.bar(np.arange(len(bins)) + w/2, attack_vals, w, label='PosePert', color='tab:red', hatch='//')
    ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels([b[2] for b in bins])
    ax.set_xlabel('Closest benign dist (m)'); ax.set_ylabel('Avg IoU')
    ax.set_title('(a) Benign distance', fontweight='bold'); ax.set_ylim(0, 1.0); ax.legend()

def draw_b(ax):
    ax.bar(np.arange(len(valid_n)) - w/2, normal_b, w, label='Normal', color='tab:blue')
    ax.bar(np.arange(len(valid_n)) + w/2, attack_b, w, label='PosePert', color='tab:red', hatch='//')
    ax.set_xticks(np.arange(len(valid_n))); ax.set_xticklabels(valid_n)
    ax.set_xlabel('# benign vehicles'); ax.set_ylabel('Avg IoU')
    ax.set_title('(b) Benign count', fontweight='bold'); ax.set_ylim(0, 1.0); ax.legend()

def draw_c(ax):
    ax.bar(np.arange(len(pt_bins)) - w/2, normal_p, w, label='Normal', color='tab:blue')
    ax.bar(np.arange(len(pt_bins)) + w/2, attack_p, w, label='PosePert', color='tab:red', hatch='//')
    ax.set_xticks(np.arange(len(pt_bins))); ax.set_xticklabels([b[2] for b in pt_bins])
    ax.set_xlabel('Benign points on target'); ax.set_ylabel('Avg IoU')
    ax.set_title('(c) Benign point density', fontweight='bold'); ax.set_ylim(0, 1.0); ax.legend()

def draw_d(ax):
    ax.plot(xs, avg_ious, 'o-', color='tab:red', linewidth=2.5, markersize=10)
    for xv, yv in zip(xs, avg_ious):
        ax.annotate(f'{yv:.2f}', (xv, yv), textcoords="offset points", xytext=(-5, 12), ha='center', fontsize=11)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='IoU=0.5')
    ax.set_xticks(xs); ax.set_xticklabels(['0\n(normal)', '1', '2', '3'])
    ax.set_xlabel('# attackers'); ax.set_ylabel('Avg IoU with target')
    ax.set_title('(d) Attacker count', fontweight='bold'); ax.set_ylim(0, 1.0); ax.legend()

save_subfig(draw_a, 'fig_factor_distance_a.pdf')
save_subfig(draw_b, 'fig_factor_distance_b.pdf')
save_subfig(draw_c, 'fig_factor_distance_c.pdf')
save_subfig(draw_d, 'fig_factor_distance_d.pdf')
