"""Generate Figure 6: Defense score distributions (2x2).

Row 1: shift 0.5-2m (existing data)
Row 2: shift <0.5m (new small-shift data)
Columns: (a/c) CAD score, (b/d) PoseGuard L1 score
"""
import os, sys, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = os.path.join(os.path.dirname(__file__), '..')
result_dir = os.path.join(root, 'results_paper/D_pp_attentive')

# Load existing large-shift data
with open(os.path.join(result_dir, 'defense_full_results.pkl'), 'rb') as f:
    large_shift = pickle.load(f)
with open(os.path.join(result_dir, 'cad_normal_results.pkl'), 'rb') as f:
    cad_normal = pickle.load(f)

# Load small-shift data
with open(os.path.join(result_dir, 'small_shift_defense.pkl'), 'rb') as f:
    small_shift = pickle.load(f)

# === Large shift data ===
# CAD normal: max_spoof from cad_normal_results
cad_normal_scores = [r['max_spoof'] for r in cad_normal if r['max_spoof'] >= 0]
# CAD attack: cad_target_spoof from defense_full_results (only valid ones)
cad_attack_large = [r['cad_target_spoof'] for r in large_shift if r.get('cad_target_spoof', -1) >= 0]
# PoseGuard normal/attack
l1_normal_large = [r['lucia_local_max_l1_normal'] for r in large_shift if r['lucia_local_max_l1_normal'] > 0]
l1_attack_large = [r['lucia_local_max_l1_attack'] for r in large_shift if r['lucia_local_max_l1_attack'] > 0]

# === Small shift data ===
cad_attack_small = [r['cad_target_spoof'] for r in small_shift if r.get('cad_target_spoof', -1) >= 0]
l1_normal_small = [r['lucia_local_max_l1_normal'] for r in small_shift if r['lucia_local_max_l1_normal'] > 0]
l1_attack_small = [r['lucia_local_max_l1_attack'] for r in small_shift if r['lucia_local_max_l1_attack'] > 0]

# CAD threshold
CAD_THRES = 2.7

fig, axes = plt.subplots(2, 2, figsize=(8, 4))

# Common style
hist_kw_n = dict(alpha=0.5, color='cornflowerblue', label='Normal', density=True)
hist_kw_a = dict(alpha=0.5, color='indianred', label='Attack', density=True)

# Row 1: Large shift (0.5-2m)
ax = axes[0, 0]
bins_cad = np.linspace(0, 11, 40)
ax.hist(cad_normal_scores, bins=bins_cad, **hist_kw_n)
ax.hist(cad_attack_large, bins=bins_cad, **hist_kw_a)
ax.axvline(CAD_THRES, color='black', linestyle='--', linewidth=1.5, label=f'Threshold ({CAD_THRES})')
ax.set_xlabel('CAD spoof area')
ax.set_ylabel('Density')
ax.set_title('(a) CAD score (shift 0.5--2m)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7)

ax = axes[0, 1]
bins_l1 = np.linspace(0, 3500, 50)
ax.hist(l1_normal_large, bins=bins_l1, **hist_kw_n)
ax.hist(l1_attack_large, bins=bins_l1, **hist_kw_a)
ax.set_xlabel('Local L1 anomaly score')
ax.set_ylabel('Density')
ax.set_title('(b) PoseGuard score (shift 0.5--2m)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7)

# Row 2: Small shift (<0.5m)
ax = axes[1, 0]
ax.hist(cad_normal_scores, bins=bins_cad, **hist_kw_n)
ax.hist(cad_attack_small, bins=bins_cad, **hist_kw_a)
ax.axvline(CAD_THRES, color='black', linestyle='--', linewidth=1.5, label=f'Threshold ({CAD_THRES})')
ax.set_xlabel('CAD spoof area')
ax.set_ylabel('Density')
ax.set_title('(c) CAD score (shift <0.5m)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7)

ax = axes[1, 1]
ax.hist(l1_normal_small, bins=bins_l1, **hist_kw_n)
ax.hist(l1_attack_small, bins=bins_l1, **hist_kw_a)
ax.set_xlabel('Local L1 anomaly score')
ax.set_ylabel('Density')
ax.set_title('(d) PoseGuard score (shift <0.5m)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7)

plt.tight_layout()
out_path = os.path.join(root, os.path.join(os.environ.get('FIG_OUT_DIR', 'results_paper/figures'), 'fig_defense_dist.pdf'))
fig.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"Saved to {out_path}")

# Print stats
n_large = len(large_shift)
n_small = len(small_shift)
cad_det_large = sum(1 for r in large_shift if r.get('cad_detected') == True)
cad_avail_large = sum(1 for r in large_shift if r.get('cad_detected') is not None)
cad_det_small = sum(1 for r in small_shift if r.get('cad_detected') == True)
cad_avail_small = sum(1 for r in small_shift if r.get('cad_detected') is not None)

print(f"\nLarge shift (0.5-2m): {n_large} cases")
print(f"  CAD detected: {cad_det_large}/{cad_avail_large} ({100*cad_det_large/cad_avail_large:.1f}%)")
print(f"  L1 normal mean: {np.mean(l1_normal_large):.0f}, attack mean: {np.mean(l1_attack_large):.0f}")

print(f"\nSmall shift (<0.5m): {n_small} cases")
print(f"  CAD detected: {cad_det_small}/{cad_avail_small} ({100*cad_det_small/cad_avail_small:.1f}%)")
print(f"  L1 normal mean: {np.mean(l1_normal_small):.0f}, attack mean: {np.mean(l1_attack_small):.0f}")

# PoseGuard detection rate at 5% FPR threshold
all_normal = sorted(l1_normal_large + l1_normal_small)
fpr_idx = int(len(all_normal) * 0.95)
thres_l1 = all_normal[fpr_idx] if fpr_idx < len(all_normal) else all_normal[-1]
det_large = sum(1 for v in l1_attack_large if v > thres_l1) / len(l1_attack_large) * 100
det_small = sum(1 for v in l1_attack_small if v > thres_l1) / len(l1_attack_small) * 100
print(f"\nPoseGuard L1 threshold at 5% FPR: {thres_l1:.0f}")
print(f"  Large shift detection: {det_large:.1f}%")
print(f"  Small shift detection: {det_small:.1f}%")
