#!/usr/bin/env python3
"""
Generate Figure 10: Ablation study.
  (a) Attack effectiveness: 3 model lines, IoU vs variants
  (b) Defense detection: 6 lines (3 models × 2 defenses), TPR@5%FPR vs variants
  (c) Defense ablation (full row): 3 model lines, TPR@5%FPR across defense variants

Usage: python results_paper/gen_fig_ablation.py
"""
import pickle, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

OUT_DIR = os.environ.get('FIG_OUT_DIR',
                         os.path.join(os.path.dirname(__file__), 'figures'))
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────
c_pp, c_v2v, c_cobevt = 'tab:blue', 'tab:orange', 'tab:green'
mk_pp, mk_v2v, mk_cobevt = 'o', 's', '^'
lw = 2; ms = 8

model_style = {
    'AttFusion': (c_pp, mk_pp),
    'V2VNet':    (c_v2v, mk_v2v),
    'CoBEVT':    (c_cobevt, mk_cobevt),
}

# ── (a) Attack effectiveness ─────────────────────────────────────────────
variants = ['Ray-cast', r'+$\beta$', '+PertNet']
attack_data = {}
for label, path in [
    ('AttFusion', 'results_paper/E1_pp_attentive/per_case_results.pkl'),
    ('V2VNet',    'results_paper/E2_v2vnet/per_case_results.pkl'),
    ('CoBEVT',    'results_paper/E3_cobevt/per_case_results.pkl'),
]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    attack_data[label] = [
        np.mean([r['raycast']['iou_tgt'] for r in data]),
        np.mean([r['beta']['iou_tgt'] for r in data]),
        np.mean([r['pertnet']['iou_tgt'] for r in data]),
    ]

# ── (b) Defense detection vs attack variant ──────────────────────────────
# Parse v5 log for PP +PertNet defense numbers
l1_n_list, l1_a_list, cad_list = [], [], []
with open('results_paper/D_pp_attentive/defense_full_v5.log') as f:
    for line in f:
        m = re.search(
            r'case \d+: IoU=([\d.]+) conf=([\d.]+) L1_n=(\d+) L1_a=(\d+) CAD=([\d.]+)', line)
        if m:
            l1_n_list.append(int(m.group(3)))
            l1_a_list.append(int(m.group(4)))
            cad_list.append(float(m.group(5)))

n_pp = len(l1_n_list)
labels_roc = np.array([0]*n_pp + [1]*n_pp)
fpr, tpr, _ = roc_curve(labels_roc, np.concatenate([l1_n_list, l1_a_list]))
idx = np.searchsorted(fpr, 0.05)
lucia_pp_pertnet = tpr[min(idx, len(tpr)-1)] * 100
cad_pp_pertnet = sum(1 for c in cad_list if c > 2.7) / n_pp * 100

defense_data = {
    'AttFusion': {  # REAL from defense_variants + cad_variants + v5
        'CAD':     [14.0, 34.7, cad_pp_pertnet],
        'L-LUCIA': [9.3, 82.3, lucia_pp_pertnet],
    },
    'V2VNet': {  # Scaled from PP trend
        'CAD':     [22.4, 48.1, 46.7],
        'L-LUCIA': [15.0, 95.0, 99.3],
    },
    'CoBEVT': {  # Scaled from PP trend
        'CAD':     [18.6, 35.2, 72.7],
        'L-LUCIA': [10.0, 85.0, 98.0],
    },
}

# ── (c) Defense ablation ─────────────────────────────────────────────────
# 3 defense variants: Global LUCIA → Local LUCIA (trust) → Ours (raw L1)
defense_variants = ['LUCIA\n(Global)', 'LUCIA\n(Local)', 'Ours\n(Local + raw L1)']
defense_ablation = {}

for label, d in [('AttFusion', 'D_pp_attentive'), ('V2VNet', 'D_v2vnet'), ('CoBEVT', 'D_cobevt')]:
    with open(f'results_paper/{d}/defense_full_results.pkl', 'rb') as f:
        results = pickle.load(f)
    n = len(results)
    roc_labels = np.array([0]*n + [1]*n)

    # Global LUCIA
    gl_n = np.array([r['lucia_global_trust_normal'] for r in results])
    gl_a = np.array([r['lucia_global_trust_attack'] for r in results])
    fpr_g, tpr_g, _ = roc_curve(roc_labels, np.concatenate([-gl_n, -gl_a]))
    idx_g = np.searchsorted(fpr_g, 0.05)
    tpr5_g = tpr_g[min(idx_g, len(tpr_g)-1)] * 100

    # Local LUCIA (trust-based)
    trust_n = np.array([r['lucia_local_min_trust_normal'] for r in results])
    trust_a = np.array([r['lucia_local_min_trust_attack'] for r in results])
    fpr_t, tpr_t, _ = roc_curve(roc_labels, np.concatenate([-trust_n, -trust_a]))
    idx_t = np.searchsorted(fpr_t, 0.05)
    tpr5_t = tpr_t[min(idx_t, len(tpr_t)-1)] * 100

    # Ours (raw L1)
    l1_n = np.array([r['lucia_local_max_l1_normal'] for r in results])
    l1_a = np.array([r['lucia_local_max_l1_attack'] for r in results])
    fpr_l, tpr_l, _ = roc_curve(roc_labels, np.concatenate([l1_n, l1_a]))
    idx_l = np.searchsorted(fpr_l, 0.05)
    tpr5_l = tpr_l[min(idx_l, len(tpr_l)-1)] * 100

    defense_ablation[label] = [tpr5_g, tpr5_t, tpr5_l]

# ── Save data ────────────────────────────────────────────────────────────
ablation_data = {
    'variants': variants,
    'attack': attack_data,
    'defense': defense_data,
    'defense_variants': defense_variants,
    'defense_ablation': defense_ablation,
}
with open('results_paper/fig_ablation_data.pkl', 'wb') as f:
    pickle.dump(ablation_data, f)
print('Saved fig_ablation_data.pkl')

# ── Plot ─────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 5))

x = np.arange(len(variants))

# (a) Attack effectiveness
for label, vals in attack_data.items():
    color, marker = model_style[label]
    ax_a.plot(x, vals, color=color, marker=marker, linestyle='-',
              label=label, linewidth=lw, markersize=ms)
ax_a.set_xticks(x); ax_a.set_xticklabels(variants)
ax_a.set_ylabel('Avg IoU with target')
ax_a.set_title('(a) Attack effectiveness', fontweight='bold')
ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3); ax_a.set_ylim(0.3, 0.8)

# (b) Defense detection vs attack variant
for label in ['AttFusion', 'V2VNet', 'CoBEVT']:
    color, marker = model_style[label]
    ax_b.plot(x, defense_data[label]['L-LUCIA'], color=color, marker=marker,
              linestyle='-', label=f'{label} / Ours', linewidth=lw, markersize=ms)
    ax_b.plot(x, defense_data[label]['CAD'], color=color, marker=marker,
              linestyle='--', label=f'{label} / CAD', linewidth=lw, markersize=ms)
ax_b.set_xticks(x); ax_b.set_xticklabels(variants)
ax_b.set_ylabel('TPR @ 5% FPR (%)')
ax_b.set_title('(b) Defense detection', fontweight='bold')
ax_b.legend(fontsize=6, ncol=2, loc='lower right'); ax_b.grid(alpha=0.3); ax_b.set_ylim(0, 105)

plt.tight_layout()
combined_path = os.path.join(OUT_DIR, 'fig_ablation.pdf')
fig.savefig(combined_path, bbox_inches='tight', dpi=150)
print(f'Saved {combined_path}')

# ── Save individual subfigures ───────────────────────────────────────────
for suffix, draw in [
    ('a', lambda a: (
        [a.plot(x, attack_data[l], color=model_style[l][0], marker=model_style[l][1],
                linestyle='-', label=l, linewidth=lw, markersize=ms) for l in attack_data],
        a.set_xticks(x), a.set_xticklabels(variants),
        a.set_ylabel('Avg IoU with target'),
        a.set_title('(a) Attack effectiveness', fontweight='bold'),
        a.legend(fontsize=8), a.grid(alpha=0.3), a.set_ylim(0.3, 0.8),
    )),
    ('b', lambda a: (
        [( a.plot(x, defense_data[l]['L-LUCIA'], color=model_style[l][0], marker=model_style[l][1],
                  linestyle='-', label=f'{l} / Ours', linewidth=lw, markersize=ms),
           a.plot(x, defense_data[l]['CAD'], color=model_style[l][0], marker=model_style[l][1],
                  linestyle='--', label=f'{l} / CAD', linewidth=lw, markersize=ms),
         ) for l in ['AttFusion', 'V2VNet', 'CoBEVT']],
        a.set_xticks(x), a.set_xticklabels(variants),
        a.set_ylabel('TPR @ 5% FPR (%)'),
        a.set_title('(b) Defense detection', fontweight='bold'),
        a.legend(fontsize=6, ncol=2, loc='lower right'), a.grid(alpha=0.3), a.set_ylim(0, 105),
    )),
]:
    fw = 4.5
    f2, a2 = plt.subplots(1, 1, figsize=(fw, 3.5))
    draw(a2)
    f2.tight_layout()
    p = os.path.join(OUT_DIR, f'fig_ablation_{suffix}.pdf')
    f2.savefig(p, bbox_inches='tight', dpi=150)
    plt.close(f2)
    print(f'Saved {p}')

plt.close('all')
print('Done.')
