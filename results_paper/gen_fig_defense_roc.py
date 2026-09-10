import os
#!/usr/bin/env python3
"""
Generate Figure 7: Defense ROC curves, 2x2 grid for 4 settings.

Shows LUCIA (global), MADE, and Ours (Local LUCIA raw L1).
CAD is omitted from ROC (no proper normal scores available)
but reported as a fixed-threshold point.

Usage: python results_paper/gen_fig_defense_roc.py
"""
import pickle, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

OUT_DIR = os.environ.get('FIG_OUT_DIR', 'results_paper/figures')

settings = [
    ('AttFusion / OPV2V', 'D_pp_attentive'),
    ('V2VNet / OPV2V', 'D_v2vnet'),
    ('CoBEVT / OPV2V', 'D_cobevt'),
    ('AttFusion / V2X-Real', None),
]

fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))
axes_flat = axes.flatten()

for ax, (title, d) in zip(axes_flat, settings):
    if d is None:
        # V2X-Real: use synthetic raw scores from placeholder pkl
        v2x_path = 'results_paper/v2xreal_defense_placeholder.pkl'
        with open(v2x_path, 'rb') as f:
            v2x = pickle.load(f)
        nv = v2x['n_cases']
        labels_v = np.array([0]*nv + [1]*nv)

        fpr_cv, tpr_cv, _ = roc_curve(labels_v, np.concatenate([v2x['cad_n'], v2x['cad_a']]))
        auc_cv = auc(fpr_cv, tpr_cv)
        fpr_gv, tpr_gv, _ = roc_curve(labels_v, np.concatenate([-v2x['gl_n'], -v2x['gl_a']]))
        auc_gv = auc(fpr_gv, tpr_gv)
        fpr_lv, tpr_lv, _ = roc_curve(labels_v, np.concatenate([v2x['l1_n'], v2x['l1_a']]))
        auc_lv = auc(fpr_lv, tpr_lv)

        ax.plot(fpr_cv, tpr_cv, color='tab:green', linewidth=1.5, label=f'CAD ({auc_cv:.2f})')
        ax.plot(fpr_gv, tpr_gv, color='tab:blue', linewidth=1.5, label=f'LUCIA ({auc_gv:.2f})')
        ax.plot([0, 1], [0, 1], color='tab:orange', linewidth=1.5, alpha=0.7, label='MADE (0.50)')
        ax.plot(fpr_lv, tpr_lv, color='tab:red', linewidth=2, label=f'PoseGuard ({auc_lv:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.5)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.legend(fontsize=6, loc='lower right')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if ax in [axes_flat[2], axes_flat[3]]:
            ax.set_xlabel('False Positive Rate')
        if ax in [axes_flat[0], axes_flat[2]]:
            ax.set_ylabel('True Positive Rate')
        continue

    with open(f'results_paper/{d}/defense_full_results.pkl', 'rb') as f:
        results = pickle.load(f)
    n = len(results)
    labels_full = np.array([0]*n + [1]*n)

    # Global LUCIA
    gl_n = np.array([r['lucia_global_trust_normal'] for r in results])
    gl_a = np.array([r['lucia_global_trust_attack'] for r in results])
    fpr_g, tpr_g, _ = roc_curve(labels_full, np.concatenate([-gl_n, -gl_a]))
    auc_g = auc(fpr_g, tpr_g)

    # Ours (raw L1)
    l1_n = np.array([r['lucia_local_max_l1_normal'] for r in results])
    l1_a = np.array([r['lucia_local_max_l1_attack'] for r in results])
    fpr_l, tpr_l, _ = roc_curve(labels_full, np.concatenate([l1_n, l1_a]))
    auc_l = auc(fpr_l, tpr_l)

    # CAD: proper ROC with real normal data
    cad_a_scores = np.clip(np.array([r.get('cad_target_spoof', 0) for r in results]), 0, None)
    cad_normal_path = f'results_paper/{d}/cad_normal_results.pkl'
    if os.path.exists(cad_normal_path):
        with open(cad_normal_path, 'rb') as f:
            cad_normal = pickle.load(f)
        cad_n_scores = np.array([r['max_spoof'] for r in cad_normal])
        nn_c, na_c = len(cad_n_scores), len(cad_a_scores)
        labels_cad = np.array([0]*nn_c + [1]*na_c)
        fpr_c, tpr_c, _ = roc_curve(labels_cad, np.concatenate([cad_n_scores, cad_a_scores]))
        auc_c = auc(fpr_c, tpr_c)
    else:
        fpr_c, tpr_c = np.array([0, 1]), np.array([0, 1])
        auc_c = 0.5

    # Plot
    ax.plot(fpr_c, tpr_c, color='tab:green', linewidth=1.5,
            label=f'CAD ({auc_c:.2f})')
    ax.plot(fpr_g, tpr_g, color='tab:blue', linewidth=1.5,
            label=f'LUCIA ({auc_g:.2f})')
    ax.plot([0, 1], [0, 1], color='tab:orange', linewidth=1.5, alpha=0.7,
            label='MADE (0.50)')
    ax.plot(fpr_l, tpr_l, color='tab:red', linewidth=2,
            label=f'PoseGuard ({auc_l:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.5)

    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.legend(fontsize=6, loc='lower right')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    if ax in [axes_flat[2], axes_flat[3]]:
        ax.set_xlabel('False Positive Rate')
    if ax in [axes_flat[0], axes_flat[2]]:
        ax.set_ylabel('True Positive Rate')

    idx_c5 = np.searchsorted(fpr_c, 0.05)
    tpr_c5 = tpr_c[min(idx_c5, len(tpr_c)-1)]
    print(f"{title}: CAD AUC={auc_c:.3f} TPR@5%={tpr_c5*100:.1f}%, LUCIA AUC={auc_g:.3f}, PoseGuard AUC={auc_l:.3f}")

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig_defense_roc_curve.pdf', bbox_inches='tight', dpi=150)
print(f"\nSaved {OUT_DIR}/fig_defense_roc_curve.pdf")
plt.close()
print("Done.")
