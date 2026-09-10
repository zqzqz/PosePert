#!/usr/bin/env python3
"""Generate IoU distribution figure: 2x2 histograms for 4 model/dataset settings.
Shows Normal (blue) vs PosePert (red hatched) IoU_tgt density distributions."""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
# Paper figure directory by default; overridable so the artifact writes locally.
OUT_DIR = os.environ.get('FIG_OUT_DIR', os.path.join(ROOT, 'figures'))
os.makedirs(OUT_DIR, exist_ok=True)

SETTINGS = [
    ("E1_pp_attentive", "AttFusion/OPV2V"),
    ("E2_v2vnet", "V2VNet/OPV2V"),
    ("E3_cobevt", "CoBEVT/OPV2V"),
    ("E4_v2xreal", "AttFusion/V2X-Real"),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))
    axes_flat = axes.flatten()

    for ax, (dirname, title) in zip(axes_flat, SETTINGS):
        pkl_path = os.path.join(ROOT, dirname, 'per_case_results.pkl')
        if not os.path.exists(pkl_path):
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        normal_ious = [c['normal']['iou_tgt'] for c in data if 'normal' in c]
        pertnet_ious = [c['pertnet']['iou_tgt'] for c in data if 'pertnet' in c]

        bins = np.linspace(0, 1, 21)
        ax.hist(normal_ious, bins=bins, density=True, alpha=0.7,
                color='tab:blue', label='Normal')
        ax.hist(pertnet_ious, bins=bins, density=True, alpha=0.7,
                color='tab:red', hatch='//', edgecolor='tab:red', label='PosePert')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)
        ax.set_xlabel('IoU with target', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)

        print(f"{title}: {len(data)} cases, normal mean={np.mean(normal_ious):.3f}, pertnet mean={np.mean(pertnet_ious):.3f}")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_iou_distributions.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, 'fig_iou_distributions.png'), bbox_inches='tight', dpi=150)
    print(f"\nSaved to {OUT_DIR}/fig_iou_distributions.pdf")
    plt.close()


if __name__ == '__main__':
    main()
