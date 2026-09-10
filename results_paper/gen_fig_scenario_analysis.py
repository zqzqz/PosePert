#!/usr/bin/env python3
"""Generate scenario attack analysis figure: 2x3 grid of box plots (ADE, MinDist)."""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)

# Model configs: (prefix, dir_suffix, display_name)
MODELS = [
    ("S1", "pp", "AttFusion"),
    ("S2", "v2vnet", "V2VNet"),
    ("S3", "cobevt", "CoBEVT"),
]

CATEGORIES = ["Normal", "Ideal\nWB", "Ideal\nBB", "WB", "BB", "Transfer"]


def load_cases(directory):
    """Load all case pkl files from a directory."""
    cases = {}
    if not os.path.isdir(directory):
        print(f"  WARNING: {directory} not found")
        return cases
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("case_") and fname.endswith(".pkl"):
            case_id = fname.replace("case_", "").replace(".pkl", "")
            with open(os.path.join(directory, fname), "rb") as f:
                cases[case_id] = pickle.load(f)
    return cases


def compute_ade(pred, gt_future):
    """ADE = mean L2 between predicted and GT future trajectories."""
    gt = gt_future[:len(pred), :2]
    return np.mean(np.linalg.norm(pred - gt, axis=1))


def compute_mindist(pred, victim_obs):
    """MinDist = min L2 between any predicted point and victim's last observed position."""
    victim_pos = victim_obs[-1, :2]
    return np.min(np.linalg.norm(pred - victim_pos, axis=1))


def get_last_frame_pred(traj_dict):
    """Get prediction from the last frame in a trajectory dict."""
    if not traj_dict:
        return None
    return traj_dict[max(traj_dict.keys())]


def collect_metrics_for_model(prefix, model_suffix):
    """Collect ADE and MinDist for all 5 categories for one model."""
    wb_dir = os.path.join(ROOT, f"{prefix}_scenario_{model_suffix}_whitebox")
    bb_dir = os.path.join(ROOT, f"{prefix}_scenario_{model_suffix}")
    tr_dir = os.path.join(ROOT, f"{prefix}_scenario_{model_suffix}_transfer")

    print(f"Loading {prefix} {model_suffix}...")
    wb_cases = load_cases(wb_dir)
    bb_cases = load_cases(bb_dir)
    tr_cases = load_cases(tr_dir)

    results = {cat: {"ade": [], "mindist": []} for cat in CATEGORIES}

    # Find common case IDs across WB and BB (transfer may have fewer)
    common_ids = sorted(set(wb_cases.keys()) & set(bb_cases.keys()))
    print(f"  WB={len(wb_cases)}, BB={len(bb_cases)}, TR={len(tr_cases)}, common={len(common_ids)}")

    for cid in common_ids:
        wb = wb_cases[cid]
        bb = bb_cases[cid]

        ao_wb = wb["attack_opts"]
        ao_bb = bb["attack_opts"]

        # GT future and victim observed (should be same across WB/BB)
        if "extra" in wb:
            gt_future = wb["extra"]["gt_future_trajectory"]
            victim_obs = wb["extra"]["victim_observed_trajectory"]
            normal_pred = wb["extra"]["normal_predicted_trajectory"]
        else:
            gt_future = wb["gt_future_trajectory"]
            victim_obs = wb["victim_observed_trajectory"]
            normal_pred = wb["normal_pred"]

        # 1. Normal
        if normal_pred is not None and len(normal_pred) > 0:
            results["Normal"]["ade"].append(compute_ade(normal_pred, gt_future))
            results["Normal"]["mindist"].append(compute_mindist(normal_pred, victim_obs))

        # 2. Ideal WB (from WB ideal_predicted_trajectories)
        ideal_wb = get_last_frame_pred(ao_wb.get("ideal_predicted_trajectories", {}))
        if ideal_wb is not None:
            results["Ideal\nWB"]["ade"].append(compute_ade(ideal_wb, gt_future))
            results["Ideal\nWB"]["mindist"].append(compute_mindist(ideal_wb, victim_obs))

        # 2b. Ideal BB (from BB ideal_predicted_trajectories)
        ideal_bb = get_last_frame_pred(ao_bb.get("ideal_predicted_trajectories", {}))
        if ideal_bb is not None:
            results["Ideal\nBB"]["ade"].append(compute_ade(ideal_bb, gt_future))
            results["Ideal\nBB"]["mindist"].append(compute_mindist(ideal_bb, victim_obs))

        # 3. WB actual
        wb_pred = get_last_frame_pred(ao_wb.get("real_predicted_trajectories", {}))
        if wb_pred is not None:
            results["WB"]["ade"].append(compute_ade(wb_pred, gt_future))
            results["WB"]["mindist"].append(compute_mindist(wb_pred, victim_obs))

        # 4. BB actual
        bb_pred = get_last_frame_pred(ao_bb.get("real_predicted_trajectories", {}))
        if bb_pred is not None:
            results["BB"]["ade"].append(compute_ade(bb_pred, gt_future))
            results["BB"]["mindist"].append(compute_mindist(bb_pred, victim_obs))

        # 5. Transfer
        if cid in tr_cases:
            tr = tr_cases[cid]
            tr_pred = tr.get("attack_pred", None)
            if tr_pred is None:
                ao_tr = tr.get("attack_opts", {})
                tr_pred = get_last_frame_pred(ao_tr.get("real_predicted_trajectories", {}))
            tr_gt = tr.get("gt_future_trajectory", gt_future)
            tr_victim = tr.get("victim_observed_trajectory", victim_obs)
            if tr_pred is not None:
                results["Transfer"]["ade"].append(compute_ade(tr_pred, tr_gt))
                results["Transfer"]["mindist"].append(compute_mindist(tr_pred, tr_victim))

    for cat in CATEGORIES:
        print(f"  {cat}: n={len(results[cat]['ade'])}, "
              f"ADE={np.mean(results[cat]['ade']):.2f} +/- {np.std(results[cat]['ade']):.2f}, "
              f"MinDist={np.mean(results[cat]['mindist']):.2f}" if results[cat]['ade'] else f"  {cat}: no data")

    return results


def main():
    all_data = {}
    for prefix, model_suffix, display_name in MODELS:
        all_data[display_name] = collect_metrics_for_model(prefix, model_suffix)

    # Save intermediate data
    data_path = os.path.join(ROOT, "fig_scenario_analysis_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"\nSaved intermediate data to {data_path}")

    # Fixed y-axis limits
    ylims = {"ade": (-0.5, 25), "mindist": (-0.5, 10)}

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 4), sharey='row')

    metrics = ["ade", "mindist"]
    metric_labels = ["ADE (m)", "MinDist (m)"]
    colors = ["#7fbf7f", "#ff9999", "#ffb366", "#6699cc", "#ffcc66", "#cc99cc"]

    for col_idx, (_, _, display_name) in enumerate(MODELS):
        data = all_data[display_name]
        for row_idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx, col_idx]

            box_data = []
            labels = []
            for cat in CATEGORIES:
                vals = data[cat][metric]
                if vals:
                    box_data.append(vals)
                    labels.append(cat)

            bp = ax.boxplot(
                box_data,
                labels=labels,
                patch_artist=True,
                widths=0.6,
                showfliers=False,
                medianprops=dict(color='black', linewidth=1.5),
            )

            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

            ax.tick_params(axis='x', labelsize=8, rotation=20)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim(ylims[metric])

            if row_idx == 0:
                ax.set_title(display_name, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    fig_path = os.path.join(os.environ.get("FIG_OUT_DIR", os.path.join(PROJ, "results_paper/figures")), "fig_scenario_analysis.pdf")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {fig_path}")

    # Also save PNG for quick viewing
    png_path = fig_path.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f"Saved PNG to {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
