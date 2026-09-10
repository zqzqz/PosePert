"""Generate diverse case study figures for scenario attacks.

Produces before/after BEV trajectory plots matching the paper's case_study style:
  - Top panel: traffic scene before attack (normal observation + prediction)
  - Bottom panel: traffic scene after attack (perturbed positions + attack prediction)
  - Dot trajectories with time-varying opacity
  - Green=Victim, Blue=Target, Red=Attacker, gray=other vehicles
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.data.util import bbox_sensor_to_map


CASES = [
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 0,
        'label': 'Opposite direction',
        'description': 'Target approaches victim from opposite direction',
        'fname': 'case_study_opposite',
    },
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 35,
        'label': 'Lane intrusion (rear)',
        'description': 'Target in adjacent lane behind victim, shifted into victim lane',
        'fname': 'case_study_lane_intrusion',
    },
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 69,
        'label': 'Intersection (OPV2V)',
        'description': 'Perpendicular crossing at intersection',
        'fname': 'case_study_intersection_opv2v',
    },
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 83,
        'label': 'Follow (same direction)',
        'description': 'Same direction, target trailing behind victim',
        'fname': 'case_study_follow',
    },
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 51,
        'label': 'Opposite direction (2)',
        'description': 'Another opposite direction scenario, different road',
        'fname': 'case_study_opposite2',
    },
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 55,
        'label': 'Lane intrusion (2)',
        'description': 'Adjacent lane, target approaching from behind',
        'fname': 'case_study_lane_intrusion2',
    },
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 65,
        'label': 'Lane intrusion (3)',
        'description': 'Wide road, target in adjacent lane behind victim',
        'fname': 'case_study_lane_intrusion3',
    },
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 91,
        'label': 'Lane intrusion (curved)',
        'description': 'Same direction on curved road, attacker below',
        'fname': 'case_study_lane_curved',
    },
    {
        'dataset': 'OPV2V',
        'result_dir': 'S1_scenario_pp',
        'case_idx': 37,
        'label': 'Lane intrusion (attacker turns)',
        'description': 'Target in adjacent lane, attacker approaching from turn',
        'fname': 'case_study_lane_turn',
    },
]


def load_gt_trajectories(dataset, scenario_case, observer_id, frame_ids):
    """Extract GT trajectories for all vehicles across frames."""
    num_frames = len(frame_ids)
    unique_ids = list(set(
        sum([scenario_case[fi][observer_id]["object_ids"] for fi in frame_ids], [])
    ))
    result = {oid: np.zeros((num_frames, 7)) for oid in unique_ids + [observer_id]}

    for i, fi in enumerate(frame_ids):
        vdata = scenario_case[fi][observer_id]
        result[observer_id][i] = vdata["ego_bbox"]
        for oi, oid in enumerate(vdata["object_ids"]):
            result[oid][i] = bbox_sensor_to_map(
                vdata["gt_bboxes"][oi], vdata["lidar_pose"]
            )
    return result


def plot_trajectory_dots(ax, traj, color, alpha_range=(0.15, 0.9),
                         size_range=(15, 50), zorder=3, highlight_range=None):
    """Plot trajectory as dots with time-varying opacity and size."""
    valid = np.any(traj[:, 3:6] > 0, axis=1) if traj.shape[1] >= 6 else np.ones(len(traj), dtype=bool)
    if valid.sum() < 1:
        return

    n = valid.sum()
    positions = traj[valid, :2]
    alphas = np.linspace(alpha_range[0], alpha_range[1], n)
    sizes = np.linspace(size_range[0], size_range[1], n)

    for i in range(n):
        a = alphas[i]
        s = sizes[i]
        if highlight_range is not None:
            idx_in_full = np.where(valid)[0][i]
            if highlight_range[0] <= idx_in_full <= highlight_range[1]:
                s *= 1.8
                a = min(1.0, a * 1.3)
        ax.scatter(positions[i, 0], positions[i, 1], c=color, s=s,
                   alpha=a, zorder=zorder, edgecolors='none')


def plot_prediction_line(ax, pred, color, linewidth=2.5, alpha=0.8,
                         zorder=4, linestyle='-', label=None):
    """Plot predicted trajectory as a line with end marker."""
    if pred is None or len(pred) == 0:
        return
    ax.plot(pred[:, 0], pred[:, 1], color=color, linewidth=linewidth,
            alpha=alpha, linestyle=linestyle, zorder=zorder, label=label)
    ax.scatter(pred[-1, 0], pred[-1, 1], c=color, s=80, marker='*',
               alpha=alpha, zorder=zorder + 1, edgecolors='none')


def add_vehicle_label(ax, pos, text, color, offset=(0, 0), fontsize=12):
    """Add a text label near a vehicle position."""
    ax.annotate(text, xy=(pos[0], pos[1]),
                xytext=(pos[0] + offset[0], pos[1] + offset[1]),
                fontsize=fontsize, fontweight='bold', color=color,
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5) if (offset[0]**2 + offset[1]**2) > 0 else None,
                zorder=10)


def compute_label_offsets(victim_pos, target_pos, attacker_pos, span):
    """Compute label offsets to avoid overlap."""
    off = max(span * 0.12, 3.0)
    positions = {'victim': victim_pos, 'target': target_pos, 'attacker': attacker_pos}
    offsets = {}
    used = []

    for name, pos in positions.items():
        best_dir = None
        best_dist = -1
        for dx, dy in [(0, off), (0, -off), (-off, 0), (off, 0),
                        (-off*0.7, off*0.7), (off*0.7, off*0.7),
                        (-off*0.7, -off*0.7), (off*0.7, -off*0.7)]:
            cand = (pos[0] + dx, pos[1] + dy)
            min_d = min([np.sqrt((cand[0]-u[0])**2 + (cand[1]-u[1])**2)
                         for u in used] + [1e9])
            if min_d > best_dist:
                best_dist = min_d
                best_dir = (dx, dy)
        offsets[name] = best_dir
        used.append((pos[0] + best_dir[0], pos[1] + best_dir[1]))

    return offsets


def generate_case_study(case_info, save_dir):
    """Generate a single case study figure with before/after panels."""
    dset_name = case_info['dataset']
    result_dir = case_info['result_dir']
    ci = case_info['case_idx']
    fname = case_info['fname']

    root = 'data/OPV2V' if dset_name == 'OPV2V' else 'data/V2X-Real'
    pkl_path = f'results_paper/{result_dir}/case_{ci:03d}.pkl'

    print(f"Loading {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    ao = result['attack_opts']
    extra = result['extra']
    metrics = result['metrics']
    case_update = result['case_update']

    victim_id = ao['victim_vehicle_id']
    attacker_id = ao['attacker_vehicle_id']
    target_id = ao['target_id']
    vtid = ao.get('victim_target_track_id')

    sc_pkl_path = f'{root}/test_scenario_attacks.pkl'
    with open(sc_pkl_path, 'rb') as f:
        sc_cases = pickle.load(f)
    sc = sc_cases[ci]

    ds = OPV2VDataset(root_path=root, mode='test', dataset_name=dset_name)
    case_id = sc.get('case_id', metrics.get('case_id', 0))
    scenario_case = ds.get_case(case_id, tag='scenario')

    frame_ids = sc['frame_ids']
    total_frames = len(frame_ids)
    n_case_frames = len(scenario_case)

    frame_range = list(range(n_case_frames))
    gt_traj = load_gt_trajectories(ds, scenario_case, attacker_id, frame_range)

    gt_traj_victim_view = load_gt_trajectories(ds, scenario_case, victim_id, frame_range)

    history_frames = 20
    attack_frames = 3
    attack_start = history_frames
    attack_end = history_frames + attack_frames - 1

    normal_obs = extra['normal_observed_trajectory']
    victim_obs = extra['victim_observed_trajectory']
    normal_pred = extra['normal_predicted_trajectory']
    gt_future = extra['gt_future_trajectory']

    real_pred_dict = ao.get('real_predicted_trajectories', {})
    last_attack_frame = max(real_pred_dict.keys()) if real_pred_dict else attack_end
    attack_pred = real_pred_dict.get(last_attack_frame)

    ideal_pred_dict = ao.get('ideal_predicted_trajectories', {})
    ideal_pred = ideal_pred_dict.get(last_attack_frame)

    victim_gt = gt_traj.get(victim_id, gt_traj_victim_view.get(victim_id, np.zeros((n_case_frames, 7))))
    target_gt = gt_traj.get(target_id, np.zeros((n_case_frames, 7)))
    if np.all(target_gt[:, 3:6] == 0):
        target_gt = gt_traj_victim_view.get(target_id, target_gt)
    attacker_gt = gt_traj.get(attacker_id, np.zeros((n_case_frames, 7)))

    if np.all(victim_gt[:, 3:6] == 0) and len(victim_obs) > 0:
        victim_gt = np.zeros((len(victim_obs), 7))
        victim_gt[:, :min(7, victim_obs.shape[1])] = victim_obs[:, :7]
        victim_gt[:, 3:6] = np.maximum(victim_gt[:, 3:6], 0.1)
    if np.all(target_gt[:, 3:6] == 0) and len(normal_obs) > 0:
        target_gt = np.zeros((len(normal_obs), 7))
        target_gt[:, :min(7, normal_obs.shape[1])] = normal_obs[:, :7]
        target_gt[:, 3:6] = np.maximum(target_gt[:, 3:6], 0.1)

    other_gts = {}
    for src in [gt_traj, gt_traj_victim_view]:
        for oid, traj in src.items():
            if oid not in [victim_id, target_id, attacker_id] and oid not in other_gts:
                valid = np.any(traj[:, 3:6] > 0, axis=1)
                if valid.sum() >= 3:
                    other_gts[oid] = traj

    attacked_obs = None
    if case_update[attack_end] and victim_id in case_update[attack_end]:
        vdata = case_update[attack_end].get(victim_id, {})
        if isinstance(vdata, dict) and 'observed_trajectories' in vdata:
            if vtid in vdata['observed_trajectories']:
                attacked_obs = vdata['observed_trajectories'][vtid]

    primary_x, primary_y = [], []
    for traj in [victim_gt, target_gt]:
        valid = np.any(traj[:, 3:6] > 0, axis=1)
        if valid.sum() > 0:
            primary_x.extend(traj[valid, 0].tolist())
            primary_y.extend(traj[valid, 1].tolist())
    if attack_pred is not None:
        primary_x.extend(attack_pred[:, 0].tolist())
        primary_y.extend(attack_pred[:, 1].tolist())
    if normal_pred is not None:
        primary_x.extend(normal_pred[:, 0].tolist())
        primary_y.extend(normal_pred[:, 1].tolist())

    xmin, xmax = min(primary_x), max(primary_x)
    ymin, ymax = min(primary_y), max(primary_y)

    attacker_valid = np.any(attacker_gt[:, 3:6] > 0, axis=1)
    if attacker_valid.sum() > 0:
        a_center = attacker_gt[attacker_valid, :2].mean(axis=0)
        p_center = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
        p_span = max(xmax-xmin, ymax-ymin)
        if np.linalg.norm(a_center - p_center) < p_span * 1.5:
            a_near = attacker_gt[attacker_valid]
            dist_to_center = np.linalg.norm(a_near[:, :2] - p_center, axis=1)
            mask = dist_to_center < p_span * 1.2
            if mask.sum() > 0:
                xmin = min(xmin, a_near[mask, 0].min())
                xmax = max(xmax, a_near[mask, 0].max())
                ymin = min(ymin, a_near[mask, 1].min())
                ymax = max(ymax, a_near[mask, 1].max())

    xspan = xmax - xmin
    yspan = ymax - ymin
    span = max(xspan, yspan)
    pad = max(span * 0.25, 8.0)
    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad

    xspan = xmax - xmin
    yspan = ymax - ymin
    min_aspect = 2.0
    if xspan / max(yspan, 0.1) < min_aspect:
        extra = (yspan * min_aspect - xspan) / 2
        xmin -= extra
        xmax += extra
        xspan = xmax - xmin

    fig_w = 10.0
    fig_h_panel = fig_w * yspan / max(xspan, 1)
    fig_h_panel = max(2.5, min(5.0, fig_h_panel))

    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h_panel * 2 + 1.5))

    panel_titles = [
        'Traffic scene observed by the victim before attack',
        'Traffic scene observed by the victim after attack',
    ]

    victim_valid = np.any(victim_gt[:, 3:6] > 0, axis=1)
    target_valid = np.any(target_gt[:, 3:6] > 0, axis=1)
    attacker_valid_mask = np.any(attacker_gt[:, 3:6] > 0, axis=1)

    for panel_idx, ax in enumerate(axes):
        ax.set_facecolor('#f8f9fa')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#333333')

        ax.set_title(panel_titles[panel_idx], fontsize=13, fontweight='bold',
                     pad=10, color='#222222')

        for oid, traj in other_gts.items():
            valid = np.any(traj[:, 3:6] > 0, axis=1)
            if valid.sum() >= 2:
                t = traj[valid]
                ax.plot(t[:, 0], t[:, 1], '-', color='#cccccc', linewidth=0.8,
                        alpha=0.4, zorder=1)

        plot_trajectory_dots(ax, victim_gt, color='green',
                             alpha_range=(0.2, 0.85), size_range=(15, 50),
                             zorder=3, highlight_range=(attack_start, attack_end))

        plot_trajectory_dots(ax, attacker_gt, color='red',
                             alpha_range=(0.15, 0.6), size_range=(10, 30),
                             zorder=3)

        if panel_idx == 0:
            plot_trajectory_dots(ax, target_gt, color='blue',
                                 alpha_range=(0.2, 0.85), size_range=(15, 50),
                                 zorder=3, highlight_range=(attack_start, attack_end))

            if normal_pred is not None and len(normal_pred) > 0:
                plot_prediction_line(ax, normal_pred, color='#b8a900',
                                     linewidth=2.0, alpha=0.5, linestyle='--')

            label_idx = max(1, int(max(victim_valid.sum(), 1) * 0.2))
            v_pos = victim_gt[victim_valid][min(label_idx, max(victim_valid.sum()-1, 0)), :2] if victim_valid.sum() > 0 else victim_obs[0, :2]
            t_pos = target_gt[target_valid][min(label_idx, max(target_valid.sum()-1, 0)), :2] if target_valid.sum() > 0 else normal_obs[0, :2]
            a_pos = attacker_gt[attacker_valid_mask][min(label_idx, max(attacker_valid_mask.sum()-1, 0)), :2] if attacker_valid_mask.sum() > 0 else np.array([0, 0])

            offsets = compute_label_offsets(v_pos, t_pos, a_pos, span)
            for pos, color, name in [
                (v_pos, 'green', 'Victim'),
                (t_pos, 'blue', 'Target'),
                (a_pos, 'red', 'Attacker'),
            ]:
                off = offsets[name.lower()]
                add_vehicle_label(ax, pos, name, color, offset=off, fontsize=12)

        else:
            pert_center = None
            if attacked_obs is not None:
                valid = np.any(attacked_obs[:, 3:6] > 0, axis=1) if attacked_obs.shape[1] >= 6 else np.ones(len(attacked_obs), dtype=bool)
                obs_valid = attacked_obs[valid]
                n_obs = len(obs_valid)
                normal_end = max(0, n_obs - attack_frames)

                if normal_end > 0:
                    pre_attack = np.zeros((normal_end, target_gt.shape[1]))
                    pre_attack[:, :min(obs_valid.shape[1], target_gt.shape[1])] = obs_valid[:normal_end, :min(obs_valid.shape[1], target_gt.shape[1])]
                    pre_attack[:, 3:6] = 1.0
                    plot_trajectory_dots(ax, pre_attack, color='blue',
                                         alpha_range=(0.2, 0.6), size_range=(12, 35),
                                         zorder=3)

                if n_obs > normal_end:
                    perturbed = obs_valid[normal_end:]
                    for i in range(len(perturbed)):
                        ax.scatter(perturbed[i, 0], perturbed[i, 1], c='darkblue',
                                   s=90, alpha=0.9, zorder=5, edgecolors='blue',
                                   linewidths=2.0)

                    cx = perturbed[:, 0].mean()
                    cy = perturbed[:, 1].mean()
                    rx = (perturbed[:, 0].max() - perturbed[:, 0].min()) / 2 + 2.0
                    ry = (perturbed[:, 1].max() - perturbed[:, 1].min()) / 2 + 2.0
                    ellipse = matplotlib.patches.Ellipse(
                        (cx, cy), 2*rx, 2*ry, fill=False,
                        edgecolor='blue', linewidth=2.0, linestyle='-',
                        alpha=0.7, zorder=4)
                    ax.add_patch(ellipse)

                    pert_center = np.array([cx, cy])
                    pert_radius = np.array([rx, ry])
            else:
                plot_trajectory_dots(ax, target_gt, color='blue',
                                     alpha_range=(0.2, 0.85), size_range=(15, 50),
                                     zorder=3)

            if attack_pred is not None and len(attack_pred) > 0:
                plot_prediction_line(ax, attack_pred, color='#cc9900',
                                     linewidth=3.0, alpha=0.9, linestyle='-')

                min_dist = metrics.get('min_pred_dist_to_victim', 999)
                pred_ade = metrics.get('pred_ade', 0)
                v_ref = victim_gt[victim_valid][-1, :2] if victim_valid.sum() > 0 else None

                off_ann = max(span * 0.12, 5.0)
                has_pert = pert_center is not None
                pert_c = pert_center

                if min_dist < 5.0 and v_ref is not None:
                    dists = np.linalg.norm(attack_pred - v_ref, axis=1)
                    closest_pt = attack_pred[np.argmin(dists)]
                    toward_v = v_ref - closest_pt
                    toward_n = toward_v / (np.linalg.norm(toward_v) + 1e-6)
                    brake_pos = closest_pt + toward_n * off_ann
                    ax.annotate('Brake to yield', xy=closest_pt,
                                xytext=brake_pos,
                                fontsize=11, fontweight='bold', color='#cc6600',
                                ha='center', va='center',
                                arrowprops=dict(arrowstyle='->', color='#cc6600', lw=1.8),
                                zorder=10)
                    if has_pert:
                        away = pert_c - brake_pos
                        away_n = away / (np.linalg.norm(away) + 1e-6)
                        pert_text = pert_c + away_n * off_ann
                        ax.annotate('Perturbed object\nlocations', xy=tuple(pert_c),
                                    xytext=tuple(pert_text),
                                    fontsize=10, fontweight='bold', color='blue',
                                    ha='center', va='center',
                                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                                    zorder=10)
                else:
                    if has_pert:
                        away = pert_c - (v_ref if v_ref is not None else pert_c + np.array([0, -1]))
                        away_n = away / (np.linalg.norm(away) + 1e-6)
                        pert_text = pert_c + away_n * off_ann
                        ax.annotate('Perturbed object\nlocations', xy=tuple(pert_c),
                                    xytext=tuple(pert_text),
                                    fontsize=10, fontweight='bold', color='blue',
                                    ha='center', va='center',
                                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                                    zorder=10)
                    if pred_ade > 2.0:
                        pred_end = attack_pred[-1]
                        ax.annotate(f'High prediction\nerror ({pred_ade:.1f}m)',
                                    xy=(pred_end[0], pred_end[1]),
                                    xytext=(pred_end[0] + off_ann*0.7, pred_end[1] + off_ann*0.7),
                                    fontsize=10, fontweight='bold', color='#cc3300',
                                    ha='center', va='center',
                                    arrowprops=dict(arrowstyle='->', color='#cc3300', lw=1.5),
                                    zorder=10)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15, linestyle='--')

    plt.tight_layout(h_pad=2.0)

    os.makedirs(save_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        path = os.path.join(save_dir, f'{fname}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  Saved {path}")
    plt.close(fig)

    print(f"  minDist={metrics.get('min_pred_dist_to_victim', 0):.2f}m, "
          f"predADE={metrics.get('pred_ade', 0):.2f}m")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='*', type=int, default=None,
                        help='Case indices to generate (0-based into CASES list)')
    parser.add_argument('--save_dir', default='results_paper/case_studies')
    args = parser.parse_args()

    cases_to_run = CASES if args.cases is None else [CASES[i] for i in args.cases]

    for case_info in cases_to_run:
        print(f"\n=== {case_info['label']} ({case_info['dataset']} case {case_info['case_idx']}) ===")
        try:
            generate_case_study(case_info, args.save_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
