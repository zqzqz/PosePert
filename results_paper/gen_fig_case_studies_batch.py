"""Batch-generate case study figures for ALL scenario attack cases.

Groups cases by scenario_id to load each scenario only once.
"""
import os, sys, pickle, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.data.util import bbox_sensor_to_map


def load_gt_trajectories(scenario_case, observer_id, frame_range):
    num_frames = len(frame_range)
    unique_ids = list(set(
        sum([scenario_case[fi][observer_id]["object_ids"] for fi in frame_range], [])
    ))
    result = {oid: np.zeros((num_frames, 7)) for oid in unique_ids + [observer_id]}
    for i, fi in enumerate(frame_range):
        vdata = scenario_case[fi][observer_id]
        result[observer_id][i] = vdata["ego_bbox"]
        for oi, oid in enumerate(vdata["object_ids"]):
            result[oid][i] = bbox_sensor_to_map(
                vdata["gt_bboxes"][oi], vdata["lidar_pose"]
            )
    return result


def plot_dots(ax, traj, color, alpha_range=(0.15, 0.9), size_range=(15, 50),
              zorder=3, highlight_range=None):
    valid = np.any(traj[:, 3:6] > 0, axis=1) if traj.shape[1] >= 6 else np.ones(len(traj), dtype=bool)
    if valid.sum() < 1:
        return
    n = valid.sum()
    pos = traj[valid, :2]
    alphas = np.linspace(alpha_range[0], alpha_range[1], n)
    sizes = np.linspace(size_range[0], size_range[1], n)
    for i in range(n):
        a, s = alphas[i], sizes[i]
        if highlight_range is not None:
            idx_full = np.where(valid)[0][i]
            if highlight_range[0] <= idx_full <= highlight_range[1]:
                s *= 1.8
                a = min(1.0, a * 1.3)
        ax.scatter(pos[i, 0], pos[i, 1], c=color, s=s, alpha=a,
                   zorder=zorder, edgecolors='none')


def compute_offsets(positions, span):
    off = max(span * 0.12, 3.0)
    dirs = [(0, off), (0, -off), (-off, 0), (off, 0),
            (-off*.7, off*.7), (off*.7, off*.7),
            (-off*.7, -off*.7), (off*.7, -off*.7)]
    offsets = {}
    used = []
    for name, pos in positions.items():
        best_dir, best_dist = None, -1
        for dx, dy in dirs:
            cand = (pos[0]+dx, pos[1]+dy)
            min_d = min([np.sqrt((cand[0]-u[0])**2+(cand[1]-u[1])**2) for u in used]+[1e9])
            if min_d > best_dist:
                best_dist = min_d
                best_dir = (dx, dy)
        offsets[name] = best_dir
        used.append((pos[0]+best_dir[0], pos[1]+best_dir[1]))
    return offsets


def generate_one(ci, result, scenario_case, sc_info, save_dir):
    ao = result['attack_opts']
    extra = result['extra']
    metrics = result['metrics']
    case_update = result['case_update']

    victim_id = ao['victim_vehicle_id']
    attacker_id = ao['attacker_vehicle_id']
    target_id = ao['target_id']
    vtid = ao.get('victim_target_track_id')

    n_case_frames = len(scenario_case)
    frame_range = list(range(n_case_frames))

    gt_traj = load_gt_trajectories(scenario_case, attacker_id, frame_range)
    gt_traj_v = load_gt_trajectories(scenario_case, victim_id, frame_range)

    attack_start, attack_end = 20, 22

    normal_obs = extra['normal_observed_trajectory']
    victim_obs = extra['victim_observed_trajectory']
    normal_pred = extra['normal_predicted_trajectory']

    real_pred_dict = ao.get('real_predicted_trajectories', {})
    last_af = max(real_pred_dict.keys()) if real_pred_dict else attack_end
    attack_pred = real_pred_dict.get(last_af)

    victim_gt = gt_traj.get(victim_id, gt_traj_v.get(victim_id, np.zeros((n_case_frames, 7))))
    target_gt = gt_traj.get(target_id, np.zeros((n_case_frames, 7)))
    if np.all(target_gt[:, 3:6] == 0):
        target_gt = gt_traj_v.get(target_id, target_gt)
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
    for src in [gt_traj, gt_traj_v]:
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

    # Compute bounds from victim + target + predictions
    px, py = [], []
    for traj in [victim_gt, target_gt]:
        v = np.any(traj[:, 3:6] > 0, axis=1)
        if v.sum() > 0:
            px.extend(traj[v, 0].tolist())
            py.extend(traj[v, 1].tolist())
    if attack_pred is not None:
        px.extend(attack_pred[:, 0].tolist())
        py.extend(attack_pred[:, 1].tolist())
    if normal_pred is not None:
        px.extend(normal_pred[:, 0].tolist())
        py.extend(normal_pred[:, 1].tolist())

    xmin, xmax = min(px), max(px)
    ymin, ymax = min(py), max(py)

    a_valid = np.any(attacker_gt[:, 3:6] > 0, axis=1)
    if a_valid.sum() > 0:
        p_center = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
        p_span = max(xmax-xmin, ymax-ymin)
        a_near = attacker_gt[a_valid]
        dist_c = np.linalg.norm(a_near[:, :2] - p_center, axis=1)
        mask = dist_c < p_span * 1.2
        if mask.sum() > 0:
            xmin = min(xmin, a_near[mask, 0].min())
            xmax = max(xmax, a_near[mask, 0].max())
            ymin = min(ymin, a_near[mask, 1].min())
            ymax = max(ymax, a_near[mask, 1].max())

    span = max(xmax-xmin, ymax-ymin)
    pad = max(span * 0.25, 8.0)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
    xs, ys = xmax-xmin, ymax-ymin
    if xs / max(ys, 0.1) < 2.0:
        extra_x = (ys * 2.0 - xs) / 2
        xmin -= extra_x; xmax += extra_x; xs = xmax - xmin

    fig_w = 10.0
    fig_hp = max(2.5, min(5.0, fig_w * ys / max(xs, 1)))
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_hp * 2 + 1.5))

    victim_valid = np.any(victim_gt[:, 3:6] > 0, axis=1)
    target_valid = np.any(target_gt[:, 3:6] > 0, axis=1)
    a_valid_mask = np.any(attacker_gt[:, 3:6] > 0, axis=1)

    titles = ['Traffic scene observed by the victim before attack',
              'Traffic scene observed by the victim after attack']

    min_dist = metrics.get('min_pred_dist_to_victim', 999)

    for pi, ax in enumerate(axes):
        ax.set_facecolor('#f8f9fa')
        for sp in ax.spines.values():
            sp.set_linewidth(1.5); sp.set_color('#333')
        ax.set_title(titles[pi], fontsize=13, fontweight='bold', pad=10, color='#222')

        for oid, traj in other_gts.items():
            v = np.any(traj[:, 3:6] > 0, axis=1)
            if v.sum() >= 2:
                ax.plot(traj[v, 0], traj[v, 1], '-', color='#ccc', lw=0.8, alpha=0.4, zorder=1)

        plot_dots(ax, victim_gt, 'green', (0.2, 0.85), (15, 50), 3, (attack_start, attack_end))
        plot_dots(ax, attacker_gt, 'red', (0.15, 0.6), (10, 30), 3)

        if pi == 0:
            plot_dots(ax, target_gt, 'blue', (0.2, 0.85), (15, 50), 3, (attack_start, attack_end))
            if normal_pred is not None and len(normal_pred) > 0:
                ax.plot(normal_pred[:, 0], normal_pred[:, 1], '--', color='#b8a900',
                        lw=2.0, alpha=0.5, zorder=4)

            li = max(1, int(max(victim_valid.sum(), 1) * 0.2))
            vp = victim_gt[victim_valid][min(li, max(victim_valid.sum()-1, 0)), :2] if victim_valid.sum() > 0 else victim_obs[0, :2]
            tp = target_gt[target_valid][min(li, max(target_valid.sum()-1, 0)), :2] if target_valid.sum() > 0 else normal_obs[0, :2]
            ap = attacker_gt[a_valid_mask][min(li, max(a_valid_mask.sum()-1, 0)), :2] if a_valid_mask.sum() > 0 else np.array([0, 0])
            offs = compute_offsets({'victim': vp, 'target': tp, 'attacker': ap}, span)
            for pos, col, nm in [(vp, 'green', 'Victim'), (tp, 'blue', 'Target'), (ap, 'red', 'Attacker')]:
                o = offs[nm.lower()]
                ax.annotate(nm, xy=pos, xytext=(pos[0]+o[0], pos[1]+o[1]),
                            fontsize=12, fontweight='bold', color=col, ha='center', va='center',
                            arrowprops=dict(arrowstyle='->', color=col, lw=1.5) if (o[0]**2+o[1]**2) > 0 else None,
                            zorder=10)
        else:
            pert_center = None
            if attacked_obs is not None:
                v = np.any(attacked_obs[:, 3:6] > 0, axis=1) if attacked_obs.shape[1] >= 6 else np.ones(len(attacked_obs), dtype=bool)
                ov = attacked_obs[v]
                ne = max(0, len(ov) - 3)
                if ne > 0:
                    pre = np.zeros((ne, target_gt.shape[1]))
                    pre[:, :min(ov.shape[1], target_gt.shape[1])] = ov[:ne, :min(ov.shape[1], target_gt.shape[1])]
                    pre[:, 3:6] = 1.0
                    plot_dots(ax, pre, 'blue', (0.2, 0.6), (12, 35), 3)
                if len(ov) > ne:
                    pert = ov[ne:]
                    for i in range(len(pert)):
                        ax.scatter(pert[i, 0], pert[i, 1], c='darkblue', s=90, alpha=0.9,
                                   zorder=5, edgecolors='blue', linewidths=2.0)
                    cx, cy = pert[:, 0].mean(), pert[:, 1].mean()
                    rx = (pert[:, 0].max() - pert[:, 0].min()) / 2 + 2.0
                    ry = (pert[:, 1].max() - pert[:, 1].min()) / 2 + 2.0
                    ax.add_patch(matplotlib.patches.Ellipse(
                        (cx, cy), 2*rx, 2*ry, fill=False,
                        edgecolor='blue', linewidth=2.0, alpha=0.7, zorder=4))
                    pert_center = np.array([cx, cy])
            else:
                plot_dots(ax, target_gt, 'blue', (0.2, 0.85), (15, 50), 3)

            if attack_pred is not None and len(attack_pred) > 0:
                ax.plot(attack_pred[:, 0], attack_pred[:, 1], '-', color='#cc9900',
                        lw=3.0, alpha=0.9, zorder=4)
                ax.scatter(attack_pred[-1, 0], attack_pred[-1, 1], c='#cc9900',
                           s=80, marker='*', alpha=0.9, zorder=5)

                off_ann = max(span * 0.12, 5.0)
                v_ref = victim_gt[victim_valid][-1, :2] if victim_valid.sum() > 0 else None

                if min_dist < 5.0 and v_ref is not None:
                    dists = np.linalg.norm(attack_pred - v_ref, axis=1)
                    cp = attack_pred[np.argmin(dists)]
                    tn = (v_ref - cp); tn = tn / (np.linalg.norm(tn) + 1e-6)
                    bp = cp + tn * off_ann
                    ax.annotate('Brake to yield', xy=tuple(cp), xytext=tuple(bp),
                                fontsize=11, fontweight='bold', color='#cc6600',
                                ha='center', va='center',
                                arrowprops=dict(arrowstyle='->', color='#cc6600', lw=1.8), zorder=10)
                    if pert_center is not None:
                        away = pert_center - bp
                        away = away / (np.linalg.norm(away) + 1e-6)
                        pt = pert_center + away * off_ann
                        ax.annotate('Perturbed object\nlocations', xy=tuple(pert_center), xytext=tuple(pt),
                                    fontsize=10, fontweight='bold', color='blue',
                                    ha='center', va='center',
                                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5), zorder=10)
                else:
                    if pert_center is not None:
                        vc = victim_gt[victim_valid, :2].mean(axis=0) if victim_valid.sum() > 0 else pert_center + np.array([0, -1])
                        away = pert_center - vc; away = away / (np.linalg.norm(away) + 1e-6)
                        pt = pert_center + away * off_ann
                        ax.annotate('Perturbed object\nlocations', xy=tuple(pert_center), xytext=tuple(pt),
                                    fontsize=10, fontweight='bold', color='blue',
                                    ha='center', va='center',
                                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5), zorder=10)
                    pred_ade = metrics.get('pred_ade', 0)
                    if pred_ade > 2.0:
                        pe = attack_pred[-1]
                        ax.annotate('High prediction\nerror (%.1fm)' % pred_ade,
                                    xy=(pe[0], pe[1]),
                                    xytext=(pe[0]+off_ann*0.7, pe[1]+off_ann*0.7),
                                    fontsize=10, fontweight='bold', color='#cc3300',
                                    ha='center', va='center',
                                    arrowprops=dict(arrowstyle='->', color='#cc3300', lw=1.5), zorder=10)

        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal'); ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15, linestyle='--')

    plt.tight_layout(h_pad=2.0)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(save_dir, 'case_%03d.%s' % (ci, ext)),
                    dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return min_dist


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='results_paper/S1_scenario_pp')
    parser.add_argument('--dataset', default='OPV2V')
    parser.add_argument('--save_dir', default='results_paper/case_studies/opv2v')
    parser.add_argument('--min_dist_threshold', type=float, default=999,
                        help='Only generate for cases with minDist below this')
    args = parser.parse_args()

    root = 'data/OPV2V' if args.dataset == 'OPV2V' else 'data/V2X-Real'
    os.makedirs(args.save_dir, exist_ok=True)

    pkl_files = sorted(glob.glob(os.path.join(args.result_dir, 'case_*.pkl')))
    pkl_files = [f for f in pkl_files if 'summary' not in f]
    print("Found %d case pkl files" % len(pkl_files))

    with open(os.path.join(root, 'test_scenario_attacks.pkl'), 'rb') as f:
        sc_cases = pickle.load(f)

    # Group cases by scenario (case_id) to load each scenario once
    groups = {}
    for pkl_path in pkl_files:
        ci = int(os.path.basename(pkl_path).split('_')[1].split('.')[0])
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
        md = result['metrics'].get('min_pred_dist_to_victim', 999)
        if md > args.min_dist_threshold:
            continue
        case_id = sc_cases[ci].get('case_id', result['metrics'].get('case_id', 0))
        if case_id not in groups:
            groups[case_id] = []
        groups[case_id].append((ci, result))

    print("Loading dataset...")
    ds = OPV2VDataset(root_path=root, mode='test', dataset_name=args.dataset)

    total = sum(len(v) for v in groups.values())
    done = 0
    for case_id in sorted(groups.keys()):
        cases = groups[case_id]
        print("Scenario %d: loading (%d cases)..." % (case_id, len(cases)))
        scenario_case = ds.get_case(case_id, tag='scenario')

        for ci, result in cases:
            sc_info = sc_cases[ci]
            md = generate_one(ci, result, scenario_case, sc_info, args.save_dir)
            done += 1
            print("  case_%03d: minDist=%.2f  [%d/%d]" % (ci, md, done, total))

    print("\nDone! Generated %d figures in %s" % (done, args.save_dir))


if __name__ == '__main__':
    main()
