"""
Generate Insight 1 / Figure 2 for the paper:
(i)  Spoofing (>3m) + CAD occupancy check visualization
(ii) Small shift (<1m) + CAD occupancy check visualization
(iii) Spoofing (>3m) + feature map heatmap aligned with geometry
(iv) Small shift (<1m) + feature map heatmap aligned with geometry

Each figure: raw PDF, no axis labels/titles/legends.
"""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=2)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

import pickle, copy, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from shapely.ops import unary_union
from shapely.geometry import Polygon as ShapelyPolygon

sys.path.insert(0, '.')
sys.path.insert(0, 'test')

from mvp.perception.opencood_perception import OpencoodPerception
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.defense.perception_defender import PerceptionDefender
from mvp.tools.polygon_space import bbox_to_polygon
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor, pcd_sensor_to_map
from mvp.util import set_seed

SAVE_DIR = 'results_paper/insight_figures'
os.makedirs(SAVE_DIR, exist_ok=True)


def pick_case(dataset, test_cases):
    """Pick a good test case: target has clear detection, reasonable distance."""
    for tc in test_cases:
        case_id = tc['attack_meta']['case_id']
        case = dataset.get_case(case_id, tag='multi_frame', use_lidar=True)
        frame = case[9]
        ao = tc['attack_opts']
        atk_id = ao['attacker_vehicle_id']
        vic_id = ao['victim_vehicle_id']
        obj_id = ao['object_id']
        if obj_id in frame[atk_id]['object_ids']:
            obj_idx = frame[atk_id]['object_ids'].index(obj_id)
            bbox = frame[atk_id]['gt_bboxes'][obj_idx]
            dist = np.sqrt(bbox[0]**2 + bbox[1]**2)
            if 8 < dist < 30:
                return tc, case_id
    return test_cases[0], test_cases[0]['attack_meta']['case_id']


def run_attack(perception, dataset, voxel_attacker, case, tc, shift_distance, shift_direction):
    """Run a single-frame voxelwise attack with given shift parameters."""
    frame = copy.deepcopy(case[9])
    ao = tc['attack_opts']
    atk_id = ao['attacker_vehicle_id']
    vic_id = ao['victim_vehicle_id']
    obj_id = ao['object_id']

    obj_idx = frame[atk_id]['object_ids'].index(obj_id)
    bbox_orig = frame[atk_id]['gt_bboxes'][obj_idx].copy()

    bbox_spoof = bbox_orig.copy()
    bbox_spoof[0] += np.cos(shift_direction) * shift_distance
    bbox_spoof[1] += np.sin(shift_direction) * shift_distance

    # Get features
    set_seed(42)
    F_orig, batch_data = voxel_attacker._get_spatial_features(frame, vic_id)
    base_data_dict = perception.retrieve_base_data(frame, vic_id)
    atk_index = list(base_data_dict.keys()).index(atk_id)

    # Generate spoofed pcd
    atk_pcd = frame[atk_id]['lidar']
    pcd_spoofed = voxel_attacker._generate_spoof_pcd(
        atk_pcd.copy(), bbox_spoof, bbox_orig)

    # Get spoofed features
    frame_spoof = copy.deepcopy(frame)
    frame_spoof[atk_id]['lidar'] = pcd_spoofed
    set_seed(42)
    F_spoof, _ = voxel_attacker._get_spatial_features(frame_spoof, vic_id)

    # Active zone
    atk_pose = frame[atk_id]['lidar_pose']
    vic_pose = frame[vic_id]['lidar_pose']
    bbox_orig_vic = bbox_map_to_sensor(
        bbox_sensor_to_map(bbox_orig, atk_pose), vic_pose)
    bbox_spoof_vic = bbox_map_to_sensor(
        bbox_sensor_to_map(bbox_spoof, atk_pose), vic_pose)

    zone = voxel_attacker._bbox_to_voxel_mask(bbox_orig_vic) | \
           voxel_attacker._bbox_to_voxel_mask(bbox_spoof_vic)

    # Ray-cast features (beta=1) for feature visualization
    features_raycast = F_orig.clone()
    t_zone = torch.from_numpy(zone).to(perception.device)
    if t_zone.any():
        features_raycast[atk_index][:, t_zone] = F_spoof[atk_index][:, t_zone]

    # Beta-scaled features for CAD (actual attack output)
    features_beta = F_orig.clone()
    if t_zone.any():
        features_beta[atk_index][:, t_zone] = (
            voxel_attacker.beta * F_spoof[atk_index][:, t_zone])

    # Run perception with beta-scaled features (for CAD evaluation)
    pred_bboxes, pred_scores = voxel_attacker._run_with_features(batch_data, features_beta)

    # Also get normal detection
    pred_bboxes_normal, pred_scores_normal = perception.run(frame, vic_id)

    return {
        'bbox_orig': bbox_orig,          # in attacker sensor frame
        'bbox_spoof': bbox_spoof,        # in attacker sensor frame
        'bbox_orig_vic': bbox_orig_vic,  # in victim sensor frame
        'bbox_spoof_vic': bbox_spoof_vic,
        'pred_bboxes': pred_bboxes,
        'pred_scores': pred_scores,
        'pred_bboxes_normal': pred_bboxes_normal,
        'F_orig': F_orig.cpu(),
        'F_spoof': F_spoof.cpu(),
        'features': features_raycast.cpu(),
        'features_beta': features_beta.cpu(),
        'zone': zone,
        'atk_index': atk_index,
        'frame': frame,
        'atk_id': atk_id,
        'vic_id': vic_id,
        'atk_pose': atk_pose,
        'vic_pose': vic_pose,
    }


def draw_cad_figure(result, occupancy_data, case_id, tag, save_path):
    """Draw CAD occupancy map + conflict region. Raw image, no decorations."""
    vic_id = result['vic_id']
    vic_pose = result['vic_pose']
    frame = result['frame']

    # Get occupancy data for all vehicles
    occ = occupancy_data
    occupied_all = []
    free_all = []
    for vid, vdata in occ.items():
        occupied_all += vdata['occupied_areas']
        occupied_all.append(vdata['ego_area'])
        free_all.append(unary_union(vdata['free_areas']).difference(vdata['ego_area']))
    free_merged = unary_union(free_all)

    # GT bboxes in map frame
    gt_bboxes_list = []
    for vid, vdata in frame.items():
        gt_bboxes_list.append(bbox_sensor_to_map(vdata['gt_bboxes'], vdata['lidar_pose']))
    gt_bboxes_map = np.vstack(gt_bboxes_list)

    # Pred bboxes in map frame
    pred_bboxes_map = bbox_sensor_to_map(result['pred_bboxes'], vic_pose)

    # CAD check
    cad = PerceptionDefender()
    metrics = cad.run_core(pred_bboxes_map, gt_bboxes_map,
                           unary_union(occupied_all), free_merged,
                           occ[vic_id]['ego_area'])

    # Original and target bboxes in map frame
    bbox_gt_map = bbox_sensor_to_map(
        np.array([result['bbox_orig']]), result['atk_pose'])[0]
    bbox_tgt_map = bbox_sensor_to_map(
        np.array([result['bbox_spoof']]), result['atk_pose'])[0]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw free space (light gray)
    if hasattr(free_merged, 'geoms'):
        for geom in free_merged.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.15, color='gray')
    elif hasattr(free_merged, 'exterior'):
        x, y = free_merged.exterior.xy
        ax.fill(x, y, alpha=0.15, color='gray')

    # Draw occupied areas (blue patches)
    occ_merged = unary_union(occupied_all)
    if hasattr(occ_merged, 'geoms'):
        for geom in occ_merged.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.3, color='blue')
    elif hasattr(occ_merged, 'exterior'):
        x, y = occ_merged.exterior.xy
        ax.fill(x, y, alpha=0.3, color='blue')

    # Centers for proximity filtering
    tgt_center = np.array([bbox_tgt_map[0], bbox_tgt_map[1]])
    gt_center = np.array([bbox_gt_map[0], bbox_gt_map[1]])

    # Draw conflict regions (red) — only near the target area
    for error_area, free_area_error, _, pred_idx in metrics['spoof']:
        if free_area_error < 0.05:
            continue
        # Only show conflict near gt/target bbox
        try:
            ec = np.array(error_area.centroid.coords[0])
            near_target = (np.linalg.norm(ec - tgt_center) < 8 or
                          np.linalg.norm(ec - gt_center) < 8)
        except:
            near_target = False
        if not near_target:
            continue
        if hasattr(error_area, 'exterior'):
            x, y = error_area.exterior.xy
            ax.fill(x, y, alpha=0.6, color='red')
        elif hasattr(error_area, 'geoms'):
            for geom in error_area.geoms:
                if hasattr(geom, 'exterior'):
                    x, y = geom.exterior.xy
                    ax.fill(x, y, alpha=0.6, color='red')

    # Draw GT bbox (green)
    gt_poly = bbox_to_polygon(bbox_gt_map)
    x, y = gt_poly.exterior.xy
    ax.plot(x, y, 'g-', linewidth=2.5)

    # Draw target bbox (red)
    tgt_poly = bbox_to_polygon(bbox_tgt_map)
    x, y = tgt_poly.exterior.xy
    ax.plot(x, y, 'r-', linewidth=2.5)

    # Draw pred bboxes near target only (dashed orange)
    for pb in pred_bboxes_map:
        d = np.linalg.norm(pb[:2] - tgt_center)
        if d > 15:
            continue
        pp = bbox_to_polygon(pb)
        x, y = pp.exterior.xy
        ax.plot(x, y, '--', color='orange', linewidth=1.5)

    # Center on the attack area
    cx = (bbox_gt_map[0] + bbox_tgt_map[0]) / 2
    cy = (bbox_gt_map[1] + bbox_tgt_map[1]) / 2
    span = max(abs(bbox_gt_map[0] - bbox_tgt_map[0]),
               abs(bbox_gt_map[1] - bbox_tgt_map[1])) + 15
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  Saved CAD figure: {save_path}")
    max_spoof = max([s[1] for s in metrics['spoof']], default=0)
    print(f"  Max spoof area: {max_spoof:.2f} m² (threshold={cad.thres})")


def draw_feature_figure(result, save_path):
    """Draw feature map heatmap aligned with geometry. Raw image, no decorations."""
    F = result['features']  # (N_agents, C, H, W)

    # Feature magnitude: sum all agents, then L2 norm over channels
    # This shows the fused BEV feature map as perceived by the victim
    feat_all = F.sum(dim=0).numpy()  # (C, H, W) — sum across agents
    feat_mag = np.linalg.norm(feat_all, axis=0)  # (H, W)

    # BEV grid params
    lr = result['frame'][result['vic_id']].get('lidar_range', None)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot feature heatmap
    # Features are in victim's BEV frame. Flip vertically for correct orientation.
    im = ax.imshow(feat_mag, cmap='hot', origin='lower', aspect='equal',
                   interpolation='nearest')

    # Overlay GT and target bbox outlines in voxel coordinates
    bbox_orig_vic = result['bbox_orig_vic']
    bbox_spoof_vic = result['bbox_spoof_vic']

    # These are the standard OPV2V PointPillar BEV grid params
    _lr = np.array([-140.8, -40, -3, 140.8, 40, 1])
    _vs = np.array([0.4, 0.4, 4])
    H = int((_lr[4] - _lr[1]) / _vs[1])
    W = int((_lr[3] - _lr[0]) / _vs[0])

    def bbox_to_voxel_rect(bbox, lr, vs):
        """Convert bbox center to voxel coordinates for rectangle overlay."""
        cx = (bbox[0] - lr[0]) / vs[0]
        cy = (bbox[1] - lr[1]) / vs[1]
        hw = bbox[3] / vs[0] / 2
        hh = bbox[4] / vs[1] / 2
        yaw = bbox[6]
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        corners = corners @ rot.T + np.array([cx, cy])
        return corners

    # GT bbox in green
    c_gt = bbox_to_voxel_rect(bbox_orig_vic, _lr, _vs)
    ax.plot(c_gt[:, 0], c_gt[:, 1], 'g-', linewidth=2.5)

    # Target bbox in red
    c_tgt = bbox_to_voxel_rect(bbox_spoof_vic, _lr, _vs)
    ax.plot(c_tgt[:, 0], c_tgt[:, 1], 'r-', linewidth=2.5)

    # Zoom to attack region
    all_x = np.concatenate([c_gt[:, 0], c_tgt[:, 0]])
    all_y = np.concatenate([c_gt[:, 1], c_tgt[:, 1]])
    margin = 30
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.axis('off')

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  Saved feature figure: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--case_idx', type=int, default=None,
                        help='Index into test cases (auto-pick if None)')
    args = parser.parse_args()

    print("Loading perception model...")
    perception = OpencoodPerception(
        fusion_method='intermediate', model_name='pointpillar', dataset_name='OPV2V')
    dataset = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')
    voxel_attacker = LidarShiftVoxelwiseAttacker(perception, dataset, beta=2.0)

    # Load test cases
    with open('data/OPV2V/attack/lidar_shift.pkl', 'rb') as f:
        test_cases = pickle.load(f)

    if args.case_idx is not None:
        tc = test_cases[args.case_idx]
        case_id = tc['attack_meta']['case_id']
    else:
        tc, case_id = pick_case(dataset, test_cases)

    case = dataset.get_case(case_id, tag='multi_frame', use_lidar=True)
    ao = tc['attack_opts']
    print(f"Using case_id={case_id}, object={ao['object_id']}, "
          f"attacker={ao['attacker_vehicle_id']}, victim={ao['victim_vehicle_id']}")

    # Load precomputed occupancy map
    occ_path = f'data/OPV2V/normal/{case_id:06d}.pkl'
    if not os.path.exists(occ_path):
        print(f"ERROR: occupancy map not found at {occ_path}")
        return
    occ = pickle.load(open(occ_path, 'rb'))

    # Determine shift direction (perpendicular to target heading)
    obj_idx = case[9][ao['attacker_vehicle_id']]['object_ids'].index(ao['object_id'])
    bbox_gt = case[9][ao['attacker_vehicle_id']]['gt_bboxes'][obj_idx]
    shift_dir = ao.get('shift_direction', bbox_gt[6] + np.pi / 2)

    # === (1) Small shift attack (~0.5m) ===
    print("\n=== Small shift attack (0.5m) ===")
    result_shift = run_attack(perception, dataset, voxel_attacker, case, tc,
                              shift_distance=0.5, shift_direction=shift_dir)

    # === (2) Spoofing attack (>3m) ===
    print("\n=== Spoofing attack (4.0m) ===")
    result_spoof = run_attack(perception, dataset, voxel_attacker, case, tc,
                              shift_distance=4.0, shift_direction=shift_dir)

    # === Generate 4 figures ===
    print("\n=== Generating figures ===")

    # (i) Spoofing + CAD
    draw_cad_figure(result_spoof, occ, case_id, 'spoof',
                    os.path.join(SAVE_DIR, 'fig2_i_spoof_cad.pdf'))

    # (ii) Small shift + CAD
    draw_cad_figure(result_shift, occ, case_id, 'shift',
                    os.path.join(SAVE_DIR, 'fig2_ii_shift_cad.pdf'))

    # (iii) Spoofing + feature heatmap
    draw_feature_figure(result_spoof,
                        os.path.join(SAVE_DIR, 'fig2_iii_spoof_features.pdf'))

    # (iv) Small shift + feature heatmap
    draw_feature_figure(result_shift,
                        os.path.join(SAVE_DIR, 'fig2_iv_shift_features.pdf'))

    print(f"\nAll figures saved to {SAVE_DIR}/")

    # === Combine into 2x2 figure ===
    print("\n=== Combining into stealth_demo.pdf ===")
    combine_figure()


def combine_figure():
    """Combine 4 subfigures into a single 2x2 Figure 2."""
    import subprocess

    subfigs = [
        os.path.join(SAVE_DIR, 'fig2_i_spoof_cad.pdf'),
        os.path.join(SAVE_DIR, 'fig2_ii_shift_cad.pdf'),
        os.path.join(SAVE_DIR, 'fig2_iii_spoof_features.pdf'),
        os.path.join(SAVE_DIR, 'fig2_iv_shift_features.pdf'),
    ]

    # Convert PDFs to PNGs using pdftoppm
    png_paths = []
    for pdf_path in subfigs:
        png_stem = pdf_path.replace('.pdf', '')
        png_path = png_stem + '.png'
        subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile',
                        pdf_path, png_stem], check=True)
        png_paths.append(png_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    labels = ['(a) Spoofing + CAD', '(b) Small shift + CAD',
              '(c) Spoofing + Features', '(d) Small shift + Features']
    for ax, png, label in zip(axes.flat, png_paths, labels):
        img = plt.imread(png)
        ax.imshow(img)
        ax.set_title(label, fontsize=11, pad=4)
        ax.axis('off')
    fig.tight_layout(pad=0.5)
    out_path = os.path.join(os.environ.get('FIG_OUT_DIR', 'results_paper/figures'), 'stealth_demo.pdf')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Combined figure saved to {out_path}")


if __name__ == '__main__':
    main()
