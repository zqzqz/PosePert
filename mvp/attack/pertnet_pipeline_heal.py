"""
PertNet attack pipeline for HEAL models.
Supports HeterModelBaseline (lidar_attfuse) and HeterPyramidCollab (HEAL) on OPV2V.

Usage:
    python mvp/attack/pertnet_pipeline_heal.py --model_dir models/HEAL/lidar_attfuse --gpu 1
    python mvp/attack/pertnet_pipeline_heal.py --model_dir models/HEAL/heal_opv2v/final_infer --gpu 1
"""

import os, sys, argparse, numpy as np, torch, copy, traceback, pickle, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict, OrderedDict
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.heal_perception import HealPerception
from mvp.attack.heal_attacker import HealLidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_network import PerturbationNetwork, build_geometric_encoding, get_active_zone_bounds
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SHIFT = 1.0


def build_heal_perception(model_dir, device="cuda:0", modality_config=None,
                          vehicle_modality_fn=None):
    return HealPerception(
        model_dir=model_dir,
        modality_config=modality_config,
        device=device,
        vehicle_modality_fn=vehicle_modality_fn,
    )


def pick_target(perception, frame, ai, vi):
    gt = np.array(frame[ai]['gt_bboxes']); oids = frame[ai]['object_ids']
    pred, _ = perception.run(frame, vi)
    if len(pred) == 0: return None, None, None
    ap, vp = frame[ai]['lidar_pose'], frame[vi]['lidar_pose']
    pm = bbox_sensor_to_map(pred, vp)
    best, bd = None, float('inf')
    for i, o in enumerate(oids):
        if o in [ai, vi]: continue
        bm = bbox_sensor_to_map(gt[i], ap)
        d = np.linalg.norm(pm[:, :2] - bm[:2], axis=1).min()
        if d < bd: bd = d; best = (i, gt[i])
    if best is not None and bd < 3.0:
        return best[0], best[1], pred
    return None, None, None


def run_beta_scan(perception, dataset, betas, n_cases=100):
    attacks = dataset.attacks
    results = {}
    for beta in betas:
        atk = HealLidarShiftVoxelwiseAttacker(perception, dataset, beta=beta)
        weak, strong, ultra, total, zeros = 0, 0, 0, 0, 0
        ious = []
        for ci in range(min(n_cases * 3, len(attacks))):
            if total >= n_cases: break
            try:
                a = attacks[ci]
                case = dataset.get_case(a['case_id'], tag='multi_frame', use_lidar=True)
                frame = case[min(9, len(case) - 1)]
                ai, vi = a['attacker_vehicle_id'], a['victim_vehicle_id']
                ti, bo, pred_n = pick_target(perception, frame, ai, vi)
                if ti is None: continue
                bt = bo.copy(); bt[0] += SHIFT
                ap, vp = frame[ai]['lidar_pose'], frame[vi]['lidar_pose']
                btm = bbox_sensor_to_map(bt, ap); bom = bbox_sensor_to_map(bo, ap)
                r = atk.run_multi_vehicle(frame, {
                    'attacker_vehicle_id': ai, 'victim_vehicle_id': vi,
                    'bbox_to_remove': bo, 'bbox_to_spoof': bt})
                pa = r['pred_bboxes']; total += 1
                if len(pa) == 0: ious.append(0); zeros += 1; continue
                pam = bbox_sensor_to_map(pa, vp)
                dt = np.linalg.norm(pam[:, :2] - btm[:2], axis=1)
                idx = dt.argmin()
                it = iou3d(pam[idx], btm); io = iou3d(pam[idx], bom)
                ious.append(it)
                if it == 0: zeros += 1
                if it > 0 and it > io: weak += 1
                if it > 0.5: strong += 1
                if it > 0.7: ultra += 1
                torch.cuda.empty_cache()
            except: traceback.print_exc()
        nz = sum(1 for x in ious if x > 0)
        results[beta] = {
            'weak': 100 * weak / max(nz, 1),
            'strong': 100 * strong / max(nz, 1),
            'ultra': 100 * ultra / max(nz, 1),
            'iou': np.mean([x for x in ious if x > 0]) if nz > 0 else 0,
            'zero_pct': 100 * zeros / max(total, 1),
            'total': total, 'nz': nz,
        }
        logger.info(f"Beta={beta:.1f}: strong={results[beta]['strong']:.1f}% "
                    f"ultra={results[beta]['ultra']:.1f}% iou={results[beta]['iou']:.3f} "
                    f"zero={results[beta]['zero_pct']:.1f}%")
    return results


def select_best_beta(results):
    return max(results, key=lambda b: (results[b]['strong'], results[b]['ultra']))


def collect_training_data_heal(dataset, perception, n_cases=100,
                               shifts=None, save_dir='data/perturbation_train_heal'):
    """Collect PertNet training data using HEAL perception."""
    if shifts is None:
        n_shifts_per_case = 16
        shifts = []
        for _ in range(n_shifts_per_case):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.5, 2.0)
            shifts.append((dist * np.cos(angle), dist * np.sin(angle)))

    os.makedirs(save_dir, exist_ok=True)
    attacks = dataset.attacks

    lidar_range = np.array(perception.cav_lidar_range)
    voxel_size = np.array(perception.voxel_size)
    H = int((lidar_range[4] - lidar_range[1]) / voxel_size[1])
    W = int((lidar_range[3] - lidar_range[0]) / voxel_size[0])

    attacker_obj = HealLidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)
    samples = []
    sample_idx = 0

    for ci in range(min(n_cases, len(attacks))):
        try:
            attack = attacks[ci]
            case = dataset.get_case(attack['case_id'], tag='multi_frame', use_lidar=True)
            frame = case[min(9, len(case) - 1)]
            atk_id = attack['attacker_vehicle_id']
            vic_id = attack['victim_vehicle_id']

            gt = np.array(frame[atk_id]['gt_bboxes'])
            obj_ids = frame[atk_id]['object_ids']
            ti = next((i for i, o in enumerate(obj_ids)
                       if o not in [atk_id, vic_id]), None)
            if ti is None: continue
            bbox_orig = gt[ti]

            base_data_dict = perception.retrieve_base_data(frame, vic_id)
            attacker_index = list(base_data_dict.keys()).index(atk_id)
            ego_index = list(base_data_dict.keys()).index(vic_id)

            atk_pose = frame[atk_id]['lidar_pose']
            vic_pose = frame[vic_id]['lidar_pose']
            vehicle_poses = []
            for vid in base_data_dict.keys():
                vpose = frame[vid]['lidar_pose']
                dx = vpose[0] - vic_pose[0]
                dy = vpose[1] - vic_pose[1]
                vehicle_poses.append((dx, dy))

            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_orig, batch_data = attacker_obj._get_spatial_features(frame, vic_id)

            bbox_orig_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(bbox_orig, atk_pose), vic_pose)

            for shift_dx, shift_dy in shifts:
                bbox_tgt = bbox_orig.copy()
                bbox_tgt[0] += shift_dx
                bbox_tgt[1] += shift_dy

                bbox_tgt_ego = bbox_map_to_sensor(
                    bbox_sensor_to_map(bbox_tgt, atk_pose), vic_pose)

                # Use attacker-frame bboxes for zone bounds (features are per-vehicle)
                bounds = get_active_zone_bounds(
                    bbox_orig, bbox_tgt, lidar_range, voxel_size,
                    H, W, padding=2)
                h_lo, h_hi, w_lo, w_hi = bounds
                if h_hi <= h_lo or w_hi <= w_lo:
                    continue

                atk_pcd = frame[atk_id]['lidar']
                pcd_spoofed = attacker_obj._generate_spoof_pcd(
                    atk_pcd.copy(), bbox_tgt, bbox_orig)
                case_spoof = copy.deepcopy(frame)
                case_spoof[atk_id]['lidar'] = pcd_spoofed
                set_seed(42, set_python=False, set_numpy=False, set_torch=True)
                F_spoof_full, _ = attacker_obj._get_spatial_features(case_spoof, vic_id)
                F_diff = F_spoof_full[attacker_index] - F_orig[attacker_index]

                f_orig_crop = F_orig[attacker_index][:, h_lo:h_hi, w_lo:w_hi].cpu()
                f_diff_crop = F_diff[:, h_lo:h_hi, w_lo:w_hi].cpu()

                geo_enc = build_geometric_encoding(
                    bbox_orig, bbox_tgt, vehicle_poses,
                    ego_index, attacker_index, bounds,
                    lidar_range, voxel_size, H, W, max_vehicles=4)

                # Store HEAL-specific batch keys
                batch_keys = {}
                for k, v in batch_data['ego'].items():
                    if k in ['record_len', 'anchor_box', 'pairwise_t_matrix',
                             'label_dict', 'transformation_matrix']:
                        batch_keys[k] = v.cpu() if isinstance(v, torch.Tensor) else v
                batch_keys['agent_modality_list'] = batch_data['ego']['agent_modality_list']

                sample = {
                    'f_orig_crop': f_orig_crop,
                    'f_diff_crop': f_diff_crop,
                    'geo_encoding': geo_enc,
                    'bbox_orig_ego': bbox_orig_ego,
                    'bbox_tgt_ego': bbox_tgt_ego,
                    'bbox_orig_atk': bbox_orig,
                    'bbox_tgt_atk': bbox_tgt,
                    'bounds': bounds,
                    'attacker_index': attacker_index,
                    'ego_index': ego_index,
                    'n_agents': len(F_orig),
                    'F_orig': [f.cpu().to_sparse() for f in F_orig],
                    'record_len': batch_data['ego']['record_len'].cpu(),
                    'batch_data_keys': batch_keys,
                }

                save_path = os.path.join(save_dir, f'sample_{sample_idx:04d}.pt')
                torch.save(sample, save_path)
                samples.append(save_path)
                sample_idx += 1

            del F_orig, F_spoof_full, batch_data
            torch.cuda.empty_cache()

            if (ci + 1) % 10 == 0:
                logger.info(f'[{ci+1}/{n_cases}] {sample_idx} samples collected')

        except:
            traceback.print_exc()
            continue

    index_path = os.path.join(save_dir, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump({'sample_paths': samples, 'n_samples': len(samples)}, f)
    logger.info(f'Collected {len(samples)} samples, saved to {save_dir}')
    return samples


def compute_attack_loss_heal(perception, sample, delta, device):
    """Compute attack loss through HEAL model forward pass."""
    import torch.nn.functional as Fn

    F_orig_raw = sample['F_orig']
    if isinstance(F_orig_raw, list):
        F_orig = [f.to_dense().to(device) if f.is_sparse else f.to(device) for f in F_orig_raw]
    else:
        f = F_orig_raw.to_dense().to(device) if F_orig_raw.is_sparse else F_orig_raw.to(device)
        F_orig = [f[i] for i in range(f.shape[0])]
    attacker_index = sample['attacker_index']
    bbox_orig_ego = sample['bbox_orig_ego']
    bbox_tgt_ego = sample['bbox_tgt_ego']
    bbox_orig_atk = sample.get('bbox_orig_atk', bbox_orig_ego)

    atk_feat = F_orig[attacker_index]
    center = perception.point_to_voxel_index(bbox_orig_atk)
    feature_size = 15
    H_feat, W_feat = atk_feat.shape[1], atk_feat.shape[2]
    center[0] = max(feature_size, min(W_feat - feature_size, center[0]))
    center[1] = max(feature_size, min(H_feat - feature_size, center[1]))

    base_beta = sample.get('beta', 2.0)
    C = atk_feat.shape[0]
    h_delta, w_delta = delta.shape[1], delta.shape[2]
    full_pert = torch.zeros(C, 2 * feature_size, 2 * feature_size, device=device)
    h_off = max(0, (2 * feature_size - h_delta) // 2)
    w_off = max(0, (2 * feature_size - w_delta) // 2)
    h_end = min(2 * feature_size, h_off + h_delta)
    w_end = min(2 * feature_size, w_off + w_delta)
    full_pert[:, h_off:h_end, w_off:w_end] = delta[:, :h_end-h_off, :w_end-w_off]

    f_diff_crop = sample['f_diff_crop'].to(device)
    f_orig_crop = sample['f_orig_crop'].to(device)
    f_attack_crop = f_orig_crop + f_diff_crop

    base_pert = torch.zeros(C, 2 * feature_size, 2 * feature_size, device=device)
    base_pert[:, h_off:h_end, w_off:w_end] = (
        base_beta * f_attack_crop - f_orig_crop)[:, :h_end-h_off, :w_end-w_off]

    max_perturb = 10
    correction = torch.clamp(full_pert, -max_perturb, max_perturb)
    combined_pert = base_pert + correction

    F_attack = [f.clone() for f in F_orig]
    cy, cx = center[1], center[0]
    F_attack[attacker_index][:, cy-feature_size:cy+feature_size,
             cx-feature_size:cx+feature_size] = torch.clamp(
        F_orig[attacker_index][:, cy-feature_size:cy+feature_size,
               cx-feature_size:cx+feature_size] + combined_pert,
        min=0.0, max=30.0)

    F_orig_region = F_orig[attacker_index][:, cy-feature_size:cy+feature_size,
                           cx-feature_size:cx+feature_size].detach()
    F_attacked_region = F_attack[attacker_index][:, cy-feature_size:cy+feature_size,
                                  cx-feature_size:cx+feature_size]
    dist_from_orig = (F_attacked_region - F_orig_region).pow(2).mean()

    # Build batch_data for HEAL forward
    record_len = sample['record_len'].to(device)
    batch_keys = sample.get('batch_data_keys', {})
    batch_ego = {'record_len': record_len}
    for k, v in batch_keys.items():
        if k not in batch_ego:
            batch_ego[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    batch_data = {'ego': batch_ego}

    # Forward through HEAL model from pre-backbone features
    with torch.no_grad():
        orig_output = perception.forward_from_features(F_orig, batch_data)
    output = perception.forward_from_features(F_attack, batch_data)

    # Decode proposals
    anchor_box = perception.anchor_box
    if isinstance(anchor_box, np.ndarray):
        anchor_box = torch.from_numpy(anchor_box).to(device)
    elif isinstance(anchor_box, torch.Tensor):
        anchor_box = anchor_box.to(device)

    prob = torch.sigmoid(output['psm'].permute(0, 2, 3, 1)).reshape(-1)
    original_prob = torch.sigmoid(
        orig_output['psm'].permute(0, 2, 3, 1)).reshape(-1).detach()
    proposals = perception.post_processor.delta_to_boxes3d(
        output['rm'], anchor_box)[0]

    bbox_tensor = torch.from_numpy(bbox_orig_ego).to(device).float()
    bbox_tensor[2] += 0.5 * bbox_tensor[5]
    bbox2_tensor = torch.from_numpy(bbox_tgt_ego).to(device).float()
    bbox2_tensor[2] += 0.5 * bbox2_tensor[5]

    iou_fn = perception.iou_torch if hasattr(perception, 'iou_torch') else _iou_torch_simple

    iou_orig = torch.clip(iou_fn(
        proposals[:, [0,1,2,5,4,3,6]],
        bbox_tensor.tile((proposals.shape[0], 1))
    ), min=0, max=1)
    bbox_mask = (iou_orig >= 0.01)

    iou_tgt = torch.clip(iou_fn(
        proposals[:, [0,1,2,5,4,3,6]],
        bbox2_tensor.tile((proposals.shape[0], 1))
    ), min=0, max=1)
    box2_mask = (iou_tgt >= 0.01)

    prob_mask = (prob >= 0.1)
    mask = torch.logical_and(torch.logical_and(bbox_mask, box2_mask), prob_mask)

    if mask.sum() == 0:
        mask = torch.logical_and(bbox_mask, prob_mask)

    if mask.sum() == 0:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        info = {'loss_iou': 0.0, 'loss_prob': 0.0, 'loss_norm': 0.0, 'n_proposals': 0}
        return loss, info

    loss_iou = torch.log(1 - iou_tgt[mask] + 1e-8).sum()
    loss_prob = torch.clip(original_prob[mask] - prob[mask], 0, 1).sum()
    loss = loss_iou + 1.0 * loss_prob + 0.01 * dist_from_orig

    info = {
        'loss_iou': loss_iou.item(),
        'loss_prob': loss_prob.item(),
        'loss_norm': dist_from_orig.item(),
        'n_proposals': mask.sum().item(),
    }
    return loss, info


def _iou_torch_simple(boxes1, boxes2):
    """Simple BEV IoU for 7-DOF boxes (x,y,z,l,w,h,yaw)."""
    x1, y1 = boxes1[:, 0], boxes1[:, 1]
    l1, w1 = boxes1[:, 3], boxes1[:, 4]
    x2, y2 = boxes2[:, 0], boxes2[:, 1]
    l2, w2 = boxes2[:, 3], boxes2[:, 4]

    # Axis-aligned approximation
    x1_min, x1_max = x1 - l1/2, x1 + l1/2
    y1_min, y1_max = y1 - w1/2, y1 + w1/2
    x2_min, x2_max = x2 - l2/2, x2 + l2/2
    y2_min, y2_max = y2 - w2/2, y2 + w2/2

    inter_x = torch.clamp(torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min), min=0)
    inter_y = torch.clamp(torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min), min=0)
    inter = inter_x * inter_y
    area1 = l1 * w1
    area2 = l2 * w2
    union = area1 + area2 - inter + 1e-8
    return inter / union


def train_pertnet_heal(perception, data_dir, save_dir, best_beta,
                       model_tag, epochs=50):
    from mvp.attack.perturbation_data import PerturbationDataset

    train_ds = PerturbationDataset(data_dir)
    logger.info(f"Training samples: {len(train_ds)}")

    sample0 = train_ds[0]
    C = sample0['f_orig_crop'].shape[0]
    G = sample0['geo_encoding'].shape[0]
    device = perception.device

    net = PerturbationNetwork(feature_channels=C, geo_channels=G).to(device)
    logger.info(f"Network params: {sum(p.numel() for p in net.parameters())}")

    for p in perception.model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        net.train()
        epoch_stats = defaultdict(list)
        for si in np.random.permutation(len(train_ds)):
            sample = train_ds[si]
            sample['beta'] = best_beta
            f_orig = sample['f_orig_crop'].to(device)
            f_diff = sample['f_diff_crop'].to(device)
            geo = sample['geo_encoding'].to(device)
            delta = net(f_orig, f_diff, geo)
            loss, info = compute_attack_loss_heal(perception, sample, delta, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            for k, v in info.items(): epoch_stats[k].append(v)
            torch.cuda.empty_cache()
        scheduler.step()

        avg = {k: np.mean(v) for k, v in epoch_stats.items()}
        total_loss = avg.get('loss_iou', 0) + avg.get('loss_prob', 0)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={total_loss:.4f} "
                        f"iou={avg.get('loss_iou',0):.3f} prob={avg.get('loss_prob',0):.3f}")
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save({
                'model_state': net.state_dict(),
                'feature_channels': C, 'geo_channels': G,
                'rank': 4, 'hidden_dim': 64,
                'epoch': epoch + 1, 'loss': total_loss,
                'beta': best_beta, 'model_name': model_tag,
            }, os.path.join(save_dir, 'perturbation_net_best.pt'))

    logger.info(f"Training done. Best loss: {best_loss:.4f}")
    return net


def eval_pertnet_heal(perception, dataset, net, best_beta, model_tag,
                      n_cases=300, save_dir='results'):
    attacks = dataset.attacks
    atk_obj = HealLidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)
    device = perception.device
    lr = np.array(perception.cav_lidar_range)
    vs = np.array(perception.voxel_size)
    H = int((lr[4] - lr[1]) / vs[1]); W = int((lr[3] - lr[0]) / vs[0])

    results = []
    for ci in range(min(n_cases * 3, len(attacks))):
        if len(results) >= n_cases: break
        try:
            a = attacks[ci]
            case = dataset.get_case(a['case_id'], tag='multi_frame', use_lidar=True)
            frame = case[min(9, len(case) - 1)]
            ai, vi = a['attacker_vehicle_id'], a['victim_vehicle_id']
            ti, bo, pred_n = pick_target(perception, frame, ai, vi)
            if ti is None: continue
            bt = bo.copy(); bt[0] += SHIFT
            ap, vp = frame[ai]['lidar_pose'], frame[vi]['lidar_pose']
            btm = bbox_sensor_to_map(bt, ap); bom = bbox_sensor_to_map(bo, ap)

            pnm = bbox_sensor_to_map(pred_n, vp)
            iou_normal = iou3d(pnm[np.linalg.norm(pnm[:, :2] - btm[:2], axis=1).argmin()], btm)

            entry = {'ci': ci, 'bbox_tgt_map': btm, 'bbox_orig_map': bom,
                     'vic_pose': vp, 'iou_normal_tgt': iou_normal}

            bo_e = bbox_map_to_sensor(bbox_sensor_to_map(bo, ap), vp)
            bt_e = bbox_map_to_sensor(bbox_sensor_to_map(bt, ap), vp)
            base = perception.retrieve_base_data(frame, vi)
            aidx = list(base.keys()).index(ai)
            eidx = list(base.keys()).index(vi)
            vps = [(frame[v]['lidar_pose'][0] - vp[0], frame[v]['lidar_pose'][1] - vp[1])
                   for v in base.keys()]
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_orig, bd = atk_obj._get_spatial_features(frame, vi)
            pcd_sp = atk_obj._generate_spoof_pcd(frame[ai]['lidar'].copy(), bt, bo)
            cs = copy.deepcopy(frame); cs[ai]['lidar'] = pcd_sp
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_sp, _ = atk_obj._get_spatial_features(cs, vi)
            F_diff = F_sp[aidx] - F_orig[aidx]
            # Use attacker-frame bboxes for zone/center (features are per-vehicle)
            bounds = get_active_zone_bounds(bo, bt, lr, vs, H, W, padding=2)
            h_lo, h_hi, w_lo, w_hi = bounds
            if h_hi <= h_lo or w_hi <= w_lo:
                continue

            foc = F_orig[aidx][:, h_lo:h_hi, w_lo:w_hi]
            fdc = F_diff[:, h_lo:h_hi, w_lo:w_hi]
            geo = build_geometric_encoding(bo, bt, vps, eidx, aidx, bounds,
                                            lr, vs, H, W, max_vehicles=4).to(device)
            with torch.no_grad(): delta = net(foc, fdc, geo)

            center = perception.point_to_voxel_index(bo)
            atk_feat = F_orig[aidx]
            Hf, Wf = atk_feat.shape[1], atk_feat.shape[2]; fs = 15
            center[0] = max(fs, min(Wf - fs, center[0]))
            center[1] = max(fs, min(Hf - fs, center[1]))
            C = atk_feat.shape[0]
            fp = torch.zeros(C, 2 * fs, 2 * fs, device=device)
            hd, wd = delta.shape[1], delta.shape[2]
            ho = max(0, (2 * fs - hd) // 2); wo = max(0, (2 * fs - wd) // 2)
            he = min(2 * fs, ho + hd); we = min(2 * fs, wo + wd)
            fp[:, ho:he, wo:we] = delta[:, :he - ho, :we - wo]

            f_atk_crop = foc + fdc
            bp = torch.zeros_like(fp)
            bp[:, ho:he, wo:we] = (best_beta * f_atk_crop - foc)[:, :he - ho, :we - wo]
            correction = torch.clamp(fp, -10, 10)
            combined = bp + correction

            F_attack = [f.clone() for f in F_orig]
            cy, cx = center[1], center[0]
            F_attack[aidx][:, cy - fs:cy + fs, cx - fs:cx + fs] = torch.clamp(
                F_orig[aidx][:, cy - fs:cy + fs, cx - fs:cx + fs] + combined, min=0, max=30)

            pa_pn, _ = atk_obj._run_with_features(bd, F_attack)

            atk_beta = HealLidarShiftVoxelwiseAttacker(perception, dataset, beta=best_beta)
            r_beta = atk_beta.run_multi_vehicle(frame, {
                'attacker_vehicle_id': ai, 'victim_vehicle_id': vi,
                'bbox_to_remove': bo, 'bbox_to_spoof': bt})
            pa_beta = r_beta['pred_bboxes']

            for key, pa in [('pertnet', pa_pn), ('beta', pa_beta)]:
                if len(pa) == 0:
                    entry[f'iou_tgt_{key}'] = 0.0; entry[f'iou_orig_{key}'] = 0.0
                else:
                    pam = bbox_sensor_to_map(pa, vp)
                    dt = np.linalg.norm(pam[:, :2] - btm[:2], axis=1)
                    idx = dt.argmin()
                    entry[f'iou_tgt_{key}'] = iou3d(pam[idx], btm)
                    entry[f'iou_orig_{key}'] = iou3d(pam[idx], bom)

            results.append(entry)
            torch.cuda.empty_cache()
            if len(results) % 50 == 0:
                logger.info(f'{len(results)} cases done')
        except: traceback.print_exc()

    cache_path = os.path.join(save_dir, f'{model_tag}_attack_results.pkl')
    with open(cache_path, 'wb') as f: pickle.dump(results, f)

    n = len(results)
    logger.info(f"\n{n} test cases")
    print(f"\n{'Method':<15} | {'Imp%':>6} | {'Weak%':>6} | {'Str%':>6} | {'Ult%':>6} | {'IoU':>5} | {'Zero%':>5}")
    print("-" * 65)
    for method in ['pertnet', 'beta']:
        ious_t = [r[f'iou_tgt_{method}'] for r in results]
        ious_o = [r[f'iou_orig_{method}'] for r in results]
        ious_n = [r['iou_normal_tgt'] for r in results]
        zeros = sum(1 for t in ious_t if t == 0)
        nz = [i for i, t in enumerate(ious_t) if t > 0]
        n_nz = len(nz)
        if n_nz == 0: continue
        imp = sum(1 for i in nz if ious_t[i] > ious_n[i])
        weak = sum(1 for i in nz if ious_t[i] > ious_o[i])
        strong = sum(1 for i in nz if ious_t[i] > 0.5)
        ultra = sum(1 for i in nz if ious_t[i] > 0.7)
        iou_m = np.mean([ious_t[i] for i in nz])
        name = 'PertNet' if method == 'pertnet' else f'Beta={best_beta}'
        print(f"  {name:<13} | {100*imp/n_nz:5.1f}% | {100*weak/n_nz:5.1f}% | "
              f"{100*strong/n_nz:5.1f}% | {100*ultra/n_nz:5.1f}% | {iou_m:5.3f} | {100*zeros/n:5.1f}%")
    print("-" * 65)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for method, color, label in [('pertnet', 'steelblue', 'PertNet'),
                                  ('beta', 'tomato', f'Beta={best_beta}')]:
        ious_t = [r[f'iou_tgt_{method}'] for r in results]
        ious_o = [r[f'iou_orig_{method}'] for r in results]
        axes[0].hist(ious_t, bins=np.linspace(0, 1, 25), alpha=0.5,
                     label=f'{label} (mean={np.mean(ious_t):.3f})', color=color)
        axes[1].scatter(ious_o, ious_t, alpha=0.3, s=10, label=label, color=color)
        diff = [t - o for t, o in zip(ious_t, ious_o)]
        axes[2].hist(diff, bins=np.linspace(-0.6, 0.6, 30), alpha=0.5,
                     label=f'{label} (mean={np.mean(diff):.3f})', color=color)
    axes[0].axvline(x=0.5, color='k', ls='--', alpha=0.3)
    axes[0].axvline(x=0.7, color='r', ls='--', alpha=0.3)
    axes[0].set_xlabel('IoU with Target'); axes[0].set_title('IoU to Target'); axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel('IoU with Original'); axes[1].set_ylabel('IoU with Target')
    axes[1].set_title('IoU Target vs Original'); axes[1].legend(fontsize=8)
    axes[2].axvline(x=0, color='k', ls='--', alpha=0.3)
    axes[2].set_xlabel('IoU_tgt - IoU_orig'); axes[2].set_title('IoU Shift'); axes[2].legend(fontsize=8)
    plt.suptitle(f'{model_tag} PertNet (beta={best_beta}) on {n} test cases', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plot_path = os.path.join(save_dir, f'{model_tag}_pertnet_eval.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f"Saved {plot_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Path to HEAL model directory')
    parser.add_argument('--tag', default=None, help='Short tag for output (default: derived from model_dir)')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n_beta_scan', type=int, default=100)
    parser.add_argument('--n_train_cases', type=int, default=300)
    parser.add_argument('--n_eval', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--skip_scan', action='store_true')
    parser.add_argument('--skip_collect', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--heter', action='store_true',
                        help='Enable heterogeneous modality assignment')
    parser.add_argument('--attacker_mod', default='m1',
                        help='Attacker modality (default: m1)')
    parser.add_argument('--victim_mod', default='m1',
                        help='Victim modality (default: m1)')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model_tag = args.tag or os.path.basename(args.model_dir.rstrip('/'))
    save_dir = f'results_paper/HT_{model_tag}'
    data_dir = f'data/perturbation_train_heal_{model_tag}'
    model_save_dir = f'models/perturbation_net_heal_{model_tag}'
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"=== HEAL PertNet Pipeline: {model_tag} ===")
    logger.info(f"Model dir: {args.model_dir}")

    device = "cuda:0"
    vehicle_modality_fn = None

    if args.heter:
        atk_mod = args.attacker_mod
        vic_mod = args.victim_mod

        def vehicle_modality_fn(vehicle_id, is_ego):
            return vic_mod if is_ego else atk_mod

        modality_config = None
        logger.info(f"Heterogeneous mode: attacker={atk_mod}, victim={vic_mod}")
    else:
        modality_config = {
            'mapping_dict': {'m1': 'm1', 'm2': 'm1', 'm3': 'm1', 'm4': 'm1'},
        }

    perception = build_heal_perception(args.model_dir, device=device,
                                        modality_config=modality_config,
                                        vehicle_modality_fn=vehicle_modality_fn)
    perception.model.eval()
    logger.info(f"Model type: {perception.model_type}, modality: {perception.ego_modality}")

    dataset_test = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')
    dataset_train = OPV2VDataset(root_path='data/OPV2V', mode='train', dataset_name='OPV2V')

    # Step 1: Beta scan
    if args.beta:
        best_beta = args.beta
        logger.info(f"Using provided beta={best_beta}")
    elif args.skip_scan:
        best_beta = 2.0
        logger.info(f"Skipping scan, using default beta={best_beta}")
    else:
        logger.info(f"Step 1: Beta scan ({args.n_beta_scan} test cases)")
        betas = [1.0, 1.5, 2.0, 2.5, 3.0]
        scan_results = run_beta_scan(perception, dataset_test, betas,
                                      n_cases=args.n_beta_scan)
        best_beta = select_best_beta(scan_results)
        logger.info(f"Best beta: {best_beta}")
        with open(os.path.join(save_dir, 'beta_scan.pkl'), 'wb') as f:
            pickle.dump({'results': scan_results, 'best_beta': best_beta}, f)

    # Step 2: Collect training data
    if not args.skip_collect:
        logger.info(f"Step 2: Collecting training data ({args.n_train_cases} train cases)")
        os.makedirs(data_dir, exist_ok=True)
        collect_training_data_heal(dataset_train, perception,
                                   n_cases=args.n_train_cases, save_dir=data_dir)

    # Step 3: Train PertNet
    if not args.skip_train:
        logger.info(f"Step 3: Training PertNet (beta={best_beta}, {args.epochs} epochs)")
        net = train_pertnet_heal(perception, data_dir, model_save_dir,
                                 best_beta, model_tag, epochs=args.epochs)
    else:
        ckpt = torch.load(os.path.join(model_save_dir, 'perturbation_net_best.pt'),
                           map_location='cpu')
        net = PerturbationNetwork(feature_channels=ckpt['feature_channels'],
                                   geo_channels=ckpt['geo_channels']).to(device)
        net.load_state_dict(ckpt['model_state'])
        best_beta = ckpt.get('beta', best_beta)

    net.eval()

    # Step 4: Evaluate
    logger.info(f"Step 4: Evaluating ({args.n_eval} test cases)")
    eval_pertnet_heal(perception, dataset_test, net, best_beta, model_tag,
                      n_cases=args.n_eval, save_dir=save_dir)

    logger.info("=== Pipeline complete ===")
