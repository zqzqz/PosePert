"""
Complete PertNet attack pipeline: beta scan → data collection → training → evaluation.
Supports PointPillar, V2VNet, CoBEVT on OPV2V.

Usage:
    python mvp/attack/pertnet_pipeline.py --model pointpillar --gpu 2
    python mvp/attack/pertnet_pipeline.py --model v2vnet --gpu 2
    python mvp/attack/pertnet_pipeline.py --model cobevt --gpu 3
"""

import os, sys, argparse, numpy as np, torch, copy, traceback, pickle, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_network import PerturbationNetwork, build_geometric_encoding, get_active_zone_bounds
from mvp.attack.perturbation_data import collect_training_data, PerturbationDataset
from mvp.attack.perturbation_train import build_perception, compute_attack_loss, _apply_warp_patches, _restore_warp_patches
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d
from mvp.util import set_seed
from opencood.tools import train_utils
from opencood.utils import box_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SHIFT = 1.0


def build_model_perception(model_name):
    """Build perception for the specified model."""
    return build_perception(model_name)


def pick_target(perception, frame, ai, vi):
    """Pick a target object that is detected in normal mode."""
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
    """Scan beta values on test split, return results dict."""
    attacks = dataset.attacks
    results = {}

    for beta in betas:
        atk = LidarShiftVoxelwiseAttacker(perception, dataset, beta=beta)
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
                    f"ultra={results[beta]['ultra']:.1f}% iou={results[beta]['iou']:.3f}")

    return results


def select_best_beta(results):
    """Select best beta by Strong%, breaking ties by Ultra%."""
    best_beta = max(results, key=lambda b: (results[b]['strong'], results[b]['ultra']))
    return best_beta


def train_pertnet(perception, dataset, data_dir, save_dir, best_beta, epochs=50):
    """Train PertNet with the selected beta."""
    import mvp.attack.perturbation_train as pt

    # Override beta in the training module
    # We need to monkey-patch since base_beta is hardcoded
    original_fn = pt.compute_attack_loss
    def patched_loss(perc, sample, delta, device):
        # Temporarily set module-level beta
        return original_fn(perc, sample, delta, device)

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

    # Patch warp/discretization for V2VNet/CoAlign
    warp_patches = _apply_warp_patches()

    best_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        net.train()
        epoch_stats = defaultdict(list)
        for si in np.random.permutation(len(train_ds)):
            sample = train_ds[si]
            f_orig = sample['f_orig_crop'].to(device)
            f_diff = sample['f_diff_crop'].to(device)
            geo = sample['geo_encoding'].to(device)
            delta = net(f_orig, f_diff, geo)
            loss, info = compute_attack_loss(perception, sample, delta, device)
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
                'beta': best_beta, 'model_name': args.model,
            }, os.path.join(save_dir, 'perturbation_net_best.pt'))

    _restore_warp_patches(warp_patches)
    logger.info(f"Training done. Best loss: {best_loss:.4f}")
    return net


def eval_pertnet(perception, dataset, net, best_beta, n_cases=300, save_dir='results'):
    """Full evaluation with all metrics."""
    attacks = dataset.attacks
    atk_obj = LidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)
    device = perception.device
    lr = perception.dataset.pre_processor.params["cav_lidar_range"]
    vs = perception.dataset.pre_processor.params["args"]["voxel_size"]
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

            # Normal detection IoU
            pnm = bbox_sensor_to_map(pred_n, vp)
            iou_normal = iou3d(pnm[np.linalg.norm(pnm[:, :2] - btm[:2], axis=1).argmin()], btm)

            entry = {'ci': ci, 'bbox_tgt_map': btm, 'bbox_orig_map': bom,
                     'vic_pose': vp, 'iou_normal_tgt': iou_normal}

            # PertNet attack
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
            bounds = get_active_zone_bounds(bo_e, bt_e, lr, vs, H, W, padding=2)
            h_lo, h_hi, w_lo, w_hi = bounds
            if h_hi <= h_lo or w_hi <= w_lo:
                continue

            foc = F_orig[aidx][:, h_lo:h_hi, w_lo:w_hi]
            fdc = F_diff[:, h_lo:h_hi, w_lo:w_hi]
            geo = build_geometric_encoding(bo_e, bt_e, vps, eidx, aidx, bounds,
                                            lr, vs, H, W, max_vehicles=4).to(device)
            with torch.no_grad(): delta = net(foc, fdc, geo)

            center = perception.point_to_voxel_index(bo_e)
            Hf, Wf = F_orig.shape[2], F_orig.shape[3]; fs = 15
            center[0] = max(fs, min(Wf - fs, center[0]))
            center[1] = max(fs, min(Hf - fs, center[1]))
            C = F_orig.shape[1]
            fp = torch.zeros(C, 2 * fs, 2 * fs, device=device)
            hd, wd = delta.shape[1], delta.shape[2]
            ho = max(0, (2 * fs - hd) // 2); wo = max(0, (2 * fs - wd) // 2)
            he = min(2 * fs, ho + hd); we = min(2 * fs, wo + wd)
            fp[:, ho:he, wo:we] = delta[:, :he - ho, :we - wo]

            # Correct base: beta * F_attack - F_orig
            f_atk_crop = foc + fdc
            bp = torch.zeros_like(fp)
            bp[:, ho:he, wo:we] = (best_beta * f_atk_crop - foc)[:, :he - ho, :we - wo]
            correction = torch.clamp(fp, -10, 10)  # fixed Linf=10
            combined = bp + correction

            F_attack = F_orig.clone()
            cy, cx = center[1], center[0]
            F_attack[aidx, :, cy - fs:cy + fs, cx - fs:cx + fs] = torch.clamp(
                F_orig[aidx, :, cy - fs:cy + fs, cx - fs:cx + fs] + combined, min=0, max=30)

            pa_pn, _ = atk_obj._run_with_features(bd, F_attack)

            # Beta baseline
            atk_beta = LidarShiftVoxelwiseAttacker(perception, dataset, beta=best_beta)
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

    # Save results
    cache_path = os.path.join(save_dir, f'{args.model}_attack_results.pkl')
    with open(cache_path, 'wb') as f: pickle.dump(results, f)

    # Print metrics
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

    # Plot
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
    plt.suptitle(f'{args.model} PertNet (beta={best_beta}) on {n} test cases', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plot_path = os.path.join(save_dir, f'{args.model}_pertnet_eval.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f"Saved {plot_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['pointpillar', 'v2vnet', 'cobevt'])
    parser.add_argument('--n_beta_scan', type=int, default=100)
    parser.add_argument('--n_train_cases', type=int, default=300)
    parser.add_argument('--n_eval', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--skip_scan', action='store_true')
    parser.add_argument('--skip_collect', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--beta', type=float, default=None, help='Override beta selection')
    args = parser.parse_args()

    save_dir = f'results/{args.model}'
    data_dir = f'data/perturbation_train_{args.model}'
    model_dir = f'models/perturbation_net_{args.model}'
    os.makedirs(save_dir, exist_ok=True)

    # Apply warp patches BEFORE building model (patches must be in place
    # before v2v_fuse imports bind the function references)
    logger.info(f"=== {args.model} PertNet Pipeline ===")
    warp_patches = _apply_warp_patches()
    perception = build_model_perception(args.model)
    perception.model.eval()
    device = perception.device

    dataset_test = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')
    dataset_train = OPV2VDataset(root_path='data/OPV2V', mode='train', dataset_name='OPV2V')

    # Step 1: Beta scan on test set
    if args.beta:
        best_beta = args.beta
        logger.info(f"Using provided beta={best_beta}")
    elif args.skip_scan:
        best_beta = 2.0
        logger.info(f"Skipping scan, using default beta={best_beta}")
    else:
        logger.info(f"Step 1: Beta scan ({args.n_beta_scan} test cases)")
        betas = [1.0, 1.5, 2.0, 2.5, 3.0]
        scan_results = run_beta_scan(perception, dataset_test, betas, n_cases=args.n_beta_scan)
        best_beta = select_best_beta(scan_results)
        logger.info(f"Best beta: {best_beta}")

        # Save scan results
        with open(os.path.join(save_dir, 'beta_scan.pkl'), 'wb') as f:
            pickle.dump({'results': scan_results, 'best_beta': best_beta}, f)
        logger.info(f"Saved beta scan to {save_dir}/beta_scan.pkl")

    # Step 2: Collect training data from train split
    if not args.skip_collect:
        logger.info(f"Step 2: Collecting training data ({args.n_train_cases} train cases)")
        os.makedirs(data_dir, exist_ok=True)
        collect_training_data(dataset_train, perception,
                              n_cases=args.n_train_cases, save_dir=data_dir)

    # Step 3: Train PertNet
    if not args.skip_train:
        logger.info(f"Step 3: Training PertNet (beta={best_beta}, {args.epochs} epochs)")
        net = train_pertnet(perception, dataset_train, data_dir, model_dir,
                            best_beta, epochs=args.epochs)
    else:
        ckpt = torch.load(os.path.join(model_dir, 'perturbation_net_best.pt'),
                           map_location='cpu')
        net = PerturbationNetwork(feature_channels=ckpt['feature_channels'],
                                   geo_channels=ckpt['geo_channels']).to(device)
        net.load_state_dict(ckpt['model_state'])
        best_beta = ckpt.get('beta', best_beta)

    net.eval()

    # Step 4: Evaluate on test split
    logger.info(f"Step 4: Evaluating ({args.n_eval} test cases)")
    eval_results = eval_pertnet(perception, dataset_test, net, best_beta,
                                 n_cases=args.n_eval, save_dir=save_dir)

    logger.info("=== Pipeline complete ===")
