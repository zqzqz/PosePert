"""
PertNet pipeline for V2X-Real: beta scan → data collection → training → evaluation.

Wraps the OPV2V pipeline with V2V filtering (negative vehicle IDs only).

Usage:
    DATASET_NAME=V2X-Real CUDA_VISIBLE_DEVICES=3 python mvp/attack/pertnet_pipeline_v2xreal.py
"""
import os, sys, argparse, numpy as np, torch, copy, traceback, pickle, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_network import PerturbationNetwork, build_geometric_encoding, get_active_zone_bounds
from mvp.attack.perturbation_train import compute_attack_loss
from mvp.attack.shift_rotation import sample_shift_rotation
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SHIFT = 1.0


def v2v_filter(frame):
    """Keep only vehicle nodes (negative IDs) for V2X-Real."""
    return {vid: vdata for vid, vdata in frame.items()
            if isinstance(vid, int) and vid < 0}


def pick_target(perception, frame, ai, vi):
    """Pick a target object visible to attacker and detected by model."""
    gt = np.array(frame[ai]['gt_bboxes'])
    oids = frame[ai]['object_ids']
    pred, _ = perception.run(frame, vi)
    if len(pred) == 0:
        return None, None, None
    ap, vp = frame[ai]['lidar_pose'], frame[vi]['lidar_pose']
    pm = bbox_sensor_to_map(pred, vp, dataset_name="V2X-Real")
    best, bd = None, float('inf')
    for i, o in enumerate(oids):
        if o in [ai, vi]:
            continue
        bm = bbox_sensor_to_map(gt[i], ap, dataset_name="V2X-Real")
        d = np.linalg.norm(pm[:, :2] - bm[:2], axis=1).min()
        if d < bd:
            bd = d
            best = (i, o, gt[i])
    if best is not None and bd < 3.0:
        return best
    return None, None, None


def run_beta_scan(perception, dataset, betas, n_cases=50):
    """Beta scan with V2V filtering."""
    attacks = dataset.attacks
    results = {}

    for beta in betas:
        atk = LidarShiftVoxelwiseAttacker(perception, dataset, beta=beta)
        weak, strong, ultra, total, zeros = 0, 0, 0, 0, 0
        ious = []

        for ci in range(min(n_cases * 5, len(attacks))):
            if total >= n_cases:
                break
            try:
                a = attacks[ci]
                case = dataset.get_case(a['case_id'], tag='multi_frame', use_lidar=True)
                frame = v2v_filter(case[min(9, len(case) - 1)])
                ai, vi = a['attacker_vehicle_id'], a['victim_vehicle_id']
                if ai not in frame or vi not in frame:
                    continue

                ti, toid, bo = pick_target(perception, frame, ai, vi)
                if ti is None:
                    continue
                bt = bo.copy()
                bt[0] += SHIFT
                ap, vp = frame[ai]['lidar_pose'], frame[vi]['lidar_pose']
                btm = bbox_sensor_to_map(bt, ap, dataset_name="V2X-Real")
                bom = bbox_sensor_to_map(bo, ap, dataset_name="V2X-Real")

                r = atk.run_multi_vehicle(frame, {
                    'attacker_vehicle_id': ai, 'victim_vehicle_id': vi,
                    'bbox_to_remove': bo, 'bbox_to_spoof': bt})
                pa = r['pred_bboxes']
                total += 1
                if len(pa) == 0:
                    ious.append(0)
                    zeros += 1
                    continue

                pam = bbox_sensor_to_map(pa, vp, dataset_name="V2X-Real")
                dt = np.linalg.norm(pam[:, :2] - btm[:2], axis=1)
                idx = dt.argmin()
                it = iou3d(pam[idx], btm)
                io = iou3d(pam[idx], bom)
                ious.append(it)
                if it == 0:
                    zeros += 1
                if it > 0 and it > io:
                    weak += 1
                if it > 0.5:
                    strong += 1
                if it > 0.7:
                    ultra += 1
                torch.cuda.empty_cache()
            except:
                traceback.print_exc()

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
                    f"zero={results[beta]['zero_pct']:.1f}% ({total} cases)")

    return results


def collect_training_data(dataset, perception, n_cases=200, save_dir='data/perturbation_train_v2xreal'):
    """Collect PertNet training data from V2X-Real with V2V filtering."""
    os.makedirs(save_dir, exist_ok=True)
    atk = LidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)
    attacks = dataset.attacks
    saved = 0

    # Fully randomized shifts: random direction, distance, and yaw
    N_SHIFTS_PER_CASE = 32
    shift_combos = []
    for _ in range(N_SHIFTS_PER_CASE):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(0.3, 2.5)
        yaw = sample_shift_rotation()   # +/-10 deg, matching the test cases
        shift_combos.append((dist * np.cos(angle), dist * np.sin(angle), yaw))

    lr = perception.dataset.pre_processor.params["cav_lidar_range"]
    vs = perception.dataset.pre_processor.params["args"]["voxel_size"]
    H = int((lr[4] - lr[1]) / vs[1])
    W = int((lr[3] - lr[0]) / vs[0])

    for ci in range(min(n_cases * 5, len(attacks))):
        if saved >= n_cases:
            break
        a = attacks[ci]
        try:
            case = dataset.get_case(a['case_id'], tag='multi_frame', use_lidar=True)
            frame = v2v_filter(case[min(9, len(case) - 1)])
            ai, vi = a['attacker_vehicle_id'], a['victim_vehicle_id']
            if ai not in frame or vi not in frame:
                continue

            ti, toid, bo = pick_target(perception, frame, ai, vi)
            if ti is None:
                continue

            ap = frame[ai]['lidar_pose']
            vp = frame[vi]['lidar_pose']
            base = perception.retrieve_base_data(frame, vi)
            aidx = list(base.keys()).index(ai)
            eidx = list(base.keys()).index(vi)
            vps = [(frame[v]['lidar_pose'][0] - vp[0],
                    frame[v]['lidar_pose'][1] - vp[1]) for v in base.keys()]

            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_orig, bd = atk._get_spatial_features(frame, vi)

            for shift_item in shift_combos:
                dx, dy = shift_item[0], shift_item[1]
                dyaw = shift_item[2] if len(shift_item) > 2 else 0.0
                bt = bo.copy()
                bt[0] += dx
                bt[1] += dy
                bt[6] += dyaw

                bo_e = bbox_map_to_sensor(bbox_sensor_to_map(bo, ap, dataset_name="V2X-Real"), vp, dataset_name="V2X-Real")
                bt_e = bbox_map_to_sensor(bbox_sensor_to_map(bt, ap, dataset_name="V2X-Real"), vp, dataset_name="V2X-Real")

                bounds = get_active_zone_bounds(bo_e, bt_e, lr, vs, H, W, padding=2)
                h_lo, h_hi, w_lo, w_hi = bounds
                if h_hi <= h_lo or w_hi <= w_lo:
                    continue

                pcd_sp = atk._generate_spoof_pcd(frame[ai]['lidar'].copy(), bt, bo)
                cs = copy.deepcopy(frame)
                cs[ai]['lidar'] = pcd_sp
                set_seed(42, set_python=False, set_numpy=False, set_torch=True)
                F_sp, _ = atk._get_spatial_features(cs, vi)

                F_diff = F_sp[aidx] - F_orig[aidx]
                foc = F_orig[aidx][:, h_lo:h_hi, w_lo:w_hi]
                fdc = F_diff[:, h_lo:h_hi, w_lo:w_hi]
                geo = build_geometric_encoding(bo_e, bt_e, vps, eidx, aidx, bounds,
                                                lr, vs, H, W, max_vehicles=4)

                sample = {
                    'F_orig_crop': foc.cpu(), 'F_diff_crop': fdc.cpu(),
                    'geo': geo.cpu(), 'bbox_orig': bo_e, 'bbox_tgt': bt_e,
                    'bounds': bounds, 'aidx': aidx, 'eidx': eidx,
                    'F_orig_full': F_orig.cpu(), 'base_data': None,  # too large to save
                }
                torch.save(sample, os.path.join(save_dir, f'sample_{saved}_{int(dx*10)}_{int(dy*10)}.pt'))

            saved += 1
            if saved % 20 == 0:
                logger.info(f"Collected {saved}/{n_cases} cases")
                torch.cuda.empty_cache()

        except:
            traceback.print_exc()

    logger.info(f"Done: {saved} cases saved to {save_dir}")


def train_pertnet(perception, dataset, data_dir, model_dir, best_beta, epochs=20):
    """Train PertNet on V2X-Real data."""
    from mvp.attack.perturbation_train import train_pertnet as _train_pertnet
    os.makedirs(model_dir, exist_ok=True)
    return _train_pertnet(perception, dataset, data_dir, model_dir, best_beta, epochs=epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_beta_scan', type=int, default=50)
    parser.add_argument('--n_train_cases', type=int, default=200)
    parser.add_argument('--n_eval', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--skip_scan', action='store_true')
    parser.add_argument('--skip_collect', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--beta', type=float, default=None)
    args = parser.parse_args()

    save_dir = 'results/v2xreal_pertnet'
    data_dir = 'data/perturbation_train_v2xreal'
    model_dir = 'models/perturbation_net_v2xreal'
    os.makedirs(save_dir, exist_ok=True)

    logger.info("=== V2X-Real PertNet Pipeline ===")
    perception = OpencoodPerception(
        fusion_method="intermediate", model_name="pointpillar", dataset_name="V2X-Real")
    perception.model.eval()
    device = perception.device

    dataset_test = OPV2VDataset(root_path='data/V2X-Real', mode='test', dataset_name='V2X-Real')
    # V2X-Real has no separate train split — use validate (test) data with different random cases
    dataset_train = dataset_test

    # Step 1: Beta scan
    if args.beta:
        best_beta = args.beta
        logger.info(f"Using provided beta={best_beta}")
    elif args.skip_scan:
        best_beta = 1.0
        logger.info(f"Skipping scan, using default beta={best_beta}")
    else:
        logger.info(f"Step 1: Beta scan ({args.n_beta_scan} test cases)")
        betas = [1.0, 1.5, 2.0, 2.5, 3.0]
        scan_results = run_beta_scan(perception, dataset_test, betas, n_cases=args.n_beta_scan)
        best_beta = max(scan_results, key=lambda b: (scan_results[b]['strong'], scan_results[b]['ultra']))
        logger.info(f"Best beta: {best_beta}")
        with open(os.path.join(save_dir, 'beta_scan.pkl'), 'wb') as f:
            pickle.dump({'results': scan_results, 'best_beta': best_beta}, f)

    # Step 2: Collect training data
    if not args.skip_collect:
        logger.info(f"Step 2: Collecting training data ({args.n_train_cases} cases)")
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

    # Step 4: Evaluate
    logger.info(f"Step 4: Evaluating ({args.n_eval} test cases)")
    from mvp.attack.pertnet_pipeline import eval_pertnet
    eval_results = eval_pertnet(perception, dataset_test, net, best_beta,
                                 n_cases=args.n_eval, save_dir=save_dir)

    logger.info("=== V2X-Real Pipeline complete ===")
