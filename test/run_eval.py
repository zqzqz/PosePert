"""
Full evaluation: ray-cast baseline, +beta scaling, +PertNet on test cases.
Saves per-case results and aggregated metrics.

Usage:
  CUDA_VISIBLE_DEVICES=1 python test/run_eval.py --model pointpillar --beta 2.0 --dataset OPV2V
  CUDA_VISIBLE_DEVICES=2 python test/run_eval.py --model v2vnet --beta 3.0 --dataset OPV2V
  CUDA_VISIBLE_DEVICES=3 python test/run_eval.py --model cobevt --beta 2.0 --dataset OPV2V
"""
import os, sys, pickle, copy, numpy as np, torch, time, argparse, logging, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_network import PerturbationNetwork, build_geometric_encoding, get_active_zone_bounds
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['pointpillar', 'v2vnet', 'cobevt'])
    parser.add_argument('--beta', required=True, type=float)
    parser.add_argument('--dataset', default='OPV2V', choices=['OPV2V', 'V2X-Real'])
    parser.add_argument('--n_cases', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default=None, help='PertNet checkpoint path')
    args = parser.parse_args()

    if args.dataset == 'OPV2V':
        data_path = os.path.join(root, 'data/OPV2V')
        cache_dir = os.path.join(root, 'data/OPV2V/attack_cache_paper')
        test_pkl = os.path.join(root, 'data/OPV2V/attack/lidar_shift.pkl')
        result_dir = {
            'pointpillar': 'results_paper/E1_pp_attentive',
            'v2vnet': 'results_paper/E2_v2vnet',
            'cobevt': 'results_paper/E3_cobevt',
        }[args.model]
    else:
        data_path = os.path.join(root, 'data/V2X-Real')
        cache_dir = os.path.join(root, 'data/V2X-Real/attack_cache_paper')
        test_pkl = os.path.join(root, 'data/V2X-Real/attack/lidar_shift.pkl')
        result_dir = 'results_paper/E4_v2xreal'

    if args.dataset == 'V2X-Real':
        model_dir = os.path.join(root, f'models/perturbation_net_paper_{args.model}_V2X-Real')
    else:
        model_dir = os.path.join(root, f'models/perturbation_net_paper_{args.model}')
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(model_dir, 'perturbation_net_best.pt')

    os.makedirs(result_dir, exist_ok=True)

    logger.info(f"=== Eval: {args.model} / {args.dataset}, beta={args.beta} ===")

    warp_patches = _apply_warp_patches()
    if args.dataset == 'OPV2V':
        perception = build_perception(args.model)
    else:
        perception = OpencoodPerception(fusion_method='intermediate', model_name='pointpillar',
                                         dataset_name='V2X-Real')
    perception.model.eval()
    device = perception.device

    dataset = OPV2VDataset(root_path=data_path, mode='test', dataset_name=args.dataset)
    atk_obj = LidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)

    lr = perception.dataset.pre_processor.params["cav_lidar_range"]
    vs = perception.dataset.pre_processor.params["args"]["voxel_size"]
    H = int((lr[4] - lr[1]) / vs[1])
    W = int((lr[3] - lr[0]) / vs[0])

    # Load PertNet
    net = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        net = PerturbationNetwork(feature_channels=ckpt['feature_channels'],
                                   geo_channels=ckpt['geo_channels']).to(device)
        net.load_state_dict(ckpt['model_state'])
        net.eval()
        logger.info(f"Loaded PertNet from {ckpt_path}, epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}")
    else:
        logger.info(f"No PertNet checkpoint at {ckpt_path}, evaluating beta-only")

    with open(test_pkl, 'rb') as f:
        attacks = pickle.load(f)

    n_cases = args.n_cases or len(attacks)
    results = []
    t_start = time.time()
    use_bev = (args.dataset == 'V2X-Real')
    _dn = args.dataset if args.dataset != 'OPV2V' else None
    v2v_filter_fn = (lambda frame: {v: d for v, d in frame.items() if isinstance(v, int) and v < 0}) if args.dataset == 'V2X-Real' else None

    if use_bev:
        import cv2
        from shapely.geometry import Polygon as _Polygon
        def iou_bev(b1, b2):
            bp1 = cv2.boxPoints(((b1[0],b1[1]),(b1[3],b1[4]),b1[6]/np.pi*180))
            bp2 = cv2.boxPoints(((b2[0],b2[1]),(b2[3],b2[4]),b2[6]/np.pi*180))
            p1 = _Polygon(bp1); p2 = _Polygon(bp2)
            if not p1.is_valid or not p2.is_valid: return 0.0
            inter = p1.intersection(p2).area; union = p1.area + p2.area - inter
            return inter / max(union, 1e-6)
        iou_fn = iou_bev
    else:
        iou_fn = iou3d

    for ci in range(min(n_cases, len(attacks))):
        cache_path = os.path.join(cache_dir, f'{ci:06d}.pkl')
        if not os.path.exists(cache_path):
            continue

        meta = attacks[ci]['attack_meta']
        try:
            cached = pickle.load(open(cache_path, 'rb'))
            case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
            frame = case[min(9, len(case) - 1)]
            if v2v_filter_fn:
                frame = v2v_filter_fn(frame)
            ai, vi = meta['attacker_vehicle_id'], meta['victim_vehicle_id']
            if ai not in frame or vi not in frame:
                continue

            atk_pose = frame[ai]['lidar_pose']
            vic_pose = frame[vi]['lidar_pose']
            bbox_orig = cached['bbox_orig']
            bbox_tgt = cached['bbox_tgt']

            # Normal detection
            pred_normal, scores_normal = perception.run(frame, vi)

            # Feature extraction
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_orig, bd = atk_obj._get_spatial_features(frame, vi)

            frame_sp = copy.deepcopy(frame)
            frame_sp[ai]['lidar'] = cached['spoof_pcd']
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_spoof, _ = atk_obj._get_spatial_features(frame_sp, vi)

            base = perception.retrieve_base_data(frame, vi)
            aidx = list(base.keys()).index(ai)

            # Convert bboxes to victim frame for IoU
            bo_e = bbox_map_to_sensor(bbox_sensor_to_map(bbox_orig, atk_pose, dataset_name=_dn), vic_pose, dataset_name=_dn)
            bt_e = bbox_map_to_sensor(bbox_sensor_to_map(bbox_tgt, atk_pose, dataset_name=_dn), vic_pose, dataset_name=_dn)

            # --- Variant 1: Ray-cast only (beta=1) ---
            F_rc = F_orig.clone()
            F_rc[aidx] = F_spoof[aidx]
            pred_rc, scores_rc = atk_obj._run_with_features(bd, F_rc)

            # --- Variant 2: Ray-cast + beta scaling ---
            center = perception.point_to_voxel_index(bo_e)
            fs = 15
            Hf, Wf = F_orig.shape[2], F_orig.shape[3]
            center[0] = max(fs, min(Wf - fs, center[0]))
            center[1] = max(fs, min(Hf - fs, center[1]))
            cy, cx = center[1], center[0]

            F_beta = F_orig.clone()
            F_beta[aidx, :, cy-fs:cy+fs, cx-fs:cx+fs] = torch.clamp(
                args.beta * F_spoof[aidx, :, cy-fs:cy+fs, cx-fs:cx+fs], min=0, max=30)
            pred_beta, scores_beta = atk_obj._run_with_features(bd, F_beta)

            # --- Variant 3: Full PertNet ---
            pred_pn = np.array([]).reshape(0, 7)
            scores_pn = np.array([])
            if net is not None:
                bounds = get_active_zone_bounds(bo_e, bt_e, lr, vs, H, W, padding=2)
                h_lo, h_hi, w_lo, w_hi = bounds
                if h_hi > h_lo and w_hi > w_lo:
                    eidx = list(base.keys()).index(vi)
                    vps = [(frame[v]['lidar_pose'][0] - vic_pose[0],
                            frame[v]['lidar_pose'][1] - vic_pose[1]) for v in base.keys()]
                    F_diff = F_spoof[aidx] - F_orig[aidx]
                    foc = F_orig[aidx][:, h_lo:h_hi, w_lo:w_hi]
                    fdc = F_diff[:, h_lo:h_hi, w_lo:w_hi]
                    geo = build_geometric_encoding(bo_e, bt_e, vps, eidx, aidx, bounds,
                                                    lr, vs, H, W, max_vehicles=4).to(device)
                    with torch.no_grad():
                        delta = net(foc, fdc, geo)

                    C = F_orig.shape[1]
                    fp = torch.zeros(C, 2*fs, 2*fs, device=device)
                    hd, wd = delta.shape[1], delta.shape[2]
                    ho = max(0, (2*fs-hd)//2); wo = max(0, (2*fs-wd)//2)
                    he = min(2*fs, ho+hd); we = min(2*fs, wo+wd)
                    fp[:, ho:he, wo:we] = delta[:, :he-ho, :we-wo]
                    f_atk_crop = foc + fdc
                    bp = torch.zeros_like(fp)
                    bp[:, ho:he, wo:we] = (args.beta * f_atk_crop - foc)[:, :he-ho, :we-wo]
                    correction = torch.clamp(fp, -10, 10)
                    combined = bp + correction

                    F_pn = F_orig.clone()
                    F_pn[aidx, :, cy-fs:cy+fs, cx-fs:cx+fs] = torch.clamp(
                        F_orig[aidx, :, cy-fs:cy+fs, cx-fs:cx+fs] + combined, min=0, max=30)
                    pred_pn, scores_pn = atk_obj._run_with_features(bd, F_pn)

            # Compute IoUs
            def get_metrics(pred, scores, bt_e, bo_e):
                if len(pred) == 0:
                    return {'iou_tgt': 0.0, 'iou_orig': 0.0, 'conf': 0.0, 'n_dets': 0, 'pred_bbox': None}
                dt = np.linalg.norm(pred[:, :2] - bt_e[:2], axis=1)
                idx = dt.argmin()
                return {
                    'iou_tgt': iou_fn(pred[idx], bt_e),
                    'iou_orig': iou_fn(pred[idx], bo_e),
                    'conf': float(scores[idx]) if scores is not None and len(scores) > idx else 0.0,
                    'n_dets': len(pred),
                    'pred_bbox': pred[idx].tolist(),
                }

            entry = {
                'case_idx': ci, 'case_id': meta['case_id'],
                'attacker_id': ai, 'victim_id': vi,
                'bbox_orig': bbox_orig.tolist(), 'bbox_tgt': bbox_tgt.tolist(),
                'normal': get_metrics(pred_normal, scores_normal, bt_e, bo_e),
                'raycast': get_metrics(pred_rc, scores_rc, bt_e, bo_e),
                'beta': get_metrics(pred_beta, scores_beta, bt_e, bo_e),
                'pertnet': get_metrics(pred_pn, scores_pn, bt_e, bo_e),
            }
            results.append(entry)

            if (len(results)) % 25 == 0:
                logger.info(f"  [{len(results)}] latest: normal={entry['normal']['iou_tgt']:.3f}, "
                            f"beta={entry['beta']['iou_tgt']:.3f}, pertnet={entry['pertnet']['iou_tgt']:.3f}")

            torch.cuda.empty_cache()
        except Exception as e:
            traceback.print_exc()
            continue

    # Save per-case results
    with open(os.path.join(result_dir, 'per_case_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Aggregate
    n = len(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"{args.model} / {args.dataset}: {n} cases, beta={args.beta}")
    logger.info(f"{'='*60}")

    header = f"{'Method':<15} | {'%Imp':>6} | {'%S(0.5)':>8} | {'%S(0.7)':>8} | {'AvgIoU':>7} | {'AvgConf':>8} | {'%Zero':>6}"
    logger.info(header)
    logger.info("-" * 70)

    for method in ['normal', 'raycast', 'beta', 'pertnet']:
        ious_t = [r[method]['iou_tgt'] for r in results]
        ious_n = [r['normal']['iou_tgt'] for r in results]
        confs = [r[method]['conf'] for r in results]
        zeros = sum(1 for r in results if r[method]['n_dets'] == 0)
        improved = sum(1 for t, nn in zip(ious_t, ious_n) if t > nn)
        strong = sum(1 for t in ious_t if t > 0.5)
        ultra = sum(1 for t in ious_t if t > 0.7)
        avg_iou = np.mean(ious_t)
        avg_conf = np.mean([c for c in confs if c > 0]) if any(c > 0 for c in confs) else 0

        label = {'normal': 'Normal', 'raycast': 'Ray-cast(b=1)', 'beta': f'Beta={args.beta}',
                 'pertnet': 'PertNet'}[method]
        logger.info(f"  {label:<13} | {100*improved/n:5.1f}% | {100*strong/n:7.1f}% | "
                    f"{100*ultra/n:7.1f}% | {avg_iou:6.3f} | {avg_conf:7.3f} | {100*zeros/n:5.1f}%")

    logger.info(f"\nSaved to {result_dir}/per_case_results.pkl")
    logger.info(f"Time: {time.time()-t_start:.0f}s")

    # Save summary
    with open(os.path.join(result_dir, 'summary.txt'), 'w') as f:
        f.write(f"Model: {args.model}\nDataset: {args.dataset}\nBeta: {args.beta}\nCases: {n}\n")
        f.write(f"Checkpoint: {ckpt_path}\n\n")
        f.write(header + "\n" + "-"*70 + "\n")
        for method in ['normal', 'raycast', 'beta', 'pertnet']:
            ious_t = [r[method]['iou_tgt'] for r in results]
            ious_n = [r['normal']['iou_tgt'] for r in results]
            improved = sum(1 for t, nn in zip(ious_t, ious_n) if t > nn)
            strong = sum(1 for t in ious_t if t > 0.5)
            ultra = sum(1 for t in ious_t if t > 0.7)
            zeros = sum(1 for r in results if r[method]['n_dets'] == 0)
            label = {'normal': 'Normal', 'raycast': 'Ray-cast(b=1)', 'beta': f'Beta={args.beta}',
                     'pertnet': 'PertNet'}[method]
            f.write(f"  {label:<13} | {100*improved/n:5.1f}% | {100*strong/n:7.1f}% | "
                    f"{100*ultra/n:7.1f}% | {np.mean(ious_t):6.3f} | {100*zeros/n:5.1f}%\n")
