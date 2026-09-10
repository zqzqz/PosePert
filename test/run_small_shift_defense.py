"""Run defense on small-shift (<0.5m) attacks for Figure 6."""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=0)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

import pickle, copy, numpy as np, torch, time, logging, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.defense.lucia.local_lucia import LocalLuciaDefender
from mvp.defense.perception_defender import PerceptionDefender
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_cases', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    model = 'pointpillar'
    beta = 2.0
    epsilon = 10.0

    cache_dir = os.path.join(root, 'data/OPV2V/attack_cache_paper')
    test_pkl = os.path.join(root, 'data/OPV2V/attack/lidar_shift.pkl')
    occ_dir = os.path.join(root, 'data/OPV2V/normal')
    result_dir = os.path.join(root, 'results_paper/D_pp_attentive')
    os.makedirs(result_dir, exist_ok=True)

    logger.info(f"=== Small-Shift Defense Eval: {model}, beta={beta} ===")

    warp_patches = _apply_warp_patches()
    perception = build_perception(model)
    perception.model.eval()
    device = perception.device
    dataset = OPV2VDataset(root_path=os.path.join(root, 'data/OPV2V'),
                            mode='test', dataset_name='OPV2V')
    atk_obj = LidarShiftVoxelwiseAttacker(perception, dataset, beta=beta)

    # Load PertNet
    from mvp.attack.perturbation_network import PerturbationNetwork
    pertnet_path = os.path.join(root, 'models/perturbation_net_paper_pointpillar/perturbation_net_ep35.pt')
    if os.path.exists(pertnet_path):
        ckpt = torch.load(pertnet_path, map_location='cpu')
        pertnet = PerturbationNetwork(feature_channels=ckpt['feature_channels'],
                                       geo_channels=ckpt['geo_channels']).to(device)
        pertnet.load_state_dict(ckpt['model_state'])
        pertnet.eval()
        atk_obj.pertnet = pertnet
        atk_obj.pertnet_epsilon = epsilon
        logger.info(f"Loaded PertNet from {pertnet_path}")
    else:
        logger.error(f"PertNet not found at {pertnet_path}")
        sys.exit(1)

    # Initialize defenses
    lucia_local = LocalLuciaDefender(perception)
    cad = PerceptionDefender()
    cad.thres = 2.7

    with open(test_pkl, 'rb') as f:
        attacks = pickle.load(f)

    np.random.seed(args.seed)
    n_cases = args.n_cases
    results = []
    t_start = time.time()

    for ci in range(min(n_cases, len(attacks))):
        cache_path = os.path.join(cache_dir, f'{ci:06d}.pkl')
        if not os.path.exists(cache_path):
            continue

        meta = attacks[ci]['attack_meta']
        try:
            cached = pickle.load(open(cache_path, 'rb'))
            case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
            frame = case[9]
            ai, vi = meta['attacker_vehicle_id'], meta['victim_vehicle_id']
            if ai not in frame or vi not in frame:
                continue

            # Create small-shift bbox_tgt: shift bbox_orig by 0.1-0.5m in random direction
            shift_dist = np.random.uniform(0.1, 0.5)
            shift_dir = np.random.uniform(0, 2 * np.pi)
            bbox_tgt = cached['bbox_orig'].copy()
            bbox_tgt[0] += shift_dist * np.cos(shift_dir)
            bbox_tgt[1] += shift_dist * np.sin(shift_dir)

            # Convert to victim frame for metrics
            atk_pose = frame[ai]['lidar_pose']
            vic_pose = frame[vi]['lidar_pose']
            bbox_orig_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(np.array([cached['bbox_orig']]), atk_pose),
                vic_pose)[0]
            bbox_tgt_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(np.array([bbox_tgt]), atk_pose),
                vic_pose)[0]

            # Run full attack with small-shift target
            result = atk_obj.run_multi_vehicle(frame, {
                'attacker_vehicle_id': ai,
                'victim_vehicle_id': vi,
                'bbox_to_remove': cached['bbox_orig'],
                'bbox_to_spoof': bbox_tgt,
            })
            F_a = result['spatial_features']

            # Normal features
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_n, _ = atk_obj._get_spatial_features(frame, vi)

            base = perception.retrieve_base_data(frame, vi)
            vids = list(base.keys())
            aidx = vids.index(ai)

            # Attack predictions
            pred_a = result['pred_bboxes']
            pred_a_scores = result.get('pred_scores', np.array([]))
            frame_atk = copy.deepcopy(frame)
            frame_atk[ai]['lidar'] = cached['spoof_pcd']

            # Attack success
            from mvp.tools.iou import iou3d
            atk_iou_tgt = 0.0
            atk_conf = 0.0
            if len(pred_a) > 0:
                d_tgt = np.linalg.norm(pred_a[:, :2] - bbox_tgt_ego[:2], axis=1)
                best_idx = d_tgt.argmin()
                atk_iou_tgt = float(iou3d(pred_a[best_idx], bbox_tgt_ego))
                atk_conf = float(pred_a_scores[best_idx]) if len(pred_a_scores) > best_idx else 0.0

            entry = {
                'case_idx': ci, 'case_id': meta['case_id'],
                'shift_distance': shift_dist,
                'atk_iou_tgt': atk_iou_tgt,
                'atk_conf': atk_conf,
                'atk_n_dets': len(pred_a),
                'attacker_id': ai, 'victim_id': vi, 'attacker_idx': aidx,
            }

            # === Local LUCIA (PoseGuard) ===
            if len(pred_a) > 0:
                trust_l_n, l1_n = lucia_local.compute_local_trust(F_n, pred_a, ego_index=0)
                trust_l_a, l1_a = lucia_local.compute_local_trust(F_a, pred_a, ego_index=0)
                entry['lucia_local_max_l1_normal'] = float(l1_n[:, aidx].max()) if l1_n.shape[1] > aidx else 0.0
                entry['lucia_local_max_l1_attack'] = float(l1_a[:, aidx].max()) if l1_a.shape[1] > aidx else 0.0
            else:
                entry['lucia_local_max_l1_normal'] = 0.0
                entry['lucia_local_max_l1_attack'] = 0.0

            # === CAD ===
            occ_path = os.path.join(occ_dir, f'{meta["case_id"]:06d}.pkl')
            if os.path.exists(occ_path):
                occ_data = pickle.load(open(occ_path, 'rb'))
                for vid in frame_atk:
                    if vid in occ_data:
                        frame_atk[vid].update(occ_data[vid])

                frame_atk[vi]['pred_bboxes'] = pred_a
                try:
                    _, _, metrics_a = cad.run({9: frame_atk}, {'frame_ids': [9],
                        'vehicle_ids': [v for v in frame_atk if isinstance(v, int)]})
                    spoof_list = metrics_a[9].get(vi, {}).get('spoof', [])
                    bbox_tgt_map = bbox_sensor_to_map(
                        np.array([bbox_tgt]), frame[ai]['lidar_pose'])[0]
                    pred_map = bbox_sensor_to_map(pred_a, frame_atk[vi]['lidar_pose'])
                    tgt_spoof = 0
                    all_spoof = [m[1] for m in spoof_list]
                    for det_i, (_, area, _, _) in enumerate(spoof_list):
                        if det_i < len(pred_map):
                            dist = np.linalg.norm(pred_map[det_i][:2] - bbox_tgt_map[:2])
                            if dist < 3.0:
                                tgt_spoof = max(tgt_spoof, area)
                    entry['cad_target_spoof'] = tgt_spoof
                    entry['cad_max_spoof'] = max(all_spoof) if all_spoof else 0
                except:
                    entry['cad_target_spoof'] = -1
                    entry['cad_max_spoof'] = -1
                entry['cad_detected'] = entry.get('cad_target_spoof', 0) > cad.thres
            else:
                entry['cad_max_spoof'] = -1
                entry['cad_detected'] = None

            results.append(entry)
            torch.cuda.empty_cache()

            logger.info(
                f"  case {ci}: IoU={atk_iou_tgt:.3f} conf={atk_conf:.2f} "
                f"L1_n={entry['lucia_local_max_l1_normal']:.0f} "
                f"L1_a={entry['lucia_local_max_l1_attack']:.0f} "
                f"CAD={entry.get('cad_target_spoof', -1):.1f} "
                f"shift={shift_dist:.3f}m")

            if len(results) % 10 == 0:
                n = len(results)
                cad_avail = [r for r in results if r.get('cad_detected') is not None]
                cad_det = sum(1 for r in cad_avail if r['cad_detected'])
                l1_a_mean = np.mean([r['lucia_local_max_l1_attack'] for r in results])
                l1_n_mean = np.mean([r['lucia_local_max_l1_normal'] for r in results])
                logger.info(f"  [{n}] L1_n_mean={l1_n_mean:.0f} L1_a_mean={l1_a_mean:.0f} "
                            f"CAD_det={cad_det}/{len(cad_avail)}")

        except Exception as e:
            logger.warning(f"Case {ci}: {traceback.format_exc()}")
            continue

    # Save
    out_path = os.path.join(result_dir, 'small_shift_defense.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    # Summary
    n = len(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"Small-Shift Defense Eval: {model}, beta={beta}, {n} cases")
    logger.info(f"{'='*60}")
    shifts = np.array([r['shift_distance'] for r in results])
    ious = np.array([r['atk_iou_tgt'] for r in results])
    l1_n = np.array([r['lucia_local_max_l1_normal'] for r in results])
    l1_a = np.array([r['lucia_local_max_l1_attack'] for r in results])
    cad_avail = [r for r in results if r.get('cad_detected') is not None]
    cad_det = sum(1 for r in cad_avail if r['cad_detected'])
    logger.info(f"Shift: mean={shifts.mean():.3f}m, range=[{shifts.min():.3f}, {shifts.max():.3f}]")
    logger.info(f"Attack IoU: mean={ious.mean():.3f}, >0.5: {(ious>0.5).mean()*100:.1f}%")
    logger.info(f"L1 Normal: mean={l1_n.mean():.0f}, L1 Attack: mean={l1_a.mean():.0f}")
    logger.info(f"CAD detected: {cad_det}/{len(cad_avail)} ({100*cad_det/len(cad_avail):.1f}%)" if cad_avail else "CAD: no data")
    logger.info(f"Saved to {out_path}")
    logger.info(f"Time: {time.time()-t_start:.0f}s")
