"""
Full defense evaluation: CAD, global LUCIA, global MADE, local LUCIA, local MADE.

Usage:
  python test/run_defense_eval_full.py --model pointpillar --beta 2.0 --gpu 2
  python test/run_defense_eval_full.py --model v2vnet --beta 3.0 --gpu 2
  python test/run_defense_eval_full.py --model cobevt --beta 2.0 --gpu 3
"""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=2)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

import pickle, copy, numpy as np, torch, time, logging, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.defense.lucia.lucia import LuciaDefender
from mvp.defense.lucia.local_lucia import LocalLuciaDefender
from mvp.defense.made.made_residual_ae import MadeResidualDetector
from mvp.defense.made.local_made import LocalMadeDefender
from mvp.defense.perception_defender import PerceptionDefender
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['pointpillar', 'v2vnet', 'cobevt'])
    parser.add_argument('--beta', required=True, type=float)
    parser.add_argument('--epsilon', type=float, default=10.0,
                        help='PertNet correction clamp bound (default 10)')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--n_cases', type=int, default=None)
    parser.add_argument('--tag', type=str, default=None,
                        help='Output subdirectory tag (default: auto from beta/epsilon)')
    args = parser.parse_args()

    cache_dir = os.path.join(root, 'data/OPV2V/attack_cache_paper')
    test_pkl = os.path.join(root, 'data/OPV2V/attack/lidar_shift.pkl')
    occ_dir = os.path.join(root, 'data/OPV2V/normal')
    base_dir = {
        'pointpillar': 'results_paper/D_pp_attentive',
        'v2vnet': 'results_paper/D_v2vnet',
        'cobevt': 'results_paper/D_cobevt',
    }[args.model]
    if args.tag:
        result_dir = f'{base_dir}_{args.tag}'
    elif args.epsilon != 10.0 or args.beta != {'pointpillar': 2.0, 'v2vnet': 3.0, 'cobevt': 2.0}[args.model]:
        result_dir = f'{base_dir}_b{args.beta}_e{args.epsilon}'
    else:
        result_dir = base_dir
    os.makedirs(result_dir, exist_ok=True)

    logger.info(f"=== Full Defense Eval: {args.model}, beta={args.beta}, epsilon={args.epsilon} ===")

    warp_patches = _apply_warp_patches()
    perception = build_perception(args.model)
    perception.model.eval()
    device = perception.device
    dataset = OPV2VDataset(root_path=os.path.join(root, 'data/OPV2V'),
                            mode='test', dataset_name='OPV2V')
    atk_obj = LidarShiftVoxelwiseAttacker(perception, dataset, beta=args.beta)

    # Load PertNet for full attack features
    from mvp.attack.perturbation_network import PerturbationNetwork, build_geometric_encoding, get_active_zone_bounds
    pertnet = None
    pertnet_dir = os.path.join(root, f'models/perturbation_net_paper_{args.model}')
    if args.model == 'pointpillar':
        pertnet_path = os.path.join(pertnet_dir, 'perturbation_net_ep35.pt')
    else:
        pertnet_path = os.path.join(pertnet_dir, 'perturbation_net_best.pt')
    if not os.path.exists(pertnet_path):
        # Without PertNet this silently degrades to beta-scaling only, which is a
        # different (weaker) attack than the paper reports. Fail loudly instead.
        raise FileNotFoundError(
            f"PertNet checkpoint not found: {pertnet_path}\n"
            "Run 'python scripts/check_artifact.py --group T2' to check the layout.")
    if os.path.exists(pertnet_path):
        import torch as _torch
        ckpt = _torch.load(pertnet_path, map_location='cpu')
        pertnet = PerturbationNetwork(feature_channels=ckpt['feature_channels'],
                                       geo_channels=ckpt['geo_channels']).to(device)
        pertnet.load_state_dict(ckpt['model_state'])
        pertnet.eval()
        atk_obj.pertnet = pertnet
        atk_obj.pertnet_epsilon = args.epsilon
        logger.info(f"Loaded PertNet from {pertnet_path}, epsilon={args.epsilon}")
    lr = np.array(perception.dataset.pre_processor.params['cav_lidar_range'])
    vs = np.array(perception.dataset.pre_processor.params['args']['voxel_size'])

    # Initialize all defenses
    lucia_global = LuciaDefender(compression_ratio=32)
    lucia_local = LocalLuciaDefender(perception)
    made_global = MadeResidualDetector(perception,
                                       ae_checkpoint=os.path.join(root, 'models/MADE/residual_ae.pt'),
                                       device=device)
    made_local = LocalMadeDefender(perception)
    cad = PerceptionDefender()
    cad.thres = 2.7  # calibrated for 5% FPR on normal data


    with open(test_pkl, 'rb') as f:
        attacks = pickle.load(f)

    n_cases = args.n_cases or len(attacks)
    results = []
    t_start = time.time()

    for ci in range(min(n_cases, len(attacks))):
        cache_path = os.path.join(cache_dir, f'{ci:06d}.pkl')
        if not os.path.exists(cache_path):
            continue

        meta = attacks[ci]['attack_meta']
        ao = attacks[ci]['attack_opts']
        try:
            cached = pickle.load(open(cache_path, 'rb'))
            case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
            frame = case[9]
            ai, vi = meta['attacker_vehicle_id'], meta['victim_vehicle_id']
            if ai not in frame or vi not in frame:
                continue

            # Get target bbox in ego (victim) frame
            # cached bboxes are in attacker sensor frame → convert to victim frame
            atk_pose = frame[ai]['lidar_pose']
            vic_pose = frame[vi]['lidar_pose']
            bbox_orig_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(np.array([cached['bbox_orig']]), atk_pose),
                vic_pose)[0]
            bbox_tgt_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(np.array([cached['bbox_tgt']]), atk_pose),
                vic_pose)[0]

            # Run full attack via run_multi_vehicle (applies beta + PertNet)
            result = atk_obj.run_multi_vehicle(frame, {
                'attacker_vehicle_id': ai,
                'victim_vehicle_id': vi,
                'bbox_to_remove': cached['bbox_orig'],
                'bbox_to_spoof': cached['bbox_tgt'],
            })
            # F_a = attacked spatial features (with beta scaling + PertNet)
            F_a = result['spatial_features']

            # Normal features
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            F_n, _ = atk_obj._get_spatial_features(frame, vi)
            record_len = torch.tensor([F_n.shape[0]])

            base = perception.retrieve_base_data(frame, vi)
            vids = list(base.keys())
            aidx = vids.index(ai)

            # Attack perception output (for local defense bbox regions)
            pred_a = result['pred_bboxes']
            frame_atk = copy.deepcopy(frame)
            frame_atk[ai]['lidar'] = cached['spoof_pcd']

            # === Attack success metrics ===
            from mvp.tools.iou import iou3d
            pred_a_scores = result.get('pred_scores', np.array([]))
            atk_iou_tgt = 0.0
            atk_conf = 0.0
            if len(pred_a) > 0:
                d_tgt = np.linalg.norm(pred_a[:, :2] - bbox_tgt_ego[:2], axis=1)
                best_idx = d_tgt.argmin()
                atk_iou_tgt = float(iou3d(pred_a[best_idx], bbox_tgt_ego))
                atk_conf = float(pred_a_scores[best_idx]) if len(pred_a_scores) > best_idx else 0.0

            entry = {
                'case_idx': ci, 'case_id': meta['case_id'],
                'atk_iou_tgt': atk_iou_tgt,
                'atk_conf': atk_conf,
                'atk_n_dets': len(pred_a),
                'attacker_id': ai, 'victim_id': vi, 'attacker_idx': aidx,
            }

            # === Global LUCIA ===
            trust_n = lucia_global.compute_trust(F_n)
            trust_a = lucia_global.compute_trust(F_a)
            entry['lucia_global_trust_normal'] = float(trust_n[aidx])
            entry['lucia_global_trust_attack'] = float(trust_a[aidx])

            # === Local LUCIA ===
            if len(pred_a) > 0:
                trust_l_n, l1_n = lucia_local.compute_local_trust(F_n, pred_a, ego_index=0)
                trust_l_a, l1_a = lucia_local.compute_local_trust(F_a, pred_a, ego_index=0)
                # Min trust for attacker across all objects
                entry['lucia_local_min_trust_normal'] = float(trust_l_n[:, aidx].min()) if trust_l_n.shape[1] > aidx else 1.0
                entry['lucia_local_min_trust_attack'] = float(trust_l_a[:, aidx].min()) if trust_l_a.shape[1] > aidx else 1.0
                # Max L1 for attacker
                entry['lucia_local_max_l1_normal'] = float(l1_n[:, aidx].max()) if l1_n.shape[1] > aidx else 0.0
                entry['lucia_local_max_l1_attack'] = float(l1_a[:, aidx].max()) if l1_a.shape[1] > aidx else 0.0

                # Magnitude anomaly
                anom_a, _ = lucia_local.compute_magnitude_anomaly(F_a, pred_a, ego_index=0)
                entry['lucia_local_max_anomaly_attack'] = float(anom_a[:, aidx].max()) if anom_a.shape[1] > aidx else 0.0
            else:
                entry['lucia_local_min_trust_normal'] = 1.0
                entry['lucia_local_min_trust_attack'] = 1.0
                entry['lucia_local_max_l1_normal'] = 0.0
                entry['lucia_local_max_l1_attack'] = 0.0
                entry['lucia_local_max_anomaly_attack'] = 0.0

            # === Global MADE ===
            # scores format: (scores_list, agent_indices)
            # scores_list[k] corresponds to agent_indices[k]
            # Need to find the score for aidx by matching agent_indices
            try:
                scores_n, indices_n = made_global.compute_global_anomaly(F_n, record_len)
                scores_a, indices_a = made_global.compute_global_anomaly(F_a, record_len)
                if aidx in indices_n:
                    pos = indices_n.index(aidx)
                    entry['made_global_normal'] = float(scores_n[pos])
                    entry['made_global_attack'] = float(scores_a[pos])
                else:
                    entry['made_global_normal'] = 0
                    entry['made_global_attack'] = 0

                # Also raw residual norm (no AE)
                scores_n_raw, _ = made_global.compute_global_anomaly_no_ae(F_n, record_len)
                scores_a_raw, _ = made_global.compute_global_anomaly_no_ae(F_a, record_len)
                if aidx in indices_n:
                    pos = indices_n.index(aidx)
                    entry['made_raw_normal'] = float(scores_n_raw[pos])
                    entry['made_raw_attack'] = float(scores_a_raw[pos])
                else:
                    entry['made_raw_normal'] = 0
                    entry['made_raw_attack'] = 0
            except Exception as e:
                entry['made_global_normal'] = 0
                entry['made_global_attack'] = 0
                entry['made_raw_normal'] = 0
                entry['made_raw_attack'] = 0

            # === Local MADE ===
            try:
                influence_a, _ = made_local.compute_local_influence(frame_atk, vi)
                # Max influence of attacker
                atk_influence = influence_a.get(ai, [])
                entry['made_local_max_influence'] = max([x[0] for x in atk_influence], default=0)
                # Influence specifically at the target detection
                agent_inf, tgt_idx, _ = made_local.detect_for_target(
                    frame_atk, vi, bbox_tgt_ego)
                entry['made_local_target_influence'] = agent_inf.get(ai, 0)
            except Exception as e:
                entry['made_local_max_influence'] = 0
                entry['made_local_target_influence'] = 0

            # === CAD ===
            occ_path = os.path.join(occ_dir, f'{meta["case_id"]:06d}.pkl')
            if os.path.exists(occ_path):
                occ_data = pickle.load(open(occ_path, 'rb'))
                # Use NORMAL occupancy maps for all vehicles (including attacker)
                # The defense sees the real-world occupancy, not the spoofed one
                for vid in frame_atk:
                    if vid in occ_data:
                        frame_atk[vid].update(occ_data[vid])

                frame_atk[vi]['pred_bboxes'] = pred_a
                try:
                    _, _, metrics_a = cad.run({9: frame_atk}, {'frame_ids': [9],
                        'vehicle_ids': [v for v in frame_atk if isinstance(v, int)]})
                    spoof_list = metrics_a[9].get(vi, {}).get('spoof', [])
                    # Find the detection closest to the target bbox
                    # CAD works in map frame; convert target bbox to map frame
                    bbox_tgt_map = bbox_sensor_to_map(
                        np.array([cached['bbox_tgt']]),
                        frame[ai]['lidar_pose'])[0]
                    pred_map = bbox_sensor_to_map(pred_a, frame_atk[vi]['lidar_pose'])
                    tgt_spoof = 0
                    all_spoof = [m[1] for m in spoof_list]
                    for det_i, (_, area, _, _) in enumerate(spoof_list):
                        if det_i < len(pred_map):
                            dist = np.linalg.norm(pred_map[det_i][:2] - bbox_tgt_map[:2])
                            if dist < 3.0:  # within 3m of target
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

            # Per-case log
            logger.info(
                f"  case {ci}: IoU={atk_iou_tgt:.3f} conf={atk_conf:.2f} "
                f"L1_n={entry['lucia_local_max_l1_normal']:.0f} "
                f"L1_a={entry['lucia_local_max_l1_attack']:.0f} "
                f"CAD={entry.get('cad_target_spoof', -1):.1f} "
                f"GLtrust={entry['lucia_global_trust_attack']:.3f}")

            if len(results) % 10 == 0:
                n = len(results)
                g_lucia = sum(1 for r in results if r['lucia_global_trust_attack'] < r['lucia_global_trust_normal'] - 0.05)
                l_lucia = sum(1 for r in results if r['lucia_local_min_trust_attack'] < 0.3)
                g_made = sum(1 for r in results if r['made_global_attack'] > r['made_global_normal'] * 1.5 and r['made_global_normal'] > 0)
                l_made = sum(1 for r in results if r['made_local_target_influence'] > 0.5)
                cad_d = sum(1 for r in results if r.get('cad_detected') == True)
                logger.info(f"  [{n}] GlobalLUCIA={g_lucia} LocalLUCIA={l_lucia} "
                            f"GlobalMADE={g_made} LocalMADE={l_made} CAD={cad_d}")

        except Exception as e:
            logger.warning(f"Case {ci}: {traceback.format_exc()}")
            continue

    # Save results
    with open(os.path.join(result_dir, 'defense_full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Aggregate
    n = len(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"Full Defense Eval: {args.model}, beta={args.beta}, eps={args.epsilon}, {n} cases")
    logger.info(f"{'='*60}")

    # Attack success
    ious = np.array([r['atk_iou_tgt'] for r in results])
    confs = np.array([r['atk_conf'] for r in results])
    improved = (ious > 0).mean() * 100
    strong = (ious > 0.5).mean() * 100
    ultra = (ious > 0.7).mean() * 100
    logger.info(f"Attack: Improved={improved:.1f}%, Strong={strong:.1f}%, Ultra={ultra:.1f}%, AvgIoU={ious.mean():.3f}, AvgConf={confs.mean():.3f}")

    g_lucia = sum(1 for r in results if r['lucia_global_trust_attack'] < r['lucia_global_trust_normal'] - 0.05)
    l_lucia = sum(1 for r in results if r['lucia_local_min_trust_attack'] < 0.3)
    g_made = sum(1 for r in results if r['made_global_attack'] > r['made_global_normal'] * 1.5 and r['made_global_normal'] > 0)
    g_made_raw = sum(1 for r in results if r.get('made_raw_attack', 0) > r.get('made_raw_normal', 0) * 1.5 and r.get('made_raw_normal', 0) > 0)
    l_made = sum(1 for r in results if r['made_local_target_influence'] > 0.5)
    cad_avail = [r for r in results if r.get('cad_detected') is not None]
    cad_det = sum(1 for r in cad_avail if r['cad_detected'])

    logger.info(f"Global LUCIA (trust drop > 0.05): {g_lucia}/{n} ({100*g_lucia/n:.1f}%)")
    logger.info(f"Local LUCIA (min trust < 0.3):    {l_lucia}/{n} ({100*l_lucia/n:.1f}%)")
    logger.info(f"Global MADE AE (50% increase):    {g_made}/{n} ({100*g_made/n:.1f}%)")
    logger.info(f"Global MADE Raw (50% increase):   {g_made_raw}/{n} ({100*g_made_raw/n:.1f}%)")
    logger.info(f"Local MADE (influence > 0.5):     {l_made}/{n} ({100*l_made/n:.1f}%)")
    logger.info(f"CAD (spoof > {cad.thres}):              {cad_det}/{len(cad_avail)} ({100*cad_det/len(cad_avail):.1f}%)" if cad_avail else "CAD: no data")

    logger.info(f"\nSaved to {result_dir}/defense_full_results.pkl")
    logger.info(f"Time: {time.time()-t_start:.0f}s")

    with open(os.path.join(result_dir, 'defense_full_summary.txt'), 'w') as f:
        f.write(f"Model: {args.model}, Beta: {args.beta}, Cases: {n}\n\n")
        f.write(f"Global LUCIA detected: {g_lucia}/{n} ({100*g_lucia/n:.1f}%)\n")
        f.write(f"Local LUCIA detected:  {l_lucia}/{n} ({100*l_lucia/n:.1f}%)\n")
        f.write(f"Global MADE detected:  {g_made}/{n} ({100*g_made/n:.1f}%)\n")
        f.write(f"Local MADE detected:   {l_made}/{n} ({100*l_made/n:.1f}%)\n")
        f.write(f"CAD detected:          {cad_det}/{len(cad_avail)} ({100*cad_det/len(cad_avail):.1f}%)\n" if cad_avail else "CAD: no data\n")
