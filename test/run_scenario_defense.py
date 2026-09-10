"""
Defense evaluation on scenario attack results for Table 3.

For each scenario case, loads the attack configuration, runs the full
perception attack (beta + PertNet) on the key attack frame, and computes
L-LUCIA and CAD defense scores.

Usage:
  python test/run_scenario_defense.py --model pointpillar --gpu 2
  python test/run_scenario_defense.py --model v2vnet --gpu 3
  python test/run_scenario_defense.py --model cobevt --gpu 3
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
from mvp.defense.lucia.local_lucia import LocalLuciaDefender
from mvp.defense.perception_defender import PerceptionDefender
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_tpr_at_fpr(attack_scores, normal_scores, target_fpr=0.05):
    """Compute TPR at a given FPR threshold.
    Higher score = more anomalous for both L-LUCIA (L1) and CAD (spoof area).
    """
    normal_scores = np.array(normal_scores)
    attack_scores = np.array(attack_scores)
    threshold = np.quantile(normal_scores, 1 - target_fpr)
    tpr = (attack_scores > threshold).mean()
    return tpr, threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['pointpillar', 'v2vnet', 'cobevt'])
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--n_cases', type=int, default=None)
    parser.add_argument('--attack_frame', type=int, default=22,
                        help='Which attack frame index to evaluate (20, 21, or 22). Default: 22 (last)')
    args = parser.parse_args()

    # Model-specific settings
    beta_map = {'pointpillar': 2.0, 'v2vnet': 3.0, 'cobevt': 2.0}
    beta = beta_map[args.model]

    pertnet_paths = {
        'pointpillar': 'models/perturbation_net_paper_pointpillar/perturbation_net_ep35.pt',
        'v2vnet': 'models/perturbation_net_paper_v2vnet/perturbation_net_best.pt',
        'cobevt': 'models/perturbation_net_paper_cobevt/perturbation_net_best.pt',
    }

    scenario_dirs = {
        'pointpillar': {
            'whitebox': 'results_paper/S1_scenario_pp_whitebox',
            'blackbox': 'results_paper/S1_scenario_pp',
        },
        'v2vnet': {
            'whitebox': 'results_paper/S2_scenario_v2vnet_whitebox',
            'blackbox': 'results_paper/S2_scenario_v2vnet',
        },
        'cobevt': {
            'whitebox': 'results_paper/S3_scenario_cobevt_whitebox',
            'blackbox': 'results_paper/S3_scenario_cobevt',
        },
    }

    normal_result_dirs = {
        'pointpillar': 'results_paper/D_pp_attentive',
        'v2vnet': 'results_paper/D_v2vnet',
        'cobevt': 'results_paper/D_cobevt',
    }

    occ_dir = os.path.join(root, 'data/OPV2V/normal')
    scenario_occ_dir = os.path.join(root, 'data/OPV2V/scenario/normal')
    scenario_pkl = os.path.join(root, 'data/OPV2V/test_scenario_attacks.pkl')

    logger.info(f"=== Scenario Defense Eval: {args.model}, beta={beta} ===")

    # Build perception and attacker
    warp_patches = _apply_warp_patches()
    perception = build_perception(args.model)
    perception.model.eval()
    device = perception.device
    dataset = OPV2VDataset(root_path=os.path.join(root, 'data/OPV2V'),
                           mode='test', dataset_name='OPV2V')
    atk_obj = LidarShiftVoxelwiseAttacker(perception, dataset, beta=beta)

    # Load PertNet
    from mvp.attack.perturbation_network import PerturbationNetwork
    pertnet_path = os.path.join(root, pertnet_paths[args.model])
    if os.path.exists(pertnet_path):
        ckpt = torch.load(pertnet_path, map_location='cpu')
        pertnet = PerturbationNetwork(
            feature_channels=ckpt['feature_channels'],
            geo_channels=ckpt['geo_channels']).to(device)
        pertnet.load_state_dict(ckpt['model_state'])
        pertnet.eval()
        atk_obj.pertnet = pertnet
        atk_obj.pertnet_epsilon = 10.0
        logger.info(f"Loaded PertNet from {pertnet_path}")
    else:
        logger.warning(f"PertNet not found at {pertnet_path}, using beta-only attack")

    # Initialize defenses
    lucia_local = LocalLuciaDefender(perception)
    cad = PerceptionDefender()
    cad.thres = 2.7

    # Load scenario test cases
    with open(scenario_pkl, 'rb') as f:
        scenario_cases = pickle.load(f)
    n_scenario = len(scenario_cases)
    logger.info(f"Loaded {n_scenario} scenario test cases")

    # Attack frame index within the scenario window (20=first attack, 22=last attack)
    attack_frame_idx = args.attack_frame
    attack_start = 20  # ScenarioAttacker default: history_num_frames=20

    n_cases = args.n_cases or n_scenario

    # Process both whitebox and blackbox scenario results
    for attack_type in ['whitebox', 'blackbox']:
        scenario_dir = os.path.join(root, scenario_dirs[args.model][attack_type])
        if not os.path.exists(scenario_dir):
            logger.warning(f"Scenario dir not found: {scenario_dir}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {attack_type} scenarios from {scenario_dir}")
        logger.info(f"{'='*60}")

        results = []
        t_start = time.time()

        for ci in range(min(n_cases, n_scenario)):
            case_path = os.path.join(scenario_dir, f'case_{ci:03d}.pkl')
            if not os.path.exists(case_path):
                continue

            try:
                # Load scenario result
                with open(case_path, 'rb') as f:
                    scenario_result = pickle.load(f)

                ao = scenario_result['attack_opts']
                attacker_id = ao['attacker_vehicle_id']
                victim_id = ao['victim_vehicle_id']

                # Get target_trajectory (in MAP coordinates, shape (3, 7))
                target_trajectory = ao['target_trajectory']
                # Index within attack frames: frame 20->0, 21->1, 22->2
                traj_idx = attack_frame_idx - attack_start

                # Load the single frame from the dataset
                # The test scenario's frame_ids[attack_frame_idx] gives the
                # actual frame timestamp to load
                sc = scenario_cases[ci]
                case_id = sc['case_id']
                frame_timestamp = sc['frame_ids'][attack_frame_idx]

                frame = dataset.get_case_by_meta(
                    {'scenario_id': sc['scenario_id'],
                     'frame_id': frame_timestamp},
                    tag='multi_vehicle', use_lidar=True)

                if attacker_id not in frame or victim_id not in frame:
                    logger.warning(f"Case {ci}: vehicle {attacker_id}/{victim_id} "
                                   f"not in frame (ts={frame_timestamp})")
                    continue

                atk_pose = frame[attacker_id]['lidar_pose']
                vic_pose = frame[victim_id]['lidar_pose']

                # bbox_to_remove: GT position of the target object in attacker's
                # sensor frame. Use GT bboxes for the true (un-perturbed) position.
                target_oid = sc['target_id']
                obj_ids = frame[attacker_id].get('object_ids', [])
                if target_oid in obj_ids:
                    gt_idx = obj_ids.index(target_oid)
                    bbox_to_remove = frame[attacker_id]['gt_bboxes'][gt_idx].copy()
                else:
                    # Fallback: use real_target_trajectory from scenario result
                    rtt = ao.get('real_target_trajectory')
                    if rtt is not None and len(rtt) > traj_idx:
                        bbox_to_remove = bbox_map_to_sensor(
                            np.array([rtt[traj_idx]]), atk_pose)[0]
                    else:
                        logger.warning(f"Case {ci}: target {target_oid} not found")
                        continue

                # bbox_to_spoof: the shifted target position (MAP -> attacker sensor)
                bbox_to_spoof_map = target_trajectory[traj_idx]
                bbox_to_spoof = bbox_map_to_sensor(
                    np.array([bbox_to_spoof_map]), atk_pose)[0]

                # Run full attack
                result = atk_obj.run_multi_vehicle(frame, {
                    'attacker_vehicle_id': attacker_id,
                    'victim_vehicle_id': victim_id,
                    'bbox_to_remove': bbox_to_remove,
                    'bbox_to_spoof': bbox_to_spoof,
                })

                F_a = result['spatial_features']
                pred_a = result['pred_bboxes']
                pred_a_scores = result.get('pred_scores', np.array([]))

                # Normal features
                set_seed(42, set_python=False, set_numpy=False, set_torch=True)
                F_n, _ = atk_obj._get_spatial_features(frame, victim_id)

                # Get attacker index
                base = perception.retrieve_base_data(frame, victim_id)
                vids = list(base.keys())
                aidx = vids.index(attacker_id)

                # Target bbox in victim (ego) frame for IoU evaluation
                bbox_tgt_vic = bbox_map_to_sensor(
                    np.array([bbox_to_spoof_map]), vic_pose)[0]

                # Attack success metrics
                from mvp.tools.iou import iou3d
                atk_iou_tgt = 0.0
                atk_conf = 0.0
                if len(pred_a) > 0:
                    d_tgt = np.linalg.norm(pred_a[:, :2] - bbox_tgt_vic[:2], axis=1)
                    best_idx = d_tgt.argmin()
                    atk_iou_tgt = float(iou3d(pred_a[best_idx], bbox_tgt_vic))
                    atk_conf = float(pred_a_scores[best_idx]) if len(pred_a_scores) > best_idx else 0.0

                entry = {
                    'case_idx': ci,
                    'case_id': case_id,
                    'attack_type': attack_type,
                    'atk_iou_tgt': atk_iou_tgt,
                    'atk_conf': atk_conf,
                    'atk_n_dets': len(pred_a),
                    'attacker_id': attacker_id,
                    'victim_id': victim_id,
                    'attacker_idx': aidx,
                }

                # === Local LUCIA ===
                if len(pred_a) > 0:
                    trust_l_n, l1_n = lucia_local.compute_local_trust(
                        F_n, pred_a, ego_index=0)
                    trust_l_a, l1_a = lucia_local.compute_local_trust(
                        F_a, pred_a, ego_index=0)
                    entry['lucia_local_min_trust_normal'] = (
                        float(trust_l_n[:, aidx].min())
                        if trust_l_n.shape[1] > aidx else 1.0)
                    entry['lucia_local_min_trust_attack'] = (
                        float(trust_l_a[:, aidx].min())
                        if trust_l_a.shape[1] > aidx else 1.0)
                    entry['lucia_local_max_l1_normal'] = (
                        float(l1_n[:, aidx].max())
                        if l1_n.shape[1] > aidx else 0.0)
                    entry['lucia_local_max_l1_attack'] = (
                        float(l1_a[:, aidx].max())
                        if l1_a.shape[1] > aidx else 0.0)
                else:
                    entry['lucia_local_min_trust_normal'] = 1.0
                    entry['lucia_local_min_trust_attack'] = 1.0
                    entry['lucia_local_max_l1_normal'] = 0.0
                    entry['lucia_local_max_l1_attack'] = 0.0

                # === CAD ===
                # Load occupancy from per-scenario pre-computed data
                # Map test scenario frame to dataset scenario frame index
                ds_case_meta = dataset.cases['scenario'][case_id]
                ds_frame_idx = ds_case_meta['frame_ids'].index(
                    frame_timestamp)
                occ_path = os.path.join(
                    scenario_occ_dir,
                    f'{case_id:06d}', 'occupancy.pkl')
                if os.path.exists(occ_path):
                    if not hasattr(compute_tpr_at_fpr, '_occ_cache'):
                        compute_tpr_at_fpr._occ_cache = {}
                    if occ_path not in compute_tpr_at_fpr._occ_cache:
                        compute_tpr_at_fpr._occ_cache[occ_path] = (
                            pickle.load(open(occ_path, 'rb')))
                    occ_all = compute_tpr_at_fpr._occ_cache[occ_path]
                    occ_data = occ_all[ds_frame_idx]

                    frame_atk = copy.deepcopy(frame)
                    for vid in frame_atk:
                        if vid in occ_data:
                            frame_atk[vid].update(occ_data[vid])
                    frame_atk[victim_id]['pred_bboxes'] = pred_a

                    try:
                        _, _, metrics_a = cad.run(
                            {9: frame_atk},
                            {'frame_ids': [9],
                             'vehicle_ids': [v for v in frame_atk
                                             if isinstance(v, int)]})
                        spoof_list = metrics_a[9].get(
                            victim_id, {}).get('spoof', [])
                        pred_map = bbox_sensor_to_map(pred_a, vic_pose)
                        tgt_spoof = 0
                        all_spoof = [m[1] for m in spoof_list]
                        for det_i, (_, area, _, _) in enumerate(
                                spoof_list):
                            if det_i < len(pred_map):
                                dist = np.linalg.norm(
                                    pred_map[det_i][:2] -
                                    bbox_to_spoof_map[:2])
                                if dist < 3.0:
                                    tgt_spoof = max(tgt_spoof, area)
                        entry['cad_target_spoof'] = tgt_spoof
                        entry['cad_max_spoof'] = (
                            max(all_spoof) if all_spoof else 0)
                    except Exception:
                        entry['cad_target_spoof'] = -1
                        entry['cad_max_spoof'] = -1
                    entry['cad_detected'] = (
                        entry.get('cad_target_spoof', 0) > cad.thres)
                else:
                    entry['cad_max_spoof'] = -1
                    entry['cad_detected'] = None

                results.append(entry)
                torch.cuda.empty_cache()

                logger.info(
                    f"  case {ci}: IoU={atk_iou_tgt:.3f} conf={atk_conf:.2f} "
                    f"L1_n={entry['lucia_local_max_l1_normal']:.0f} "
                    f"L1_a={entry['lucia_local_max_l1_attack']:.0f} "
                    f"CAD={entry.get('cad_target_spoof', -1):.1f}")

                if len(results) % 10 == 0:
                    n = len(results)
                    l_lucia = sum(1 for r in results
                                  if r['lucia_local_min_trust_attack'] < 0.3)
                    cad_d = sum(1 for r in results
                                if r.get('cad_detected') == True)
                    logger.info(f"  [{n}] LocalLUCIA={l_lucia} CAD={cad_d}")

            except Exception as e:
                logger.warning(f"Case {ci}: {traceback.format_exc()}")
                continue

        # Save results
        out_path = os.path.join(
            root,
            f'results_paper/scenario_defense_{args.model}_{attack_type}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)

        # Aggregate
        n = len(results)
        if n == 0:
            logger.warning(f"No results for {attack_type}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario Defense ({attack_type}): "
                     f"{args.model}, beta={beta}, {n} cases")
        logger.info(f"{'='*60}")

        ious = np.array([r['atk_iou_tgt'] for r in results])
        improved = (ious > 0).mean() * 100
        strong = (ious > 0.5).mean() * 100
        logger.info(f"Attack: Improved={improved:.1f}%, "
                     f"Strong={strong:.1f}%, AvgIoU={ious.mean():.3f}")

        # L-LUCIA scores
        l1_attack = np.array(
            [r['lucia_local_max_l1_attack'] for r in results])
        l1_normal_atk = np.array(
            [r['lucia_local_max_l1_normal'] for r in results])
        logger.info(f"L-LUCIA L1: normal_mean={l1_normal_atk.mean():.0f}, "
                     f"attack_mean={l1_attack.mean():.0f}")

        # CAD scores
        cad_avail = [r for r in results
                     if r.get('cad_detected') is not None]
        cad_det = sum(1 for r in cad_avail if r['cad_detected'])
        if cad_avail:
            cad_scores = np.array([r['cad_target_spoof'] for r in cad_avail
                                   if r['cad_target_spoof'] >= 0])
            logger.info(
                f"CAD: detected={cad_det}/{len(cad_avail)} "
                f"({100*cad_det/len(cad_avail):.1f}%), "
                f"mean_spoof={cad_scores.mean():.2f}")

        # Compute TPR@5%FPR using normal data
        normal_dir = os.path.join(root, normal_result_dirs[args.model])

        # L-LUCIA: use normal L1 values from defense_full_results.pkl
        normal_defense_path = os.path.join(
            normal_dir, 'defense_full_results.pkl')
        if os.path.exists(normal_defense_path):
            with open(normal_defense_path, 'rb') as f:
                normal_results = pickle.load(f)
            l1_normal_all = np.array(
                [r['lucia_local_max_l1_normal'] for r in normal_results])
            tpr_lucia, thresh_lucia = compute_tpr_at_fpr(
                l1_attack, l1_normal_all, 0.05)
            logger.info(f"L-LUCIA TPR@5%FPR: {tpr_lucia:.3f} "
                         f"(threshold={thresh_lucia:.0f})")
        else:
            logger.warning(f"Normal defense results not found: "
                           f"{normal_defense_path}")

        # CAD: use normal spoof areas from cad_normal_results.pkl
        cad_normal_path = os.path.join(normal_dir, 'cad_normal_results.pkl')
        if os.path.exists(cad_normal_path):
            with open(cad_normal_path, 'rb') as f:
                cad_normal = pickle.load(f)
            cad_normal_scores = np.array(
                [r['max_spoof'] for r in cad_normal])
            cad_attack_scores = np.array(
                [r['cad_target_spoof'] for r in results
                 if r.get('cad_target_spoof', -1) >= 0])
            if len(cad_attack_scores) > 0:
                tpr_cad, thresh_cad = compute_tpr_at_fpr(
                    cad_attack_scores, cad_normal_scores, 0.05)
                logger.info(f"CAD TPR@5%FPR: {tpr_cad:.3f} "
                             f"(threshold={thresh_cad:.2f})")
        else:
            logger.warning(f"CAD normal results not found: "
                           f"{cad_normal_path}")

        logger.info(f"Saved to {out_path}")
        logger.info(f"Time: {time.time()-t_start:.0f}s")

    logger.info("\nDone.")
