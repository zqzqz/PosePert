"""
Defense evaluation on V2X-Real scenario attack results.

For each scenario case, loads the attack configuration, runs the full
perception attack (beta + PertNet) on the key attack frame, and computes
defense scores: Global/Local LUCIA, Global/Local MADE.
(No CAD — V2X-Real lacks occupancy/lane data.)

Usage:
  python test/run_scenario_defense_v2xreal.py --gpu 0
"""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=0)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
os.environ['DATASET_NAME'] = 'V2X-Real'

import pickle, copy, numpy as np, torch, time, logging, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.defense.lucia.lucia import LuciaDefender
from mvp.defense.lucia.local_lucia import LocalLuciaDefender
from mvp.defense.made.made_residual_ae import MadeResidualDetector
from mvp.defense.made.local_made import LocalMadeDefender
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BETA = 1.2
HISTORY_FRAMES = 20
ATTACK_FRAMES = 3


def compute_tpr_at_fpr(attack_scores, normal_scores, target_fpr=0.05):
    normal_scores = np.array(normal_scores)
    attack_scores = np.array(attack_scores)
    threshold = np.quantile(normal_scores, 1 - target_fpr)
    tpr = (attack_scores > threshold).mean()
    return tpr, threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_cases', type=int, default=None)
    args = parser.parse_args()

    scenario_dirs = {
        'blackbox': 'results_paper/S4_scenario_v2xreal',
        'noEoT': 'results_paper/S4_scenario_v2xreal_noEoT',
        'whitebox': 'results_paper/S4_scenario_v2xreal_whitebox',
    }

    normal_dir = os.path.join(root, 'results_paper/D_v2xreal')
    scenario_pkl = os.path.join(root, 'data/V2X-Real/test_scenario_attacks.pkl')

    logger.info(f"=== V2X-Real Scenario Defense Eval, beta={BETA} ===")

    import cv2
    from shapely.geometry import Polygon as _Polygon
    def iou_bev(b1, b2):
        bp1 = cv2.boxPoints(((b1[0],b1[1]),(b1[3],b1[4]),b1[6]/np.pi*180))
        bp2 = cv2.boxPoints(((b2[0],b2[1]),(b2[3],b2[4]),b2[6]/np.pi*180))
        p1 = _Polygon(bp1); p2 = _Polygon(bp2)
        if not p1.is_valid or not p2.is_valid: return 0.0
        inter = p1.intersection(p2).area; union = p1.area + p2.area - inter
        return inter / max(union, 1e-6)

    perception = OpencoodPerception(
        fusion_method='intermediate', model_name='pointpillar',
        dataset_name='V2X-Real')
    perception.model.eval()
    device = perception.device
    dataset = OPV2VDataset(root_path=os.path.join(root, 'data/V2X-Real'),
                           mode='test', dataset_name='V2X-Real')
    atk_obj = LidarShiftVoxelwiseAttacker(perception, dataset, beta=BETA)

    # Load PertNet
    from mvp.attack.perturbation_network import PerturbationNetwork
    pertnet_dir = 'models/perturbation_net_paper_pointpillar_V2X-Real'
    pertnet_path = os.path.join(root, pertnet_dir, 'perturbation_net_best.pt')
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

    # Initialize defenses
    lucia_global = LuciaDefender(compression_ratio=32)
    lucia_local = LocalLuciaDefender(perception)
    made_global = MadeResidualDetector(perception,
                                       ae_checkpoint=os.path.join(root, 'models/MADE/residual_ae.pt'),
                                       device=device)
    made_local = LocalMadeDefender(perception)

    with open(scenario_pkl, 'rb') as f:
        scenario_cases = pickle.load(f)
    n_scenario = len(scenario_cases)
    logger.info(f"Loaded {n_scenario} scenario test cases")

    n_cases = args.n_cases or n_scenario

    for attack_type, scenario_dir_name in scenario_dirs.items():
        scenario_dir = os.path.join(root, scenario_dir_name)
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
                with open(case_path, 'rb') as f:
                    scenario_result = pickle.load(f)

                ao = scenario_result['attack_opts']
                attacker_id = ao['attacker_vehicle_id']
                victim_id = ao['victim_vehicle_id']

                target_trajectory = ao.get('target_trajectory')
                if target_trajectory is None:
                    logger.warning(f"Case {ci}: no target_trajectory, skipping")
                    continue

                sc = scenario_cases[ci]
                case_id = sc['case_id']

                custom_meta = {
                    'scenario_id': sc['scenario_id'],
                    'frame_ids': sc['frame_ids'],
                }
                case = dataset.get_case_by_meta(custom_meta, tag='scenario',
                                                use_lidar=True)
                attack_frame_actual = min(
                    HISTORY_FRAMES + ATTACK_FRAMES - 1, len(case) - 1)
                frame = case[attack_frame_actual]

                # V2X-Real: filter to V2V vehicles only
                frame = {v: d for v, d in frame.items()
                         if isinstance(v, int) and v < 0}

                if attacker_id not in frame or victim_id not in frame:
                    logger.warning(f"Case {ci}: vehicle {attacker_id}/{victim_id} not in frame")
                    continue

                atk_pose = frame[attacker_id]['lidar_pose']
                vic_pose = frame[victim_id]['lidar_pose']

                # Get bbox_to_remove (GT target in attacker sensor frame)
                target_oid = sc['target_id']
                obj_ids = frame[attacker_id].get('object_ids', [])
                if target_oid in obj_ids:
                    gt_idx = obj_ids.index(target_oid)
                    bbox_to_remove = frame[attacker_id]['gt_bboxes'][gt_idx].copy()
                else:
                    logger.warning(f"Case {ci}: target {target_oid} not in object_ids, skipping")
                    continue

                # bbox_to_spoof: shifted target (MAP → attacker sensor)
                traj_idx = ATTACK_FRAMES - 1
                if traj_idx >= len(target_trajectory):
                    traj_idx = len(target_trajectory) - 1
                bbox_to_spoof_map = target_trajectory[traj_idx]
                bbox_to_spoof = bbox_map_to_sensor(
                    np.array([bbox_to_spoof_map]), atk_pose,
                    dataset_name='V2X-Real')[0]

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
                record_len = torch.tensor([F_n.shape[0]])

                base = perception.retrieve_base_data(frame, victim_id)
                vids = list(base.keys())
                aidx = vids.index(attacker_id)

                # Target bbox in victim frame for IoU
                bbox_tgt_vic = bbox_map_to_sensor(
                    np.array([bbox_to_spoof_map]), vic_pose,
                    dataset_name='V2X-Real')[0]

                # Attack success
                atk_iou_tgt = 0.0
                atk_conf = 0.0
                if len(pred_a) > 0:
                    d_tgt = np.linalg.norm(pred_a[:, :2] - bbox_tgt_vic[:2], axis=1)
                    best_idx = d_tgt.argmin()
                    atk_iou_tgt = float(iou_bev(pred_a[best_idx], bbox_tgt_vic))
                    atk_conf = float(pred_a_scores[best_idx]) if len(pred_a_scores) > best_idx else 0.0

                entry = {
                    'case_idx': ci, 'case_id': case_id,
                    'attack_type': attack_type,
                    'atk_iou_tgt': atk_iou_tgt, 'atk_conf': atk_conf,
                    'atk_n_dets': len(pred_a),
                    'attacker_id': attacker_id, 'victim_id': victim_id,
                    'attacker_idx': aidx,
                }

                # === Global LUCIA ===
                trust_n = lucia_global.compute_trust(F_n)
                trust_a = lucia_global.compute_trust(F_a)
                entry['lucia_global_trust_normal'] = float(trust_n[aidx])
                entry['lucia_global_trust_attack'] = float(trust_a[aidx])

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
                    anom_a, _ = lucia_local.compute_magnitude_anomaly(
                        F_a, pred_a, ego_index=0)
                    entry['lucia_local_max_anomaly_attack'] = (
                        float(anom_a[:, aidx].max())
                        if anom_a.shape[1] > aidx else 0.0)
                else:
                    entry['lucia_local_min_trust_normal'] = 1.0
                    entry['lucia_local_min_trust_attack'] = 1.0
                    entry['lucia_local_max_l1_normal'] = 0.0
                    entry['lucia_local_max_l1_attack'] = 0.0
                    entry['lucia_local_max_anomaly_attack'] = 0.0

                # === Global MADE ===
                try:
                    scores_n, indices_n = made_global.compute_global_anomaly(
                        F_n, record_len)
                    scores_a, indices_a = made_global.compute_global_anomaly(
                        F_a, record_len)
                    if aidx in indices_n:
                        pos = indices_n.index(aidx)
                        entry['made_global_normal'] = float(scores_n[pos])
                        entry['made_global_attack'] = float(scores_a[pos])
                    else:
                        entry['made_global_normal'] = 0
                        entry['made_global_attack'] = 0
                    scores_n_raw, _ = made_global.compute_global_anomaly_no_ae(
                        F_n, record_len)
                    scores_a_raw, _ = made_global.compute_global_anomaly_no_ae(
                        F_a, record_len)
                    if aidx in indices_n:
                        pos = indices_n.index(aidx)
                        entry['made_raw_normal'] = float(scores_n_raw[pos])
                        entry['made_raw_attack'] = float(scores_a_raw[pos])
                    else:
                        entry['made_raw_normal'] = 0
                        entry['made_raw_attack'] = 0
                except Exception:
                    entry['made_global_normal'] = 0
                    entry['made_global_attack'] = 0
                    entry['made_raw_normal'] = 0
                    entry['made_raw_attack'] = 0

                # === Local MADE ===
                try:
                    frame_atk = copy.deepcopy(frame)
                    frame_atk[victim_id]['pred_bboxes'] = pred_a
                    influence_a, _ = made_local.compute_local_influence(
                        frame_atk, victim_id)
                    atk_influence = influence_a.get(attacker_id, [])
                    entry['made_local_max_influence'] = max(
                        [x[0] for x in atk_influence], default=0)
                    agent_inf, tgt_idx, _ = made_local.detect_for_target(
                        frame_atk, victim_id, bbox_tgt_vic)
                    entry['made_local_target_influence'] = agent_inf.get(
                        attacker_id, 0)
                except Exception:
                    entry['made_local_max_influence'] = 0
                    entry['made_local_target_influence'] = 0

                results.append(entry)
                torch.cuda.empty_cache()

                logger.info(
                    f"  case {ci}: IoU={atk_iou_tgt:.3f} conf={atk_conf:.2f} "
                    f"L1_n={entry['lucia_local_max_l1_normal']:.0f} "
                    f"L1_a={entry['lucia_local_max_l1_attack']:.0f} "
                    f"GLtrust={entry['lucia_global_trust_attack']:.3f}")

                if len(results) % 10 == 0:
                    n = len(results)
                    g_lucia = sum(1 for r in results
                                  if r['lucia_global_trust_attack'] <
                                  r['lucia_global_trust_normal'] - 0.05)
                    l_lucia = sum(1 for r in results
                                  if r.get('lucia_local_max_anomaly_attack', 0) > 0.5)
                    g_made = sum(1 for r in results
                                if r['made_global_attack'] >
                                r['made_global_normal'] * 1.5
                                and r['made_global_normal'] > 0)
                    l_made = sum(1 for r in results
                                if r['made_local_target_influence'] > 0.5)
                    logger.info(f"  [{n}] GLucia={g_lucia} LLucia={l_lucia} "
                                f"GMADE={g_made} LMADE={l_made}")

            except Exception as e:
                logger.warning(f"Case {ci}: {traceback.format_exc()}")
                continue

        # Save results
        out_path = os.path.join(
            root, f'results_paper/scenario_defense_v2xreal_{attack_type}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)

        # Aggregate
        n = len(results)
        if n == 0:
            logger.warning(f"No results for {attack_type}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"V2X-Real Scenario Defense ({attack_type}): "
                     f"beta={BETA}, {n} cases")
        logger.info(f"{'='*60}")

        ious = np.array([r['atk_iou_tgt'] for r in results])
        improved = (ious > 0).mean() * 100
        strong = (ious > 0.5).mean() * 100
        logger.info(f"Attack: Improved={improved:.1f}%, "
                     f"Strong={strong:.1f}%, AvgIoU={ious.mean():.3f}")

        # L-LUCIA
        l1_attack = np.array(
            [r['lucia_local_max_l1_attack'] for r in results])
        l1_normal_atk = np.array(
            [r['lucia_local_max_l1_normal'] for r in results])
        logger.info(f"L-LUCIA L1: normal_mean={l1_normal_atk.mean():.0f}, "
                     f"attack_mean={l1_attack.mean():.0f}")

        # Load normal baselines for TPR@FPR
        normal_defense_path = os.path.join(normal_dir, 'defense_full_results.pkl')
        if os.path.exists(normal_defense_path):
            with open(normal_defense_path, 'rb') as f:
                normal_results = pickle.load(f)

            # L-LUCIA TPR
            l1_normal_all = np.array(
                [r['lucia_local_max_l1_normal'] for r in normal_results
                 if 'lucia_local_max_l1_normal' in r])
            if len(l1_normal_all) > 0:
                tpr_lucia, thresh_lucia = compute_tpr_at_fpr(
                    l1_attack, l1_normal_all, 0.05)
                logger.info(f"L-LUCIA TPR@5%FPR: {tpr_lucia:.3f} "
                             f"(threshold={thresh_lucia:.0f})")

            # Global LUCIA TPR
            g_trust_normal = np.array(
                [r.get('lucia_global_trust_normal', 1.0) for r in normal_results])
            g_trust_attack = np.array(
                [r['lucia_global_trust_attack'] for r in results])
            if len(g_trust_normal) > 0:
                tpr_gl, _ = compute_tpr_at_fpr(
                    -g_trust_attack, -g_trust_normal, 0.05)
                logger.info(f"G-LUCIA TPR@5%FPR: {tpr_gl:.3f}")

            # MADE TPR
            made_normal = np.array(
                [r.get('made_global_normal', 0) for r in normal_results])
            made_attack_vals = np.array(
                [r['made_global_attack'] for r in results])
            if len(made_normal) > 0 and made_normal.max() > 0:
                tpr_made, _ = compute_tpr_at_fpr(
                    made_attack_vals, made_normal, 0.05)
                logger.info(f"G-MADE TPR@5%FPR: {tpr_made:.3f}")
        else:
            logger.warning(f"Normal defense results not found: {normal_defense_path}")

        logger.info(f"Saved to {out_path}")
        logger.info(f"Time: {time.time()-t_start:.0f}s")

    logger.info("\nDone.")
