"""
Scenario attack: PP-Attentive on V2X-Real (v2, long frames).
Uses tag='scenario' for full frame sequences (60 frames, like OPV2V).
History=20, attack=3, predict=20 (same as OPV2V).
Saves 'extra' data with normal baselines for delta metrics.

Usage:
  python results_paper/S4_scenario_v2xreal/run_scenario_test.py --n_cases 100 --gpu 0
  # Whitebox:
  python results_paper/S4_scenario_v2xreal/run_scenario_test.py --n_cases 100 --gpu 0 --attack_type whitebox --save_dir results_paper/S4_scenario_v2xreal_whitebox
  # No uncertainty (noEoT):
  python results_paper/S4_scenario_v2xreal/run_scenario_test.py --n_cases 100 --gpu 0 --no_uncertainty --save_dir results_paper/S4_scenario_v2xreal_noEoT
"""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=0)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
os.environ['DATASET_NAME'] = 'V2X-Real'

import pickle, copy, time, logging, traceback, signal
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '.')
sys.path.insert(0, 'test')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.scenario_shift_movein_attacker import ScenarioShiftMoveinAttacker
from mvp.data.util import bbox_sensor_to_map

SAVE_DIR = 'results_paper/S4_scenario_v2xreal'

HISTORY_FRAMES = 20
ATTACK_FRAMES = 3
PREDICT_FRAMES = 20


def compute_metrics(case, case_update, attack_opts, attacker):
    """Compute scenario attack metrics for one case."""
    victim_id = attack_opts["victim_vehicle_id"]
    victim_target_track_id = attack_opts.get("victim_target_track_id")
    attack_end = attacker.attack_end_frame_id

    metrics = {}

    pert = attack_opts.get("perturbation", np.zeros((3, 2)))
    metrics["pert_norm"] = np.linalg.norm(pert, axis=1).tolist()
    metrics["pert_max"] = float(np.max(np.linalg.norm(pert, axis=1)))

    target_traj = attack_opts.get("target_trajectory")
    real_traj = attack_opts.get("real_target_trajectory", np.zeros((3, 7)))
    if target_traj is not None and len(target_traj) > 0:
        metrics["desired_shift"] = float(np.linalg.norm(
            target_traj[-1, :2] - target_traj[0, :2]))
    if np.any(real_traj != 0):
        metrics["real_shift"] = float(np.linalg.norm(
            real_traj[-1, :2] - real_traj[0, :2]))

    ideal_pred = attack_opts.get("ideal_predicted_trajectories", {})
    real_pred = attack_opts.get("real_predicted_trajectories", {})
    if attack_end in ideal_pred and attack_end in real_pred:
        ip = ideal_pred[attack_end]
        rp = real_pred[attack_end]
        min_len = min(len(ip), len(rp))
        if min_len > 0:
            metrics["pred_ade"] = float(np.mean(
                np.linalg.norm(ip[:min_len, :2] - rp[:min_len, :2], axis=1)))
            metrics["pred_fde"] = float(np.linalg.norm(
                ip[min_len-1, :2] - rp[min_len-1, :2]))

    if victim_target_track_id is not None and case_update[attack_end][victim_id]:
        pred = case_update[attack_end][victim_id].get("predicted_trajectories", {})
        obs = case_update[attack_end][victim_id].get("observed_trajectories", {})
        if victim_target_track_id in pred and victim_id in obs:
            target_pred = pred[victim_target_track_id]
            victim_pos = obs.get(victim_id)
            if victim_pos is not None and len(target_pred) > 0:
                victim_last = victim_pos[-1, :2]
                min_dist = np.min(np.linalg.norm(
                    target_pred[:, :2] - victim_last, axis=1))
                metrics["min_pred_dist_to_victim"] = float(min_dist)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cases', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--pert_bound', type=float, default=0.5)
    parser.add_argument('--opt_iters', type=int, default=5)
    parser.add_argument('--attack_type', type=str, default='blackbox',
                        choices=['blackbox', 'whitebox'])
    parser.add_argument('--no_uncertainty', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    save_dir = args.save_dir or SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    use_uncertainty = not args.no_uncertainty
    variant_name = 'ours' if use_uncertainty else 'no_uncertainty'

    with open('data/V2X-Real/test_scenario_attacks.pkl', 'rb') as f:
        scenario_attacks = pickle.load(f)
    logger.info(f"Loaded {len(scenario_attacks)} scenario test cases")

    logger.info(f"Building pointpillar perception (beta={args.beta}, "
                f"variant={variant_name}, attack_type={args.attack_type})...")

    perception = OpencoodPerception(
        fusion_method='intermediate',
        model_name='pointpillar',
        dataset_name='V2X-Real',
    )
    dataset = OPV2VDataset(
        root_path='data/V2X-Real', mode='test', dataset_name='V2X-Real')

    voxel_attacker = LidarShiftVoxelwiseAttacker(
        perception, dataset, beta=args.beta)

    # Load V2X-Real PertNet
    pertnet_dir = 'models/perturbation_net_paper_pointpillar_V2X-Real'
    pertnet_best = os.path.join(pertnet_dir, 'perturbation_net_best.pt')
    pertnet_path = pertnet_best if os.path.exists(pertnet_best) else None
    if pertnet_path is None:
        pertnet_files = sorted([f for f in os.listdir(pertnet_dir)
                                if f.startswith('perturbation_net_ep')
                                and f.endswith('.pt')],
                               key=lambda x: int(x.split('ep')[1].split('.')[0]))
        if pertnet_files:
            pertnet_path = os.path.join(pertnet_dir, pertnet_files[-1])
    if pertnet_path:
        import torch
        from mvp.attack.perturbation_network import PerturbationNetwork
        ckpt = torch.load(pertnet_path, map_location='cpu')
        pertnet = PerturbationNetwork(
            feature_channels=ckpt['feature_channels'],
            geo_channels=ckpt['geo_channels']).to(perception.device)
        pertnet.load_state_dict(ckpt['model_state'])
        pertnet.eval()
        voxel_attacker.pertnet = pertnet
        voxel_attacker.pertnet_epsilon = 10.0
        logger.info(f"Loaded PertNet from {pertnet_path}")

    attacker = ScenarioShiftMoveinAttacker(
        dataset_name='V2X-Real',
        perception_attacker=voxel_attacker,
        perturbation_type='location',
        optimization_type='sign',
        use_uncertainty=use_uncertainty,
        history_num_frames=HISTORY_FRAMES,
        attack_num_frames=ATTACK_FRAMES,
        predict_num_frames=PREDICT_FRAMES,
        attack_type=args.attack_type,
    )
    attacker.location_bound = args.pert_bound
    attacker.opt_iterations = args.opt_iters

    all_results = []
    n_success = 0
    n_fail = 0

    for idx in range(min(args.n_cases, len(scenario_attacks))):
        attack_case = scenario_attacks[idx]
        case_id = attack_case["case_id"]
        logger.info(f"\n=== Case {idx}: case_id={case_id}, "
                     f"attacker={attack_case['attacker_vehicle_id']}, "
                     f"victim={attack_case['victim_vehicle_id']}, "
                     f"target={attack_case['target_id']} ===")

        attack_opts = {
            "victim_vehicle_id": attack_case["victim_vehicle_id"],
            "attacker_vehicle_id": attack_case["attacker_vehicle_id"],
            "target_id": attack_case["target_id"],
            "gt": False,
        }

        try:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Case {idx} timed out")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)

            t0 = time.time()

            custom_meta = {
                'scenario_id': attack_case['scenario_id'],
                'frame_ids': attack_case['frame_ids'],
            }
            case = dataset.get_case_by_meta(custom_meta, tag='scenario',
                                            use_lidar=True)

            attacker.preprocess(case, attack_opts)

            case_before = copy.deepcopy(case)

            attack_info = attacker.attack(case, attack_opts)
            signal.alarm(0)

            elapsed = time.time() - t0
            opts = attack_info["attack_opts"]
            case_update = attack_info["update"]

            metrics = compute_metrics(case_before, case_update, opts, attacker)
            metrics["elapsed"] = elapsed
            metrics["idx"] = idx
            metrics["case_id"] = case_id

            logger.info(f"  Time: {elapsed:.1f}s, Pert: {metrics['pert_max']:.3f}m")
            if "real_shift" in metrics:
                logger.info(f"  Real shift: {metrics['real_shift']:.3f}m")
            if "pred_ade" in metrics:
                logger.info(f"  ADE: {metrics['pred_ade']:.3f}m, "
                            f"FDE: {metrics['pred_fde']:.3f}m")
            if "min_pred_dist_to_victim" in metrics:
                logger.info(f"  Min dist to victim: "
                            f"{metrics['min_pred_dist_to_victim']:.2f}m")

            # Save extra data for delta metrics (matching OPV2V format)
            victim_id = opts['victim_vehicle_id']
            target_id = attack_case['target_id']
            vtid = opts.get('victim_target_track_id')
            attack_end = attacker.attack_end_frame_id

            extra = {}

            gt_future = []
            for fid in range(attack_end + 1, min(attack_end + 21, len(case_before))):
                fb = case_before[fid]
                if victim_id in fb and target_id in fb[victim_id].get('object_ids', []):
                    oi = fb[victim_id]['object_ids'].index(target_id)
                    gt_bbox = bbox_sensor_to_map(
                        fb[victim_id]['gt_bboxes'][oi], fb[victim_id]['lidar_pose'])
                    gt_future.append(gt_bbox)
                else:
                    gt_future.append(np.zeros(7))
            extra['gt_future_trajectory'] = np.array(gt_future) if gt_future else np.zeros((0, 7))

            if vtid is not None and vtid in case_before[attack_end][victim_id].get('predicted_trajectories', {}):
                extra['normal_predicted_trajectory'] = case_before[attack_end][victim_id]['predicted_trajectories'][vtid]
            else:
                extra['normal_predicted_trajectory'] = None

            if vtid is not None and vtid in case_before[attack_end][victim_id].get('observed_trajectories', {}):
                extra['normal_observed_trajectory'] = case_before[attack_end][victim_id]['observed_trajectories'][vtid]
            else:
                extra['normal_observed_trajectory'] = None

            if victim_id in case_before[attack_end][victim_id].get('observed_trajectories', {}):
                extra['victim_observed_trajectory'] = case_before[attack_end][victim_id]['observed_trajectories'][victim_id]
            else:
                extra['victim_observed_trajectory'] = None

            extra['attack_detections'] = {}
            for fid in range(attacker.attack_start_frame_id, attack_end + 1):
                if case_update[fid][victim_id] and 'detections' in case_update[fid][victim_id]:
                    extra['attack_detections'][fid] = case_update[fid][victim_id]['detections']

            result = {
                "attack_opts": opts,
                "metrics": metrics,
                "case_update": case_update,
                "extra": extra,
            }
            with open(os.path.join(save_dir, f'case_{idx:03d}.pkl'), 'wb') as f:
                pickle.dump(result, f)

            all_results.append(metrics)
            n_success += 1

        except KeyboardInterrupt:
            break
        except Exception as e:
            n_fail += 1
            logger.error(f"Case {idx} failed: {traceback.format_exc()}")
            signal.alarm(0)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Scenario Attack Summary ({variant_name}, {args.attack_type}): "
                f"{n_success} success, {n_fail} fail")
    if all_results:
        shifts = [m.get("real_shift", 0) for m in all_results if "real_shift" in m]
        ades = [m["pred_ade"] for m in all_results if "pred_ade" in m]
        fdes = [m["pred_fde"] for m in all_results if "pred_fde" in m]
        dists = [m["min_pred_dist_to_victim"] for m in all_results
                 if "min_pred_dist_to_victim" in m]
        perts = [m["pert_max"] for m in all_results]
        times = [m["elapsed"] for m in all_results]

        if shifts:
            logger.info(f"Real shift: mean={np.mean(shifts):.3f}m")
        if ades:
            logger.info(f"Prediction ADE: mean={np.mean(ades):.3f}m, "
                        f"FDE: mean={np.mean(fdes):.3f}m")
        if dists:
            d = np.array(dists)
            logger.info(f"Min dist to victim: mean={d.mean():.2f}m, "
                        f"<3m: {(d<3).sum()}/{len(d)}, <5m: {(d<5).sum()}/{len(d)}")
        logger.info(f"Perturbation max: mean={np.mean(perts):.3f}m")
        logger.info(f"Avg time per case: {np.mean(times):.1f}s")

    with open(os.path.join(save_dir, 'summary.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
