"""
Scenario attack on all OPV2V perception models (PP-Attentive, V2VNet, CoBEVT).
Uses precomputed detection/tracking/prediction features (model-independent).

Usage:
  python results_paper/run_scenario_all.py --model pointpillar --gpu 1 --n_cases 36
  python results_paper/run_scenario_all.py --model v2vnet --gpu 2 --n_cases 36
  python results_paper/run_scenario_all.py --model cobevt --gpu 3 --n_cases 36
"""
import os, sys, argparse

# Parse GPU arg and set CUDA_VISIBLE_DEVICES BEFORE any torch imports
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=1)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

import pickle, copy, time, logging, traceback, signal
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, 'test')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.scenario_shift_movein_attacker import ScenarioShiftMoveinAttacker
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.data.util import bbox_sensor_to_map
from mvp.visualize.general import draw_trajectories

# Model configs: (model_name, beta)
MODEL_CONFIGS = {
    'pointpillar': {'beta': 2.0, 'tag': 'S1_scenario_pp'},
    'v2vnet':      {'beta': 3.0, 'tag': 'S2_scenario_v2vnet'},
    'cobevt':      {'beta': 2.0, 'tag': 'S3_scenario_cobevt'},
}


def visualize_scenario(case, case_update, attack_opts, attacker, idx, save_dir):
    """BEV trajectory visualization for a scenario attack."""
    victim_id = attack_opts["victim_vehicle_id"]
    attacker_vehicle_id = attack_opts["attacker_vehicle_id"]
    target_id = attack_opts.get("target_id")
    victim_target_track_id = attack_opts.get("victim_target_track_id")

    total_frames = attacker.total_num_frames
    attack_end = attacker.attack_end_frame_id

    gt_traj = attacker.get_gt_trajectories(
        case, attacker_vehicle_id, frame_ids=list(range(total_frames)))

    fig, ax = plt.subplots(figsize=(16, 16))

    for obj_id, traj in gt_traj.items():
        valid = np.any(traj[:, 3:6] > 0, axis=1)
        if valid.sum() < 2:
            continue
        t = traj[valid]
        if obj_id == victim_id:
            ax.plot(t[:, 0], t[:, 1], 'g-', linewidth=2, alpha=0.4, label='Victim GT')
            ax.plot(t[0, 0], t[0, 1], 'g^', markersize=10)
            ax.plot(t[-1, 0], t[-1, 1], 'gs', markersize=10)
        elif obj_id == attacker_vehicle_id:
            ax.plot(t[:, 0], t[:, 1], 'r-', linewidth=2, alpha=0.4, label='Attacker GT')
        elif obj_id == target_id:
            ax.plot(t[:, 0], t[:, 1], 'b-', linewidth=2, alpha=0.4, label='Target GT')
            ax.plot(t[0, 0], t[0, 1], 'b^', markersize=10)
            ax.plot(t[-1, 0], t[-1, 1], 'bs', markersize=10)
        else:
            ax.plot(t[:, 0], t[:, 1], 'k-', linewidth=0.5, alpha=0.2)

    target_traj = attack_opts.get("target_trajectory")
    if target_traj is not None and len(target_traj) > 0:
        ax.plot(target_traj[:, 0], target_traj[:, 1], 'b--', linewidth=3,
                alpha=0.8, label='Manipulated target')
        ax.plot(target_traj[-1, 0], target_traj[-1, 1], 'b*', markersize=15)

    if victim_target_track_id is not None:
        try:
            orig_pred = case[attack_end][victim_id]["predicted_trajectories"][victim_target_track_id]
            if len(orig_pred) > 0:
                ax.plot(orig_pred[:, 0], orig_pred[:, 1], 'y-', linewidth=2,
                        alpha=0.4, label='Original prediction')
        except (KeyError, IndexError):
            pass

    if victim_target_track_id is not None and case_update[attack_end][victim_id]:
        try:
            attack_pred = case_update[attack_end][victim_id]["predicted_trajectories"][victim_target_track_id]
            if len(attack_pred) > 0:
                ax.plot(attack_pred[:, 0], attack_pred[:, 1], 'm-', linewidth=3,
                        alpha=0.8, label='Attack prediction')
                ax.plot(attack_pred[-1, 0], attack_pred[-1, 1], 'm*', markersize=12)
        except (KeyError, IndexError):
            pass

    ideal_pred = attack_opts.get("ideal_predicted_trajectories", {})
    if attack_end in ideal_pred:
        ip = ideal_pred[attack_end]
        if len(ip) > 0:
            ax.plot(ip[:, 0], ip[:, 1], 'y--', linewidth=2,
                    alpha=0.7, label='Ideal prediction')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    dir_name = os.path.basename(save_dir)
    ax.set_title(f'Scenario Attack Case {idx} ({dir_name}, OPV2V)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, f'case_{idx:03d}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_metrics(case, case_update, attack_opts, attacker):
    """Compute scenario attack metrics for one case."""
    victim_id = attack_opts["victim_vehicle_id"]
    victim_target_track_id = attack_opts.get("victim_target_track_id")
    attack_end = attacker.attack_end_frame_id

    metrics = {}

    pert = attack_opts.get("perturbation", np.zeros((3, 2)))
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
    parser.add_argument('--model', type=str, required=True,
                        choices=['pointpillar', 'v2vnet', 'cobevt'])
    parser.add_argument('--n_cases', type=int, default=36)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--pert_bound', type=float, default=0.5)
    parser.add_argument('--opt_iters', type=int, default=5)
    parser.add_argument('--variant', type=str, default='ours',
                        choices=['ours', 'late', 'no_uncertainty', 'no_online_update'],
                        help='ours=voxelwise intermediate, late=late-fusion baseline, '
                             'no_uncertainty=ours without EoT, no_online_update=optimize once only')
    parser.add_argument('--prediction', type=str, default='grip',
                        choices=['grip', 'trajectron'],
                        help='Prediction model for scenario attack')
    parser.add_argument('--attack_type', type=str, default='blackbox',
                        choices=['blackbox', 'whitebox'],
                        help='Optimization type: blackbox (finite diff) or whitebox (gradient)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip cases that already have result files')
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    variant_suffix = {'ours': '', 'late': '_late', 'no_uncertainty': '_noEoT', 'no_online_update': '_noUpdate'}
    pred_suffix = '' if args.prediction == 'grip' else f'_{args.prediction}'
    atk_suffix = '' if args.attack_type == 'blackbox' else f'_{args.attack_type}'
    save_dir = f'results_paper/{cfg["tag"]}{variant_suffix[args.variant]}{pred_suffix}{atk_suffix}'
    os.makedirs(save_dir, exist_ok=True)

    # Load test cases
    with open('data/OPV2V/test_scenario_attacks.pkl', 'rb') as f:
        scenario_attacks = pickle.load(f)
    logger.info(f"Loaded {len(scenario_attacks)} scenario test cases")

    # Build perception model
    logger.info(f"Building {args.model} perception (beta={cfg['beta']}, variant={args.variant})...")
    warp_patches = _apply_warp_patches()

    dataset = OPV2VDataset(
        root_path='data/OPV2V', mode='test', dataset_name='OPV2V')

    # Build the perception attacker based on variant
    if args.variant == 'late':
        # For late-fusion, pass perception_attacker=None so ScenarioAttacker
        # creates its own LidarShiftLateAttacker with the built-in late-fusion
        # perception model (not intermediate). This is critical because
        # attack_late() requires late-fusion preprocessing.
        perception_attacker = None
    else:
        perception = build_perception(args.model)
        perception.model.eval()
        for p in perception.model.parameters():
            p.requires_grad = False
        perception_attacker = LidarShiftVoxelwiseAttacker(
            perception, dataset, beta=cfg['beta'])

        # Load PertNet checkpoint
        import torch
        from mvp.attack.perturbation_network import PerturbationNetwork
        pertnet_dir = f'models/perturbation_net_paper_{args.model}'
        # Use best checkpoint, or ep35 for pointpillar
        if args.model == 'pointpillar':
            pertnet_path = os.path.join(pertnet_dir, 'perturbation_net_ep35.pt')
        else:
            pertnet_path = os.path.join(pertnet_dir, 'perturbation_net_best.pt')
        if os.path.exists(pertnet_path):
            ckpt = torch.load(pertnet_path, map_location='cpu')
            pertnet = PerturbationNetwork(
                feature_channels=ckpt['feature_channels'],
                geo_channels=ckpt['geo_channels']).to(perception.device)
            pertnet.load_state_dict(ckpt['model_state'])
            pertnet.eval()
            perception_attacker.pertnet = pertnet
            logger.info(f"Loaded PertNet from {pertnet_path} (epoch {ckpt.get('epoch')})")
        else:
            logger.warning(f"No PertNet at {pertnet_path}, using beta-only")

    use_uncertainty = (args.variant not in ('no_uncertainty', 'no_online_update'))
    no_online_update = (args.variant == 'no_online_update')

    # Select prediction function
    if args.prediction == 'trajectron':
        from mvp.attack.scenario_attacker_util import prediction_trajectron
        prediction_func = prediction_trajectron
    else:
        from mvp.attack.scenario_attacker_util import prediction_grip
        prediction_func = prediction_grip

    attacker = ScenarioShiftMoveinAttacker(
        dataset_name='OPV2V',
        perception_attacker=perception_attacker,
        prediction_func=prediction_func,
        perturbation_type='location',
        optimization_type='sign',
        use_uncertainty=use_uncertainty,
        no_online_update=no_online_update,
        attack_type=args.attack_type,
    )
    attacker.location_bound = args.pert_bound
    attacker.opt_iterations = args.opt_iters

    # Override prediction model API for non-GRIP models
    if args.prediction == 'trajectron':
        from mvp.config import model_root
        import importlib
        # Trajectron needs its own path context for internal imports
        traj_src = os.path.join(os.path.dirname(__file__), '..', 'third_party',
                                'AdvTrajectoryPrediction', 'prediction', 'model',
                                'Trajectron', 'Trajectron-plus-plus', 'trajectron')
        traj_exp = os.path.join(os.path.dirname(__file__), '..', 'third_party',
                                'AdvTrajectoryPrediction', 'prediction', 'model',
                                'Trajectron', 'Trajectron-plus-plus', 'experiments', 'nuScenes')
        for p in [traj_src, traj_exp]:
            if p not in sys.path:
                sys.path.insert(0, p)
        # Remove conflicting 'model' module if present
        if 'model' in sys.modules and not hasattr(sys.modules['model'], 'model_registrar'):
            del sys.modules['model']
        from prediction.model.Trajectron.interface import TrajectronInterface
        from environment import Environment, Scene
        # OPV2V model was trained with VEHICLE only — create matching environment
        traj_api = TrajectronInterface.__new__(TrajectronInterface)
        traj_api.obs_length = 20
        traj_api.pred_length = 20
        traj_api.seq_length = 40
        traj_api.time_step = 0.1
        traj_api.smooth = 0
        traj_api.dataset = None
        traj_api.test_vars = []
        traj_api.dev = 'cuda:0'
        traj_api.standardization = {
            'VEHICLE': {
                'position': {'x': {'mean': 0, 'std': 80}, 'y': {'mean': 0, 'std': 80}},
                'velocity': {'x': {'mean': 0, 'std': 15}, 'y': {'mean': 0, 'std': 15}, 'norm': {'mean': 0, 'std': 15}},
                'acceleration': {'x': {'mean': 0, 'std': 4}, 'y': {'mean': 0, 'std': 4}, 'norm': {'mean': 0, 'std': 4}},
                'heading': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}, '°': {'mean': 0, 'std': np.pi}, 'd°': {'mean': 0, 'std': 1}}
            }
        }
        env = Environment(node_type_list=['VEHICLE'], standardization=traj_api.standardization)
        env.attention_radius = {(env.NodeType.VEHICLE, env.NodeType.VEHICLE): 30.0}
        env.robot_type = env.NodeType.VEHICLE
        env.scenes = [Scene(timesteps=40, dt=0.1, name="", aug_func=None)]
        traj_api.env = env
        traj_api.model, traj_api.hyperparams = traj_api.load_model(
            os.path.join(model_root, 'Trajectron/OPV2V'))
        from prediction.model.Trajectron.dataloader import TrajectronDataLoader
        traj_api.dataloader = TrajectronDataLoader(20, 20)
        attacker.prediction_model_api = traj_api
        logger.info("Using Trajectron++ prediction model")

    all_results = []
    n_success = 0
    n_fail = 0

    for idx in range(min(args.n_cases, len(scenario_attacks))):
        if args.skip_existing and os.path.exists(os.path.join(save_dir, f'case_{idx:03d}.pkl')):
            try:
                r = pickle.load(open(os.path.join(save_dir, f'case_{idx:03d}.pkl'), 'rb'))
                all_results.append(r['metrics'])
                n_success += 1
                logger.info(f"Skipping case {idx} (already exists)")
                continue
            except Exception:
                pass
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
            signal.alarm(1800 if args.prediction == 'trajectron' else 600)

            t0 = time.time()

            case = dataset.get_case(case_id, tag="scenario", use_lidar=True)

            # Load precomputed features (same for all models)
            normal_dir = f"data/OPV2V/scenario/normal/{case_id:06d}"
            for feat_name in ["pointpillar_intermediate", "ab3dmot", "grip"]:
                feat_path = os.path.join(normal_dir, f"{feat_name}.pkl")
                if os.path.exists(feat_path):
                    case = dataset.load_feature(
                        case, pickle.load(open(feat_path, "rb")))

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
                logger.info(f"  ADE: {metrics['pred_ade']:.3f}m, FDE: {metrics['pred_fde']:.3f}m")
            if "min_pred_dist_to_victim" in metrics:
                logger.info(f"  Min dist to victim: {metrics['min_pred_dist_to_victim']:.2f}m")

            visualize_scenario(
                case_before, case_update, opts, attacker, idx, save_dir)

            # Collect all intermediate results for analysis
            victim_id = opts['victim_vehicle_id']
            attacker_id = opts['attacker_vehicle_id']
            target_id = attack_case['target_id']
            vtid = opts.get('victim_target_track_id')
            attack_end = attacker.attack_end_frame_id  # 22

            extra = {}

            # 1. GT future trajectory of target (frames 23-42, map frame)
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

            # 2. Normal (pre-attack) predicted trajectory of target
            if vtid is not None and vtid in case_before[attack_end][victim_id].get('predicted_trajectories', {}):
                extra['normal_predicted_trajectory'] = case_before[attack_end][victim_id]['predicted_trajectories'][vtid]
            else:
                extra['normal_predicted_trajectory'] = None

            # 3. Normal observed trajectory of target (pre-attack tracking)
            if vtid is not None and vtid in case_before[attack_end][victim_id].get('observed_trajectories', {}):
                extra['normal_observed_trajectory'] = case_before[attack_end][victim_id]['observed_trajectories'][vtid]
            else:
                extra['normal_observed_trajectory'] = None

            # 4. Normal predicted trajectory of victim (for min-dist baseline)
            if victim_id in case_before[attack_end][victim_id].get('observed_trajectories', {}):
                extra['victim_observed_trajectory'] = case_before[attack_end][victim_id]['observed_trajectories'][victim_id]
            else:
                extra['victim_observed_trajectory'] = None

            # 5. Per-frame attack detections (what perception attack produced)
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
        except Exception:
            n_fail += 1
            if n_fail <= 5:
                logger.error(f"Case {idx} failed: {traceback.format_exc()}")
            signal.alarm(0)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Scenario Attack Summary ({args.model}): {n_success} success, {n_fail} fail")
    if all_results:
        shifts = [m["real_shift"] for m in all_results if "real_shift" in m]
        ades = [m["pred_ade"] for m in all_results if "pred_ade" in m]
        fdes = [m["pred_fde"] for m in all_results if "pred_fde" in m]
        dists = [m["min_pred_dist_to_victim"] for m in all_results
                 if "min_pred_dist_to_victim" in m]
        times = [m["elapsed"] for m in all_results]

        if shifts:
            logger.info(f"Real shift: mean={np.mean(shifts):.3f}m")
        if ades:
            logger.info(f"Prediction ADE: mean={np.mean(ades):.3f}m, FDE: mean={np.mean(fdes):.3f}m")
        if dists:
            d = np.array(dists)
            logger.info(f"Min dist to victim: mean={d.mean():.2f}m, "
                        f"<3m: {(d<3).sum()}/{len(d)}, <5m: {(d<5).sum()}/{len(d)}")
        logger.info(f"Avg time per case: {np.mean(times):.1f}s")

    with open(os.path.join(save_dir, 'summary.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
