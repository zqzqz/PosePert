"""
Transfer attack evaluation for Table 3.

Takes white-box scenario attack results (optimized against GRIP++) and
re-evaluates them using Trajectron++ as the prediction model.

No perception re-run is needed: we directly use the attacked observed
trajectories stored in the white-box result files.

Usage:
  python test/run_transfer_eval.py --model pointpillar --gpu 0
  python test/run_transfer_eval.py --model v2vnet --gpu 0
  python test/run_transfer_eval.py --model cobevt --gpu 0
  python test/run_transfer_eval.py --all --gpu 0
"""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=1)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

import pickle, copy, time, logging, traceback
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'test')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model name -> (whitebox dir, save dir tag)
MODEL_CONFIGS = {
    'pointpillar': {
        'wb_dir': 'results_paper/S1_scenario_pp_whitebox',
        'save_dir': 'results_paper/S1_scenario_pp_transfer',
        'tag': 'S1',
    },
    'v2vnet': {
        'wb_dir': 'results_paper/S2_scenario_v2vnet_whitebox',
        'save_dir': 'results_paper/S2_scenario_v2vnet_transfer',
        'tag': 'S2',
    },
    'cobevt': {
        'wb_dir': 'results_paper/S3_scenario_cobevt_whitebox',
        'save_dir': 'results_paper/S3_scenario_cobevt_transfer',
        'tag': 'S3',
    },
}


def build_trajectron():
    """Load and return Trajectron++ model interface."""
    from mvp.config import model_root

    adv_root = os.path.join('third_party', 'AdvTrajectoryPrediction')
    traj_src = os.path.join(adv_root, 'prediction',
                            'model', 'Trajectron', 'Trajectron-plus-plus', 'trajectron')
    traj_exp = os.path.join(adv_root, 'prediction',
                            'model', 'Trajectron', 'Trajectron-plus-plus', 'experiments', 'nuScenes')
    for p in [adv_root, traj_src, traj_exp]:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Avoid module collision with opencood's 'model' package
    if 'model' in sys.modules and not hasattr(sys.modules['model'], 'model_registrar'):
        del sys.modules['model']

    from prediction.model.Trajectron.interface import TrajectronInterface
    from prediction.model.Trajectron.dataloader import TrajectronDataLoader
    from environment import Environment, Scene

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
            'velocity': {'x': {'mean': 0, 'std': 15}, 'y': {'mean': 0, 'std': 15},
                         'norm': {'mean': 0, 'std': 15}},
            'acceleration': {'x': {'mean': 0, 'std': 4}, 'y': {'mean': 0, 'std': 4},
                             'norm': {'mean': 0, 'std': 4}},
            'heading': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1},
                        '\u00b0': {'mean': 0, 'std': np.pi}, 'd\u00b0': {'mean': 0, 'std': 1}},
        }
    }
    env = Environment(node_type_list=['VEHICLE'], standardization=traj_api.standardization)
    env.attention_radius = {(env.NodeType.VEHICLE, env.NodeType.VEHICLE): 30.0}
    env.robot_type = env.NodeType.VEHICLE
    env.scenes = [Scene(timesteps=40, dt=0.1, name="", aug_func=None)]
    traj_api.env = env
    traj_api.model, traj_api.hyperparams = traj_api.load_model(
        os.path.join(model_root, 'Trajectron/OPV2V'))
    traj_api.dataloader = TrajectronDataLoader(20, 20)

    logger.info("Loaded Trajectron++ prediction model")
    return traj_api


def run_trajectron_prediction(pred_api, observed_trajectories, object_ids=None, num_frames=20):
    """Run Trajectron++ prediction on observed trajectories."""
    from mvp.attack.scenario_attacker_util import prediction_trajectron
    return prediction_trajectron(
        observed_trajectories,
        model_args={'model_api': pred_api},
        num_frames=num_frames,
        object_ids=object_ids,
    )


def eval_model(model_name, pred_api, n_cases=None):
    """Run transfer evaluation for one perception model."""
    cfg = MODEL_CONFIGS[model_name]
    wb_dir = cfg['wb_dir']
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Find all whitebox case files
    case_files = sorted([f for f in os.listdir(wb_dir) if f.startswith('case_') and f.endswith('.pkl')])
    if n_cases is not None:
        case_files = case_files[:n_cases]

    logger.info(f"Transfer eval for {model_name}: {len(case_files)} cases from {wb_dir}")

    all_results = []
    n_success = 0
    n_fail = 0
    t_start = time.time()

    for case_file in case_files:
        idx = int(case_file.replace('case_', '').replace('.pkl', ''))
        try:
            with open(os.path.join(wb_dir, case_file), 'rb') as f:
                wb_data = pickle.load(f)

            opts = wb_data['attack_opts']
            extra = wb_data['extra']
            case_update = wb_data['case_update']
            wb_metrics = wb_data['metrics']

            victim_id = opts['victim_vehicle_id']
            target_track_id = int(opts['victim_target_track_id'])
            case_id = wb_metrics.get('case_id', idx)

            # Determine the last attack frame (frame with tracking data)
            attack_end = None
            for frame_idx in [22, 21, 20]:
                if (frame_idx < len(case_update) and
                        victim_id in case_update[frame_idx] and
                        'observed_trajectories' in case_update[frame_idx][victim_id]):
                    attack_end = frame_idx
                    break

            if attack_end is None:
                logger.warning(f"Case {idx}: no valid attack frame found, skipping")
                n_fail += 1
                continue

            # Get attacked observed trajectories at the last attack frame
            atk_obs = case_update[attack_end][victim_id]['observed_trajectories']

            if target_track_id not in atk_obs:
                logger.warning(f"Case {idx}: target track {target_track_id} not in attacked observations, skipping")
                n_fail += 1
                continue

            # Build normal observed trajectories:
            # same as attacked, but with the target's trajectory replaced by the normal one
            normal_target_traj = extra['normal_observed_trajectory']
            normal_obs = copy.deepcopy(atk_obs)
            normal_obs[target_track_id] = normal_target_traj

            # Get victim trajectory for min-distance computation
            victim_traj = extra['victim_observed_trajectory']
            victim_last_pos = victim_traj[-1, :2]

            # --- Run Trajectron++ predictions ---
            # 1. Normal prediction (unattacked target trajectory)
            normal_pred = run_trajectron_prediction(
                pred_api, normal_obs, object_ids=[target_track_id])

            # 2. Attacked prediction (WB-shifted target trajectory)
            attack_pred = run_trajectron_prediction(
                pred_api, atk_obs, object_ids=[target_track_id])

            metrics = {
                'idx': idx,
                'case_id': case_id,
            }

            # Copy WB perception-level metrics for reference
            for key in ['pert_max', 'desired_shift', 'real_shift']:
                if key in wb_metrics:
                    metrics[key] = wb_metrics[key]

            # Normal Trajectron prediction metrics
            if target_track_id in normal_pred:
                normal_traj_pred = normal_pred[target_track_id]
                metrics['normal_min_dist'] = float(np.min(
                    np.linalg.norm(normal_traj_pred[:, :2] - victim_last_pos, axis=1)))
                metrics['normal_pred'] = normal_traj_pred
            else:
                logger.warning(f"Case {idx}: Trajectron++ produced no normal prediction for target")
                n_fail += 1
                continue

            # Attacked Trajectron prediction metrics
            if target_track_id in attack_pred:
                attack_traj_pred = attack_pred[target_track_id]
                metrics['attack_min_dist'] = float(np.min(
                    np.linalg.norm(attack_traj_pred[:, :2] - victim_last_pos, axis=1)))
                metrics['attack_pred'] = attack_traj_pred

                # ADE and FDE between normal and attacked predictions
                min_len = min(len(normal_traj_pred), len(attack_traj_pred))
                metrics['pred_ade'] = float(np.mean(
                    np.linalg.norm(normal_traj_pred[:min_len, :2] - attack_traj_pred[:min_len, :2], axis=1)))
                metrics['pred_fde'] = float(np.linalg.norm(
                    normal_traj_pred[min_len - 1, :2] - attack_traj_pred[min_len - 1, :2]))

                # MinDist delta (how much closer the attacked prediction is to victim)
                metrics['min_pred_dist_to_victim'] = metrics['attack_min_dist']
                metrics['min_dist_delta'] = metrics['normal_min_dist'] - metrics['attack_min_dist']

                # Is the attack "improved" (moved prediction closer to victim)?
                metrics['improved'] = metrics['attack_min_dist'] < metrics['normal_min_dist']
                # Is the attack "dangerous" (prediction within 3m of victim)?
                metrics['danger_3m'] = metrics['attack_min_dist'] < 3.0
                metrics['danger_5m'] = metrics['attack_min_dist'] < 5.0
            else:
                logger.warning(f"Case {idx}: Trajectron++ produced no attack prediction for target")
                n_fail += 1
                continue

            # Reference: WB GRIP++ metrics for comparison
            metrics['wb_pred_ade'] = wb_metrics.get('pred_ade', None)
            metrics['wb_pred_fde'] = wb_metrics.get('pred_fde', None)
            metrics['wb_min_dist'] = wb_metrics.get('min_pred_dist_to_victim', None)

            # Save per-case result
            case_result = {
                'attack_opts': {k: v for k, v in opts.items()
                                if k not in ('perturbation',)},  # keep metadata, skip large arrays
                'metrics': {k: v for k, v in metrics.items()
                            if k not in ('normal_pred', 'attack_pred')},
                'normal_pred': metrics.get('normal_pred'),
                'attack_pred': metrics.get('attack_pred'),
                'normal_observed_trajectory': normal_target_traj,
                'attack_observed_trajectory': atk_obs[target_track_id],
                'victim_observed_trajectory': victim_traj,
                'gt_future_trajectory': extra.get('gt_future_trajectory'),
            }
            save_path = os.path.join(save_dir, case_file)
            with open(save_path, 'wb') as f:
                pickle.dump(case_result, f)

            all_results.append(metrics)
            n_success += 1

            if n_success % 20 == 0 or n_success <= 5:
                logger.info(
                    f"  [{n_success}] case {idx}: "
                    f"ADE={metrics['pred_ade']:.3f}m "
                    f"FDE={metrics['pred_fde']:.3f}m "
                    f"MinDist={metrics['attack_min_dist']:.2f}m "
                    f"(normal={metrics['normal_min_dist']:.2f}m) "
                    f"[WB: ADE={metrics.get('wb_pred_ade', '?')}, "
                    f"MinDist={metrics.get('wb_min_dist', '?')}]"
                )

        except Exception:
            n_fail += 1
            if n_fail <= 5:
                logger.error(f"Case {idx} failed:\n{traceback.format_exc()}")

    elapsed = time.time() - t_start

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Transfer Eval Summary ({model_name}): "
                f"{n_success} success, {n_fail} fail, {elapsed:.0f}s")

    if all_results:
        ades = [m['pred_ade'] for m in all_results if 'pred_ade' in m]
        fdes = [m['pred_fde'] for m in all_results if 'pred_fde' in m]
        atk_dists = [m['attack_min_dist'] for m in all_results if 'attack_min_dist' in m]
        normal_dists = [m['normal_min_dist'] for m in all_results if 'normal_min_dist' in m]
        improved = [m['improved'] for m in all_results if 'improved' in m]
        danger_3 = [m['danger_3m'] for m in all_results if 'danger_3m' in m]
        danger_5 = [m['danger_5m'] for m in all_results if 'danger_5m' in m]

        # WB reference
        wb_ades = [m['wb_pred_ade'] for m in all_results if m.get('wb_pred_ade') is not None]
        wb_fdes = [m['wb_pred_fde'] for m in all_results if m.get('wb_pred_fde') is not None]
        wb_dists = [m['wb_min_dist'] for m in all_results if m.get('wb_min_dist') is not None]

        def fmt_delta(atk_vals, normal_vals):
            """Format: mean (delta from normal)."""
            a = np.mean(atk_vals)
            n = np.mean(normal_vals)
            return f"{a:.3f} (delta={a-n:+.3f})"

        logger.info(f"  ADE:     {np.mean(ades):.3f}m")
        logger.info(f"  FDE:     {np.mean(fdes):.3f}m")
        logger.info(f"  MinDist: {fmt_delta(atk_dists, normal_dists)}m "
                     f"(normal={np.mean(normal_dists):.3f}m)")
        logger.info(f"  %%Improved: {sum(improved)}/{len(improved)} "
                     f"({100*sum(improved)/len(improved):.1f}%%)")
        logger.info(f"  %%Danger<3m: {sum(danger_3)}/{len(danger_3)} "
                     f"({100*sum(danger_3)/len(danger_3):.1f}%%)")
        logger.info(f"  %%Danger<5m: {sum(danger_5)}/{len(danger_5)} "
                     f"({100*sum(danger_5)/len(danger_5):.1f}%%)")

        if wb_ades:
            logger.info(f"  [WB ref] ADE={np.mean(wb_ades):.3f}m "
                         f"FDE={np.mean(wb_fdes):.3f}m "
                         f"MinDist={np.mean(wb_dists):.3f}m")

    # Save summary
    summary_metrics = [{k: v for k, v in m.items()
                        if k not in ('normal_pred', 'attack_pred')}
                       for m in all_results]
    with open(os.path.join(save_dir, 'summary.pkl'), 'wb') as f:
        pickle.dump(summary_metrics, f)
    logger.info(f"Results saved to {save_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        choices=['pointpillar', 'v2vnet', 'cobevt'])
    parser.add_argument('--all', action='store_true',
                        help='Run all 3 models')
    parser.add_argument('--n_cases', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    if not args.all and args.model is None:
        parser.error("Specify --model or --all")

    # Build Trajectron++ (shared across all models)
    pred_api = build_trajectron()

    models = ['pointpillar', 'v2vnet', 'cobevt'] if args.all else [args.model]

    all_model_results = {}
    for model_name in models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running transfer eval for {model_name}")
        logger.info(f"{'='*70}")
        results = eval_model(model_name, pred_api, n_cases=args.n_cases)
        all_model_results[model_name] = results

    # Final combined summary
    if len(models) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("COMBINED TRANSFER EVAL SUMMARY")
        logger.info(f"{'='*70}")
        for model_name in models:
            results = all_model_results[model_name]
            if not results:
                logger.info(f"  {model_name}: no results")
                continue
            ades = [m['pred_ade'] for m in results if 'pred_ade' in m]
            fdes = [m['pred_fde'] for m in results if 'pred_fde' in m]
            dists = [m['attack_min_dist'] for m in results if 'attack_min_dist' in m]
            normal_dists = [m['normal_min_dist'] for m in results if 'normal_min_dist' in m]
            improved = [m['improved'] for m in results if 'improved' in m]
            danger_3 = [m['danger_3m'] for m in results if 'danger_3m' in m]
            logger.info(
                f"  {model_name:12s}: "
                f"ADE={np.mean(ades):.3f} "
                f"FDE={np.mean(fdes):.3f} "
                f"MinDist={np.mean(dists):.3f} "
                f"(normal={np.mean(normal_dists):.3f}) "
                f"Improved={100*sum(improved)/len(improved):.1f}%% "
                f"Danger<3m={100*sum(danger_3)/len(danger_3):.1f}%%"
            )


if __name__ == '__main__':
    main()
