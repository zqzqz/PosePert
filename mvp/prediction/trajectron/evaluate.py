"""
Evaluate Trajectron++ on OPV2V data.

Computes ADE/FDE metrics on validation or test data.

Usage:
  python mvp/prediction/trajectron/evaluate.py \
      --model_dir models/Trajectron/OPV2V \
      --data data/prediction/Trajectron/val.pkl \
      --epoch 100
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import dill
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TRAJECTRON_ROOT = os.path.join(
    PROJECT_ROOT,
    "third_party/AdvTrajectoryPrediction/prediction/model/Trajectron/Trajectron-plus-plus/trajectron")
NUSCENES_ROOT = os.path.join(
    PROJECT_ROOT,
    "third_party/AdvTrajectoryPrediction/prediction/model/Trajectron/Trajectron-plus-plus/experiments/nuScenes")
sys.path.insert(0, TRAJECTRON_ROOT)
sys.path.insert(0, NUSCENES_ROOT)

from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/Trajectron/OPV2V")
    parser.add_argument("--data", type=str, default="data/prediction/Trajectron/val.pkl")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch to load. If None, loads best_epoch.txt or latest")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of prediction samples (1=deterministic mode)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path, "r") as f:
        hyperparams = json.load(f)

    # Load data
    logging.info(f"Loading data from {args.data}")
    with open(args.data, "rb") as f:
        eval_env = dill.load(f, encoding="latin1")

    if eval_env.robot_type is None:
        eval_env.robot_type = eval_env.NodeType[0]
        for scene in eval_env.scenes:
            scene.add_robot_from_nodes(eval_env.robot_type)

    for scene in eval_env.scenes:
        scene.calculate_scene_graph(
            eval_env.attention_radius,
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"])

    # Load model
    epoch = args.epoch
    if epoch is None:
        best_path = os.path.join(args.model_dir, "best_epoch.txt")
        if os.path.exists(best_path):
            epoch = int(open(best_path).read().strip())
        else:
            # Find latest checkpoint
            import glob
            ckpts = glob.glob(os.path.join(args.model_dir, "model_registrar-*.pt"))
            if ckpts:
                epochs = [int(os.path.basename(c).split("-")[1].split(".")[0]) for c in ckpts]
                epoch = max(epochs)

    logging.info(f"Loading model from {args.model_dir}, epoch {epoch}")
    model_registrar = ModelRegistrar(args.model_dir, device)

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment(eval_env)
    trajectron.set_annealing_params()

    model_registrar.load_models(epoch)
    model_registrar.to(device)

    ph = hyperparams["prediction_horizon"]

    # Evaluate
    all_ade = []
    all_fde = []
    all_ade_per_t = [[] for _ in range(ph)]

    with torch.no_grad():
        for scene_idx, scene in enumerate(eval_env.scenes):
            # Sample all valid timesteps
            timesteps = np.arange(
                hyperparams["minimum_history_length"],
                scene.timesteps - ph)

            if len(timesteps) == 0:
                continue

            predictions = trajectron.predict(
                scene, timesteps, ph,
                num_samples=args.num_samples,
                min_future_timesteps=ph,
                z_mode=(args.num_samples == 1),
                gmm_mode=(args.num_samples == 1),
                full_dist=False)

            for ts, node_preds in predictions.items():
                for node, pred in node_preds.items():
                    # pred: (num_samples, 1, ph, 2) or (1, 1, ph, 2)
                    if args.num_samples > 1:
                        pred_pos = pred[:, 0]  # (num_samples, ph, 2)
                    else:
                        pred_pos = pred[0, 0]  # (ph, 2)

                    gt = node.get(
                        np.array([ts + 1, ts + ph]),
                        {"position": ["x", "y"]})
                    if gt is None or len(gt) < ph or np.any(np.isnan(gt)):
                        continue

                    if args.num_samples > 1:
                        # Best-of-N: take sample with lowest ADE
                        errors_all = np.linalg.norm(
                            pred_pos - gt[np.newaxis], axis=2)  # (N, ph)
                        best_idx = errors_all.mean(axis=1).argmin()
                        errors = errors_all[best_idx]
                    else:
                        errors = np.linalg.norm(pred_pos - gt, axis=1)

                    all_ade.append(errors.mean())
                    all_fde.append(errors[-1])
                    for t in range(ph):
                        all_ade_per_t[t].append(errors[t])

            if (scene_idx + 1) % 5 == 0:
                logging.info(f"Evaluated {scene_idx+1}/{len(eval_env.scenes)} scenes")

    ade = np.mean(all_ade) if all_ade else float("nan")
    fde = np.mean(all_fde) if all_fde else float("nan")
    ade_median = np.median(all_ade) if all_ade else float("nan")
    fde_median = np.median(all_fde) if all_fde else float("nan")

    print(f"\n{'=' * 50}")
    print(f"Trajectron++ Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Model: {args.model_dir} (epoch {epoch})")
    print(f"Data: {args.data}")
    print(f"Samples: {args.num_samples}")
    print(f"Objects evaluated: {len(all_ade)}")
    print(f"\nADE (mean): {ade:.4f} m")
    print(f"ADE (median): {ade_median:.4f} m")
    print(f"FDE (mean): {fde:.4f} m")
    print(f"FDE (median): {fde_median:.4f} m")
    print(f"\nError per timestep (0.5s intervals):")
    for t in range(ph):
        t_err = np.mean(all_ade_per_t[t]) if all_ade_per_t[t] else float("nan")
        print(f"  t={t+1} ({(t+1)*0.5:.1f}s): {t_err:.4f} m")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
