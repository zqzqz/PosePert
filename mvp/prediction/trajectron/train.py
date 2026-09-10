"""
Train Trajectron++ on OPV2V data.

Self-contained training script that wraps the Trajectron++ training loop
without needing to run from within the Trajectron directory.

Usage:
  python mvp/prediction/trajectron/train.py \
      --train_data data/prediction/Trajectron/train.pkl \
      --eval_data data/prediction/Trajectron/val.pkl \
      --out_dir models/Trajectron/OPV2V \
      --epochs 100
"""

import os
import sys
import argparse
import json
import time
import random
import pathlib
import logging
import numpy as np
import torch
from torch import nn, optim, utils
import dill
from tqdm import tqdm

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
from model.dataset import EnvironmentDataset, collate
import evaluation

logging.basicConfig(level=logging.INFO)

# Default config (based on nuScenes.json, no map encoding)
DEFAULT_CONFIG = {
    "batch_size": 256,
    "grad_clip": 1.0,
    "learning_rate_style": "exp",
    "learning_rate": 0.003,
    "min_learning_rate": 0.00001,
    "learning_decay_rate": 0.9999,
    "prediction_horizon": 6,
    "minimum_history_length": 1,
    "maximum_history_length": 8,
    "k": 1,
    "k_eval": 1,
    "kl_min": 0.07,
    "kl_weight": 100.0,
    "kl_weight_start": 0,
    "kl_decay_rate": 0.99995,
    "kl_crossover": 400,
    "kl_sigmoid_divisor": 4,
    "rnn_kwargs": {"dropout_keep_prob": 0.75},
    "MLP_dropout_keep_prob": 0.9,
    "enc_rnn_dim_edge": 32,
    "enc_rnn_dim_edge_influence": 32,
    "enc_rnn_dim_history": 32,
    "enc_rnn_dim_future": 32,
    "dec_rnn_dim": 128,
    "q_z_xy_MLP_dims": None,
    "p_z_x_MLP_dims": 32,
    "GMM_components": 1,
    "log_p_yt_xz_max": 6,
    "N": 1,
    "K": 25,
    "tau_init": 2.0,
    "tau_final": 0.05,
    "tau_decay_rate": 0.997,
    "use_z_logit_clipping": True,
    "z_logit_clip_start": 0.05,
    "z_logit_clip_final": 5.0,
    "z_logit_clip_crossover": 300,
    "z_logit_clip_divisor": 5,
    "dynamic": {
        "VEHICLE": {
            "name": "Unicycle",
            "distribution": True,
            "limits": {
                "max_a": 4, "min_a": -5,
                "max_heading_change": 0.7, "min_heading_change": -0.7
            }
        }
    },
    "state": {
        "VEHICLE": {
            "position": ["x", "y"],
            "velocity": ["x", "y"],
            "acceleration": ["x", "y"],
            "heading": ["\u00b0", "d\u00b0"]
        }
    },
    "pred_state": {
        "VEHICLE": {"position": ["x", "y"]}
    },
    "log_histograms": False,
    # Defaults for missing args
    "dynamic_edges": "yes",
    "edge_state_combine_method": "sum",
    "edge_influence_combine_method": "attention",
    "edge_addition_filter": [0.25, 0.5, 0.75, 1.0],
    "edge_removal_filter": [1.0, 0.0],
    "offline_scene_graph": "yes",
    "incl_robot_node": False,
    "node_freq_mult_train": False,
    "node_freq_mult_eval": False,
    "scene_freq_mult_train": False,
    "scene_freq_mult_eval": False,
    "scene_freq_mult_viz": False,
    "edge_encoding": True,
    "use_map_encoding": False,
    "augment": False,
    "override_attention_radius": [],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str,
                        default="data/prediction/Trajectron/train.pkl")
    parser.add_argument("--eval_data", type=str,
                        default="data/prediction/Trajectron/val.pkl")
    parser.add_argument("--out_dir", type=str,
                        default="models/Trajectron/OPV2V")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    hyperparams = DEFAULT_CONFIG.copy()
    hyperparams["batch_size"] = args.batch_size

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    # Load training data
    logging.info(f"Loading training data from {args.train_data}")
    with open(args.train_data, "rb") as f:
        train_env = dill.load(f, encoding="latin1")

    if train_env.robot_type is None:
        train_env.robot_type = train_env.NodeType[0]
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    # Offline scene graph
    logging.info("Computing scene graphs...")
    for i, scene in enumerate(train_env.scenes):
        scene.calculate_scene_graph(
            train_env.attention_radius,
            hyperparams["edge_addition_filter"],
            hyperparams["edge_removal_filter"])

    train_dataset = EnvironmentDataset(
        train_env,
        hyperparams["state"],
        hyperparams["pred_state"],
        scene_freq_mult=hyperparams["scene_freq_mult_train"],
        node_freq_mult=hyperparams["node_freq_mult_train"],
        hyperparams=hyperparams,
        min_history_timesteps=hyperparams["minimum_history_length"],
        min_future_timesteps=hyperparams["prediction_horizon"],
        return_robot=not hyperparams["incl_robot_node"])

    train_data_loader = {}
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue
        node_type_dataloader = utils.data.DataLoader(
            node_type_data_set,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    total_samples = sum(len(dl.dataset) for dl in train_data_loader.values())
    logging.info(f"Training samples: {total_samples}")

    # Load eval data if available
    eval_env = None
    eval_data_loader = {}
    eval_scenes = []
    if os.path.exists(args.eval_data):
        logging.info(f"Loading eval data from {args.eval_data}")
        with open(args.eval_data, "rb") as f:
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
        eval_scenes = eval_env.scenes

        eval_dataset = EnvironmentDataset(
            eval_env,
            hyperparams["state"],
            hyperparams["pred_state"],
            scene_freq_mult=hyperparams["scene_freq_mult_eval"],
            node_freq_mult=hyperparams["node_freq_mult_eval"],
            hyperparams=hyperparams,
            min_history_timesteps=hyperparams["minimum_history_length"],
            min_future_timesteps=hyperparams["prediction_horizon"],
            return_robot=not hyperparams["incl_robot_node"])

        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue
            node_type_dataloader = utils.data.DataLoader(
                node_type_data_set,
                collate_fn=collate,
                pin_memory=False,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        eval_total = sum(len(dl.dataset) for dl in eval_data_loader.values())
        logging.info(f"Eval samples: {eval_total}")

    # Create model
    model_registrar = ModelRegistrar(args.out_dir, device)
    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment(train_env)
    trajectron.set_annealing_params()

    eval_trajectron = None
    if eval_env is not None:
        eval_trajectron = Trajectron(model_registrar, hyperparams, None, device)
        eval_trajectron.set_environment(eval_env)
        eval_trajectron.set_annealing_params()

    param_count = sum(p.numel() for p in model_registrar.parameters())
    logging.info(f"Model parameters: {param_count}")

    # Optimizer
    optimizer = {}
    lr_scheduler = {}
    for node_type in train_env.NodeType:
        if node_type not in hyperparams["pred_state"]:
            continue
        optimizer[node_type] = optim.Adam(
            [{"params": model_registrar.get_all_but_name_match("map_encoder").parameters()},
             {"params": model_registrar.get_name_match("map_encoder").parameters(), "lr": 0.0008}],
            lr=hyperparams["learning_rate"])
        lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
            optimizer[node_type], gamma=hyperparams["learning_decay_rate"])

    # Training loop
    curr_iter_node_type = {nt: 0 for nt in train_data_loader.keys()}
    best_eval_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model_registrar.to(device)

        # Train
        epoch_losses = []
        for node_type, data_loader in train_data_loader.items():
            curr_iter = curr_iter_node_type[node_type]
            pbar = tqdm(data_loader, ncols=80, desc=f"Epoch {epoch} {node_type}")
            for batch in pbar:
                trajectron.set_curr_iter(curr_iter)
                trajectron.step_annealers(node_type)
                optimizer[node_type].zero_grad()
                train_loss = trajectron.train_loss(batch, node_type)
                pbar.set_description(
                    f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f}")
                train_loss.backward()
                if hyperparams["grad_clip"] is not None:
                    nn.utils.clip_grad_value_(
                        model_registrar.parameters(), hyperparams["grad_clip"])
                optimizer[node_type].step()
                lr_scheduler[node_type].step()
                epoch_losses.append(train_loss.item())
                curr_iter += 1
            curr_iter_node_type[node_type] = curr_iter

        avg_loss = np.mean(epoch_losses)
        lr = list(optimizer.values())[0].param_groups[0]["lr"]
        logging.info(f"Epoch {epoch}/{args.epochs}: train_loss={avg_loss:.4f} lr={lr:.6f}")

        # Save
        if epoch % args.save_every == 0:
            model_registrar.save_models(epoch)
            logging.info(f"  Saved checkpoint at epoch {epoch}")

        # Evaluate
        if eval_trajectron is not None and epoch % args.eval_every == 0:
            ph = hyperparams["prediction_horizon"]
            with torch.no_grad():
                eval_losses = []
                for node_type, data_loader in eval_data_loader.items():
                    for batch in data_loader:
                        eval_loss = eval_trajectron.eval_loss(batch, node_type)
                        eval_losses.append(eval_loss.item())

                avg_eval_loss = np.mean(eval_losses) if eval_losses else float("nan")

                # ADE/FDE on eval scenes
                all_ade = []
                all_fde = []
                for scene in eval_scenes:
                    timesteps = np.arange(
                        max(hyperparams["minimum_history_length"], 1),
                        scene.timesteps - ph)
                    if len(timesteps) == 0:
                        continue
                    # Sample a subset
                    if len(timesteps) > 32:
                        timesteps = np.random.choice(timesteps, 32, replace=False)
                    predictions = eval_trajectron.predict(
                        scene, timesteps, ph,
                        num_samples=1,
                        min_future_timesteps=ph,
                        z_mode=True, gmm_mode=True,
                        full_dist=False)
                    for ts, node_preds in predictions.items():
                        for node, pred in node_preds.items():
                            # pred: (1, 1, ph, 2)
                            pred_pos = pred[0, 0]  # (ph, 2)
                            gt = node.get(
                                np.array([ts + 1, ts + ph]),
                                {"position": ["x", "y"]})
                            if gt is None or len(gt) < ph or np.any(np.isnan(gt)):
                                continue
                            errors = np.linalg.norm(pred_pos - gt, axis=1)
                            all_ade.append(errors.mean())
                            all_fde.append(errors[-1])

                ade = np.mean(all_ade) if all_ade else float("nan")
                fde = np.mean(all_fde) if all_fde else float("nan")

                logging.info(
                    f"  Eval: loss={avg_eval_loss:.4f} "
                    f"ADE={ade:.4f}m FDE={fde:.4f}m ({len(all_ade)} objects)")

                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    model_registrar.save_models(epoch)
                    # Also save as "best"
                    best_path = os.path.join(args.out_dir, "best_epoch.txt")
                    with open(best_path, "w") as f:
                        f.write(f"{epoch}\n")
                    logging.info(f"  New best model (eval_loss={avg_eval_loss:.4f})")

    logging.info(f"Training complete. Models saved to {args.out_dir}")


if __name__ == "__main__":
    main()
