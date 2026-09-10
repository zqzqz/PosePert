"""
Train GRIP trajectory prediction model on OPV2V data.

Uses the same architecture and training procedure as the original GRIP,
but with data prepared from OPV2V scenarios.

Usage:
  python mvp/prediction/grip/train.py \
      --train_data data/OPV2V/grip_train.pkl \
      --out_dir models/GRIP/OPV2V_v2 \
      --epochs 50
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import random
import itertools
import logging
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
GRIP_ROOT = os.path.join(PROJECT_ROOT, "third_party/AdvTrajectoryPrediction/prediction/model/GRIP/GRIP")
sys.path.insert(0, GRIP_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from model import Model
from xin_feeder_baidu import Feeder

logging.basicConfig(level=logging.INFO)

HISTORY_FRAMES = 6
FUTURE_FRAMES = 6
GRAPH_ARGS = {"max_hop": 2, "num_node": 120}


def seed_torch(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_data(ori_data, rescale_xy, device):
    """Extract velocity features from raw data."""
    feature_id = [3, 4, 9, 10]  # x, y, heading, mask
    ori_data_sel = ori_data[:, feature_id].detach()
    data = ori_data_sel.detach().clone()

    # Convert positions to velocities (differences)
    new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0)
    data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
    data[:, :2, 0] = 0

    object_type = ori_data[:, 2:3]

    data = data.float().to(device)
    ori_data_sel = ori_data_sel.float().to(device)
    object_type = object_type.to(device)
    data[:, :2] = data[:, :2] / rescale_xy

    return data, ori_data_sel, object_type


def compute_rmse(pred, gt, mask, error_order=2):
    pred = pred * mask
    gt = gt * mask
    x2y2 = torch.sum(torch.abs(pred - gt) ** error_order, dim=1)
    overall_sum_time = x2y2.sum(dim=-1)
    overall_num = mask.sum(dim=1).sum(dim=-1)
    return overall_sum_time, overall_num, x2y2


def train_epoch(model, data_loader, optimizer, device, rescale_xy, epoch, total_epochs):
    model.train()
    total_losses = []

    for iteration, (ori_data, A, _) in enumerate(data_loader):
        data, _, _ = preprocess_data(ori_data, rescale_xy, device)

        for now_history in range(1, data.shape[-2]):
            input_data = data[:, :, :now_history, :]
            output_gt = data[:, :2, now_history:, :]
            output_mask = data[:, -1:, now_history:, :]

            A = A.float().to(device)
            predicted = model(
                pra_x=input_data, pra_A=A,
                pra_pred_length=output_gt.shape[-2],
                pra_teacher_forcing_ratio=0,
                pra_teacher_location=output_gt
            )

            overall_sum, overall_num, _ = compute_rmse(
                predicted, output_gt, output_mask, error_order=1)
            loss = torch.sum(overall_sum) / torch.max(
                torch.sum(overall_num), torch.ones(1).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_losses.append(loss.item())

    return np.mean(total_losses)


def val_epoch(model, data_loader, device, rescale_xy):
    model.eval()
    all_sum = []
    all_num = []

    with torch.no_grad():
        for ori_data, A, _ in data_loader:
            data, no_norm_loc, _ = preprocess_data(ori_data, rescale_xy, device)

            input_data = data[:, :, :HISTORY_FRAMES, :]
            output_gt = data[:, :2, HISTORY_FRAMES:, :]
            output_mask = data[:, -1:, HISTORY_FRAMES:, :]
            ori_output_gt = no_norm_loc[:, :2, HISTORY_FRAMES:, :]
            ori_output_last = no_norm_loc[:, :2, HISTORY_FRAMES-1:HISTORY_FRAMES, :]

            A = A.float().to(device)
            predicted = model(
                pra_x=input_data, pra_A=A,
                pra_pred_length=output_gt.shape[-2],
                pra_teacher_forcing_ratio=0,
                pra_teacher_location=output_gt
            )

            # Convert velocity predictions back to positions
            predicted = predicted * rescale_xy
            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind-1:ind+1], dim=-2)
            predicted += ori_output_last

            overall_sum, overall_num, x2y2 = compute_rmse(
                predicted, ori_output_gt, output_mask)
            x2y2_np = x2y2.detach().cpu().numpy().sum(axis=-1)
            all_sum.extend(x2y2_np)
            all_num.extend(overall_num.detach().cpu().numpy())

    all_sum = np.array(all_sum)
    all_num = np.array(all_num)

    # RMSE per timestep
    sum_per_t = np.sum(all_sum ** 0.5, axis=0)
    num_per_t = np.sum(all_num, axis=0)
    rmse_per_t = sum_per_t / np.maximum(num_per_t, 1)

    return rmse_per_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/prediction/GRIP/train.pkl")
    parser.add_argument("--out_dir", type=str, default="models/GRIP/OPV2V_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_decay_epoch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rescale_x", type=float, default=1.0)
    parser.add_argument("--rescale_y", type=float, default=1.0)
    args = parser.parse_args()

    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    rescale_xy = torch.ones((1, 2, 1, 1)).to(device)
    rescale_xy[:, 0] = args.rescale_x
    rescale_xy[:, 1] = args.rescale_y

    # Load data
    logging.info(f"Loading training data from {args.train_data}")
    train_loader = torch.utils.data.DataLoader(
        Feeder(data_path=args.train_data, graph_args=GRAPH_ARGS, train_val_test="train"),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        Feeder(data_path=args.train_data, graph_args=GRAPH_ARGS, train_val_test="val"),
        batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    logging.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Model
    model = Model(in_channels=4, graph_args=GRAPH_ARGS, edge_importance_weighting=True)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {param_count}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=0.5)

    best_val_rmse = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, rescale_xy, epoch, args.epochs)
        scheduler.step()

        val_rmse = val_epoch(model, val_loader, device, rescale_xy)
        val_total = np.sum(val_rmse)

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f} "
            f"val_RMSE=[{' '.join(f'{r:.3f}' for r in val_rmse)}] "
            f"total={val_total:.3f} lr={lr:.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"model_epoch_{epoch:04d}.pt")
        torch.save({"xin_graph_seq2seq_model": model.state_dict()}, ckpt_path)

        if val_total < best_val_rmse:
            best_val_rmse = val_total
            best_epoch = epoch
            best_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({"xin_graph_seq2seq_model": model.state_dict()}, best_path)
            logging.info(f"  New best model saved (RMSE={val_total:.3f})")

    logging.info(f"\nBest: epoch {best_epoch+1}, val_RMSE={best_val_rmse:.3f}")
    logging.info(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
