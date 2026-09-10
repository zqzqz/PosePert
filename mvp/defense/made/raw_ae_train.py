"""
Raw Feature Autoencoder (baseline reconstruction detector).

Trains a UNet to reconstruct single-agent raw spatial features (64-ch,
pre-backbone). NOT the original MADE — this is a simpler baseline.
See made_residual_ae.py for the true MADE residual AE.

Usage:
    python mvp/defense/made/raw_ae_train.py \
        --root data/OPV2V --mode train \
        --out models/MADE/raw_ae_64ch.pt \
        --epochs 50 --n_cases 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.util import set_seed
from mvp.defense.made.unet_utils import DoubleConv, Down, Up, OutConv

logging.basicConfig(level=logging.INFO)


class UNet64(nn.Module):
    """UNet for 64-channel spatial features (pre-backbone)."""
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        self.down4 = Down(64, 64)
        self.up1 = Up(64, 64, bilinear=False, ch_factor=1.5)
        self.up2 = Up(64, 64, bilinear=False, ch_factor=1.5)
        self.up3 = Up(64, 64, bilinear=False, ch_factor=1.5)
        self.up4 = Up(64, 64, bilinear=False, ch_factor=1.5)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class SpatialFeatureDataset(Dataset):
    """Dataset of clean spatial features for autoencoder training."""
    def __init__(self, features_list):
        self.features = features_list  # list of (C, H, W) tensors

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collect_clean_features(dataset, perception, n_cases=100, frame_id=9):
    """
    Collect clean spatial features from the dataset.
    Returns list of (C, H, W) numpy arrays, one per agent per case.
    """
    from opencood.tools import train_utils

    features = []
    case_ids = list(range(min(n_cases, len(dataset.attacks) // 3)))

    for i, case_id in enumerate(case_ids):
        try:
            case = dataset.get_case(case_id, tag="multi_frame", use_lidar=True)
            if frame_id >= len(case):
                continue
            frame = case[frame_id]
            vids = list(frame.keys())
            ego_id = vids[0]

            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            batch = perception.preprocessors[perception.fusion_method](frame, ego_id)
            batch_data = perception.dataset.collate_batch_test([batch])
            batch_data = train_utils.to_device(batch_data, perception.device)

            bd = {
                'voxel_features': batch_data['ego']['processed_lidar']['voxel_features'],
                'voxel_coords': batch_data['ego']['processed_lidar']['voxel_coords'],
                'voxel_num_points': batch_data['ego']['processed_lidar']['voxel_num_points'],
                'record_len': batch_data['ego']['record_len'],
            }
            perception.model.pillar_vfe(bd)
            perception.model.scatter(bd)

            sf = bd['spatial_features'].detach().cpu()  # (N_agents, C, H, W)
            for agent_idx in range(sf.shape[0]):
                features.append(sf[agent_idx].numpy())

            del bd, batch_data, sf
            torch.cuda.empty_cache()

        except Exception as e:
            if i < 3:
                logging.warning(f"Case {case_id} failed: {e}")
            continue

        if (i + 1) % 20 == 0:
            logging.info(f"Collected {len(features)} features from {i+1} cases")

    logging.info(f"Total: {len(features)} clean features from {len(case_ids)} cases")
    return features


def train_ae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OPV2VDataset(root_path=args.root, mode=args.mode, dataset_name="OPV2V")
    perception = OpencoodPerception(
        fusion_method="intermediate", model_name="pointpillar", dataset_name="OPV2V")

    # Collect clean features
    logging.info("Collecting clean features...")
    raw_features = collect_clean_features(dataset, perception, n_cases=args.n_cases)

    if len(raw_features) == 0:
        logging.error("No features collected!")
        return

    # Convert to tensors
    all_features = [torch.from_numpy(f).float() for f in raw_features]

    # Split train/val (90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(all_features))
    split = int(0.9 * len(indices))
    train_features = [all_features[i] for i in indices[:split]]
    val_features = [all_features[i] for i in indices[split:]]

    train_loader = DataLoader(SpatialFeatureDataset(train_features),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SpatialFeatureDataset(val_features),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    logging.info(f"Train: {len(train_features)}, Val: {len(val_features)}")

    # Model
    C = all_features[0].shape[0]
    model = UNet64(in_channels=C, out_channels=C).to(device)
    logging.info(f"UNet64: {sum(p.numel() for p in model.parameters())} params, input channels={C}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_loss = float("inf")
    train_losses_all = []
    val_losses_all = []

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        val_recon_losses = []  # per-sample recon loss for calibration
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = F.mse_loss(recon, batch)
                val_losses.append(loss.item())
                # Per-sample reconstruction loss
                per_sample = F.mse_loss(recon, batch, reduction="none").sum(dim=(1, 2, 3))
                val_recon_losses.extend(per_sample.cpu().numpy().tolist())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        train_losses_all.append(tl)
        val_losses_all.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch = epoch
            # Compute calibration threshold
            val_recon_arr = np.array(val_recon_losses)
            threshold_95 = np.percentile(val_recon_arr, 95)
            threshold_99 = np.percentile(val_recon_arr, 99)

            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": vl,
                "threshold_95": float(threshold_95),
                "threshold_99": float(threshold_99),
                "calibration_scores": val_recon_arr,
                "in_channels": C,
            }, args.out)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1}/{args.epochs}: train={tl:.6f} val={vl:.6f} "
                         f"best={best_val_loss:.6f}@{best_epoch+1}")

    logging.info(f"\nBest: epoch {best_epoch+1}, val_loss={best_val_loss:.6f}")
    logging.info(f"Threshold 95th: {threshold_95:.4f}")
    logging.info(f"Threshold 99th: {threshold_99:.4f}")
    logging.info(f"Saved to {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/OPV2V")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--out", type=str, default="models/made_ae_64ch.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_cases", type=int, default=100)
    args = parser.parse_args()
    train_ae(args)


if __name__ == "__main__":
    main()
