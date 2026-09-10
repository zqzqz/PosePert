"""
Train MADE's residual autoencoder.

Collects residual features (fused_with_agent - fused_without_agent) from
clean OPV2V data, trains an AE to reconstruct them.

Usage:
    python mvp/defense/made/train_residual_ae.py \
        --out models/MADE/residual_ae.pt --epochs 50 --n_cases 100
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.defense.made.made_residual_ae import ResidualAutoencoder, MadeResidualDetector

logging.basicConfig(level=logging.INFO)


class ResidualDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collect_residuals(dataset, perception, n_cases=100, frame_id=9):
    """Collect clean residual features from training data.

    For each case/frame, extracts pre-backbone spatial features via
    pillar_vfe + scatter, then computes post-backbone residuals using
    MadeResidualDetector.compute_residuals_from_features.
    """
    from opencood.tools import train_utils

    detector = MadeResidualDetector(perception)
    all_residuals = []

    n_total = len(dataset.attacks) // 3 if hasattr(dataset, 'attacks') else len(dataset)
    for ci in range(min(n_cases, n_total)):
        try:
            case = dataset.get_case(ci, tag="multi_frame", use_lidar=True)
            if frame_id >= len(case):
                continue
            frame = case[frame_id]
            vids = list(frame.keys())
            if len(vids) < 2:
                continue  # need at least ego + 1 agent for residuals
            ego_id = vids[0]

            # Extract pre-backbone spatial features (pillar_vfe + scatter)
            batch = perception.preprocessors[perception.fusion_method](
                frame, ego_id)
            batch_data = perception.dataset.collate_batch_test([batch])
            batch_data = train_utils.to_device(batch_data, perception.device)

            with torch.no_grad():
                bd = {
                    'voxel_features': batch_data['ego']['processed_lidar']['voxel_features'],
                    'voxel_coords': batch_data['ego']['processed_lidar']['voxel_coords'],
                    'voxel_num_points': batch_data['ego']['processed_lidar']['voxel_num_points'],
                    'record_len': batch_data['ego']['record_len'],
                }
                perception.model.pillar_vfe(bd)
                perception.model.scatter(bd)
                spatial_features = bd['spatial_features'].clone().detach()
                record_len = bd['record_len']

            # Compute post-backbone residuals
            _, residuals, _ = detector.compute_residuals_from_features(
                spatial_features, record_len)
            for res in residuals:
                all_residuals.append(res.cpu())

            torch.cuda.empty_cache()
        except Exception as e:
            if ci < 3:
                logging.warning(f"Case {ci}: {e}")
            continue

        if (ci + 1) % 20 == 0:
            logging.info(f"Collected {len(all_residuals)} residuals from {ci+1} cases")

    logging.info(f"Total: {len(all_residuals)} residual features")
    return all_residuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/OPV2V")
    parser.add_argument("--dataset", type=str, default="OPV2V")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--out", type=str, default="models/MADE/residual_ae.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_cases", type=int, default=100)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OPV2VDataset(root_path=args.root, mode=args.mode,
                           dataset_name=args.dataset)
    perception = OpencoodPerception(fusion_method="intermediate",
                                    model_name="pointpillar",
                                    dataset_name=args.dataset)

    logging.info("Collecting residual features...")
    raw_residuals = collect_residuals(dataset, perception, n_cases=args.n_cases)

    if len(raw_residuals) == 0:
        logging.error("No residuals collected!")
        return

    in_channels = raw_residuals[0].shape[0]
    logging.info(f"Residual shape: {raw_residuals[0].shape}, in_channels={in_channels}")

    # Compute data mean for normalization
    all_stacked = torch.stack(raw_residuals)
    data_mean = all_stacked.mean(dim=0)  # (C, H, W)
    logging.info(f"Data mean norm: {data_mean.norm():.2f}")

    # Normalize
    normalized = [r - data_mean for r in raw_residuals]

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(normalized))
    split = int(0.9 * len(indices))
    train_data = [normalized[i] for i in indices[:split]]
    val_data = [normalized[i] for i in indices[split:]]

    train_loader = DataLoader(ResidualDataset(train_data),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ResidualDataset(val_data),
                            batch_size=args.batch_size, shuffle=False)

    logging.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    model = ResidualAutoencoder(in_channels, args.base_channels, args.latent_dim)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logging.info(f"ResidualAutoencoder: {param_count} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        val_per_sample = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = F.mse_loss(recon, batch)
                val_losses.append(loss.item())
                per_sample = F.mse_loss(recon, batch, reduction="none").sum(dim=(1, 2, 3))
                val_per_sample.extend(per_sample.cpu().numpy().tolist())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)

        if vl < best_val:
            best_val = vl
            best_epoch = epoch
            val_arr = np.array(val_per_sample)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": vl,
                "in_channels": in_channels,
                "base_channels": args.base_channels,
                "latent_dim": args.latent_dim,
                "data_mean": data_mean.numpy(),
                "threshold_95": float(np.percentile(val_arr, 95)),
                "threshold_99": float(np.percentile(val_arr, 99)),
            }, args.out)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1}/{args.epochs}: train={tl:.6f} val={vl:.6f} "
                         f"best={best_val:.6f}@{best_epoch+1}")

    logging.info(f"Best: epoch {best_epoch+1}, val={best_val:.6f}")
    logging.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
