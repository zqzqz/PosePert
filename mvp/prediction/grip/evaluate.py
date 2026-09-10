"""
Evaluate GRIP trajectory prediction on OPV2V data.

Computes ADE (Average Displacement Error) and FDE (Final Displacement Error)
at multiple time horizons. Also evaluates on scenario attack data to measure
prediction quality under adversarial conditions.

Usage:
  python mvp/prediction/grip/evaluate.py --model models/GRIP/OPV2V_v2/best_model.pt
  python mvp/prediction/grip/evaluate.py --model models/GRIP/OPV2V/checkpoint.pt --data data/OPV2V/grip_train.pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import logging

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


def preprocess_data(ori_data, rescale_xy, device):
    feature_id = [3, 4, 9, 10]
    ori_data_sel = ori_data[:, feature_id].detach()
    data = ori_data_sel.detach().clone()
    new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0)
    data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
    data[:, :2, 0] = 0
    data = data.float().to(device)
    ori_data_sel = ori_data_sel.float().to(device)
    data[:, :2] = data[:, :2] / rescale_xy
    return data, ori_data_sel


def evaluate(model, data_loader, device, rescale_xy):
    """
    Evaluate model and return per-object ADE/FDE.

    Returns:
      results dict with keys: ade, fde, rmse_per_t, num_objects
    """
    model.eval()
    all_ade = []
    all_fde = []
    all_rmse_per_t = []
    total_objects = 0

    with torch.no_grad():
        for ori_data, A, mean_xy in data_loader:
            data, no_norm_loc = preprocess_data(ori_data, rescale_xy, device)

            if data.shape[2] < HISTORY_FRAMES + FUTURE_FRAMES:
                continue

            input_data = data[:, :, :HISTORY_FRAMES, :]
            output_mask = data[:, -1:, HISTORY_FRAMES:, :]
            ori_output_gt = no_norm_loc[:, :2, HISTORY_FRAMES:, :]
            ori_output_last = no_norm_loc[:, :2, HISTORY_FRAMES-1:HISTORY_FRAMES, :]

            A = A.float().to(device)
            predicted = model(
                pra_x=input_data, pra_A=A,
                pra_pred_length=FUTURE_FRAMES,
                pra_teacher_forcing_ratio=0,
                pra_teacher_location=None
            )

            # Velocity -> position
            predicted = predicted * rescale_xy
            for ind in range(1, predicted.shape[-2]):
                predicted[:, :, ind] = torch.sum(predicted[:, :, ind-1:ind+1], dim=-2)
            predicted += ori_output_last

            # Compute per-object errors
            pred_np = predicted.cpu().numpy()  # (N, 2, T, V)
            gt_np = ori_output_gt.cpu().numpy()  # (N, 2, T, V)
            mask_np = output_mask.cpu().numpy()  # (N, 1, T, V)

            for n in range(pred_np.shape[0]):
                for v in range(pred_np.shape[3]):
                    obj_mask = mask_np[n, 0, :, v]
                    if obj_mask.sum() == 0:
                        continue

                    pred_xy = pred_np[n, :, :, v].T  # (T, 2)
                    gt_xy = gt_np[n, :, :, v].T  # (T, 2)
                    valid = obj_mask > 0

                    if valid.sum() == 0:
                        continue

                    errors = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=1))  # (T,)
                    valid_errors = errors[valid]

                    all_ade.append(valid_errors.mean())
                    all_fde.append(valid_errors[-1])
                    total_objects += 1

            # Per-timestep RMSE
            diff = pred_np - gt_np  # (N, 2, T, V)
            sq_err = (diff ** 2).sum(axis=1)  # (N, T, V)
            mask_2d = mask_np[:, 0, :, :]  # (N, T, V)
            for t in range(FUTURE_FRAMES):
                valid = mask_2d[:, t, :].flatten() > 0
                if valid.sum() > 0:
                    t_errors = sq_err[:, t, :].flatten()[valid]
                    if len(all_rmse_per_t) <= t:
                        all_rmse_per_t.append([])
                    all_rmse_per_t[t].extend(t_errors.tolist())

    ade = np.mean(all_ade) if all_ade else float("nan")
    fde = np.mean(all_fde) if all_fde else float("nan")

    rmse_per_t = []
    for t_errors in all_rmse_per_t:
        rmse_per_t.append(np.sqrt(np.mean(t_errors)))

    return {
        "ade": ade,
        "fde": fde,
        "rmse_per_t": rmse_per_t,
        "num_objects": total_objects,
        "ade_median": np.median(all_ade) if all_ade else float("nan"),
        "fde_median": np.median(all_fde) if all_fde else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/GRIP/OPV2V/checkpoint.pt")
    parser.add_argument("--data", type=str, default="data/prediction/GRIP/train.pkl",
                        help="Path to GRIP-format data pickle")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rescale_x", type=float, default=1.0)
    parser.add_argument("--rescale_y", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rescale_xy = torch.ones((1, 2, 1, 1)).to(device)
    rescale_xy[:, 0] = args.rescale_x
    rescale_xy[:, 1] = args.rescale_y

    # Load model
    model = Model(in_channels=4, graph_args=GRAPH_ARGS, edge_importance_weighting=True)
    model.to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt["xin_graph_seq2seq_model"])
    logging.info(f"Loaded model from {args.model}")

    # Load data
    loader = torch.utils.data.DataLoader(
        Feeder(data_path=args.data, graph_args=GRAPH_ARGS, train_val_test=args.split),
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    logging.info(f"Evaluating on {len(loader.dataset)} samples ({args.split} split)")

    results = evaluate(model, loader, device, rescale_xy)

    print(f"\n{'=' * 50}")
    print(f"GRIP Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data} ({args.split})")
    print(f"Objects evaluated: {results['num_objects']}")
    print(f"\nADE (mean): {results['ade']:.4f} m")
    print(f"ADE (median): {results['ade_median']:.4f} m")
    print(f"FDE (mean): {results['fde']:.4f} m")
    print(f"FDE (median): {results['fde_median']:.4f} m")
    print(f"\nRMSE per timestep (0.5s intervals):")
    for t, rmse in enumerate(results["rmse_per_t"]):
        print(f"  t={t+1} ({(t+1)*0.5:.1f}s): {rmse:.4f} m")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
