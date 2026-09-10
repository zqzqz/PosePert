"""
Convert OPV2V ground-truth trajectories to GRIP training format.

Uses OPV2V train split scenarios (tag="scenario") which provide full-length
trajectories (100+ frames per scenario). Adds noise augmentation to simulate
detection/tracking errors.

Usage:
  python mvp/prediction/grip/prepare_data.py --out data/prediction/GRIP/train.pkl
  python mvp/prediction/grip/prepare_data.py --mode test --out data/prediction/GRIP/test.pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
from scipy import spatial
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from mvp.data.opv2v_dataset import OPV2VDataset

logging.basicConfig(level=logging.INFO)

HISTORY_FRAMES = 6
FUTURE_FRAMES = 6
TOTAL_FRAMES = HISTORY_FRAMES + FUTURE_FRAMES
MAX_NUM_OBJECT = 120
NEIGHBOR_DISTANCE = 10
TOTAL_FEATURE_DIM = 11

POSITION_NOISE_STD = 0.15
HEADING_NOISE_STD = 0.03
SIZE_NOISE_STD = 0.05


def extract_gt_from_scenario(dataset, scenario_idx):
    """
    Extract per-frame GT object states from a full OPV2V scenario.

    Returns:
      frame_dict: {frame_idx: {object_id: 10-dim feature vector}}
      where feature = [frame_idx, obj_id, obj_type, x, y, z, l, w, h, heading]
    """
    try:
        case = dataset.get_case(scenario_idx, tag="scenario",
                                use_lidar=False, use_camera=False)
    except Exception:
        return {}

    frame_dict = {}
    for frame_idx, frame in enumerate(case):
        if not frame:
            continue
        ego_vid = list(frame.keys())[0]
        ego_data = frame[ego_vid]
        if "gt_bboxes" not in ego_data:
            continue

        gt_bboxes = np.array(ego_data["gt_bboxes"])
        object_ids = ego_data["object_ids"]
        if gt_bboxes.ndim != 2 or gt_bboxes.shape[0] == 0:
            continue

        frame_dict[frame_idx] = {}
        for i, oid in enumerate(object_ids):
            bbox = gt_bboxes[i]
            frame_dict[frame_idx][oid] = np.array([
                frame_idx, oid, 1,
                bbox[0], bbox[1], bbox[2],
                bbox[3], bbox[4], bbox[5],
                bbox[6]
            ], dtype=float)

    return frame_dict


def add_noise(frame_dict, rng):
    """Add detection/tracking-like noise to GT trajectories."""
    noisy = {}
    obj_bias = {}
    for fid in sorted(frame_dict.keys()):
        noisy[fid] = {}
        for oid, feat in frame_dict[fid].items():
            if oid not in obj_bias:
                obj_bias[oid] = {
                    'pos': rng.normal(0, POSITION_NOISE_STD * 0.5, size=2),
                    'heading': rng.normal(0, HEADING_NOISE_STD * 0.5),
                }
            feat_noisy = feat.copy()
            feat_noisy[3] += obj_bias[oid]['pos'][0] + rng.normal(0, POSITION_NOISE_STD)
            feat_noisy[4] += obj_bias[oid]['pos'][1] + rng.normal(0, POSITION_NOISE_STD)
            feat_noisy[9] += obj_bias[oid]['heading'] + rng.normal(0, HEADING_NOISE_STD)
            feat_noisy[6] += rng.normal(0, SIZE_NOISE_STD)
            feat_noisy[7] += rng.normal(0, SIZE_NOISE_STD)
            feat_noisy[8] += rng.normal(0, SIZE_NOISE_STD)
            noisy[fid][oid] = feat_noisy
    return noisy


def process_window(frame_dict, start_frame, end_frame, observed_last):
    if observed_last not in frame_dict:
        return None, None, None

    visible_ids = list(frame_dict[observed_last].keys())
    num_visible = len(visible_ids)
    if num_visible == 0:
        return None, None, None

    visible_values = np.array([frame_dict[observed_last][oid] for oid in visible_ids])
    xy = visible_values[:, 3:5]
    mean_xy = np.mean(xy, axis=0)
    mean_vec = np.zeros(10, dtype=float)
    mean_vec[3:5] = mean_xy

    dist_xy = spatial.distance.cdist(xy, xy)
    neighbor_matrix = np.zeros((MAX_NUM_OBJECT, MAX_NUM_OBJECT))
    neighbor_matrix[:num_visible, :num_visible] = (dist_xy < NEIGHBOR_DISTANCE).astype(int)

    all_ids_in_window = set()
    for fid in range(start_frame, end_frame):
        if fid in frame_dict:
            all_ids_in_window.update(frame_dict[fid].keys())
    non_visible_ids = list(all_ids_in_window - set(visible_ids))
    all_ordered_ids = visible_ids + non_visible_ids

    T = end_frame - start_frame
    object_feature_list = []
    for fid in range(start_frame, end_frame):
        frame_features = {}
        if fid in frame_dict:
            for oid in frame_dict[fid]:
                raw = frame_dict[fid][oid] - mean_vec
                mask = 1 if oid in visible_ids else 0
                frame_features[oid] = list(raw) + [mask]
        row = np.array([
            frame_features.get(oid, np.zeros(TOTAL_FEATURE_DIM))
            for oid in all_ordered_ids
        ])
        object_feature_list.append(row)

    object_feature_list = np.array(object_feature_list)
    num_obj = len(all_ordered_ids)
    object_frame_feature = np.zeros((MAX_NUM_OBJECT, T, TOTAL_FEATURE_DIM))
    object_frame_feature[:min(num_obj, MAX_NUM_OBJECT)] = np.transpose(
        object_feature_list[:, :MAX_NUM_OBJECT], (1, 0, 2))

    return object_frame_feature, neighbor_matrix, mean_xy


def generate_samples(frame_dict, is_train=True):
    frame_ids = sorted(frame_dict.keys())
    if len(frame_ids) < TOTAL_FRAMES:
        return [], [], []

    all_features, all_adjacency, all_mean_xy = [], [], []

    if is_train:
        for start in range(frame_ids[0], frame_ids[-1] - TOTAL_FRAMES + 2):
            end = start + TOTAL_FRAMES
            observed_last = start + HISTORY_FRAMES - 1
            if not all(f in frame_dict for f in range(start, end)):
                continue
            feat, adj, mxy = process_window(frame_dict, start, end, observed_last)
            if feat is not None:
                all_features.append(feat)
                all_adjacency.append(adj)
                all_mean_xy.append(mxy)
    else:
        for start in range(frame_ids[0], frame_ids[-1] - TOTAL_FRAMES + 2, HISTORY_FRAMES):
            end = start + TOTAL_FRAMES
            observed_last = start + HISTORY_FRAMES - 1
            if not all(f in frame_dict for f in range(start, end)):
                continue
            feat, adj, mxy = process_window(frame_dict, start, end, observed_last)
            if feat is not None:
                all_features.append(feat)
                all_adjacency.append(adj)
                all_mean_xy.append(mxy)

    return all_features, all_adjacency, all_mean_xy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/OPV2V")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--noise_augment", type=int, default=3,
                        help="Number of noisy copies per scenario (train only, 0=clean only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"data/prediction/GRIP/{args.mode}.pkl"

    rng = np.random.RandomState(args.seed)
    dataset = OPV2VDataset(root_path=args.root, mode=args.mode, dataset_name="OPV2V")
    n_scenarios = dataset.case_number("scenario")
    is_train = (args.mode == "train")

    logging.info(f"Mode={args.mode}, {n_scenarios} scenarios, "
                 f"noise_augment={args.noise_augment if is_train else 0}")

    all_features, all_adjacency, all_mean_xy = [], [], []

    for sid in range(n_scenarios):
        frame_dict = extract_gt_from_scenario(dataset, sid)
        if frame_dict is None or len(frame_dict) < TOTAL_FRAMES:
            continue

        # Clean GT
        feats, adjs, mxys = generate_samples(frame_dict, is_train=is_train)
        all_features.extend(feats)
        all_adjacency.extend(adjs)
        all_mean_xy.extend(mxys)

        # Noisy augmented copies (train only)
        if is_train:
            for _ in range(args.noise_augment):
                noisy_dict = add_noise(frame_dict, rng)
                feats, adjs, mxys = generate_samples(noisy_dict, is_train=True)
                all_features.extend(feats)
                all_adjacency.extend(adjs)
                all_mean_xy.extend(mxys)

        if (sid + 1) % 10 == 0 or sid == n_scenarios - 1:
            logging.info(f"Processed {sid+1}/{n_scenarios} scenarios, "
                         f"{len(all_features)} samples")

    if len(all_features) == 0:
        logging.error("No samples generated!")
        return

    all_features = np.transpose(np.array(all_features), (0, 3, 2, 1))
    all_adjacency = np.array(all_adjacency)
    all_mean_xy = np.array(all_mean_xy)

    logging.info(f"features: {all_features.shape}, adjacency: {all_adjacency.shape}, "
                 f"mean_xy: {all_mean_xy.shape}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump([all_features, all_adjacency, all_mean_xy], f)
    logging.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
