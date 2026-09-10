"""
Convert OPV2V ground-truth trajectories to Trajectron++ training format.

Uses OPV2V train split scenarios (tag="scenario") which provide full-length
trajectories (100+ frames per scenario). Adds noise augmentation to simulate
detection/tracking errors.

Usage:
  python mvp/prediction/trajectron/prepare_data.py --out data/prediction/Trajectron/train.pkl
  python mvp/prediction/trajectron/prepare_data.py --mode test --out data/prediction/Trajectron/test.pkl
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import dill
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TRAJECTRON_ROOT = os.path.join(
    PROJECT_ROOT,
    "third_party/AdvTrajectoryPrediction/prediction/model/Trajectron/Trajectron-plus-plus/trajectron")
sys.path.insert(0, TRAJECTRON_ROOT)
sys.path.insert(0, os.path.join(
    PROJECT_ROOT,
    "third_party/AdvTrajectoryPrediction/prediction/model/Trajectron/Trajectron-plus-plus/experiments/nuScenes"))
sys.path.insert(0, PROJECT_ROOT)

from environment import Environment, Scene, Node, derivative_of
from mvp.data.opv2v_dataset import OPV2VDataset

logging.basicConfig(level=logging.INFO)

DT = 0.5
POSITION_NOISE_STD = 0.15
HEADING_NOISE_STD = 0.03

STANDARDIZATION = {
    'VEHICLE': {
        'position': {'x': {'mean': 0, 'std': 80}, 'y': {'mean': 0, 'std': 80}},
        'velocity': {'x': {'mean': 0, 'std': 15}, 'y': {'mean': 0, 'std': 15},
                     'norm': {'mean': 0, 'std': 15}},
        'acceleration': {'x': {'mean': 0, 'std': 4}, 'y': {'mean': 0, 'std': 4},
                         'norm': {'mean': 0, 'std': 4}},
        'heading': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1},
                    '°': {'mean': 0, 'std': np.pi}, 'd°': {'mean': 0, 'std': 1}},
    }
}


def extract_gt_trajectories_from_scenario(dataset, scenario_idx):
    """
    Extract per-object continuous trajectories from a full OPV2V scenario.

    Returns:
      trajectories: {obj_id: {'positions': (T,2), 'headings': (T,), 'start_frame': int}}
      total_frames: int
    """
    try:
        case = dataset.get_case(scenario_idx, tag="scenario",
                                use_lidar=False, use_camera=False)
    except Exception:
        return {}, 0

    frame_objects = {}
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

        frame_objects[frame_idx] = {}
        for i, oid in enumerate(object_ids):
            bbox = gt_bboxes[i]
            frame_objects[frame_idx][oid] = (bbox[0], bbox[1], bbox[6])

    all_oids = set()
    for fobj in frame_objects.values():
        all_oids.update(fobj.keys())

    trajectories = {}
    for oid in all_oids:
        frames = sorted([f for f in frame_objects if oid in frame_objects[f]])
        if len(frames) < 3:
            continue

        segments = []
        seg_start = 0
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] != 1:
                if i - seg_start >= 3:
                    segments.append(frames[seg_start:i])
                seg_start = i
        if len(frames) - seg_start >= 3:
            segments.append(frames[seg_start:])

        for seg_idx, seg_frames in enumerate(segments):
            positions = np.array([
                [frame_objects[f][oid][0], frame_objects[f][oid][1]]
                for f in seg_frames
            ])
            headings = np.array([frame_objects[f][oid][2] for f in seg_frames])

            key = f"{oid}_{seg_idx}" if len(segments) > 1 else oid
            trajectories[key] = {
                'positions': positions,
                'headings': headings,
                'start_frame': seg_frames[0],
            }

    total_frames = len(case)
    return trajectories, total_frames


def add_noise_to_trajectories(trajectories, rng):
    noisy = {}
    for key, tdata in trajectories.items():
        T = len(tdata['positions'])
        bias_pos = rng.normal(0, POSITION_NOISE_STD * 0.5, size=2)
        bias_heading = rng.normal(0, HEADING_NOISE_STD * 0.5)

        noisy_pos = tdata['positions'].copy()
        noisy_pos += bias_pos + rng.normal(0, POSITION_NOISE_STD, size=(T, 2))

        noisy_heading = tdata['headings'].copy()
        noisy_heading += bias_heading + rng.normal(0, HEADING_NOISE_STD, size=T)

        noisy[key] = {
            'positions': noisy_pos,
            'headings': noisy_heading,
            'start_frame': tdata['start_frame'],
        }
    return noisy


def build_scene(trajectories, scene_name, total_timesteps, env):
    scene = Scene(timesteps=total_timesteps, dt=DT, name=scene_name)

    data_columns_vehicle = pd.MultiIndex.from_product(
        [['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(
        pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns_vehicle = data_columns_vehicle.append(
        pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

    for track_id, tdata in trajectories.items():
        positions = tdata['positions']
        headings = tdata['headings']
        start_frame = tdata['start_frame']

        if len(positions) < 3:
            continue

        x = positions[:, 0].astype(float)
        y = positions[:, 1].astype(float)
        heading = headings.astype(float)

        vx = derivative_of(x, DT)
        vy = derivative_of(y, DT)
        ax = derivative_of(vx, DT)
        ay = derivative_of(vy, DT)

        v = np.stack((vx, vy), axis=-1)
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))

        data_dict = {
            ('position', 'x'): x, ('position', 'y'): y,
            ('velocity', 'x'): vx, ('velocity', 'y'): vy,
            ('velocity', 'norm'): np.linalg.norm(v, axis=-1),
            ('acceleration', 'x'): ax, ('acceleration', 'y'): ay,
            ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
            ('heading', 'x'): heading_v[:, 0], ('heading', 'y'): heading_v[:, 1],
            ('heading', '°'): heading,
            ('heading', 'd°'): derivative_of(heading, DT, radian=True),
        }
        node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        node = Node(node_type=env.NodeType.VEHICLE, node_id=str(track_id),
                    data=node_data, frequency_multiplier=1)
        node.first_timestep = start_frame
        scene.nodes.append(node)

    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/OPV2V")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--noise_augment", type=int, default=3,
                        help="Number of noisy copies per scenario (train only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"data/prediction/Trajectron/{args.mode}.pkl"

    rng = np.random.RandomState(args.seed)
    dataset = OPV2VDataset(root_path=args.root, mode=args.mode, dataset_name="OPV2V")

    env = Environment(node_type_list=['VEHICLE'], standardization=STANDARDIZATION)
    env.attention_radius = {(env.NodeType.VEHICLE, env.NodeType.VEHICLE): 30.0}
    env.robot_type = env.NodeType.VEHICLE

    n_scenarios = dataset.case_number("scenario")
    is_train = (args.mode == "train")

    logging.info(f"Mode={args.mode}, {n_scenarios} scenarios, "
                 f"noise_augment={args.noise_augment if is_train else 0}")

    scenes = []
    total_nodes = 0

    for sid in range(n_scenarios):
        trajectories, total_frames = extract_gt_trajectories_from_scenario(dataset, sid)
        if not trajectories:
            continue

        scene = build_scene(trajectories, f"opv2v_{sid:04d}", total_frames, env)
        if len(scene.nodes) > 0:
            scenes.append(scene)
            total_nodes += len(scene.nodes)

        if is_train:
            for aug_i in range(args.noise_augment):
                noisy_traj = add_noise_to_trajectories(trajectories, rng)
                scene_aug = build_scene(noisy_traj, f"opv2v_{sid:04d}_aug{aug_i}",
                                        total_frames, env)
                if len(scene_aug.nodes) > 0:
                    scenes.append(scene_aug)
                    total_nodes += len(scene_aug.nodes)

        if (sid + 1) % 10 == 0 or sid == n_scenarios - 1:
            logging.info(f"Processed {sid+1}/{n_scenarios} scenarios, "
                         f"{len(scenes)} scenes, {total_nodes} nodes")

    env.scenes = scenes
    logging.info(f"Total: {len(scenes)} scenes, {total_nodes} nodes")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
    logging.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
