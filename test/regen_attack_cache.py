"""Regenerate the ray-cast attack cache (data/<dataset>/attack_cache_paper/).

Each entry holds the spoofed point cloud plus the original and target boxes for one
test case. run_eval.py and run_defense_eval_full.py read this cache rather than
re-casting rays, so the cache -- not the consuming script -- is where the spoofed
target pose is fixed. Regenerate it after changing test cases or the shift
parameters, otherwise the boxes and the rendered points disagree.

Ray casting uses no ground plane (uneven terrain) and subsamples to <=1200 points.

Usage:
  DATASET_NAME=V2X-Real python test/regen_attack_cache.py --gpu 0
  DATASET_NAME=OPV2V     python test/regen_attack_cache.py --gpu 0
"""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=1)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
os.environ.setdefault('DATASET_NAME', 'V2X-Real')

import pickle, numpy as np, time, logging

sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.data.util import sort_lidar_points
from mvp.attack.shift_rotation import apply_shift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    cache_dir = 'data/V2X-Real/attack_cache_paper'
    backup_dir = 'data/V2X-Real/attack_cache_paper_old'

    # Backup old cache
    if os.path.exists(cache_dir) and not os.path.exists(backup_dir):
        os.rename(cache_dir, backup_dir)
        logger.info(f"Backed up old cache to {backup_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    with open('data/V2X-Real/attack/lidar_shift.pkl', 'rb') as f:
        attacks = pickle.load(f)
    logger.info(f"Loaded {len(attacks)} V2X-Real test cases")

    # Build perception just to get the attacker with correct _generate_spoof_pcd
    perception = OpencoodPerception(
        fusion_method='intermediate', model_name='pointpillar', dataset_name='V2X-Real')
    dataset = OPV2VDataset(root_path='data/V2X-Real', mode='test', dataset_name='V2X-Real')
    attacker = LidarShiftVoxelwiseAttacker(perception, dataset, beta=1.0)

    t_start = time.time()
    n_done = 0

    for ci, a in enumerate(attacks):
        cache_path = os.path.join(cache_dir, f'{ci:06d}.pkl')
        if os.path.exists(cache_path):
            n_done += 1
            continue

        meta = a['attack_meta']
        ao = a['attack_opts']
        try:
            case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
            frame = case[0]  # V2X-Real uses frame 0
            ai = ao['attacker_vehicle_id']

            if ai not in frame:
                logger.warning(f"Case {ci}: attacker {ai} not in frame")
                continue

            obj_id = ao['object_id']
            if obj_id not in frame[ai]['object_ids']:
                logger.warning(f"Case {ci}: object {obj_id} not found")
                continue

            obj_idx = frame[ai]['object_ids'].index(obj_id)
            bbox_orig = frame[ai]['gt_bboxes'][obj_idx].copy()
            # apply_shift applies translation AND the yaw term. This used to inline the
            # two translation lines only, so attack_opts['rotation'] never reached the
            # rendered point cloud and every cached case was effectively unrotated.
            bbox_tgt = apply_shift(bbox_orig, ao)

            pcd = frame[ai]['lidar']

            np.random.seed(ci)
            spoof_pcd = attacker._generate_spoof_pcd(pcd.copy(), bbox_tgt, bbox_orig)

            entry = {
                'spoof_pcd': spoof_pcd,
                'bbox_orig': bbox_orig,
                'bbox_tgt': bbox_tgt,
                'case_id': meta['case_id'],
                'attacker_vehicle_id': ai,
                'victim_vehicle_id': ao['victim_vehicle_id'],
                'attack_idx': ci,
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)

            n_pts = len(spoof_pcd)
            n_orig = len(pcd)
            n_diff = n_pts - n_orig
            n_done += 1

            if n_done % 20 == 0 or ci < 5:
                logger.info(f"  [{n_done}/{len(attacks)}] case {ci}: "
                            f"orig={n_orig} pts, spoof={n_pts} pts (delta={n_diff:+d}), "
                            f"{time.time()-t_start:.0f}s")

        except Exception as e:
            logger.warning(f"Case {ci} failed: {e}")

    logger.info(f"Done: {n_done}/{len(attacks)} cases, {time.time()-t_start:.0f}s")
    logger.info(f"Cache saved to {cache_dir}")


if __name__ == '__main__':
    main()
