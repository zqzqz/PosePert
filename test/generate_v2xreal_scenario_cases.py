"""
Generate V2X-Real scenario test cases with proper frame counts and target selection.

Key improvements over previous version:
- Uses tag='scenario' to get full frame sequences (98-283 frames, like OPV2V's 60+)
- Selects target with highest attack fitness (closest predicted collision with victim),
  not just best detection score
- Uses history=20 frames like OPV2V (not 7)
- Slices long scenarios into windows with sufficient frames

Usage:
  CUDA_VISIBLE_DEVICES=1 python test/generate_v2xreal_scenario_cases.py --gpu 1
"""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=1)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
os.environ['DATASET_NAME'] = 'V2X-Real'

import pickle, copy, time, logging, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
root = os.path.join(os.path.dirname(__file__), '..')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.scenario_shift_movein_attacker import ScenarioShiftMoveinAttacker
from mvp.attack.lidar_shift_voxelwise_attacker import LidarShiftVoxelwiseAttacker
from mvp.data.util import bbox_sensor_to_map

HISTORY_FRAMES = 20
ATTACK_FRAMES = 3
PREDICT_FRAMES = 20
TOTAL_FRAMES = HISTORY_FRAMES + ATTACK_FRAMES + PREDICT_FRAMES  # 43
MIN_CASE_FRAMES = 43
WINDOW_SIZE = 60
WINDOW_STRIDE = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--max_cases', type=int, default=100)
    parser.add_argument('--output', type=str,
                        default='data/V2X-Real/test_scenario_attacks.pkl')
    args = parser.parse_args()

    dataset = OPV2VDataset(root_path=os.path.join(root, 'data/V2X-Real'),
                           mode='test', dataset_name='V2X-Real')

    perception = OpencoodPerception(
        fusion_method='intermediate',
        model_name='pointpillar',
        dataset_name='V2X-Real',
    )
    voxel_attacker = LidarShiftVoxelwiseAttacker(
        perception, dataset, beta=1.2)

    attacker = ScenarioShiftMoveinAttacker(
        dataset_name='V2X-Real',
        perception_attacker=voxel_attacker,
        perturbation_type='location',
        optimization_type='sign',
        use_uncertainty=False,
        history_num_frames=HISTORY_FRAMES,
        attack_num_frames=ATTACK_FRAMES,
        predict_num_frames=PREDICT_FRAMES,
        attack_type='blackbox',
    )
    attacker.location_bound = 0.5

    def is_vehicle(vid):
        return isinstance(vid, int) and vid < 0

    n_scenarios = len(dataset.cases['scenario'])
    logger.info(f"V2X-Real: {n_scenarios} scenarios")

    all_candidates = []

    for si in range(n_scenarios):
        sc = dataset.cases['scenario'][si]
        scenario_id = sc['scenario_id']
        all_frame_ids = sc['frame_ids']
        vehicle_ids = dataset.meta[scenario_id]['vehicle_ids']
        v2v_ids = [v for v in vehicle_ids if is_vehicle(v)]

        if len(v2v_ids) < 2:
            logger.info(f"  Scenario {si} ({scenario_id}): skip, only {len(v2v_ids)} V2V vehicles")
            continue

        n_frames = len(all_frame_ids)
        logger.info(f"  Scenario {si} ({scenario_id}): {n_frames} frames, "
                     f"V2V vehicles: {v2v_ids}")

        if n_frames < MIN_CASE_FRAMES:
            logger.info(f"    Skip: too few frames ({n_frames} < {MIN_CASE_FRAMES})")
            continue

        windows = []
        for start in range(0, n_frames - WINDOW_SIZE + 1, WINDOW_STRIDE):
            windows.append((start, start + WINDOW_SIZE))
        if not windows:
            windows.append((0, min(n_frames, WINDOW_SIZE)))
        if windows[-1][1] < n_frames and n_frames - windows[-1][0] >= MIN_CASE_FRAMES:
            windows.append((n_frames - WINDOW_SIZE, n_frames))

        for win_start, win_end in windows:
            win_frame_ids = all_frame_ids[win_start:win_end]

            for attacker_id in v2v_ids:
                for victim_id in v2v_ids:
                    if attacker_id == victim_id:
                        continue

                    all_candidates.append({
                        'scenario_idx': si,
                        'scenario_id': scenario_id,
                        'frame_ids': win_frame_ids,
                        'vehicle_ids': vehicle_ids,
                        'attacker_vehicle_id': attacker_id,
                        'victim_vehicle_id': victim_id,
                        'window': (win_start, win_end),
                    })

    logger.info(f"\nTotal candidate windows: {len(all_candidates)}")

    scored_cases = []
    for ci, cand in enumerate(all_candidates):
        si = cand['scenario_idx']
        scenario_id = cand['scenario_id']
        attacker_id = cand['attacker_vehicle_id']
        victim_id = cand['victim_vehicle_id']
        win = cand['window']

        logger.info(f"\n--- Candidate {ci}/{len(all_candidates)}: "
                     f"scenario={si}, atk={attacker_id}, vic={victim_id}, "
                     f"window={win} ({len(cand['frame_ids'])} frames) ---")

        try:
            custom_meta = {
                'scenario_id': scenario_id,
                'frame_ids': cand['frame_ids'],
            }
            case = dataset.get_case_by_meta(custom_meta, tag='scenario',
                                            use_lidar=True)

            attack_opts_base = {
                'victim_vehicle_id': victim_id,
                'attacker_vehicle_id': attacker_id,
                'gt': False,
            }

            attacker.preprocess(case, attack_opts_base)

            attack_end = attacker.attack_end_frame_id
            atk_frame = case[attack_end][attacker_id]
            gt_object_ids = atk_frame.get('object_ids', [])
            candidate_targets = [oid for oid in gt_object_ids
                                 if oid not in [attacker_id, victim_id]]

            if not candidate_targets:
                logger.info(f"  No candidate GT objects visible to attacker")
                continue

            best_fitness = -float('inf')
            best_target_id = None
            best_vtid = None
            best_init_dist = None

            for tgt_id in candidate_targets:
                try:
                    opts_try = copy.deepcopy(attack_opts_base)
                    opts_try['target_id'] = tgt_id
                    seed = attacker.init_seed_default(case, opts_try)
                    fitness = float(attacker.attack_projected_fitness(
                        case, seed, simple=True))

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_target_id = tgt_id
                        best_vtid = seed.get('victim_target_track_id')

                        obs = case[attack_end][victim_id].get(
                            'observed_trajectories', {})
                        if best_vtid in obs and victim_id in obs:
                            target_pos = obs[best_vtid][-1, :2]
                            victim_pos = obs[victim_id][-1, :2]
                            best_init_dist = float(
                                np.linalg.norm(target_pos - victim_pos))
                except Exception as e:
                    logger.debug(f"    target {tgt_id} failed: {e}")
                    continue

            if best_target_id is None:
                logger.info(f"  All candidate targets failed")
                continue

            entry = {
                'case_id': si,
                'scenario_id': scenario_id,
                'frame_ids': cand['frame_ids'],
                'vehicle_ids': cand['vehicle_ids'],
                'attacker_vehicle_id': attacker_id,
                'victim_vehicle_id': victim_id,
                'target_id': best_target_id,
                'fitness': best_fitness,
                'init_dist': best_init_dist,
                'window': win,
            }
            scored_cases.append(entry)

            logger.info(f"  Best target: {best_target_id} (track {best_vtid}), "
                         f"fitness={best_fitness:.3f}, "
                         f"init_dist={best_init_dist:.1f}m"
                         if best_init_dist else
                         f"  Best target: {best_target_id}, "
                         f"fitness={best_fitness:.3f}")

        except Exception as e:
            logger.error(f"  Failed: {traceback.format_exc()}")
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Scored {len(scored_cases)} candidate cases")

    scored_cases.sort(key=lambda x: -x['fitness'])

    seen_keys = set()
    selected = []
    for entry in scored_cases:
        key = (entry['scenario_id'], entry['attacker_vehicle_id'],
               entry['victim_vehicle_id'], entry['window'])
        if key in seen_keys:
            continue

        selected.append(entry)
        seen_keys.add(key)
        if len(selected) >= args.max_cases:
            break

    logger.info(f"Selected {len(selected)} cases (deduped, sorted by fitness)")

    if selected:
        fitnesses = [e['fitness'] for e in selected]
        dists = [e['init_dist'] for e in selected if e['init_dist'] is not None]
        logger.info(f"  Fitness: min={min(fitnesses):.3f}, max={max(fitnesses):.3f}, "
                     f"mean={np.mean(fitnesses):.3f}")
        if dists:
            d = np.array(dists)
            logger.info(f"  Init dist: min={d.min():.1f}m, max={d.max():.1f}m, "
                         f"mean={d.mean():.1f}m, median={np.median(d):.1f}m")
            logger.info(f"    <5m: {(d<5).sum()}, <10m: {(d<10).sum()}, "
                         f"<15m: {(d<15).sum()}, <20m: {(d<20).sum()}")

    for e in selected:
        e.pop('fitness', None)
        e.pop('init_dist', None)
        e.pop('window', None)
        e.pop('scenario_idx', None)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(selected, f)
    logger.info(f"Saved {len(selected)} cases to {args.output}")


if __name__ == '__main__':
    main()
