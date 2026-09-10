"""
Run CAD on normal (unattacked) detections to get true FP distribution.

For each test case, run normal perception and compute CAD spoof area
for every detection. This gives the normal spoof score distribution
needed for proper CAD ROC curves.

Usage:
  python test/run_cad_normal.py --model pointpillar --gpu 0
  python test/run_cad_normal.py --model v2vnet --gpu 0
  python test/run_cad_normal.py --model cobevt --gpu 0
"""
import os, sys, argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=1)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

import pickle, numpy as np, time, logging, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.defense.perception_defender import PerceptionDefender
from mvp.data.util import bbox_sensor_to_map
from mvp.util import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['pointpillar', 'v2vnet', 'cobevt'])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--n_cases', type=int, default=None)
    args = parser.parse_args()

    test_pkl = os.path.join(root, 'data/OPV2V/attack/lidar_shift.pkl')
    occ_dir = os.path.join(root, 'data/OPV2V/normal')
    out_dir = {
        'pointpillar': 'results_paper/D_pp_attentive',
        'v2vnet': 'results_paper/D_v2vnet',
        'cobevt': 'results_paper/D_cobevt',
    }[args.model]
    os.makedirs(out_dir, exist_ok=True)

    warp_patches = _apply_warp_patches()
    perception = build_perception(args.model)
    perception.model.eval()
    dataset = OPV2VDataset(root_path=os.path.join(root, 'data/OPV2V'),
                            mode='test', dataset_name='OPV2V')

    cad = PerceptionDefender()

    with open(test_pkl, 'rb') as f:
        attacks = pickle.load(f)

    n_cases = args.n_cases or len(attacks)
    results = []
    t0 = time.time()

    for ci in range(min(n_cases, len(attacks))):
        meta = attacks[ci]['attack_meta']
        try:
            case = dataset.get_case(meta['case_id'], tag='multi_frame', use_lidar=True)
            frame = case[9]
            vi = meta['victim_vehicle_id']

            if vi not in frame:
                continue

            # Run NORMAL perception (no attack)
            set_seed(42, set_python=False, set_numpy=False, set_torch=True)
            pred_bboxes, pred_scores = perception.run(frame, vi)

            if len(pred_bboxes) == 0:
                results.append({'case_idx': ci, 'n_dets': 0, 'spoof_areas': [], 'max_spoof': 0})
                logger.info(f"  case {ci}: 0 dets, max_spoof=0.0")
                continue

            # Load occupancy data
            occ_path = os.path.join(occ_dir, f'{meta["case_id"]:06d}.pkl')
            if not os.path.exists(occ_path):
                continue

            occ_data = pickle.load(open(occ_path, 'rb'))
            frame_normal = frame.copy()
            for vid in frame_normal:
                if vid in occ_data:
                    frame_normal[vid].update(occ_data[vid])
            frame_normal[vi]['pred_bboxes'] = pred_bboxes

            # Run CAD on normal detections
            _, _, metrics = cad.run({9: frame_normal}, {'frame_ids': [9],
                'vehicle_ids': [v for v in frame_normal if isinstance(v, int)]})

            spoof_list = metrics[9].get(vi, {}).get('spoof', [])
            spoof_areas = [m[1] for m in spoof_list]

            entry = {
                'case_idx': ci,
                'case_id': meta['case_id'],
                'n_dets': len(pred_bboxes),
                'spoof_areas': spoof_areas,
                'max_spoof': max(spoof_areas) if spoof_areas else 0,
            }
            results.append(entry)

            logger.info(f"  case {ci}: {len(pred_bboxes)} dets, "
                       f"max_spoof={entry['max_spoof']:.2f}, "
                       f"n_spoof>2.7={sum(1 for s in spoof_areas if s > 2.7)}")

        except Exception as e:
            logger.warning(f"Case {ci}: {traceback.format_exc()}")
            continue

    # Save
    out_path = os.path.join(out_dir, 'cad_normal_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

    # Summary
    all_spoof = [s for r in results for s in r['spoof_areas']]
    max_per_case = [r['max_spoof'] for r in results]
    logger.info(f"\n{'='*60}")
    logger.info(f"CAD Normal: {args.model}, {len(results)} cases")
    logger.info(f"  Per-detection spoof: n={len(all_spoof)}, mean={np.mean(all_spoof):.2f}, "
                f"p95={np.percentile(all_spoof, 95):.2f}, max={max(all_spoof):.2f}")
    logger.info(f"  Per-case max spoof: mean={np.mean(max_per_case):.2f}, "
                f"p95={np.percentile(max_per_case, 95):.2f}")
    logger.info(f"  FPR@2.7: {sum(1 for m in max_per_case if m > 2.7)/len(max_per_case)*100:.1f}%")
    logger.info(f"  Time: {time.time()-t0:.0f}s")
    logger.info(f"  Saved to {out_path}")
