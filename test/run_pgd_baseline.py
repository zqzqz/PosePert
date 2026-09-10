"""
PGD baseline evaluation using test_shift_attack.py's proven code pattern.

Usage:
  CUDA_VISIBLE_DEVICES=2 python test/run_pgd_baseline.py --model pointpillar
"""
import os, sys, pickle, copy, numpy as np, torch, time, logging, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.join(os.path.dirname(__file__), "..")

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_shift_intermediate_attacker import LidarShiftIntermediateAttacker
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.iou import iou3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pointpillar', choices=['pointpillar', 'v2vnet', 'cobevt'])
    parser.add_argument('--n_cases', type=int, default=None)
    args = parser.parse_args()

    from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
    warp_patches = _apply_warp_patches()

    dataset_name = "OPV2V"
    dataset = OPV2VDataset(root_path=os.path.join(root, f"data/{dataset_name}"),
                           mode="test", dataset_name=dataset_name)
    perception = build_perception(args.model)

    # PGD baseline: online=True, sync=1 (async), init=True, step=2
    attacker = LidarShiftIntermediateAttacker(
        perception, dataset, learn_rate=0.2, step=2,
        sync=1, init=True, online=True, aggr=1, attn="none", debug=False)

    with open(os.path.join(root, 'data/OPV2V/attack/lidar_shift.pkl'), 'rb') as f:
        test_cases = pickle.load(f)

    n_cases = args.n_cases or len(test_cases)
    results = []

    for ci in range(min(n_cases, len(test_cases))):
        tc = test_cases[ci]
        ao = tc['attack_opts']
        meta = tc['attack_meta']
        case_id = meta['case_id']
        attacker_id = ao['attacker_vehicle_id']
        victim_id = ao['victim_vehicle_id']

        try:
            case = dataset.get_case(case_id, tag='multi_frame', use_lidar=True)
            frame9 = case[9]

            if attacker_id not in frame9 or victim_id not in frame9:
                continue

            # Set up attack_opts as test_shift_attack does
            attack_opts = copy.deepcopy(ao)
            attack_opts['victim_vehicle_id'] = victim_id
            # online=True with sync=1: optimize on frames 0-8, apply on frame 9
            attack_opts['frame_ids'] = [i for i in range(10)]

            # init=True: load pre-computed ray tracing data
            if attacker.init:
                early_dir = os.path.join(root, "data/OPV2V/multi_frame/attack/lidar_shift_early_Sampled_dense1")
                init_path = os.path.join(early_dir, f"{ci:06d}", "attack_info.pkl")
                if os.path.exists(init_path):
                    init_info = pickle.load(open(init_path, 'rb'))
                    attack_opts["attack_info"] = [
                        init_info[fid].get(attacker_id, {}).get("lidar_update", {})
                        for fid in range(len(init_info))
                    ]
                else:
                    # Skip cases without pre-computed init data
                    continue

            t0 = time.time()
            attack_case, attack_info = attacker.run(case, attack_opts)
            elapsed = time.time() - t0

            # Extract predictions (same as analyze_attacks)
            pred_bboxes = attack_info[9][victim_id]["pred_bboxes"]
            pred_scores = attack_info[9][victim_id]["pred_scores"]

            # Target bbox in victim frame (same as analyze_attacks line 313)
            attack_bbox_orig = bbox_map_to_sensor(
                bbox_sensor_to_map(meta["bboxes"][-1], frame9[attacker_id]["lidar_pose"]),
                frame9[victim_id]["lidar_pose"])
            attack_bbox_tgt = bbox_map_to_sensor(
                bbox_sensor_to_map(meta["new_bboxes"][-1], frame9[attacker_id]["lidar_pose"]),
                frame9[victim_id]["lidar_pose"])

            # Compute IoU with target
            best_iou_tgt = 0.0
            best_conf = 0.0
            for pb, ps in zip(pred_bboxes, pred_scores):
                iou = iou3d(pb, attack_bbox_tgt)
                if iou > best_iou_tgt:
                    best_iou_tgt = iou
                    best_conf = float(ps)

            results.append({
                'case_idx': ci, 'case_id': case_id,
                'iou_tgt': best_iou_tgt, 'conf': best_conf,
                'n_dets': len(pred_bboxes), 'time': elapsed,
            })

            if (ci + 1) % 10 == 0 or ci < 5:
                logger.info(f"Case {ci}: IoU={best_iou_tgt:.3f}, conf={best_conf:.3f}, "
                            f"dets={len(pred_bboxes)}, time={elapsed:.1f}s")

        except KeyboardInterrupt:
            break
        except Exception:
            logger.warning(f"Case {ci} failed: {traceback.format_exc()}")

    # Summary
    n = len(results)
    ious = np.array([r['iou_tgt'] for r in results])
    confs = np.array([r['conf'] for r in results])
    times = np.array([r['time'] for r in results])

    logger.info(f"\n{'='*60}")
    logger.info(f"PGD Baseline: {args.model}, {n} cases")
    logger.info(f"  %S(.5)={100*(ious>0.5).mean():.1f}%, %S(.7)={100*(ious>0.7).mean():.1f}%")
    logger.info(f"  AvgIoU={ious.mean():.3f}, AvgConf={confs.mean():.3f}")
    logger.info(f"  AvgTime={times.mean():.1f}s")

    result_dir = f'results_paper/PGD_{args.model}'
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved to {result_dir}")
