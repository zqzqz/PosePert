"""
Example: Run end-to-end scenario attack.

Demonstrates the observe-predict-plan-execute loop:
  1. Load a scenario test case
  2. Run the scenario attacker over multiple frames
  3. Measure trajectory prediction deflection and danger metrics

Usage:
  CUDA_VISIBLE_DEVICES=0 python test/test_scenario_attack.py
"""
import os, sys, pickle, copy, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.perturbation_train import build_perception, _apply_warp_patches
from mvp.attack.scenario_shift_movein_attacker import ScenarioShiftMoveinAttacker
from mvp.attack.perturbation_network import PerturbationNetwork
from mvp.util import set_seed


def main():
    # --- Setup ---
    warp_patches = _apply_warp_patches()
    perception = build_perception('pointpillar')
    perception.model.eval()
    dataset = OPV2VDataset(root_path='data/OPV2V', mode='test', dataset_name='OPV2V')

    # Load scenario test cases
    scenario_pkl = 'data/OPV2V/test_scenario_attacks.pkl'
    if not os.path.exists(scenario_pkl):
        print(f"Scenario test cases not found at {scenario_pkl}")
        print("Generate them first using the data preparation scripts.")
        return

    with open(scenario_pkl, 'rb') as f:
        scenario_cases = pickle.load(f)

    print(f"Loaded {len(scenario_cases)} scenario test cases")

    # Load PertNet
    pertnet_path = 'models/perturbation_net_paper_pointpillar/perturbation_net_ep35.pt'
    pertnet = None
    if os.path.exists(pertnet_path):
        ckpt = torch.load(pertnet_path, map_location='cpu')
        pertnet = PerturbationNetwork(
            feature_channels=ckpt['feature_channels'],
            geo_channels=ckpt['geo_channels']).to(perception.device)
        pertnet.load_state_dict(ckpt['model_state'])
        pertnet.eval()
        print(f"Loaded PertNet from {pertnet_path}")

    # Initialize scenario attacker
    attacker = ScenarioShiftMoveinAttacker(
        perception, dataset,
        beta=2.0,
        pertnet=pertnet,
        pertnet_epsilon=10.0,
        location_bound=0.5,     # max 0.5m shift per frame
        opt_iterations=5,       # PGD iterations per frame
        use_uncertainty=True,   # expectation over transformation
        attack_type='blackbox', # 'whitebox' or 'blackbox'
    )

    # Run on first case
    sc = scenario_cases[0]
    case_id = sc['case_id']
    print(f"\nRunning scenario attack on case {case_id}...")

    case = dataset.get_case(case_id, tag='multi_frame', use_lidar=True)

    try:
        result = attacker.run(case, sc)

        # Extract metrics
        metrics = result['metrics']
        print(f"\n{'='*50}")
        print(f"Scenario Attack Results:")
        print(f"  Real shift:  {metrics['real_shift']:.2f} m")
        print(f"  Pred ADE:    {metrics['pred_ade']:.2f} m")
        print(f"  Pred FDE:    {metrics['pred_fde']:.2f} m")
        print(f"  Min distance to victim: {metrics['min_pred_dist_to_victim']:.2f} m")
        if metrics['min_pred_dist_to_victim'] < 3.0:
            print(f"  ** DANGER: predicted collision risk! **")
        print(f"  Elapsed:     {metrics['elapsed']:.1f} s")
        print(f"{'='*50}")

    except Exception as e:
        print(f"Scenario attack failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
