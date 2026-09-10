"""
MADE defense for collaborative perception.

Implements Malicious Agent Detection from:
https://github.com/shengyin1224/MADE

MADE detects adversarial agents using dual-hypothesis testing:
1. Match Cost Test: Compare detections with/without each agent
2. Reconstruction Test: Autoencoder on feature residuals (optional, needs training)
3. Benjamini-Hochberg procedure for FDR control

Simplified implementation for our pipeline using test 1 (match cost) only,
which requires no training and directly detects if an agent's contribution
changes detections suspiciously.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
import logging

from mvp.tools.iou import iou3d


class BenjaminiHochberg:
    """
    Benjamini-Hochberg procedure for multiple hypothesis testing.
    Controls False Discovery Rate (FDR) at level alpha.
    """
    def __init__(self, alpha=0.05, n_tests=1, dependent=True):
        self.alpha = alpha
        self.n_tests = n_tests
        self.dependent = dependent
        # Constant for dependent tests
        if dependent:
            self.const = sum(1.0 / (j + 1) for j in range(n_tests))
        else:
            self.const = 1.0
        self.calibration_dists = [None] * n_tests

    def calibrate(self, calibration_scores):
        """
        Set calibration distributions from clean (no-attack) data.

        Args:
            calibration_scores: list of arrays, one per test.
                Each array has shape (n_calibration_samples,).
        """
        assert len(calibration_scores) == self.n_tests
        self.calibration_dists = [np.sort(s) for s in calibration_scores]

    def conformal_pvalue(self, score, test_idx):
        """Compute conformal p-value for a score against calibration."""
        dist = self.calibration_dists[test_idx]
        if dist is None:
            return 1.0  # No calibration → cannot reject
        n = len(dist)
        # Number of calibration scores >= observed score
        n_ge = np.sum(dist >= score)
        return (n_ge + 1) / (n + 1)

    def test(self, scores):
        """
        Test if an agent is malicious.

        Args:
            scores: array of shape (n_tests,) — one score per test

        Returns:
            rejected: bool — True if detected as malicious
        """
        pvalues = [self.conformal_pvalue(scores[i], i) for i in range(self.n_tests)]

        # BH procedure
        fdr = self.alpha / self.const
        max_pv = max(pvalues)
        min_pv = min(pvalues)

        if self.n_tests == 1:
            return pvalues[0] <= fdr
        else:
            # For K tests: reject if max_pv ≤ fdr or min_pv ≤ fdr/K
            return max_pv <= fdr or min_pv <= fdr / self.n_tests


class MadeDefender:
    """
    MADE: Malicious Agent Detection for collaborative perception.

    Test 1 (Match Cost): Runs perception with and without each agent,
    compares detection results via IoU-based Hungarian matching.
    If an agent's contribution causes large detection changes, it's suspicious.

    Test 2 (Reconstruction, optional): Autoencoder on feature residuals.
    Requires pre-trained autoencoder — not included in this simplified version.
    """
    def __init__(self, perception, alpha=0.05, calibration_scores=None):
        """
        Args:
            perception: OpencoodPerception instance
            alpha: FDR level for BH test
            calibration_scores: pre-computed match cost scores on clean data
                If None, uses a fixed threshold (less principled but works)
        """
        self.perception = perception
        self.bh = BenjaminiHochberg(alpha=alpha, n_tests=1, dependent=False)
        if calibration_scores is not None:
            self.bh.calibrate([calibration_scores])
            self.calibrated = True
        else:
            self.calibrated = False
            self.fixed_threshold = 0.5  # fallback: flag if match cost > 0.5

    def compute_match_cost(self, multi_vehicle_case, ego_id):
        """
        Compute match cost for each non-ego agent.

        For each agent i ≠ ego:
        1. Run perception with ALL agents → detections_all
        2. Run perception WITHOUT agent i → detections_without_i
        3. Match cost = how much detections change when agent i is removed

        If removing agent i barely changes detections → agent is benign
        If removing agent i causes large changes → agent is influential (suspicious or important)

        Returns:
            match_costs: dict mapping agent_id → match cost score
        """
        from opencood.utils import box_utils

        # Full perception (all agents)
        pb_all, ps_all = self.perception.run(multi_vehicle_case, ego_id)

        match_costs = {}
        vehicle_ids = list(multi_vehicle_case.keys())

        for vid in vehicle_ids:
            if vid == ego_id:
                match_costs[vid] = 0.0
                continue

            # Perception without this agent
            case_without = {k: v for k, v in multi_vehicle_case.items() if k != vid}
            if len(case_without) < 1:
                match_costs[vid] = 0.0
                continue

            try:
                pb_without, ps_without = self.perception.run(case_without, ego_id)
            except Exception:
                match_costs[vid] = 0.0
                continue

            # Compute match cost via IoU matching
            cost = self._hungarian_match_cost(pb_all, pb_without)
            match_costs[vid] = cost

        return match_costs

    def _hungarian_match_cost(self, bboxes_a, bboxes_b):
        """
        Compute match cost between two sets of detections using IoU.

        Low cost = detections are similar (agent is benign)
        High cost = detections differ significantly (agent is suspicious)
        """
        if len(bboxes_a) == 0 and len(bboxes_b) == 0:
            return 0.0
        if len(bboxes_a) == 0 or len(bboxes_b) == 0:
            return 1.0  # One set is empty → maximum change

        n_a, n_b = len(bboxes_a), len(bboxes_b)

        # Compute IoU matrix
        iou_matrix = np.zeros((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                iou_matrix[i, j] = iou3d(bboxes_a[i], bboxes_b[j])

        # Greedy matching (simplified Hungarian)
        matched_ious = []
        used_b = set()
        for i in range(n_a):
            best_j = -1
            best_iou = 0
            for j in range(n_b):
                if j not in used_b and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j
            if best_j >= 0 and best_iou > 0.1:
                matched_ious.append(best_iou)
                used_b.add(best_j)

        # Match cost: 1 - average matched IoU, penalized by unmatched detections
        n_matched = len(matched_ious)
        n_total = max(n_a, n_b)
        if n_total == 0:
            return 0.0

        avg_iou = np.mean(matched_ious) if matched_ious else 0.0
        match_ratio = n_matched / n_total
        cost = 1.0 - avg_iou * match_ratio

        return cost

    def detect(self, multi_vehicle_case, ego_id):
        """
        Detect malicious agents.

        Args:
            multi_vehicle_case: dict of vehicle data for one frame
            ego_id: ego vehicle ID

        Returns:
            is_malicious: dict mapping agent_id → bool
            match_costs: dict mapping agent_id → float
        """
        match_costs = self.compute_match_cost(multi_vehicle_case, ego_id)

        is_malicious = {}
        for vid, cost in match_costs.items():
            if vid == ego_id:
                is_malicious[vid] = False
                continue

            if self.calibrated:
                is_malicious[vid] = self.bh.test(np.array([cost]))
            else:
                is_malicious[vid] = cost > self.fixed_threshold

        return is_malicious, match_costs

    def run_with_defense(self, multi_vehicle_case, ego_id):
        """
        Run perception with MADE defense: detect and remove malicious agents.

        Returns:
            pred_bboxes, pred_scores: detections after removing malicious agents
            defense_info: dict with match costs and detection results
        """
        is_malicious, match_costs = self.detect(multi_vehicle_case, ego_id)

        # Remove detected malicious agents
        clean_case = {vid: data for vid, data in multi_vehicle_case.items()
                      if not is_malicious.get(vid, False)}

        if len(clean_case) < 1:
            # All agents flagged — fall back to ego-only
            clean_case = {ego_id: multi_vehicle_case[ego_id]}

        pred_bboxes, pred_scores = self.perception.run(clean_case, ego_id)

        return pred_bboxes, pred_scores, {
            'match_costs': match_costs,
            'is_malicious': is_malicious,
            'n_removed': sum(1 for v in is_malicious.values() if v),
        }
