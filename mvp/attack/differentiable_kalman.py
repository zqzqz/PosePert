import torch
import numpy as np


class DifferentiableKalmanSurrogate:
    """
    Differentiable surrogate for AB3DMOT's Kalman filter (x,y dimensions only).

    Matches the actual AB3DMOT behavior where t_last_predict is never updated
    (stays at 0), so dt = absolute_time for each predict call. F and Q are
    recomputed per-frame with the correct dt.

    P and K are measurement-independent, so we pre-compute them in numpy.
    The forward pass uses torch ops so gradients flow through:
        x_new = x_pred + K @ (z - H @ x_pred)
    """

    def __init__(self):
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # R: measurement noise (kalman_filter.py:37)
        self.R = np.diag([1.0, 1.0])

    def _make_F(self, dt):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    def _make_Q(self, dt):
        # kalman_filter.py:65-67: Q = (diag([3,3,3,...,10,10,10]) * dt)^2
        return np.diag([
            (3 * dt) ** 2,
            (3 * dt) ** 2,
            (10 * dt) ** 2,
            (10 * dt) ** 2,
        ])

    def precompute_gains(self, frame_times, prior_P):
        """
        Pre-compute F, K for each attack frame using the actual dt values.

        Args:
            frame_times: list of absolute times for each attack frame
            prior_P: 4x4 numpy covariance from the tracker
        Returns:
            list of (F, K) tuples per frame
        """
        P = prior_P.copy()
        results = []
        for t in frame_times:
            F = self._make_F(t)
            Q = self._make_Q(t)
            P_pred = F @ P @ F.T + Q
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            P = (np.eye(4) - K @ self.H) @ P_pred
            results.append((F, K))
        return results

    def forward(self, prior_state, measurements, frame_times, prior_P,
                device="cuda:0"):
        """
        Run measurements through differentiable Kalman filter.

        Args:
            prior_state: numpy array [x, y, vx, vy]
            measurements: torch tensor (n_frames, 2) with requires_grad
            frame_times: list of absolute times (matching AB3DMOT's t_last_predict=0)
            prior_P: 4x4 numpy covariance
            device: torch device

        Returns:
            filtered_positions: torch tensor (n_frames, 2)
        """
        n_frames = measurements.shape[0]
        fk_pairs = self.precompute_gains(frame_times, prior_P)

        x = torch.tensor(prior_state, dtype=torch.float64, device=device)
        H_t = torch.tensor(self.H, dtype=torch.float64, device=device)
        filtered = []

        for t in range(n_frames):
            F_np, K_np = fk_pairs[t]
            F_t = torch.tensor(F_np, dtype=torch.float64, device=device)
            K_t = torch.tensor(K_np, dtype=torch.float64, device=device)

            x_pred = F_t @ x
            z = measurements[t].to(torch.float64)
            innovation = z - H_t @ x_pred
            x = x_pred + K_t @ innovation
            filtered.append(x[:2])

        return torch.stack(filtered).to(measurements.dtype)

    def extract_prior_state(self, tracks, target_track_id):
        """
        Extract [x, y, vx, vy] and P from an Ab3dmotTracker for a given track ID.

        Args:
            tracks: Ab3dmotTracker object (from case[frame][vehicle]["tracks"])
            target_track_id: integer track ID

        Returns:
            state: numpy array [x, y, vx, vy]
            P: 4x4 numpy covariance submatrix
        """
        for trk in tracks.tracker.trackers:
            if trk.id + 1 == target_track_id:
                kf_x = trk.kf.x.flatten()
                state = np.array([kf_x[0], kf_x[1], kf_x[7], kf_x[8]])
                kf_P = np.array(trk.kf.P)
                idx = [0, 1, 7, 8]
                P = kf_P[np.ix_(idx, idx)]
                return state, P
        return None, None
