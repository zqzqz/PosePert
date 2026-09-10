"""
3D Kalman filter for bounding box tracking.

State: [x, y, z, theta, l, w, h, vx, vy, vz, d_theta]
Measurement: [x, y, z, theta, l, w, h]

Uses constant velocity model with heading rate.
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBox3D:
    """Kalman filter for a single 3D bounding box."""

    count = 0

    def __init__(self, bbox):
        """
        Args:
            bbox: [x, y, z, l, w, h, theta] initial detection
        """
        # State: [x, y, z, theta, l, w, h, vx, vy, vz, d_theta]
        self.kf = KalmanFilter(dim_x=11, dim_z=7)

        # State transition (constant velocity)
        self.kf.F = np.eye(11)
        self.kf.F[0, 7] = 1   # x += vx
        self.kf.F[1, 8] = 1   # y += vy
        self.kf.F[2, 9] = 1   # z += vz
        self.kf.F[3, 10] = 1  # theta += d_theta

        # Measurement matrix: observe [x, y, z, theta, l, w, h]
        self.kf.H = np.zeros((7, 11))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # z
        self.kf.H[3, 3] = 1  # theta
        self.kf.H[4, 4] = 1  # l
        self.kf.H[5, 5] = 1  # w
        self.kf.H[6, 6] = 1  # h

        # Measurement noise
        self.kf.R *= 0.1
        self.kf.R[3, 3] = 0.01  # theta more precise

        # Process noise
        self.kf.Q[7:, 7:] *= 0.01  # velocity uncertainty
        self.kf.Q[4:7, 4:7] *= 0.001  # size nearly constant

        # Initial covariance
        self.kf.P[7:, 7:] *= 100  # high uncertainty on initial velocity
        self.kf.P *= 10

        # Initialize state: [x, y, z, theta, l, w, h, 0, 0, 0, 0]
        self.kf.x[:7, 0] = bbox  # x, y, z, l, w, h, theta -> x, y, z, theta, l, w, h
        self._reorder_init(bbox)

        self.time_since_update = 0
        self.id = KalmanBox3D.count
        KalmanBox3D.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def _reorder_init(self, bbox):
        """bbox is [x,y,z,l,w,h,theta], state is [x,y,z,theta,l,w,h,...]"""
        self.kf.x[0, 0] = bbox[0]  # x
        self.kf.x[1, 0] = bbox[1]  # y
        self.kf.x[2, 0] = bbox[2]  # z
        self.kf.x[3, 0] = bbox[6]  # theta
        self.kf.x[4, 0] = bbox[3]  # l
        self.kf.x[5, 0] = bbox[4]  # w
        self.kf.x[6, 0] = bbox[5]  # h

    def predict(self):
        self.kf.predict()
        # Wrap theta to [-pi, pi]
        self.kf.x[3, 0] = self._wrap_angle(self.kf.x[3, 0])
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox):
        """Update with [x, y, z, l, w, h, theta] detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Convert to measurement format [x, y, z, theta, l, w, h]
        z = np.array([bbox[0], bbox[1], bbox[2], bbox[6],
                       bbox[3], bbox[4], bbox[5]])

        # Handle angle wrapping
        predicted_theta = self.kf.x[3, 0]
        z[3] = self._nearest_angle(z[3], predicted_theta)

        self.kf.update(z)
        self.kf.x[3, 0] = self._wrap_angle(self.kf.x[3, 0])

    def get_state(self):
        """Return [x, y, z, l, w, h, theta]."""
        s = self.kf.x
        return np.array([s[0, 0], s[1, 0], s[2, 0],
                         s[4, 0], s[5, 0], s[6, 0], s[3, 0]])

    def get_velocity(self):
        """Return [vx, vy, vz]."""
        return np.array([self.kf.x[7, 0], self.kf.x[8, 0], self.kf.x[9, 0]])

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _nearest_angle(a, ref):
        """Adjust angle a to be closest to ref (handle wrapping)."""
        diff = a - ref
        while diff > np.pi:
            a -= 2 * np.pi
            diff = a - ref
        while diff < -np.pi:
            a += 2 * np.pi
            diff = a - ref
        return a
