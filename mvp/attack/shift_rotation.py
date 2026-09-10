"""
Yaw perturbation for the shift attack's spoofed bounding box.

The shift attack moves a target object by `shift_distance` along `shift_direction`
and additionally rotates it by `rotation` radians. Rotation matters because a pure
translation leaves the spoofed box axis-aligned with the real one, which is both
less representative of a real mis-detection and easier for a consistency check to
match against the original.

`attack_opts["rotation"]` carries the per-case value in radians. The shipped
`lidar_shift.pkl` files were produced by an older generator that stored 0.0 for
every case, so every published number was measured with no rotation; the
generators below now sample from this range instead.

Range: +/- SHIFT_ROTATION_DEG degrees, uniform. Small enough that the spoofed pose
stays physically plausible for a vehicle on a lane, large enough to change which
box a matcher associates.
"""
import numpy as np

SHIFT_ROTATION_DEG = 10.0
SHIFT_ROTATION_RAD = np.radians(SHIFT_ROTATION_DEG)


def sample_shift_rotation(rng=None):
    """Draw a yaw perturbation in radians, uniform over +/- SHIFT_ROTATION_DEG.

    Pass a seeded ``numpy.random.RandomState``/``Generator`` for reproducible test
    case generation; with ``None`` the global numpy stream is used.
    """
    draw = (rng or np.random).uniform
    return float(draw(-SHIFT_ROTATION_RAD, SHIFT_ROTATION_RAD))


def apply_shift(bbox, attack_opts):
    """Return a copy of `bbox` translated and rotated per `attack_opts`.

    Single definition of the spoofed target pose. `_get_bboxes` in the voxelwise and
    intermediate attackers apply the same three lines inline; the ray-cast cache
    builder used to apply only the two translation terms, so a non-zero rotation in
    the test case silently did not reach the rendered point cloud.
    """
    out = np.copy(bbox).astype(np.float64)
    out[0] += np.cos(attack_opts["shift_direction"]) * attack_opts["shift_distance"]
    out[1] += np.sin(attack_opts["shift_direction"]) * attack_opts["shift_distance"]
    out[6] += attack_opts.get("rotation", 0.0)
    return out
