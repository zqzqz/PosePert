"""
Checks for the shift attack's yaw perturbation. No GPU or dataset needed.

    python test/test_shift_rotation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mvp.attack.shift_rotation import (SHIFT_ROTATION_DEG, apply_shift,
                                       sample_shift_rotation)

PASS, FAIL = [], []


def check(name, fn):
    try:
        info = fn()
        PASS.append(name)
        print("  [PASS] %-38s %s" % (name, info or ""))
    except Exception as e:
        FAIL.append((name, repr(e)))
        print("  [FAIL] %-38s %r" % (name, e))


BBOX = np.array([1.0, 2.0, 3.0, 4.5, 2.0, 1.6, 0.5])


def t_sampler_range():
    r = np.random.RandomState(0)
    xs = [np.degrees(sample_shift_rotation(r)) for _ in range(20000)]
    assert max(abs(min(xs)), abs(max(xs))) <= SHIFT_ROTATION_DEG + 1e-9, (min(xs), max(xs))
    assert min(xs) < -SHIFT_ROTATION_DEG * 0.95, "range not covered below"
    assert max(xs) > SHIFT_ROTATION_DEG * 0.95, "range not covered above"
    assert abs(np.mean(xs)) < 0.5, "not centred on zero"
    return "%.2f..%.2f deg, mean %.3f" % (min(xs), max(xs), np.mean(xs))


def t_sampler_is_seedable():
    a = [sample_shift_rotation(np.random.RandomState(7)) for _ in range(3)]
    b = [sample_shift_rotation(np.random.RandomState(7)) for _ in range(3)]
    assert a == b, "same seed gave different draws"
    return "same seed reproduces"


def t_apply_shift_translation():
    ao = {"shift_direction": 0.0, "shift_distance": 2.0, "rotation": 0.0}
    out = apply_shift(BBOX, ao)
    assert abs(out[0] - (BBOX[0] + 2.0)) < 1e-9
    assert abs(out[1] - BBOX[1]) < 1e-9
    return "dx=+2.0 along direction 0"


def t_apply_shift_rotation():
    ao = {"shift_direction": 0.0, "shift_distance": 0.0, "rotation": np.radians(10.0)}
    out = apply_shift(BBOX, ao)
    assert abs(np.degrees(out[6] - BBOX[6]) - 10.0) < 1e-9, out[6]
    return "yaw +10 deg"


def t_apply_shift_does_not_mutate_input():
    before = BBOX.copy()
    apply_shift(BBOX, {"shift_direction": 1.0, "shift_distance": 3.0, "rotation": 0.3})
    assert np.array_equal(BBOX, before), "input bbox was modified in place"
    return "input untouched"


def t_missing_rotation_key_is_zero():
    """Cases from the shipped pkl files predate the rotation field in some paths."""
    out = apply_shift(BBOX, {"shift_direction": 0.0, "shift_distance": 1.0})
    assert out[6] == BBOX[6]
    return "defaults to no rotation"


def t_extent_and_z_untouched():
    out = apply_shift(BBOX, {"shift_direction": 1.2, "shift_distance": 1.0,
                             "rotation": np.radians(-8)})
    assert np.array_equal(out[2:6], BBOX[2:6]), "z or extent changed"
    return "z and extent preserved"


for _name, _fn in sorted(globals().items()):
    if _name.startswith("t_"):
        check(_name[2:], _fn)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
sys.exit(1 if FAIL else 0)
