"""Tests for COMPASS metrics (pure functions, numpy)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import (brier_score, confusion, dice_from, hd95,
                         expected_calibration_error, iou_from, pixel_accuracy,
                         precision_from, recall_from, vacuity_mean)


def test_confusion_and_scores():
    gt = np.zeros((10, 10), dtype=int)
    gt[2:5, 2:5] = 1                      # 9 pixels
    pred = np.zeros((10, 10), dtype=int)
    pred[2:5, 3:6] = 1                    # 9 pixels, 6 overlap

    tp, fp, fn = confusion(pred, gt, 1)
    assert (tp, fp, fn) == (6, 3, 3)
    assert abs(dice_from(tp, fp, fn) - 2 / 3) < 1e-9   # 12/18
    assert abs(iou_from(tp, fp, fn) - 0.5) < 1e-9      # 6/12
    assert abs(precision_from(tp, fp, fn) - 2 / 3) < 1e-9
    assert abs(recall_from(tp, fp, fn) - 2 / 3) < 1e-9
    assert 0.9 < pixel_accuracy(pred, gt) < 1.0
    print("  [confusion/scores] OK")

    # empty conventions
    assert dice_from(0, 0, 0) == 1.0 and iou_from(0, 0, 0) == 1.0
    assert precision_from(0, 0, 3) == 0.0      # gt present, no prediction
    assert recall_from(0, 3, 0) == 1.0         # gt absent
    print("  [empty conventions] OK")


def test_hd95():
    gt = np.zeros((16, 16), dtype=bool)
    gt[4:8, 4:8] = True

    perfect = hd95(gt.copy(), gt)
    assert perfect == 0.0

    shifted = np.roll(gt, 2, axis=1)           # 2 px right (2 * 1.5 mm)
    d = hd95(shifted, gt)
    assert 0 < d <= 6.0, d                     # between 3mm and 6mm

    empty = np.zeros_like(gt)
    both_empty = hd95(empty, empty)
    assert np.isnan(both_empty)
    one_empty = hd95(empty, gt)
    assert one_empty > 0                        # diagonal penalty
    print(f"  [hd95] perfect=0 shifted={d:.2f}mm penalty={one_empty:.1f} OK")


def test_calibration():
    # perfectly calibrated: conf 0.7, 70% correct
    rng = np.random.default_rng(0)
    conf = np.full(1000, 0.7)
    correct = (rng.random(1000) < 0.7).astype(float)
    ece = expected_calibration_error(conf, correct)
    assert ece < 0.04, ece                     # ~ sampling noise

    # maximally miscalibrated: conf 0.9, 0% correct
    ece_bad = expected_calibration_error(np.full(100, 0.9), np.zeros(100))
    assert abs(ece_bad - 0.9) < 1e-9
    print(f"  [ece] good={ece:.4f} bad={ece_bad:.4f} OK")

    # brier: perfect / uniform
    probs = np.zeros((5, 4, 4))
    probs[2] = 1.0
    target = np.full((4, 4), 2)
    assert brier_score(probs, target) == 0.0
    uni = np.full((5, 4, 4), 0.2)
    assert abs(brier_score(uni, target) - (0.8 ** 2 + 4 * 0.04)) < 1e-9
    print("  [brier] OK")

    # vacuity: alpha0 = 10 uniform -> C/alpha0 = 0.5
    alpha = np.full((5, 4, 4), 2.0)
    assert abs(vacuity_mean(alpha) - 0.5) < 1e-9
    print("  [vacuity] OK")


if __name__ == "__main__":
    print("=== test_confusion_and_scores ===")
    test_confusion_and_scores()
    print("=== test_hd95 ===")
    test_hd95()
    print("=== test_calibration ===")
    test_calibration()
    print("\nALL METRIC TESTS PASSED")
