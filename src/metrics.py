"""Evaluation metrics for COMPASS (paper Metrics section + supervisor extras).

Per-organ (classes 1-4, background excluded), computed per patient and
averaged across patients:
- Paper: Dice, HD95, ECE, Vacuity, Brier score
- Supervisor requirement: Dice, IoU, pixel accuracy, precision, recall

Conventions (documented):
- HD95 is computed per slice in 2D with the template in-plane spacing
  (1.5 mm) and averaged over the patient's slices; a slice where the
  organ is present in GT but absent in the prediction is penalized with
  the slice diagonal; slices empty in both are skipped.
- ECE uses 10 equal-width confidence bins over softmax outputs.
- Vacuity = C / alpha_0 (Dirichlet evidence).
- Brier = multiclass mean squared error vs one-hot labels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

HD95_SPACING = 1.5        # mm, in-plane template spacing
ECE_BINS = 10


# --------------------------------------------------------------- confusion

def confusion(pred: np.ndarray, gt: np.ndarray, cls: int) -> tuple[int, int, int]:
    """TP, FP, FN for one class on boolean/int masks."""
    p = pred == cls
    g = gt == cls
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    return tp, fp, fn


def dice_from(tp: int, fp: int, fn: int) -> float:
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 1.0


def iou_from(tp: int, fp: int, fn: int) -> float:
    den = tp + fp + fn
    return tp / den if den else 1.0


def precision_from(tp: int, fp: int, fn: int) -> float:
    if tp + fp:
        return tp / (tp + fp)
    return 1.0 if fn == 0 else 0.0


def recall_from(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn else 1.0


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).mean())


# -------------------------------------------------------------------- HD95

def hd95(pred: np.ndarray, gt: np.ndarray,
         spacing: float = HD95_SPACING) -> float:
    """95th-percentile symmetric Hausdorff distance (2D, mm)."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    if not p.any() and not g.any():
        return float("nan")               # organ absent -> excluded from avg
    if not p.any() or not g.any():
        # one empty: penalize with the slice diagonal
        return float(np.hypot(*p.shape) * spacing)

    gt_dt = ndimage.distance_transform_edt(~g, sampling=spacing)
    pred_dt = ndimage.distance_transform_edt(~p, sampling=spacing)
    pred_surf = p & ~ndimage.binary_erosion(p)
    gt_surf = g & ~ndimage.binary_erosion(g)
    d = np.concatenate([gt_dt[pred_surf], pred_dt[gt_surf]])
    return float(np.percentile(d, 95))


# ------------------------------------------------------- calibration metrics

def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               n_bins: int = ECE_BINS) -> float:
    """ECE over equal-width confidence bins (arrays of per-pixel values)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(correct[m].mean())
                                          - float(conf[m].mean()))
    return ece


def brier_score(probs: np.ndarray, target: np.ndarray) -> float:
    """Multiclass Brier: mean over pixels of sum_c (p_c - y_c)^2.

    probs: (C, H, W) softmax output; target: (H, W) int labels.
    """
    C = probs.shape[0]
    onehot = np.zeros_like(probs)
    onehot[target.astype(int)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=0).mean())


def vacuity_mean(alpha: np.ndarray) -> float:
    """Mean vacuity C / alpha_0 over pixels. alpha: (C, H, W) evidence."""
    return float((alpha.shape[0] / alpha.sum(axis=0)).mean())
