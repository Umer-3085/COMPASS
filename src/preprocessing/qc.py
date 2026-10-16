"""Automatic geometry/intensity/label QC and overlay figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from .dicom_utils import geometry_compatible
from .jsonutil import dump_json


LABEL_COLORS = {
    1: (1.0, 0.15, 0.1),
    2: (0.15, 0.55, 1.0),
    3: (0.2, 0.9, 0.3),
    4: (1.0, 0.85, 0.1),
}


def _pick_qc_indices(label: sitk.Image | None, n_image: int) -> list[int]:
    if n_image <= 0:
        return []
    superior = 0
    inferior = n_image - 1
    middle = n_image // 2
    indices = [superior, middle, inferior]
    if label is not None:
        arr = sitk.GetArrayFromImage(label)
        fg = [i for i in range(arr.shape[0]) if np.any(arr[i] > 0)]
        if fg:
            indices.append(fg[len(fg) // 2])
            indices.append(fg[0])
            indices.append(fg[-1])
    # unique preserve order
    seen = set()
    out = []
    for i in indices:
        i = int(np.clip(i, 0, n_image - 1))
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _overlay_labels(ax, image2d: np.ndarray, label2d: np.ndarray | None, title: str) -> None:
    ax.imshow(image2d, cmap="gray")
    if label2d is not None:
        overlay = np.zeros(label2d.shape + (4,), dtype=float)
        for lab, color in LABEL_COLORS.items():
            m = label2d == lab
            overlay[m, 0] = color[0]
            overlay[m, 1] = color[1]
            overlay[m, 2] = color[2]
            overlay[m, 3] = 0.45
        if np.any(overlay[..., 3] > 0):
            ax.imshow(overlay)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _slice2d(img: sitk.Image, index: int) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)
    index = int(np.clip(index, 0, arr.shape[0] - 1))
    sl = arr[index]
    sl = sl.astype(np.float32)
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        return sl
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def write_ct_qc(out_qc: Path, case_id: int, ct: sitk.Image, labels: sitk.Image | None) -> list[str]:
    out_qc.mkdir(parents=True, exist_ok=True)
    n = sitk.GetArrayFromImage(ct).shape[0]
    indices = _pick_qc_indices(labels, n)
    paths = []
    ncols = 2
    fig, axes = plt.subplots(len(indices), ncols, figsize=(8, 3 * len(indices)))
    if len(indices) == 1:
        axes = np.array([axes])
    for r, idx in enumerate(indices):
        img = _slice2d(ct, idx)
        lab = sitk.GetArrayFromImage(labels)[idx] if labels is not None else None
        _overlay_labels(axes[r, 0], img, None, f"CT z={idx}")
        _overlay_labels(axes[r, 1], img, lab, f"CT + liver z={idx}")
    fig.suptitle(f"CT case {case_id}", fontsize=11)
    fig.tight_layout()
    dest = out_qc / f"ct_case_{case_id:02d}.png"
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    paths.append(str(dest))
    return paths


def write_mr_qc(
    out_qc: Path,
    case_id: int,
    t1: sitk.Image,
    t2: sitk.Image,
    labels_t1: sitk.Image | None,
    labels_t2: sitk.Image | None,
    t2_on_t1: bool,
    registration_failed: bool,
) -> list[str]:
    out_qc.mkdir(parents=True, exist_ok=True)
    n = sitk.GetArrayFromImage(t1).shape[0]
    indices = _pick_qc_indices(labels_t1, n)
    ncols = 5 if t2_on_t1 else 4
    fig, axes = plt.subplots(len(indices), ncols, figsize=(3.2 * ncols, 3 * len(indices)))
    if len(indices) == 1:
        axes = np.array([axes])
    t2n = sitk.GetArrayFromImage(t2).shape[0]
    for r, idx in enumerate(indices):
        t1s = _slice2d(t1, idx)
        t2idx = idx if t2_on_t1 else min(idx, t2n - 1)
        t2s = _slice2d(t2, t2idx)
        l1 = sitk.GetArrayFromImage(labels_t1)[idx] if labels_t1 is not None else None
        l2 = None
        if labels_t2 is not None and t2_on_t1:
            l2 = sitk.GetArrayFromImage(labels_t2)[idx]
        elif labels_t2 is not None:
            l2 = sitk.GetArrayFromImage(labels_t2)[t2idx]
        _overlay_labels(axes[r, 0], t1s, None, f"T1 z={idx}")
        _overlay_labels(axes[r, 1], t2s, None, f"T2 z={t2idx}")
        _overlay_labels(axes[r, 2], t1s, l1, f"T1 + T1-GT z={idx}")
        _overlay_labels(axes[r, 3], t2s, l2, f"T2 + T2-GT z={t2idx}")
        if t2_on_t1:
            rgb = np.stack([t1s, t2s, np.zeros_like(t1s)], axis=-1)
            axes[r, 4].imshow(np.clip(rgb, 0, 1))
            axes[r, 4].set_title(f"T1/T2 overlay z={idx}", fontsize=9)
            axes[r, 4].axis("off")
    title = f"MR case {case_id}"
    if registration_failed:
        title += " [REGISTRATION FAILED]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    dest = out_qc / f"mr_case_{case_id:02d}.png"
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return [str(dest)]


def validate_ct_result(result: dict[str, Any]) -> list[str]:
    issues = []
    ct = result["ct"]
    labels = result["labels"]
    spacing = np.array(ct.GetSpacing())
    if not np.allclose(spacing, 1.5, atol=1e-4):
        issues.append(f"CT spacing {tuple(spacing)} != 1.5")
    arr = sitk.GetArrayFromImage(ct)
    if arr.min() < -1e-5 or arr.max() > 1.0 + 1e-5:
        issues.append(f"CT normalized range [{arr.min()}, {arr.max()}] outside [0,1]")
    if labels is not None:
        if not geometry_compatible(ct, labels, origin_tol=1e-3):
            issues.append("CT/label geometry mismatch")
        uniq = set(int(v) for v in np.unique(sitk.GetArrayFromImage(labels)))
        if not uniq.issubset({0, 1}):
            issues.append(f"CT labels {sorted(uniq)}")
        if sitk.GetArrayFromImage(labels).sum() == 0:
            issues.append("CT empty liver mask")
    return issues


def validate_mr_result(result: dict[str, Any]) -> list[str]:
    issues = []
    t1 = result["t1_in"]
    t2 = result["t2"]
    if not np.allclose(t1.GetSpacing(), (1.5, 1.5, 1.5), atol=1e-4):
        issues.append(f"T1 spacing {t1.GetSpacing()}")
    if not np.allclose(t2.GetSpacing(), (1.5, 1.5, 1.5), atol=1e-4):
        issues.append(f"T2 spacing {t2.GetSpacing()}")
    if result["t2_on_t1_grid"]:
        if not geometry_compatible(t1, t2, origin_tol=1e-2):
            issues.append("T2 not on T1 physical grid after registration")
        if result["labels_t2"] is not None and not geometry_compatible(t1, result["labels_t2"], origin_tol=1e-2):
            issues.append("T2 labels not on T1 grid")
    if result["labels_t1"] is not None:
        uniq = set(int(v) for v in np.unique(sitk.GetArrayFromImage(result["labels_t1"])))
        if not uniq.issubset({0, 1, 2, 3, 4}):
            issues.append(f"T1 labels {sorted(uniq)}")
        if not geometry_compatible(t1, result["labels_t1"], origin_tol=1e-3):
            issues.append("T1/label geometry mismatch")
    if result.get("registration_failed"):
        issues.append("T2->T1 registration QC failed (low organ overlap)")
    return issues


def save_report(path: Path, report: dict[str, Any]) -> None:
    dump_json(path, report)
