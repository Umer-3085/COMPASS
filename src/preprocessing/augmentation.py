"""Training-time augmentation utilities.

These are NOT applied in the offline preprocessing pipeline.
Call them from a dataloader on the training split only.

Spatial transforms must be identical for every image channel and the label
of the same sample. Use continuous interpolation for images and nearest-neighbor
for labels. Do not augment validation or test data.

The paper specifies "random physical distortions (rotations, scaling, and
elastic stretching) identically across the CT, MRI, and label slices" —
use combine_augmentations() to build one transform per sample and apply it
to every channel with apply_same_affine / apply_same_elastic.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk


def apply_same_affine(
    images: Sequence[sitk.Image],
    labels: Sequence[sitk.Image],
    transform: sitk.Transform,
) -> tuple[list[sitk.Image], list[sitk.Image]]:
    """Resample images (linear) and labels (nearest) with one transform."""
    out_images = []
    out_labels = []
    for img in images:
        ref = images[0]
        out_images.append(
            sitk.Resample(img, ref, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        )
    for lab in labels:
        ref = images[0]
        out_labels.append(
            sitk.Resample(lab, ref, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        )
    return out_images, out_labels


def random_rotate_scale_2d(
    angle_deg: float,
    scale: float,
    size_xy: tuple[int, int],
) -> sitk.Transform:
    t = sitk.Similarity2DTransform()
    t.SetAngle(np.deg2rad(angle_deg))
    t.SetScale(scale)
    t.SetCenter((size_xy[0] / 2.0, size_xy[1] / 2.0))
    return t


def random_elastic_2d(
    size_yx: tuple[int, int],
    noise_std: float = 3.0,
    smoothing_sigma: float = 8.0,
    max_displacement: float = 8.0,
    rng: np.random.Generator | None = None,
) -> sitk.Transform:
    """Smooth random displacement field for elastic stretching (2D).

    A Gaussian-blurred white-noise vector field is scaled so displacements
    stay within +-max_displacement pixels. Deterministic given rng.
    """
    rng = rng or np.random.default_rng()
    h, w = size_yx
    noise = rng.normal(0.0, noise_std, size=(h, w, 2))
    field = _gaussian_smooth(noise, smoothing_sigma)
    peak = float(np.abs(field).max())
    if peak > 1e-8:
        field *= max_displacement / peak
    # sitk displacement field: index [y, x] -> vector (dx, dy)
    # DisplacementFieldTransform requires vector<float64>
    field_itk = np.zeros((h, w, 2), dtype=np.float64)
    field_itk[..., 0] = field[..., 1]  # x displacement from noise[...,1]
    field_itk[..., 1] = field[..., 0]  # y displacement from noise[...,0]
    disp = sitk.GetImageFromArray(field_itk, isVector=True)
    disp.SetSpacing((1.0, 1.0))
    return sitk.DisplacementFieldTransform(disp)


def _gaussian_smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a (..., 2) vector field (no scipy needed)."""
    if sigma <= 0:
        return field
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    out = field.astype(np.float32)
    # convolve along each spatial axis for each vector component
    for axis in (0, 1):
        out = np.stack(
            [_convolve1d(out[..., c], kernel, axis) for c in range(out.shape[-1])],
            axis=-1,
        )
    return out


def _convolve1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    padded = np.pad(arr, [(pad, pad) if a == axis else (0, 0) for a in range(arr.ndim)])
    out = np.zeros_like(arr, dtype=np.float32)
    for i, k in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + arr.shape[axis])
        out += k * padded[tuple(sl)]
    return out


def apply_same_elastic(
    images: Sequence[sitk.Image],
    labels: Sequence[sitk.Image],
    transform: sitk.Transform,
) -> tuple[list[sitk.Image], list[sitk.Image]]:
    """Apply one elastic displacement field to images and labels identically."""
    ref = images[0]
    out_images = [
        sitk.Resample(img, ref, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        for img in images
    ]
    out_labels = [
        sitk.Resample(lab, ref, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        for lab in labels
    ]
    return out_images, out_labels


def sample_augmentation_params(
    rng: np.random.Generator,
    angle_range: float = 10.0,
    scale_range: float = 0.1,
    elastic_prob: float = 0.5,
    elastic_max: float = 8.0,
) -> dict:
    """Sample one set of augmentation parameters for a training sample.

    Probabilities/ranges follow common abdominal CT/MR practice; adjust in
    the dataloader config as needed.
    """
    return {
        "angle_deg": float(rng.uniform(-angle_range, angle_range)),
        "scale": float(rng.uniform(1.0 - scale_range, 1.0 + scale_range)),
        "apply_elastic": bool(rng.random() < elastic_prob),
        "elastic_max": float(elastic_max),
    }
