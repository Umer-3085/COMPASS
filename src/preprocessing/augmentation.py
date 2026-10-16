"""Training-time augmentation utilities.

These are NOT applied in the offline preprocessing pipeline.
Call them from a dataloader on the training split only.

Spatial transforms must be identical for every image channel and the label
of the same sample. Use continuous interpolation for images and nearest-neighbor
for labels. Do not augment validation or test data.
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
