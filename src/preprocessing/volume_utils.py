"""Physical resampling, intensity ops, N4, z-score. No PIL resizing."""

from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk

from .config import PreprocessConfig
from .dicom_utils import geometry_dict


def copy_image_with_array(reference: sitk.Image, array_zyx: np.ndarray, pixel_type=None) -> sitk.Image:
    img = sitk.GetImageFromArray(array_zyx)
    img.CopyInformation(reference)
    if pixel_type is not None:
        img = sitk.Cast(img, pixel_type)
    return img


def make_isotropic_reference(image: sitk.Image, spacing: tuple[float, float, float]) -> sitk.Image:
    old_spacing = np.array(image.GetSpacing(), dtype=float)
    old_size = np.array(image.GetSize(), dtype=float)
    physical = old_spacing * np.maximum(old_size - 1.0, 0.0)
    new_spacing = np.array(spacing, dtype=float)
    new_size = np.round(physical / new_spacing + 1.0).astype(int)
    new_size = np.maximum(new_size, 1)
    ref = sitk.Image([int(v) for v in new_size], sitk.sitkFloat32)
    ref.SetSpacing(tuple(float(s) for s in new_spacing))
    ref.SetOrigin(image.GetOrigin())
    ref.SetDirection(image.GetDirection())
    return ref


def resample_to_reference(image: sitk.Image, reference: sitk.Image, is_label: bool) -> sitk.Image:
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default = 0
    out_type = sitk.sitkUInt8 if is_label else sitk.sitkFloat32
    source = sitk.Cast(image, sitk.sitkUInt8 if is_label else sitk.sitkFloat32)
    return sitk.Resample(
        source,
        reference,
        sitk.Transform(),
        interpolator,
        default,
        out_type,
    )


def intensity_stats(array: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is not None:
        values = array[mask]
    else:
        values = array.reshape(-1)
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p1": None,
            "p99": None,
            "count": 0,
        }
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p1": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
        "count": int(values.size),
    }


def ct_window_normalize(hu: sitk.Image, cfg: PreprocessConfig) -> tuple[sitk.Image, dict[str, Any]]:
    arr = sitk.GetArrayFromImage(hu).astype(np.float32)
    raw = intensity_stats(arr)
    n = arr.size
    below = float(np.mean(arr < cfg.ct_window_min) * 100.0)
    above = float(np.mean(arr > cfg.ct_window_max) * 100.0)
    clipped = np.clip(arr, cfg.ct_window_min, cfg.ct_window_max)
    width = cfg.ct_window_max - cfg.ct_window_min
    norm = (clipped - cfg.ct_window_min) / width
    out = copy_image_with_array(hu, norm.astype(np.float32), sitk.sitkFloat32)
    meta = {
        "raw": raw,
        "percent_below_window": below,
        "percent_above_window": above,
        "clipped": intensity_stats(clipped),
        "normalized": intensity_stats(norm),
        "window": [cfg.ct_window_min, cfg.ct_window_max],
        "voxel_count": int(n),
    }
    return out, meta


def otsu_foreground_mask(image: sitk.Image, cfg: PreprocessConfig) -> sitk.Image:
    img = sitk.Cast(image, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)
    # Otsu on a subsampled histogram; exclude exact zeros (air padding)
    nonzero = arr[arr > 0]
    if nonzero.size < 100:
        mask = (arr > 0).astype(np.uint8)
        return copy_image_with_array(image, mask, sitk.sitkUInt8)
    # SimpleITK Otsu
    otsu = sitk.OtsuThreshold(img, 0, 1, cfg.foreground_otsu_bins)
    # OtsuThreshold: inside=0 outside=1 by default with these args? 
    # sitk.OtsuThreshold(image, insideValue, outsideValue): pixels BELOW threshold get insideValue.
    # We want foreground=1. If more zeros assigned to 1, invert.
    m = sitk.GetArrayFromImage(otsu)
    if m.mean() > 0.5:
        m = 1 - m
    m = (m > 0).astype(np.uint8)
    # Keep largest component-ish: also require intensity > 0
    m = np.logical_and(m, arr > 0).astype(np.uint8)
    return copy_image_with_array(image, m, sitk.sitkUInt8)


def n4_correct(image: sitk.Image, cfg: PreprocessConfig) -> tuple[sitk.Image, sitk.Image, dict[str, Any]]:
    img = sitk.Cast(image, sitk.sitkFloat32)
    mask = otsu_foreground_mask(img, cfg)
    pre = intensity_stats(sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(mask).astype(bool))
    status = {"applied": False, "error": None}
    if not cfg.mri_n4:
        return img, mask, {"applied": False, "reason": "disabled", "pre": pre, "post": pre}

    try:
        filt = sitk.N4BiasFieldCorrectionImageFilter()
        filt.SetMaximumNumberOfIterations(list(cfg.n4_iterations))
        filt.SetConvergenceThreshold(cfg.n4_convergence_threshold)
        if cfg.n4_shrink_factor and cfg.n4_shrink_factor > 1:
            shrink = sitk.Shrink(img, [cfg.n4_shrink_factor] * img.GetDimension())
            mask_s = sitk.Resample(mask, shrink, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            corr_s = filt.Execute(shrink, mask_s)
            log_bias = filt.GetLogBiasFieldAsImage(img)
            corr = img / sitk.Exp(log_bias)
        else:
            corr = filt.Execute(img, mask)
        corr = sitk.Cast(corr, sitk.sitkFloat32)
        status["applied"] = True
    except Exception as exc:
        corr = img
        status["applied"] = False
        status["error"] = f"{type(exc).__name__}: {exc}"

    post = intensity_stats(sitk.GetArrayFromImage(corr), sitk.GetArrayFromImage(mask).astype(bool))
    status.update({"pre": pre, "post": post, "foreground_voxels": int(sitk.GetArrayFromImage(mask).sum())})
    return corr, mask, status


def foreground_zscore(image: sitk.Image, mask: sitk.Image | None, cfg: PreprocessConfig) -> tuple[sitk.Image, dict[str, Any]]:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if mask is None:
        m = arr > 0
    else:
        # resample mask if needed
        if tuple(mask.GetSize()) != tuple(image.GetSize()):
            mask = resample_to_reference(mask, image, is_label=True)
        m = sitk.GetArrayFromImage(mask).astype(bool)
        if m.sum() < 10:
            m = arr > 0
    values = arr[m]
    meta: dict[str, Any] = {"method": "foreground_zscore", "foreground_voxel_count": int(m.sum())}
    if not cfg.mri_zscore:
        meta["applied"] = False
        return sitk.Cast(image, sitk.sitkFloat32), meta
    if values.size == 0:
        meta.update({"mean": 0.0, "std": 0.0, "applied": False, "reason": "empty_foreground"})
        return sitk.Cast(image, sitk.sitkFloat32), meta
    mu = float(values.mean())
    sd = float(values.std())
    meta["mean"] = mu
    meta["std"] = sd
    if sd < 1e-8:
        z = arr - mu
        meta["applied"] = True
        meta["reason"] = "std_near_zero_centered_only"
    else:
        z = np.zeros_like(arr)
        z[m] = (arr[m] - mu) / sd
        # keep background at 0
        meta["applied"] = True
    out = copy_image_with_array(image, z.astype(np.float32), sitk.sitkFloat32)
    meta["after"] = intensity_stats(z, m)
    return out, meta


def assert_same_physical_grid(a: sitk.Image, b: sitk.Image, name_a: str, name_b: str, origin_tol: float = 1e-3) -> None:
    if tuple(a.GetSize()) != tuple(b.GetSize()):
        raise RuntimeError(f"Size mismatch {name_a} {a.GetSize()} vs {name_b} {b.GetSize()}")
    if not np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5, rtol=0):
        raise RuntimeError(f"Spacing mismatch {name_a} {a.GetSpacing()} vs {name_b} {b.GetSpacing()}")
    if not np.allclose(a.GetOrigin(), b.GetOrigin(), atol=origin_tol, rtol=0):
        raise RuntimeError(f"Origin mismatch {name_a} {a.GetOrigin()} vs {name_b} {b.GetOrigin()}")
    if not np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5, rtol=0):
        raise RuntimeError(f"Direction mismatch {name_a} vs {name_b}")


def numpy_axial_axis(image: sitk.Image) -> int:
    """Axis in numpy array (z,y,x) corresponding to axial slices.

    SimpleITK numpy layout is [z, y, x] = ITK axes [2, 1, 0].
    """
    itk_axis = axial_axis_from_direction_safe(image)
    return {0: 2, 1: 1, 2: 0}[itk_axis]


def axial_axis_from_direction_safe(image: sitk.Image) -> int:
    from .dicom_utils import axial_axis_from_direction

    return axial_axis_from_direction(image)


def extract_axial_slices(image: sitk.Image, indices: list[int]) -> list[np.ndarray]:
    arr = sitk.GetArrayFromImage(image)
    axis = numpy_axial_axis(image)
    if axis != 0:
        arr = np.moveaxis(arr, axis, 0)
    return [np.ascontiguousarray(arr[i]) for i in indices]


def write_nifti(path, image: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)


def resample_notes(original: sitk.Image, processed: sitk.Image) -> dict[str, Any]:
    return {
        "original": geometry_dict(original),
        "processed": geometry_dict(processed),
    }
