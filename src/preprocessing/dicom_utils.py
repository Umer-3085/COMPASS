"""DICOM series loading via GDCM / SimpleITK. Never rely on filename sort alone."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


def _get_tag(file_reader: sitk.ImageFileReader, key: str, default: str | None = None) -> str | None:
    try:
        if file_reader.HasMetaDataKey(key):
            return file_reader.GetMetaData(key)
    except Exception:
        return default
    return default


def read_file_tags(path: str | Path) -> dict[str, Any]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    iop = _get_tag(reader, "0020|0037")
    ipp = _get_tag(reader, "0020|0032")
    return {
        "path": str(path),
        "series_uid": _get_tag(reader, "0020|000e"),
        "sop_uid": _get_tag(reader, "0008|0018"),
        "instance_number": _get_tag(reader, "0020|0013"),
        "slice_thickness": _get_tag(reader, "0018|0050"),
        "spacing_between_slices": _get_tag(reader, "0018|0088"),
        "pixel_spacing": _get_tag(reader, "0028|0030"),
        "image_orientation_patient": _parse_float_list(iop),
        "image_position_patient": _parse_float_list(ipp),
        "rescale_slope": _parse_float(_get_tag(reader, "0028|1053"), 1.0),
        "rescale_intercept": _parse_float(_get_tag(reader, "0028|1052"), 0.0),
        "pixel_representation": _get_tag(reader, "0028|0103"),
        "bits_stored": _get_tag(reader, "0028|0101"),
        "photometric": _get_tag(reader, "0028|0004"),
        "rows": reader.GetSize()[1] if len(reader.GetSize()) >= 2 else None,
        "cols": reader.GetSize()[0] if len(reader.GetSize()) >= 1 else None,
    }


def _parse_float(text: str | None, default: float) -> float:
    if text is None or text.strip() == "":
        return default
    try:
        return float(text.split("\\")[0])
    except ValueError:
        return default


def _parse_float_list(text: str | None) -> list[float] | None:
    if not text:
        return None
    parts = [p for p in text.replace(",", "\\").split("\\") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def geometry_dict(image: sitk.Image) -> dict[str, Any]:
    return {
        "size": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "dimension": int(image.GetDimension()),
    }


def load_dicom_series(directory: str | Path) -> tuple[sitk.Image, list[str], str, dict[str, Any]]:
    directory = str(Path(directory))
    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(directory) or [])
    if not series_ids:
        raise RuntimeError(f"No GDCM DICOM series found in {directory}")

    file_counts = {
        sid: reader.GetGDCMSeriesFileNames(directory, sid) for sid in series_ids
    }
    series_uid = max(file_counts, key=lambda s: len(file_counts[s]))
    files = list(file_counts[series_uid])
    if len(series_ids) > 1:
        extra = {s: len(file_counts[s]) for s in series_ids}

    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    tags = [read_file_tags(f) for f in files]
    audit = _build_series_audit(image, files, series_uid, tags, series_ids)
    if len(series_ids) > 1:
        audit["multiple_series"] = extra
        audit["selected_series_reason"] = "largest_file_count"
    return image, files, series_uid, audit


def _orientation_normal(iop: list[float] | None) -> list[float] | None:
    if not iop or len(iop) < 6:
        return None
    r = np.array(iop[:3], dtype=float)
    c = np.array(iop[3:6], dtype=float)
    n = np.cross(r, c)
    norm = np.linalg.norm(n)
    if norm == 0:
        return None
    return (n / norm).tolist()


def _build_series_audit(
    image: sitk.Image,
    files: list[str],
    series_uid: str,
    tags: list[dict[str, Any]],
    series_ids: list[str],
) -> dict[str, Any]:
    arr = sitk.GetArrayFromImage(image)
    positions = [t["image_position_patient"] for t in tags]
    orientations = [t["image_orientation_patient"] for t in tags]
    normals = [_orientation_normal(o) for o in orientations]
    base_normal = next((n for n in normals if n is not None), None)

    projections = []
    if base_normal is not None:
        n = np.array(base_normal, dtype=float)
        for p in positions:
            if p and len(p) >= 3:
                projections.append(float(np.dot(np.array(p[:3], dtype=float), n)))
            else:
                projections.append(float("nan"))

    steps = []
    if len(projections) >= 2:
        steps = [
            projections[i + 1] - projections[i]
            for i in range(len(projections) - 1)
            if not (math.isnan(projections[i]) or math.isnan(projections[i + 1]))
        ]
    monotonic = False
    if steps:
        if all(s > 1e-4 for s in steps):
            monotonic = True
            direction = "increasing"
        elif all(s < -1e-4 for s in steps):
            monotonic = True
            direction = "decreasing"
        else:
            direction = "non_monotonic"
    else:
        direction = "unknown"

    orients = [tuple(o) for o in orientations if o]
    orientation_consistent = len(set(orients)) <= 1

    slopes = sorted({round(float(t["rescale_slope"]), 6) for t in tags})
    intercepts = sorted({round(float(t["rescale_intercept"]), 6) for t in tags})
    representations = sorted({str(t["pixel_representation"]) for t in tags})

    instances = []
    for t in tags:
        try:
            instances.append(int(float(t["instance_number"]))) if t["instance_number"] else None
        except (TypeError, ValueError):
            instances.append(None)

    return {
        "series_uid": series_uid,
        "all_series_uids": series_ids,
        "file_count": len(files),
        "files": files,
        "size_xyz": list(image.GetSize()),
        "array_shape_zyx": list(arr.shape),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "intensity_min": float(arr.min()) if arr.size else None,
        "intensity_max": float(arr.max()) if arr.size else None,
        "image_orientation_patient": orientations[0] if orientations else None,
        "orientation_consistent": orientation_consistent,
        "slice_normal": base_normal,
        "positions_available": all(p is not None for p in positions),
        "position_projection_min": min(projections) if projections else None,
        "position_projection_max": max(projections) if projections else None,
        "position_step_median": float(np.median(steps)) if steps else None,
        "position_step_min": float(np.min(steps)) if steps else None,
        "position_step_max": float(np.max(steps)) if steps else None,
        "position_monotonic": monotonic,
        "position_direction": direction,
        "rescale_slopes": slopes,
        "rescale_intercepts": intercepts,
        "pixel_representations": representations,
        "instance_numbers": instances,
        "first_tags": {k: v for k, v in tags[0].items() if k != "path"} if tags else {},
        "last_tags": {k: v for k, v in tags[-1].items() if k != "path"} if tags else {},
    }


def decide_hu_conversion(array: np.ndarray, slope: float, intercept: float) -> dict[str, Any]:
    """SimpleITK series reading typically already applies rescale slope/intercept."""
    amin = float(array.min())
    amax = float(array.max())
    already = True
    method = "sitk_already_applied_rescale"
    if abs(slope - 1.0) < 1e-6 and abs(intercept) < 1e-6:
        method = "identity_rescale_tags"
        already = True
    elif intercept <= -100 and amin <= intercept + 250:
        method = "sitk_already_applied_rescale"
        already = True
    elif intercept <= -500 and amin >= -50:
        method = "manual_apply_rescale"
        already = False
    return {
        "already_hu": already,
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "array_min": amin,
        "array_max": amax,
    }


def apply_hu_if_needed(image: sitk.Image, decision: dict[str, Any]) -> sitk.Image:
    if decision["already_hu"]:
        return sitk.Cast(image, sitk.sitkFloat32)
    slope = float(decision["slope"])
    intercept = float(decision["intercept"])
    img = sitk.Cast(image, sitk.sitkFloat32)
    return img * slope + intercept


def geometry_compatible(a: sitk.Image, b: sitk.Image, origin_tol: float = 1e-3, spacing_tol: float = 1e-4) -> bool:
    if tuple(a.GetSize()) != tuple(b.GetSize()):
        return False
    if not np.allclose(a.GetSpacing(), b.GetSpacing(), atol=spacing_tol, rtol=0):
        return False
    if not np.allclose(a.GetOrigin(), b.GetOrigin(), atol=origin_tol, rtol=0):
        return False
    if not np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5, rtol=0):
        return False
    return True


def geometry_mismatch(a: sitk.Image, b: sitk.Image) -> dict[str, Any]:
    return {
        "size_equal": tuple(a.GetSize()) == tuple(b.GetSize()),
        "spacing_a": list(a.GetSpacing()),
        "spacing_b": list(b.GetSpacing()),
        "origin_a": list(a.GetOrigin()),
        "origin_b": list(b.GetOrigin()),
        "direction_equal": np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5),
        "compatible": geometry_compatible(a, b),
    }


def axial_axis_from_direction(image: sitk.Image) -> int:
    """Return ITK axis (0=x,1=y,2=z) most aligned with LPS superior-inferior (z)."""
    d = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
    # columns are direction cosines of index axes in physical space
    z_abs = np.abs(d[2, :])
    return int(np.argmax(z_abs))
