"""Unpaired CT preprocessing: HU, window, resample, register to CT template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .config import PreprocessConfig
from .dicom_utils import apply_hu_if_needed, decide_hu_conversion, geometry_dict, load_dicom_series
from .label_utils import (
    ct_binary_mapper,
    label_statistics,
    match_ct_mask_files,
    retained_slice_indices,
    stack_masks_as_image,
)
from .paths import ProjectPaths
from .registration import register_to_template
from .volume_utils import (
    extract_axial_slices,
    make_isotropic_reference,
    numpy_axial_axis,
    resample_to_reference,
    write_nifti,
    ct_window_normalize,
)


def process_ct_case(
    paths: ProjectPaths,
    cfg: PreprocessConfig,
    case_id: int,
    split: str,
    official_test: bool = False,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = paths.ct_case_raw(case_id, official_test=official_test)
    dicom_dir = case_dir / "DICOM_anon"
    ground_dir = case_dir / "Ground"

    image, files, series_uid, audit = load_dicom_series(dicom_dir)
    arr = sitk.GetArrayFromImage(image)
    slope = float(audit["rescale_slopes"][0]) if audit["rescale_slopes"] else 1.0
    intercept = float(audit["rescale_intercepts"][0]) if audit["rescale_intercepts"] else 0.0
    hu_decision = decide_hu_conversion(arr, slope, intercept)
    hu = apply_hu_if_needed(image, hu_decision)
    windowed, intensity_meta = ct_window_normalize(hu, cfg)

    labels = None
    label_meta: dict[str, Any] = {"present": False}
    if ground_dir.is_dir() and any(ground_dir.glob("*.png")):
        pngs, match_method = match_ct_mask_files(
            files, ground_dir, audit.get("instance_numbers") or []
        )
        labels, stack_info = stack_masks_as_image(pngs, image, ct_binary_mapper)
        unique = set(stack_info["unique_after_map"])
        if not unique.issubset({0, 1}):
            raise RuntimeError(f"CT {case_id}: labels {sorted(unique)} not subset of {{0,1}}")
        label_meta = {
            "present": True,
            "match_method": match_method,
            "stack": stack_info,
            "original": label_statistics(labels, [0, 1]),
        }

    reference = make_isotropic_reference(windowed, cfg.target_spacing)
    ct_iso = resample_to_reference(windowed, reference, is_label=False)
    labels_iso = None
    if labels is not None:
        labels_iso = resample_to_reference(labels, reference, is_label=True)
        label_meta["processed"] = label_statistics(labels_iso, [0, 1])
        if not set(label_meta["processed"]["unique"]).issubset({0, 1}):
            raise RuntimeError(f"CT {case_id}: resampled labels invalid")

    # Paper: all scans standardized onto the common CT reference grid.
    template_registration: dict[str, Any] = {
        "performed": False,
        "reason": "no_template" if template is None else None,
    }
    if template is not None and cfg.register_to_template:
        reg = register_to_template(
            moving=ct_iso,
            template=template["image"],
            template_liver_mask=template.get("liver_mask"),
            moving_label=labels_iso,
            cfg=cfg,
            moving_name="ct",
            case_name=f"ct_case_{case_id:02d}",
        )
        ct_iso = reg["warped_image"]
        if reg["warped_label"] is not None:
            labels_iso = reg["warped_label"]
            if labels is not None:
                label_meta["processed"] = label_statistics(labels_iso, [0, 1])
        template_registration = {
            "performed": True,
            "on_template_grid": True,
            **{k: v for k, v in reg.items() if k not in ("warped_image", "warped_label")},
        }

    out_split = "official_test" if official_test else split
    out_dir = paths.processed_case("ct", out_split, case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_nifti(out_dir / "ct.nii.gz", ct_iso)
    if labels_iso is not None:
        write_nifti(out_dir / "labels.nii.gz", labels_iso)

    slice_method = cfg.slice_method_for_split("test" if official_test else split)
    if labels_iso is not None:
        retained = retained_slice_indices(labels_iso, slice_method)
    else:
        n = sitk.GetArrayFromImage(ct_iso).shape[0]
        retained = list(range(n))
        slice_method = "all_unlabeled"

    slice_files: list[str] = []
    if cfg.extract_slices:
        slice_dir = out_dir / "slices"
        slice_dir.mkdir(parents=True, exist_ok=True)
        images = extract_axial_slices(ct_iso, retained)
        labs = extract_axial_slices(labels_iso, retained) if labels_iso is not None else None
        for sl, img in zip(retained, images):
            np.save(slice_dir / f"image_{sl:04d}.npy", img.astype(np.float32))
            slice_files.append(f"image_{sl:04d}.npy")
            if labs is not None:
                np.save(slice_dir / f"label_{sl:04d}.npy", labs[retained.index(sl)].astype(np.uint8))

    meta = {
        "case_id": case_id,
        "modality": "CT",
        "split": out_split,
        "series_uid": series_uid,
        "unpaired": True,
        "ct_mr_registration": False,
        "original_size": geometry_dict(image)["size"],
        "original_spacing": geometry_dict(image)["spacing"],
        "original_origin": geometry_dict(image)["origin"],
        "original_direction": geometry_dict(image)["direction"],
        "processed_size": geometry_dict(ct_iso)["size"],
        "processed_spacing": geometry_dict(ct_iso)["spacing"],
        "processed_origin": geometry_dict(ct_iso)["origin"],
        "processed_direction": geometry_dict(ct_iso)["direction"],
        "dicom_audit": {k: v for k, v in audit.items() if k != "files"},
        "hu": hu_decision,
        "intensity": intensity_meta,
        "labels": label_meta,
        "registration": {"performed": False, "type": "none", "reason": "ct_only_unpaired"},
        "template_registration": template_registration,
        "on_template_grid": bool(template_registration.get("on_template_grid")),
        "slice_selection": {
            "method": slice_method,
            "retained_indices": retained,
            "n_retained": len(retained),
            "axial_numpy_axis": numpy_axial_axis(ct_iso),
        },
        "n_slices_original": int(image.GetSize()[2]),
        "n_slices_processed": int(ct_iso.GetSize()[2]),
    }
    return {
        "meta": meta,
        "out_dir": out_dir,
        "ct": ct_iso,
        "labels": labels_iso,
        "original_ct": image,
        "original_labels": labels,
        "template_registration": template_registration,
    }
