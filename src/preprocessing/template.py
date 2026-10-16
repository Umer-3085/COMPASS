"""Common CT mean template: one anatomical grid for CT, T1, T2, and labels.

CHAOS CT and MR are unpaired (different patients), so the paper's "align the
MRI sequences to the CT reference volume" is implemented as a study-specific
CT mean template built from the *training* CT scans:

1. Every training CT is HU-windowed and resampled to its 1.5 mm isotropic grid.
2. All CTs are registered to a seed CT with ANTs Affine (Mattes MI).
3. The warped CTs are averaged -> template; warped liver labels are fused by
   majority vote -> template liver mask (used later as MR registration QC).
4. Steps 2-3 repeat for cfg.template_rounds iterations (mean-template
   refinement). All cases are finally registered to the converged template.

Every downstream volume (training/val/test CT, T1, T2) is then resampled onto
this single template grid, so slice indices are identical across modalities
and labels, as required by the paper.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .config import PreprocessConfig
from .dicom_utils import apply_hu_if_needed, decide_hu_conversion, load_dicom_series
from .jsonutil import dump_json
from .label_utils import (
    ct_binary_mapper,
    label_statistics,
    match_ct_mask_files,
    stack_masks_as_image,
)
from .paths import ProjectPaths
from .registration import ants_available, ants_register_to_fixed
from .volume_utils import (
    ct_window_normalize,
    make_isotropic_reference,
    resample_to_reference,
    write_nifti,
)


def load_windowed_ct(
    paths: ProjectPaths,
    cfg: PreprocessConfig,
    case_id: int,
    official_test: bool = False,
) -> dict[str, Any]:
    """Load one CT case and apply HU windowing + isotropic resample.

    Returns windowed isotropic image, liver labels (or None), and audit meta.
    Reused by the template builder; mirrors the front half of
    process_ct_case without writing anything.
    """
    case_dir = paths.ct_case_raw(case_id, official_test=official_test)
    image, files, series_uid, audit = load_dicom_series(case_dir / "DICOM_anon")
    arr = sitk.GetArrayFromImage(image)
    slope = float(audit["rescale_slopes"][0]) if audit["rescale_slopes"] else 1.0
    intercept = float(audit["rescale_intercepts"][0]) if audit["rescale_intercepts"] else 0.0
    hu_decision = decide_hu_conversion(arr, slope, intercept)
    hu = apply_hu_if_needed(image, hu_decision)
    windowed, intensity_meta = ct_window_normalize(hu, cfg)

    labels = None
    ground_dir = case_dir / "Ground"
    if ground_dir.is_dir() and any(ground_dir.glob("*.png")):
        pngs, _ = match_ct_mask_files(files, ground_dir, audit.get("instance_numbers") or [])
        labels, _ = stack_masks_as_image(pngs, image, ct_binary_mapper)

    reference = make_isotropic_reference(windowed, cfg.target_spacing)
    ct_iso = resample_to_reference(windowed, reference, is_label=False)
    labels_iso = None
    if labels is not None:
        labels_iso = resample_to_reference(labels, reference, is_label=True)

    return {
        "case_id": case_id,
        "ct": ct_iso,
        "labels": labels_iso,
        "series_uid": series_uid,
        "intensity": intensity_meta,
        "native_size": tuple(int(v) for v in image.GetSize()),
    }


def _mean_template(
    volumes: list[sitk.Image],
    labels: list[sitk.Image | None],
    reference: sitk.Image,
    majority_threshold: float,
) -> tuple[sitk.Image, sitk.Image, dict[str, Any]]:
    """Average warped CT volumes and majority-vote warp labels on ref grid."""
    acc = np.zeros(
        (int(reference.GetSize()[2]), int(reference.GetSize()[1]), int(reference.GetSize()[0])),
        dtype=np.float64,
    )
    votes = np.zeros_like(acc, dtype=np.uint16)
    n_labels = 0
    for vol, lab in zip(volumes, labels):
        arr = sitk.GetArrayFromImage(vol).astype(np.float64)
        if arr.shape != acc.shape:
            raise RuntimeError(
                f"Template build: warped volume shape {arr.shape} != reference {acc.shape}"
            )
        acc += arr
        if lab is not None:
            votes += (sitk.GetArrayFromImage(lab) > 0).astype(np.uint16)
            n_labels += 1
    mean_arr = (acc / max(len(volumes), 1)).astype(np.float32)
    template = sitk.GetImageFromArray(mean_arr)
    template.CopyInformation(reference)

    if n_labels > 0:
        mask_arr = (votes >= max(1, int(np.ceil(majority_threshold * n_labels)))).astype(np.uint8)
    else:
        mask_arr = np.zeros_like(acc, dtype=np.uint8)
    mask = sitk.GetImageFromArray(mask_arr)
    mask.Cast(sitk.sitkUInt8)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(reference)

    stats = {
        "n_volumes": len(volumes),
        "n_labeled_volumes": n_labels,
        "majority_threshold": majority_threshold,
        "template_intensity": {
            "min": float(mean_arr.min()),
            "max": float(mean_arr.max()),
            "mean": float(mean_arr.mean()),
        },
        "liver_mask_voxels": int(mask_arr.sum()),
        "liver_mask_foreground_slices": int(
            sum(1 for sl in mask_arr if np.any(sl > 0))
        ),
    }
    return template, mask, stats


def build_template(
    paths: ProjectPaths,
    cfg: PreprocessConfig,
    ct_ids: list[int],
    seed_case: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Build the CT mean template from the given (training) CT case IDs."""
    if not ants_available():
        raise RuntimeError(
            "Template build requires ANTs (antspyx). Install antspyx or use "
            "the Python version that ships an antspyx wheel."
        )
    if not ct_ids:
        raise RuntimeError("Template build: no CT cases given")
    ids = sorted(int(i) for i in ct_ids)
    seed = int(seed_case if seed_case is not None else ids[0])
    if seed not in ids:
        raise RuntimeError(f"Template seed {seed} not among build cases {ids}")

    t0 = time.time()
    log(f"Loading {len(ids)} CT cases for template build (seed={seed})...")
    cases: dict[int, dict[str, Any]] = {}
    for cid in ids:
        cases[cid] = load_windowed_ct(paths, cfg, cid)
        lab = cases[cid]["labels"]
        log(
            f"  CT {cid}: grid {cases[cid]['ct'].GetSize()} "
            f"labels={'yes' if lab is not None else 'no'}"
        )

    reference = cases[seed]["ct"]
    current_template = reference
    current_mask: sitk.Image | None = None
    rounds_meta: list[dict[str, Any]] = []

    for rnd in range(max(1, cfg.template_rounds)):
        rt0 = time.time()
        log(f"Template round {rnd + 1}/{cfg.template_rounds}: registering {len(ids)} CTs...")
        warped_vols: list[sitk.Image] = []
        warped_labs: list[sitk.Image | None] = []
        per_case: dict[str, Any] = {}
        for cid in ids:
            c = cases[cid]
            reg = ants_register_to_fixed(
                fixed=current_template,
                moving=c["ct"],
                moving_label=c["labels"],
                type_of_transform="Affine",
            )
            warped_vols.append(reg["warped_image"])
            warped_labs.append(reg["warped_label"])
            per_case[str(cid)] = {
                "transform_type": reg["type"],
                "backend": reg["backend"],
            }
            log(f"  CT {cid} registered ({time.time() - rt0:.0f}s elapsed)")

        template, mask, stats = _mean_template(
            warped_vols, warped_labs, reference, cfg.template_majority_threshold
        )
        # Per-case Dice of warped liver vs the fused mask (build QC only).
        for cid, wlab in zip(ids, warped_labs):
            if wlab is None:
                continue
            from .label_utils import dice_score

            d = dice_score(wlab, mask, 1)
            per_case[str(cid)]["liver_dice_vs_fused_mask"] = (
                round(d, 4) if d is not None else None
            )

        current_template = template
        current_mask = mask
        round_meta = {
            "round": rnd + 1,
            "seconds": round(time.time() - rt0, 1),
            "stats": stats,
            "per_case": per_case,
        }
        rounds_meta.append(round_meta)
        log(
            f"Round {rnd + 1} done in {round_meta['seconds']}s; "
            f"liver mask {stats['liver_mask_voxels']} voxels, "
            f"{stats['liver_mask_foreground_slices']} slices"
        )

    assert current_mask is not None
    write_nifti(paths.template_image, current_template)
    write_nifti(paths.template_liver_mask, current_mask)

    dices = [
        r["per_case"][str(cid)].get("liver_dice_vs_fused_mask")
        for r in rounds_meta
        for cid in ids
        if r["per_case"][str(cid)].get("liver_dice_vs_fused_mask") is not None
    ]
    meta = {
        "purpose": (
            "Common CT mean template (paper: CT reference volume / common "
            "anatomical coordinate system). Built from training CTs only."
        ),
        "build_case_ids": ids,
        "seed_case": seed,
        "target_spacing": list(cfg.target_spacing),
        "template_grid": {
            "size": [int(v) for v in current_template.GetSize()],
            "spacing": [float(v) for v in current_template.GetSpacing()],
            "origin": [float(v) for v in current_template.GetOrigin()],
            "direction": [float(v) for v in current_template.GetDirection()],
        },
        "rounds": rounds_meta,
        "final_liver_dice_vs_fused_mask": {
            "min": round(min(dices), 4) if dices else None,
            "mean": round(float(np.mean(dices)), 4) if dices else None,
            "max": round(max(dices), 4) if dices else None,
        },
        "backend": "ants",
        "transform": "Affine",
        "total_seconds": round(time.time() - t0, 1),
    }
    dump_json(paths.template_meta, meta)
    log(f"Template saved: {paths.template_image} ({meta['total_seconds']}s total)")
    return {
        "image": current_template,
        "liver_mask": current_mask,
        "meta": meta,
        "image_path": paths.template_image,
        "liver_mask_path": paths.template_liver_mask,
    }


def load_template(paths: ProjectPaths) -> dict[str, Any] | None:
    """Load an existing template, or None if absent/incomplete."""
    if not (paths.template_image.is_file() and paths.template_liver_mask.is_file()):
        return None
    image = sitk.ReadImage(str(paths.template_image))
    mask = sitk.ReadImage(str(paths.template_liver_mask))
    meta: dict[str, Any] = {}
    if paths.template_meta.is_file():
        import json

        meta = json.loads(paths.template_meta.read_text(encoding="utf-8"))
    return {"image": image, "liver_mask": mask, "meta": meta}


def ensure_template(
    paths: ProjectPaths,
    cfg: PreprocessConfig,
    build_case_ids: list[int],
    rebuild: bool = False,
    log=print,
) -> dict[str, Any]:
    """Return existing template, building it if missing (or rebuild=True)."""
    if not rebuild:
        existing = load_template(paths)
        if existing is not None:
            log(f"Using existing template: {paths.template_image}")
            return existing
    return build_template(
        paths, cfg, build_case_ids, seed_case=cfg.template_seed_case, log=log
    )
