"""MR preprocessing: N4, z-score, register T1/T2 to the common CT template.

Paper requirement: "align the MRI sequences to the CT reference volume ...
same slice indices applied across CT, MRI, and corresponding labels."

CHAOS CT/MR are unpaired patients, so alignment goes through the CT mean
template: T1 (with out-phase as extra moving image) and T2 are each
registered to the template grid with ANTs (Affine, escalating to SyN on QC
failure). After warping, every MR output shares the template grid with all
CT outputs, so slice indices are identical across modalities and labels.

Legacy path (template=None): old T2→T1 registration, kept for tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk

from .config import PreprocessConfig
from .dicom_utils import geometry_dict, geometry_mismatch, load_dicom_series
from .label_utils import (
    UnexpectedLabelError,
    label_statistics,
    make_mr_mapper,
    match_mr_mask_files,
    retained_slice_indices,
    stack_masks_as_image,
)
from .paths import ProjectPaths
from .registration import evaluate_registration, register_t2_to_t1, register_to_template
from .volume_utils import (
    assert_same_physical_grid,
    extract_axial_slices,
    foreground_zscore,
    make_isotropic_reference,
    n4_correct,
    numpy_axial_axis,
    otsu_foreground_mask,
    resample_to_reference,
    write_nifti,
)


def _load_mr_sequence(folder) -> tuple[sitk.Image, list[str], str, dict[str, Any]]:
    return load_dicom_series(folder)


def _load_mr_labels(dicom_files, mask_dir, reference, cfg, case_id):
    if not mask_dir.is_dir() or not any(mask_dir.glob("*.png")):
        return None, {"present": False}
    pngs, method = match_mr_mask_files(dicom_files, mask_dir)
    mapper = make_mr_mapper(cfg, case_id)
    labels, stack_info = stack_masks_as_image(pngs, reference, mapper)
    unique = set(stack_info["unique_after_map"])
    if not unique.issubset({0, 1, 2, 3, 4}):
        raise UnexpectedLabelError(f"MR {case_id}: mapped labels {sorted(unique)}")
    return labels, {
        "present": True,
        "match_method": method,
        "stack": stack_info,
        "original": label_statistics(labels, [0, 1, 2, 3, 4]),
    }


def process_mr_case(
    paths: ProjectPaths,
    cfg: PreprocessConfig,
    case_id: int,
    split: str,
    official_test: bool = False,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = paths.mr_case_raw(case_id, official_test=official_test)
    t1_in, t1_in_files, t1_in_uid, t1_in_audit = _load_mr_sequence(
        case_dir / "T1DUAL" / "DICOM_anon" / "InPhase"
    )
    t1_out, t1_out_files, t1_out_uid, t1_out_audit = _load_mr_sequence(
        case_dir / "T1DUAL" / "DICOM_anon" / "OutPhase"
    )
    t2, t2_files, t2_uid, t2_audit = _load_mr_sequence(case_dir / "T2SPIR" / "DICOM_anon")

    t1_out_geom = geometry_mismatch(t1_in, t1_out)
    if not t1_out_geom["compatible"]:
        t1_out = resample_to_reference(t1_out, t1_in, is_label=False)
        t1_out_resampled_to_in = True
    else:
        t1_out_resampled_to_in = False

    labels_t1, labels_t1_meta = _load_mr_labels(
        t1_in_files, case_dir / "T1DUAL" / "Ground", t1_in, cfg, case_id
    )
    labels_t2, labels_t2_meta = _load_mr_labels(
        t2_files, case_dir / "T2SPIR" / "Ground", t2, cfg, case_id
    )

    t1_in_n4, t1_mask, t1_n4_meta = n4_correct(t1_in, cfg)
    t1_out_n4, t1_out_mask, t1_out_n4_meta = n4_correct(t1_out, cfg)
    t2_n4, t2_mask, t2_n4_meta = n4_correct(t2, cfg)

    if template is not None and cfg.register_to_template:
        return _finish_mr_on_template(
            paths=paths,
            cfg=cfg,
            case_id=case_id,
            split=split,
            official_test=official_test,
            template=template,
            original=(t1_in, t1_out, t2),
            uids=(t1_in_uid, t1_out_uid, t2_uid),
            audits=(t1_in_audit, t1_out_audit, t2_audit),
            n4_images=(t1_in_n4, t1_out_n4, t2_n4),
            n4_metas=(t1_n4_meta, t1_out_n4_meta, t2_n4_meta),
            labels=(labels_t1, labels_t2),
            label_metas=(labels_t1_meta, labels_t2_meta),
            t1_out_geom=t1_out_geom,
            t1_out_resampled_to_in=t1_out_resampled_to_in,
        )

    # ---------------- Legacy path: T2 -> T1, native grids ----------------
    registration_meta: dict[str, Any] = {
        "performed": False,
        "type": "none",
        "ct_mr_registration": False,
    }
    t2_for_grid = t2_n4
    labels_t2_for_grid = labels_t2
    t2_on_t1_grid = False
    registration_qc = {"checked": False}

    if cfg.register_t2_to_t1:
        reg = register_t2_to_t1(t1_in_n4, t2_n4, labels_t2, cfg)
        t2_warped = reg["warped_image"]
        labels_t2_warped = reg.get("warped_label")
        if labels_t1 is not None and labels_t2_warped is not None:
            registration_qc = evaluate_registration(
                labels_t1, labels_t2_warped, cfg.registration_min_liver_dice
            )
        if registration_qc.get("failed"):
            registration_meta = {
                "performed": True,
                "type": reg.get("type"),
                "backend": reg.get("backend_used"),
                "failed": True,
                "qc": registration_qc,
                "ct_mr_registration": False,
                "kept_t2_native_grid": True,
            }
        else:
            t2_for_grid = t2_warped
            labels_t2_for_grid = labels_t2_warped
            t2_on_t1_grid = True
            registration_meta = {
                "performed": True,
                "type": reg.get("type"),
                "backend": reg.get("backend_used"),
                "failed": False,
                "qc": registration_qc,
                "ct_mr_registration": False,
                "kept_t2_native_grid": False,
            }

    reference = make_isotropic_reference(t1_in_n4, cfg.target_spacing)
    t1_in_iso = resample_to_reference(t1_in_n4, reference, is_label=False)
    t1_out_iso = resample_to_reference(t1_out_n4, reference, is_label=False)
    t1_mask_iso = resample_to_reference(t1_mask, reference, is_label=True)

    if t2_on_t1_grid:
        t2_iso = resample_to_reference(t2_for_grid, reference, is_label=False)
        t2_mask_on_t1 = resample_to_reference(t2_mask, t1_in_n4, is_label=True)
        t2_mask_iso = resample_to_reference(t2_mask_on_t1, reference, is_label=True)
    else:
        t2_ref = make_isotropic_reference(t2_n4, cfg.target_spacing)
        t2_iso = resample_to_reference(t2_n4, t2_ref, is_label=False)
        t2_mask_iso = resample_to_reference(t2_mask, t2_ref, is_label=True)

    labels_t1_iso = None
    if labels_t1 is not None:
        labels_t1_iso = resample_to_reference(labels_t1, reference, is_label=True)
        labels_t1_meta["processed"] = label_statistics(labels_t1_iso, [0, 1, 2, 3, 4])
        if not set(labels_t1_meta["processed"]["unique"]).issubset({0, 1, 2, 3, 4}):
            raise RuntimeError(f"MR {case_id}: T1 resampled labels invalid")

    labels_t2_iso = None
    if labels_t2_for_grid is not None:
        if t2_on_t1_grid:
            labels_t2_iso = resample_to_reference(labels_t2_for_grid, reference, is_label=True)
        else:
            t2_ref = make_isotropic_reference(t2_n4, cfg.target_spacing)
            labels_t2_iso = resample_to_reference(labels_t2, t2_ref, is_label=True)
        labels_t2_meta["processed"] = label_statistics(labels_t2_iso, [0, 1, 2, 3, 4])
        if not set(labels_t2_meta["processed"]["unique"]).issubset({0, 1, 2, 3, 4}):
            raise RuntimeError(f"MR {case_id}: T2 resampled labels invalid")

    t1_in_z, t1_in_zmeta = foreground_zscore(t1_in_iso, t1_mask_iso, cfg)
    t1_out_z, t1_out_zmeta = foreground_zscore(t1_out_iso, t1_mask_iso, cfg)
    t2_z, t2_zmeta = foreground_zscore(t2_iso, t2_mask_iso, cfg)

    if t2_on_t1_grid:
        assert_same_physical_grid(t1_in_z, t2_z, "t1_inphase", "t2_spir")
        if labels_t1_iso is not None:
            assert_same_physical_grid(t1_in_z, labels_t1_iso, "t1_inphase", "labels_t1")
        if labels_t2_iso is not None:
            assert_same_physical_grid(t1_in_z, labels_t2_iso, "t1_inphase", "labels_t2")

    return _write_mr_outputs(
        paths=paths,
        cfg=cfg,
        case_id=case_id,
        out_split=("official_test" if official_test else split),
        images=(t1_in_z, t1_out_z, t2_z),
        labels=(labels_t1_iso, labels_t2_iso),
        label_metas=(labels_t1_meta, labels_t2_meta),
        zmetas=(t1_in_zmeta, t1_out_zmeta, t2_zmeta),
        n4_metas=(t1_n4_meta, t1_out_n4_meta, t2_n4_meta),
        originals=(t1_in, t1_out, t2),
        uids=(t1_in_uid, t1_out_uid, t2_uid),
        audits=(t1_in_audit, t1_out_audit, t2_audit),
        registration_meta=registration_meta,
        template_registration={"performed": False, "reason": "no_template"},
        on_template_grid=False,
        t2_on_t1_grid=t2_on_t1_grid,
        t1_out_geom=t1_out_geom,
        t1_out_resampled_to_in=t1_out_resampled_to_in,
    )


def _finish_mr_on_template(
    *,
    paths: ProjectPaths,
    cfg: PreprocessConfig,
    case_id: int,
    split: str,
    official_test: bool,
    template: dict[str, Any],
    original: tuple[sitk.Image, sitk.Image, sitk.Image],
    uids: tuple[str, str, str],
    audits: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    n4_images: tuple[sitk.Image, sitk.Image, sitk.Image],
    n4_metas: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    labels: tuple[sitk.Image | None, sitk.Image | None],
    label_metas: tuple[dict[str, Any], dict[str, Any]],
    t1_out_geom: dict[str, Any],
    t1_out_resampled_to_in: bool,
) -> dict[str, Any]:
    """Register T1 (+out-phase) and T2 to the CT template; shared grid."""
    t1_in_n4, t1_out_n4, t2_n4 = n4_images
    labels_t1, labels_t2 = labels
    labels_t1_meta, labels_t2_meta = label_metas
    tpl_img = template["image"]
    tpl_mask = template.get("liver_mask")

    # --- T1 InPhase -> template (OutPhase rides along on the same transform)
    reg_t1 = register_to_template(
        moving=t1_in_n4,
        template=tpl_img,
        template_liver_mask=tpl_mask,
        moving_label=labels_t1,
        cfg=cfg,
        moving_name="t1_inphase",
        case_name=f"mr_case_{case_id:02d}_t1",
        extra_movings={"t1_outphase": t1_out_n4},
    )
    t1_in_src = reg_t1["warped_image"]
    t1_out_src = reg_t1["warped_extras"].get("t1_outphase", t1_in_src)
    labels_t1_warp = reg_t1["warped_label"]

    # --- T2 SPIR -> template (independent registration, QC-gated)
    reg_t2 = register_to_template(
        moving=t2_n4,
        template=tpl_img,
        template_liver_mask=tpl_mask,
        moving_label=labels_t2,
        cfg=cfg,
        moving_name="t2_spir",
        case_name=f"mr_case_{case_id:02d}_t2",
    )
    t2_src = reg_t2["warped_image"]
    labels_t2_warp = reg_t2["warped_label"]

    # --- intensity standardization on the shared template grid
    t1_mask_tpl = otsu_foreground_mask(t1_in_src, cfg)
    t2_mask_tpl = otsu_foreground_mask(t2_src, cfg)
    t1_in_z, t1_in_zmeta = foreground_zscore(t1_in_src, t1_mask_tpl, cfg)
    t1_out_z, t1_out_zmeta = foreground_zscore(t1_out_src, t1_mask_tpl, cfg)
    t2_z, t2_zmeta = foreground_zscore(t2_src, t2_mask_tpl, cfg)

    # --- everything must sit on one physical grid
    assert_same_physical_grid(t1_in_z, t2_z, "t1_inphase", "t2_spir")
    assert_same_physical_grid(t1_in_z, t1_out_z, "t1_inphase", "t1_outphase")
    if labels_t1_warp is not None:
        assert_same_physical_grid(t1_in_z, labels_t1_warp, "t1_inphase", "labels_t1")
    if labels_t2_warp is not None:
        assert_same_physical_grid(t1_in_z, labels_t2_warp, "t1_inphase", "labels_t2")
    assert_same_physical_grid(t1_in_z, tpl_img, "t1_inphase", "template")

    if labels_t1_warp is not None:
        labels_t1_meta["processed"] = label_statistics(labels_t1_warp, [0, 1, 2, 3, 4])
        if not set(labels_t1_meta["processed"]["unique"]).issubset({0, 1, 2, 3, 4}):
            raise RuntimeError(f"MR {case_id}: T1 template-warped labels invalid")
    if labels_t2_warp is not None:
        labels_t2_meta["processed"] = label_statistics(labels_t2_warp, [0, 1, 2, 3, 4])
        if not set(labels_t2_meta["processed"]["unique"]).issubset({0, 1, 2, 3, 4}):
            raise RuntimeError(f"MR {case_id}: T2 template-warped labels invalid")

    template_registration = {
        "performed": True,
        "on_template_grid": True,
        "template_seed_case": template.get("meta", {}).get("seed_case"),
        "t1": {
            k: v
            for k, v in reg_t1.items()
            if k not in ("warped_image", "warped_label", "warped_extras")
        },
        "t2": {
            k: v
            for k, v in reg_t2.items()
            if k not in ("warped_image", "warped_label", "warped_extras")
        },
        "failed": bool(reg_t1.get("failed") or reg_t2.get("failed")),
    }

    return _write_mr_outputs(
        paths=paths,
        cfg=cfg,
        case_id=case_id,
        out_split=("official_test" if official_test else split),
        images=(t1_in_z, t1_out_z, t2_z),
        labels=(labels_t1_warp, labels_t2_warp),
        label_metas=(labels_t1_meta, labels_t2_meta),
        zmetas=(t1_in_zmeta, t1_out_zmeta, t2_zmeta),
        n4_metas=n4_metas,
        originals=original,
        uids=uids,
        audits=audits,
        registration_meta=template_registration,
        template_registration=template_registration,
        on_template_grid=True,
        t2_on_t1_grid=True,  # all modalities share the template grid
        t1_out_geom=t1_out_geom,
        t1_out_resampled_to_in=t1_out_resampled_to_in,
    )


def _write_mr_outputs(
    *,
    paths: ProjectPaths,
    cfg: PreprocessConfig,
    case_id: int,
    out_split: str,
    images: tuple[sitk.Image, sitk.Image, sitk.Image],
    labels: tuple[sitk.Image | None, sitk.Image | None],
    label_metas: tuple[dict[str, Any], dict[str, Any]],
    zmetas: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    n4_metas: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    originals: tuple[sitk.Image, sitk.Image, sitk.Image],
    uids: tuple[str, str, str],
    audits: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    registration_meta: dict[str, Any],
    template_registration: dict[str, Any],
    on_template_grid: bool,
    t2_on_t1_grid: bool,
    t1_out_geom: dict[str, Any],
    t1_out_resampled_to_in: bool,
) -> dict[str, Any]:
    """Validate grid, write volumes + slices + meta for one MR case."""
    t1_in_z, t1_out_z, t2_z = images
    labels_t1_iso, labels_t2_iso = labels
    labels_t1_meta, labels_t2_meta = label_metas
    t1_in, t1_out, t2 = originals

    out_dir = paths.processed_case("mr", out_split, case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_nifti(out_dir / "t1_inphase.nii.gz", t1_in_z)
    write_nifti(out_dir / "t1_outphase.nii.gz", t1_out_z)
    write_nifti(out_dir / "t2_spir.nii.gz", t2_z)
    if labels_t1_iso is not None:
        write_nifti(out_dir / "labels_t1.nii.gz", labels_t1_iso)
    if labels_t2_iso is not None:
        write_nifti(out_dir / "labels_t2.nii.gz", labels_t2_iso)

    slice_method = cfg.slice_method_for_split(
        "test" if out_split == "official_test" else out_split
    )
    label_for_slices = labels_t1_iso if labels_t1_iso is not None else labels_t2_iso
    if label_for_slices is not None:
        retained = retained_slice_indices(label_for_slices, slice_method)
    else:
        retained = list(range(sitk.GetArrayFromImage(t1_in_z).shape[0]))
        slice_method = "all_unlabeled"

    # On a shared grid (template or T2-on-T1), all sequences use the SAME
    # indices; otherwise T2 keeps its native indices (legacy path only).
    if cfg.extract_slices:
        slice_dir = out_dir / "slices"
        slice_dir.mkdir(parents=True, exist_ok=True)
        t1i = extract_axial_slices(t1_in_z, retained)
        t1o = extract_axial_slices(t1_out_z, retained)
        t2s = extract_axial_slices(t2_z, retained) if t2_on_t1_grid else None
        l1 = extract_axial_slices(labels_t1_iso, retained) if labels_t1_iso is not None else None
        l2 = extract_axial_slices(labels_t2_iso, retained) if (labels_t2_iso is not None and t2_on_t1_grid) else None
        for j, sl in enumerate(retained):
            np.save(slice_dir / f"t1_inphase_{sl:04d}.npy", t1i[j].astype(np.float32))
            np.save(slice_dir / f"t1_outphase_{sl:04d}.npy", t1o[j].astype(np.float32))
            if t2s is not None:
                np.save(slice_dir / f"t2_spir_{sl:04d}.npy", t2s[j].astype(np.float32))
            if l1 is not None:
                np.save(slice_dir / f"labels_t1_{sl:04d}.npy", l1[j].astype(np.uint8))
            if l2 is not None:
                np.save(slice_dir / f"labels_t2_{sl:04d}.npy", l2[j].astype(np.uint8))
        if not t2_on_t1_grid:
            t2_indices = list(range(sitk.GetArrayFromImage(t2_z).shape[0]))
            t2s_all = extract_axial_slices(t2_z, t2_indices)
            for sl, img in zip(t2_indices, t2s_all):
                np.save(slice_dir / f"t2_spir_native_{sl:04d}.npy", img.astype(np.float32))
            if labels_t2_iso is not None:
                l2n = extract_axial_slices(labels_t2_iso, t2_indices)
                for sl, img in zip(t2_indices, l2n):
                    np.save(slice_dir / f"labels_t2_native_{sl:04d}.npy", img.astype(np.uint8))

    meta = {
        "case_id": case_id,
        "modality": "MR",
        "split": out_split,
        "unpaired": True,
        "ct_mr_registration": False,
        "series_uids": {
            "t1_inphase": uids[0],
            "t1_outphase": uids[1],
            "t2_spir": uids[2],
        },
        "original": {
            "t1_inphase": geometry_dict(t1_in),
            "t1_outphase": geometry_dict(t1_out),
            "t2_spir": geometry_dict(t2),
        },
        "t1_in_out_geometry": t1_out_geom,
        "t1_out_resampled_to_in": t1_out_resampled_to_in,
        "t1_t2_slice_count_mismatch": int(t1_in.GetSize()[2]) != int(t2.GetSize()[2]),
        "processed": {
            "t1_inphase": geometry_dict(t1_in_z),
            "t1_outphase": geometry_dict(t1_out_z),
            "t2_spir": geometry_dict(t2_z),
        },
        "processed_size": geometry_dict(t1_in_z)["size"],
        "processed_spacing": geometry_dict(t1_in_z)["spacing"],
        "processed_origin": geometry_dict(t1_in_z)["origin"],
        "processed_direction": geometry_dict(t1_in_z)["direction"],
        "on_template_grid": on_template_grid,
        "n4": {"t1_inphase": n4_metas[0], "t1_outphase": n4_metas[1], "t2_spir": n4_metas[2]},
        "intensity": {
            "method": "n4_then_foreground_zscore",
            "t1_inphase": zmetas[0],
            "t1_outphase": zmetas[1],
            "t2_spir": zmetas[2],
        },
        "labels_t1": labels_t1_meta,
        "labels_t2": labels_t2_meta,
        "registration": registration_meta,
        "template_registration": template_registration,
        "t2_on_t1_grid": t2_on_t1_grid,
        "slice_selection": {
            "method": slice_method,
            "retained_indices": retained,
            "n_retained": len(retained),
            "axial_numpy_axis": numpy_axial_axis(t1_in_z),
        },
        "dicom_audit": {
            "t1_inphase": {k: v for k, v in audits[0].items() if k != "files"},
            "t1_outphase": {k: v for k, v in audits[1].items() if k != "files"},
            "t2_spir": {k: v for k, v in audits[2].items() if k != "files"},
        },
        "n_slices_original": {
            "t1_inphase": int(t1_in.GetSize()[2]),
            "t1_outphase": int(t1_in.GetSize()[2]) if t1_out_resampled_to_in else int(t1_out.GetSize()[2]),
            "t2_spir": int(t2.GetSize()[2]),
        },
        "n_slices_processed": {
            "t1_inphase": int(t1_in_z.GetSize()[2]),
            "t2_spir": int(t2_z.GetSize()[2]),
        },
    }
    return {
        "meta": meta,
        "out_dir": out_dir,
        "t1_in": t1_in_z,
        "t1_out": t1_out_z,
        "t2": t2_z,
        "labels_t1": labels_t1_iso,
        "labels_t2": labels_t2_iso,
        "t2_on_t1_grid": t2_on_t1_grid,
        "on_template_grid": on_template_grid,
        "registration_failed": bool(registration_meta.get("failed")),
        "template_registration": template_registration,
    }
