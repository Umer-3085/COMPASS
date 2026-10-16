"""Dataset discovery and DICOM metadata audit."""

from __future__ import annotations

from typing import Any

from .dicom_utils import load_dicom_series
from .jsonutil import dump_json
from .paths import ProjectPaths, list_numeric_case_dirs


def discover_dataset(paths: ProjectPaths) -> dict[str, Any]:
    ct_labeled = list_numeric_case_dirs(paths.ct_train)
    mr_labeled = list_numeric_case_dirs(paths.mr_train)
    ct_official = list_numeric_case_dirs(paths.ct_official_test)
    mr_official = list_numeric_case_dirs(paths.mr_official_test)
    overlapping_ids = sorted(set(ct_labeled) & set(mr_labeled))
    return {
        "ct_labeled": ct_labeled,
        "mr_labeled": mr_labeled,
        "ct_official_test": ct_official,
        "mr_official_test": mr_official,
        "shared_folder_ids_not_same_patient": overlapping_ids,
        "note": (
            "Folder IDs are independent. CT/1 and MR/1 are different patients. "
            "Do not pair or register across CT and MR."
        ),
    }


def audit_ct_case(paths: ProjectPaths, case_id: int, official_test: bool = False) -> dict[str, Any]:
    case_dir = paths.ct_case_raw(case_id, official_test=official_test)
    image, files, uid, audit = load_dicom_series(case_dir / "DICOM_anon")
    ground = case_dir / "Ground"
    pngs = sorted(ground.glob("*.png")) if ground.is_dir() else []
    audit["case_id"] = case_id
    audit["modality"] = "CT"
    audit["has_ground"] = bool(pngs)
    audit["ground_count"] = len(pngs)
    audit["ground_matches_dicom"] = (len(pngs) == len(files)) if pngs else None
    return audit


def audit_mr_case(paths: ProjectPaths, case_id: int, official_test: bool = False) -> dict[str, Any]:
    case_dir = paths.mr_case_raw(case_id, official_test=official_test)
    out: dict[str, Any] = {"case_id": case_id, "modality": "MR"}
    for seq, rel in (
        ("t1_inphase", case_dir / "T1DUAL" / "DICOM_anon" / "InPhase"),
        ("t1_outphase", case_dir / "T1DUAL" / "DICOM_anon" / "OutPhase"),
        ("t2_spir", case_dir / "T2SPIR" / "DICOM_anon"),
    ):
        image, files, uid, audit = load_dicom_series(rel)
        out[seq] = {k: v for k, v in audit.items() if k != "files"}
        out[seq]["n_files"] = len(files)
        out[seq]["series_uid"] = uid
    t1g = list((case_dir / "T1DUAL" / "Ground").glob("*.png")) if (case_dir / "T1DUAL" / "Ground").is_dir() else []
    t2g = list((case_dir / "T2SPIR" / "Ground").glob("*.png")) if (case_dir / "T2SPIR" / "Ground").is_dir() else []
    out["t1_ground_count"] = len(t1g)
    out["t2_ground_count"] = len(t2g)
    out["t1_slice_mismatch_t2"] = out["t1_inphase"]["n_files"] != out["t2_spir"]["n_files"]
    out["t1_in_out_count_equal"] = out["t1_inphase"]["n_files"] == out["t1_outphase"]["n_files"]
    return out


def run_metadata_audit(
    paths: ProjectPaths,
    ct_ids: list[int] | None = None,
    mr_ids: list[int] | None = None,
) -> dict[str, Any]:
    discovery = discover_dataset(paths)
    ct_ids = ct_ids if ct_ids is not None else discovery["ct_labeled"]
    mr_ids = mr_ids if mr_ids is not None else discovery["mr_labeled"]
    report: dict[str, Any] = {"discovery": discovery, "ct": {}, "mr": {}, "warnings": []}
    for cid in ct_ids:
        try:
            report["ct"][str(cid)] = {k: v for k, v in audit_ct_case(paths, cid).items() if k != "files"}
        except Exception as exc:
            report["warnings"].append({"ct": cid, "error": f"{type(exc).__name__}: {exc}"})
    for mid in mr_ids:
        try:
            report["mr"][str(mid)] = audit_mr_case(paths, mid)
        except Exception as exc:
            report["warnings"].append({"mr": mid, "error": f"{type(exc).__name__}: {exc}"})
    dump_json(paths.qc / "reports" / "dicom_audit.json", report)
    return report
