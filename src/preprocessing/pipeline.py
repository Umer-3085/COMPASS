"""CLI entry: audit → splits → CT/MR preprocess → QC. No CT↔MR pairing."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from .audit import discover_dataset, run_metadata_audit
from .config import DEFAULT_CONFIG, PreprocessConfig, save_config
from .ct_preprocessing import process_ct_case
from .jsonutil import dump_json
from .mr_preprocessing import process_mr_case
from .paths import ProjectPaths
from .qc import save_report, validate_ct_result, validate_mr_result, write_ct_qc, write_mr_qc
from .splits import build_and_save_splits, split_lookup


def parse_id_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Preprocess the local CHAOS dataset. CT and MR are unpaired; "
            "CT↔MR registration is not performed."
        )
    )
    p.add_argument("--root", type=Path, default=None, help="Project root (default: repo root)")
    p.add_argument("--only-ct", type=str, default=None, help="Comma-separated CT IDs")
    p.add_argument("--only-mr", type=str, default=None, help="Comma-separated MR IDs")
    p.add_argument("--include-official-test", action="store_true")
    p.add_argument("--no-register-t2-to-t1", action="store_true")
    p.add_argument("--registration-backend", choices=["ants", "sitk"], default="ants")
    p.add_argument("--skip-slices", action="store_true")
    p.add_argument("--skip-qc-images", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--splits-only", action="store_true")
    return p


def run(args: argparse.Namespace | None = None) -> dict[str, Any]:
    parser = build_parser()
    ns = args if args is not None else parser.parse_args()

    from .config import project_root_from_here

    root = Path(ns.root).resolve() if ns.root else project_root_from_here()
    paths = ProjectPaths(root)
    cfg = PreprocessConfig(
        register_t2_to_t1=not ns.no_register_t2_to_t1,
        registration_backend=ns.registration_backend,
        extract_slices=not ns.skip_slices,
        write_qc_images=not ns.skip_qc_images,
    )

    paths.preprocessed.mkdir(parents=True, exist_ok=True)
    paths.qc.mkdir(parents=True, exist_ok=True)

    discovery = discover_dataset(paths)
    print("Discovered labeled CT:", discovery["ct_labeled"])
    print("Discovered labeled MR:", discovery["mr_labeled"])
    print("Shared folder IDs are NOT the same patients:", discovery["shared_folder_ids_not_same_patient"])

    splits = build_and_save_splits(paths, cfg)
    print("CT split", splits["ct"])
    print("MR split", splits["mr"])

    save_config(
        cfg,
        paths.preprocessed / "config.json",
        extra={"splits": splits, "discovery": discovery},
    )

    only_ct = parse_id_list(ns.only_ct)
    only_mr = parse_id_list(ns.only_mr)

    audit_ct = only_ct if only_ct is not None else discovery["ct_labeled"]
    audit_mr = only_mr if only_mr is not None else discovery["mr_labeled"]
    print("Running DICOM metadata audit...")
    run_metadata_audit(paths, ct_ids=audit_ct, mr_ids=audit_mr)

    if ns.audit_only or ns.splits_only:
        return {"discovery": discovery, "splits": splits}

    report: dict[str, Any] = {
        "successful_cases": [],
        "failed_cases": [],
        "warnings": [],
        "qc_issues": [],
        "note": (
            "CT and MR are preprocessed independently because CHAOS CT and MR studies "
            "are unpaired. Intra-subject MR sequences (T1/T2) may be registered when required."
        ),
        "ct_mr_registration": False,
        "splits": splits,
    }

    ct_lookup = split_lookup(splits["ct"])
    mr_lookup = split_lookup(splits["mr"])
    if only_ct is None and only_mr is None:
        ct_ids = discovery["ct_labeled"]
        mr_ids = discovery["mr_labeled"]
    else:
        ct_ids = only_ct or []
        mr_ids = only_mr or []

    for cid in ct_ids:
        split = ct_lookup[cid]
        print(f"\n=== CT {cid} ({split}) ===")
        try:
            result = process_ct_case(paths, cfg, cid, split)
            dump_json(result["out_dir"] / "meta.json", result["meta"])
            issues = validate_ct_result(result)
            if cfg.write_qc_images:
                write_ct_qc(paths.qc / "ct", cid, result["ct"], result["labels"])
            if issues:
                report["qc_issues"].append({"modality": "CT", "case_id": cid, "issues": issues})
                print("  QC issues:", issues)
            report["successful_cases"].append({"modality": "CT", "case_id": cid, "split": split})
            print("  OK", result["out_dir"])
        except Exception as exc:
            info = {
                "modality": "CT",
                "case_id": cid,
                "split": split,
                "stage": "ct_preprocessing",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            report["failed_cases"].append(info)
            print("  FAILED", info["error"])

    for mid in mr_ids:
        split = mr_lookup[mid]
        print(f"\n=== MR {mid} ({split}) ===")
        try:
            result = process_mr_case(paths, cfg, mid, split)
            dump_json(result["out_dir"] / "meta.json", result["meta"])
            issues = validate_mr_result(result)
            if cfg.write_qc_images:
                write_mr_qc(
                    paths.qc / "mr",
                    mid,
                    result["t1_in"],
                    result["t2"],
                    result["labels_t1"],
                    result["labels_t2"],
                    result["t2_on_t1_grid"],
                    result["registration_failed"],
                )
                if result["t2_on_t1_grid"]:
                    write_mr_qc(
                        paths.qc / "registration",
                        mid,
                        result["t1_in"],
                        result["t2"],
                        result["labels_t1"],
                        result["labels_t2"],
                        True,
                        result["registration_failed"],
                    )
            if issues:
                report["qc_issues"].append({"modality": "MR", "case_id": mid, "issues": issues})
                print("  QC issues:", issues)
            if result["registration_failed"]:
                report["warnings"].append(
                    {"modality": "MR", "case_id": mid, "warning": "T2->T1 registration QC failed"}
                )
            report["successful_cases"].append({"modality": "MR", "case_id": mid, "split": split})
            print("  OK", result["out_dir"])
        except Exception as exc:
            info = {
                "modality": "MR",
                "case_id": mid,
                "split": split,
                "stage": "mr_preprocessing",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            report["failed_cases"].append(info)
            print("  FAILED", info["error"])

    if ns.include_official_test:
        for cid in discovery["ct_official_test"]:
            print(f"\n=== official CT test {cid} ===")
            try:
                result = process_ct_case(paths, cfg, cid, "test", official_test=True)
                dump_json(result["out_dir"] / "meta.json", result["meta"])
                report["successful_cases"].append(
                    {"modality": "CT", "case_id": cid, "split": "official_test"}
                )
            except Exception as exc:
                report["failed_cases"].append(
                    {
                        "modality": "CT",
                        "case_id": cid,
                        "split": "official_test",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        for mid in discovery["mr_official_test"]:
            print(f"\n=== official MR test {mid} ===")
            try:
                result = process_mr_case(paths, cfg, mid, "test", official_test=True)
                dump_json(result["out_dir"] / "meta.json", result["meta"])
                report["successful_cases"].append(
                    {"modality": "MR", "case_id": mid, "split": "official_test"}
                )
            except Exception as exc:
                report["failed_cases"].append(
                    {
                        "modality": "MR",
                        "case_id": mid,
                        "split": "official_test",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    save_report(paths.qc / "preprocessing_report.json", report)
    print("\nDone. Report:", paths.qc / "preprocessing_report.json")
    print("Successful:", len(report["successful_cases"]), "Failed:", len(report["failed_cases"]))
    return report


def main() -> None:
    run(None)


if __name__ == "__main__":
    main()
