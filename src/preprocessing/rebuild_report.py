"""Rebuild preprocessing_report.json from existing meta.json files."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "preprocessed"
    report = {
        "successful_cases": [],
        "failed_cases": [],
        "warnings": [],
        "qc_issues": [],
        "ct_mr_registration": False,
        "note": (
            "CT and MR are preprocessed independently because CHAOS CT and MR studies "
            "are unpaired. Intra-subject MR sequences (T1/T2) may be registered when required."
        ),
    }
    dice_rows = []
    for meta in sorted(root.glob("**/meta.json")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        report["successful_cases"].append(
            {"modality": data["modality"], "case_id": data["case_id"], "split": data["split"]}
        )
        if data["modality"] == "MR":
            reg = data.get("registration") or {}
            qc = reg.get("qc") or {}
            if qc.get("dice"):
                dice_rows.append((data["case_id"], qc["dice"], reg.get("failed"), data.get("t2_on_t1_grid")))
            if reg.get("failed"):
                report["warnings"].append(
                    {"modality": "MR", "case_id": data["case_id"], "warning": "T2->T1 registration QC failed"}
                )
                report["qc_issues"].append(
                    {
                        "modality": "MR",
                        "case_id": data["case_id"],
                        "issues": ["T2->T1 registration QC failed (low organ overlap)"],
                    }
                )
    ct_n = sum(1 for x in report["successful_cases"] if x["modality"] == "CT")
    mr_n = sum(1 for x in report["successful_cases"] if x["modality"] == "MR")
    print(f"CT {ct_n} MR {mr_n}")
    print("T2->T1 liver Dice:")
    for cid, dice, failed, on in sorted(dice_rows):
        liver = dice.get("liver")
        print(f"  MR {cid:2d} liver={liver} failed={failed} t2_on_t1={on}")
    dest = root / "qc" / "preprocessing_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
