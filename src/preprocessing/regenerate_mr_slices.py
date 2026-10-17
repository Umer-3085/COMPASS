"""Regenerate MR slice .npy files from correct .nii.gz volumes.

Fixes stale slices left by the pre-template run. Reads each case's
NIfTI volumes (verified correct on the template grid), extracts the
retained axial slices, and overwrites the .npy files.

Usage:
    python -m src.preprocessing.regenerate_mr_slices            # all MR
    python -m src.preprocessing.regenerate_mr_slices --only 3,15,36
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.preprocessing.paths import ProjectPaths
from src.preprocessing.volume_utils import extract_axial_slices


def regenerate_case(case_dir: Path) -> dict:
    """Regenerate all slice .npy files for one MR case from its .nii.gz volumes."""
    meta_path = case_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    retained = meta["slice_selection"]["retained_indices"]
    axis = meta["slice_selection"]["axial_numpy_axis"]
    slice_dir = case_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    # Load volumes (source of truth — verified on template grid)
    t1i_vol = sitk.ReadImage(str(case_dir / "t1_inphase.nii.gz"))
    t1o_vol = sitk.ReadImage(str(case_dir / "t1_outphase.nii.gz"))
    t2_vol = sitk.ReadImage(str(case_dir / "t2_spir.nii.gz"))
    lab1_path = case_dir / "labels_t1.nii.gz"
    lab2_path = case_dir / "labels_t2.nii.gz"
    lab1_vol = sitk.ReadImage(str(lab1_path)) if lab1_path.exists() else None
    lab2_vol = sitk.ReadImage(str(lab2_path)) if lab2_path.exists() else None

    # Extract retained axial slices
    t1i = extract_axial_slices(t1i_vol, retained)
    t1o = extract_axial_slices(t1o_vol, retained)
    t2s = extract_axial_slices(t2_vol, retained)
    l1 = extract_axial_slices(lab1_vol, retained) if lab1_vol is not None else None
    l2 = extract_axial_slices(lab2_vol, retained) if lab2_vol is not None else None

    written = 0
    for j, sl in enumerate(retained):
        np.save(slice_dir / f"t1_inphase_{sl:04d}.npy", t1i[j].astype(np.float32))
        np.save(slice_dir / f"t1_outphase_{sl:04d}.npy", t1o[j].astype(np.float32))
        np.save(slice_dir / f"t2_spir_{sl:04d}.npy", t2s[j].astype(np.float32))
        if l1 is not None:
            np.save(slice_dir / f"labels_t1_{sl:04d}.npy", l1[j].astype(np.uint8))
        if l2 is not None:
            np.save(slice_dir / f"labels_t2_{sl:04d}.npy", l2[j].astype(np.uint8))
        written += 1

    # Remove stale files at indices NOT in retained (leftover from old runs)
    removed = 0
    retained_set = set(retained)
    for f in slice_dir.glob("*.npy"):
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1])
            if idx not in retained_set:
                f.unlink()
                removed += 1

    return {"written": written, "removed_stale": removed, "axis": axis}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None, help="Comma-separated case IDs")
    args = ap.parse_args()

    paths = ProjectPaths(Path(__file__).resolve().parents[2])
    mr_root = paths.preprocessed / "mr"
    only = [int(x) for x in args.only.split(",")] if args.only else None

    for split in ("train", "val", "test"):
        split_dir = mr_root / split
        if not split_dir.is_dir():
            continue
        for case_dir in sorted(split_dir.iterdir()):
            if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
                continue
            cid = int(case_dir.name.split("_")[1])
            if only and cid not in only:
                continue
            result = regenerate_case(case_dir)
            print(f"{split}/{case_dir.name}: wrote {result['written']} slices, "
                  f"removed {result['removed_stale']} stale files")

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
