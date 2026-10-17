"""Tests for src/kaggle_setup.py (mount discovery, hotfix overlay, sanity)."""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kaggle_setup import (MARKER, discover, find_hotfix_roots, overlay,
                              prepare, sanity_check)

OK = (235, 235)
STALE = (323, 323)


def _write_case(case_dir: Path, shape) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "meta.json").write_text(json.dumps({"fake": True}))
    sl = case_dir / "slices"
    sl.mkdir(exist_ok=True)
    np.save(sl / "t1_inphase_0000.npy", np.zeros(shape, dtype=np.uint8))


def _make_fixture(tmp: Path) -> tuple[Path, Path]:
    """Nested mount (as in the user's dataset) + a case_01fix hotfix."""
    data = (tmp / "inputs" / "datasets" / "compass-preprocessed"
            / "compass_preprocessed" / "preprocessed")
    (data / "template").mkdir(parents=True)
    (data / "template" / "ct_template.nii.gz").write_bytes(b"fake")
    _write_case(data / "mr" / "train" / "case_01", STALE)   # stale
    _write_case(data / "mr" / "train" / "case_02", OK)
    _write_case(data / "ct" / "train" / "case_05", OK)

    fix = (tmp / "inputs" / "datasets" / "compass-preprocessed"
           / "case_01fix")
    _write_case(fix / "mr" / "train" / "case_01", OK)       # fixed
    return tmp / "inputs", data


def test_discovery_and_overlay(tmp: Path, inputs: Path, data: Path) -> None:
    work = tmp / "work" / "compass_data"

    # discovery finds the nested tree, not the mount root
    found = discover(inputs)
    assert found == data

    # hotfix root detected, and NOT the data root itself
    fixes = find_hotfix_roots(inputs, found)
    assert [f.name for f in fixes] == ["case_01fix"]

    prepare(inputs, work)

    # stale case_01 was replaced by the hotfix (235, not 323)
    arr = np.load(work / "mr" / "train" / "case_01" / "slices"
                  / "t1_inphase_0000.npy")
    assert arr.shape == OK
    # untouched cases still present
    assert (work / "mr" / "train" / "case_02" / "meta.json").exists()
    assert (work / "ct" / "train" / "case_05" / "meta.json").exists()
    assert (work / "template" / "ct_template.nii.gz").exists()
    assert (work / MARKER).read_text().strip() == str(work)
    print("  [discovery + overlay] OK")


def test_idempotent_rerun(tmp: Path, inputs: Path) -> None:
    work = tmp / "work2" / "compass_data"
    prepare(inputs, work)
    prepare(inputs, work)                      # second run: no re-copy crash
    arr = np.load(work / "mr" / "train" / "case_01" / "slices"
                  / "t1_inphase_0000.npy")
    assert arr.shape == OK
    print("  [idempotent re-run] OK")


def test_stale_without_hotfix_fails(tmp: Path, data: Path) -> None:
    work = tmp / "work3" / "compass_data"
    try:
        prepare(data.parent, work)             # only the stale tree
    except RuntimeError as exc:
        assert "template grid" in str(exc)
        print("  [stale data caught by sanity check] OK")
        return
    raise AssertionError("sanity check should have failed on stale data")


def test_explicit_data_root(tmp: Path, data: Path) -> None:
    work = tmp / "work4" / "compass_data"
    found = discover(tmp / "nonexistent", explicit=data)   # skips search
    assert found == data
    # overlay of zero roots on explicit root is a clean no-op
    n = overlay(work if work.exists() else data, [])
    assert n == 0
    print("  [explicit data root] OK")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="kaggle_setup_test_"))
    try:
        inputs, data = _make_fixture(tmp)
        print("=== test_discovery_and_overlay ===")
        test_discovery_and_overlay(tmp, inputs, data)
        print("=== test_idempotent_rerun ===")
        test_idempotent_rerun(tmp, inputs)
        print("=== test_stale_without_hotfix_fails ===")
        test_stale_without_hotfix_fails(tmp, data)
        print("=== test_explicit_data_root ===")
        test_explicit_data_root(tmp, data)
        print("\nALL KAGGLE-SETUP TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
