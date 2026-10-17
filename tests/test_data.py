"""Smoke test for src/data.py — verifies shapes, labels, and augmentation."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ChaosSliceDataset, make_dataloader

PREPROCESSED = ROOT / "preprocessed"
TEMPLATE = PREPROCESSED / "template" / "ct_template.nii.gz"

assert TEMPLATE.exists(), f"Template missing: {TEMPLATE}"


def test_dataset_basic():
    ds = ChaosSliceDataset(PREPROCESSED, "train", TEMPLATE, augment=False)
    print(f"Train dataset size: {len(ds)}")
    assert len(ds) > 0, "Empty dataset"

    # Check a few samples
    for i in [0, len(ds) // 2, len(ds) - 1]:
        s = ds[i]
        assert s["ct"].shape == (1, 235, 235), f"CT shape {s['ct'].shape}"
        assert s["t1"].shape == (1, 235, 235), f"T1 shape {s['t1'].shape}"
        assert s["t2"].shape == (1, 235, 235), f"T2 shape {s['t2'].shape}"
        assert s["label"].shape == (235, 235), f"Label shape {s['label'].shape}"
        assert s["label"].dtype == torch.int64
        assert s["mask"].shape == (3,)
        labels = s["label"].unique().tolist()
        assert all(0 <= l <= 4 for l in labels), f"Bad labels: {labels}"
        print(f"  [{i}] {s['modality']} case={s['case_id']} slice={s['slice_idx']} "
              f"labels={labels} mask={s['mask'].tolist()}")

    # Check modality distribution
    mr_count = sum(1 for mod, _, _ in ds.samples if mod == "MR")
    ct_count = len(ds.samples) - mr_count
    print(f"  MR slices: {mr_count}, CT slices: {ct_count}")
    assert mr_count > 0 and ct_count > 0, "Need both MR and CT samples"

    # Check MR patient has all 3 modalities present
    for i, (mod, _, _) in enumerate(ds.samples):
        if mod == "MR":
            s = ds[i]
            assert s["mask"].tolist() == [1.0, 1.0, 1.0], "MR mask wrong"
            assert s["t1"].abs().sum() > 0, "MR T1 is zero"
            assert s["t2"].abs().sum() > 0, "MR T2 is zero"
            assert s["ct"].abs().sum() > 0, "MR CT (template) is zero"
            break

    # Check CT patient has T1/T2 zeroed
    for i, (mod, _, _) in enumerate(ds.samples):
        if mod == "CT":
            s = ds[i]
            assert s["mask"].tolist() == [1.0, 0.0, 0.0], "CT mask wrong"
            assert s["t1"].abs().sum() == 0, "CT T1 should be zero"
            assert s["t2"].abs().sum() == 0, "CT T2 should be zero"
            assert s["ct"].abs().sum() > 0, "CT image is zero"
            # CT labels should only be 0 or 1 (liver-only)
            labels = s["label"].unique().tolist()
            assert all(l in (0, 1) for l in labels), f"CT labels not liver-only: {labels}"
            break

    print("  [OK] basic dataset test")


def test_dataloader():
    dl = make_dataloader(PREPROCESSED, "train", TEMPLATE, batch_size=4, augment=False)
    batch = next(iter(dl))
    assert batch["ct"].shape == (4, 1, 235, 235), f"CT batch {batch['ct'].shape}"
    assert batch["t1"].shape == (4, 1, 235, 235)
    assert batch["t2"].shape == (4, 1, 235, 235)
    assert batch["label"].shape == (4, 235, 235)
    assert batch["mask"].shape == (4, 3)
    print(f"  [OK] dataloader batch: ct={batch['ct'].shape} label={batch['label'].shape}")


def test_augmentation():
    ds = ChaosSliceDataset(PREPROCESSED, "train", TEMPLATE, augment=True)
    # Two calls to same sample with different epochs should give different results
    s1 = ds[0]
    ds.set_epoch(1)
    s2 = ds[0]
    assert not torch.equal(s1["ct"], s2["ct"]), "Augmentation produced identical CT"
    assert not torch.equal(s1["label"], s2["label"]), "Augmentation produced identical label"
    # But shapes should be preserved
    assert s1["ct"].shape == (1, 235, 235)
    assert s2["ct"].shape == (1, 235, 235)
    print("  [OK] augmentation produces different outputs per epoch")


def test_val_test():
    for split in ("val", "test"):
        ds = ChaosSliceDataset(PREPROCESSED, split, TEMPLATE, augment=False)
        print(f"  {split} dataset size: {len(ds)}")
        assert len(ds) > 0
        s = ds[0]
        assert s["ct"].shape == (1, 235, 235)
    print("  [OK] val/test datasets")


if __name__ == "__main__":
    print("=== test_dataset_basic ===")
    test_dataset_basic()
    print("=== test_dataloader ===")
    test_dataloader()
    print("=== test_augmentation ===")
    test_augmentation()
    print("=== test_val_test ===")
    test_val_test()
    print("\nALL TESTS PASSED")
