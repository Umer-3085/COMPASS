"""Dataset for COMPASS training.

Handles both CT and MR cases on the common template grid (Option A):
- MR patients get the CT-template slice as their CT input (paired via
  the shared template grid — slice k is the same anatomical plane).
- CT patients get their own CT slice; T1/T2 are zeroed.

Slices are loaded from preprocessed .npy files. The template volume is
loaded once from the NIfTI and cached.

Augmentation (rotation + scaling + elastic) is applied identically to
every present channel and the label, per the paper.

Modality dropout (p=0.2 for T1 and T2) is applied in the training loop,
not here — the dataset provides the full-modality sample and base mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.augmentation import (
    random_elastic_2d,
    random_rotate_scale_2d,
    sample_augmentation_params,
)

NUM_CLASSES = 5  # background + liver + right kidney + left kidney + spleen


def load_splits(preprocessed_root: Path, modality: str) -> dict[str, list[int]]:
    """Read train/val/test patient IDs from the split text files."""
    splits = {}
    for split in ("train", "val", "test"):
        f = preprocessed_root / "splits" / modality.lower() / f"{split}_patients.txt"
        splits[split] = [int(x) for x in f.read_text().split()]
    return splits


def _load_template_volume(template_nii: Path) -> np.ndarray:
    """Load CT template as (z, y, x) float32 numpy array."""
    img = sitk.ReadImage(str(template_nii))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _apply_affine_np(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    transform: sitk.Transform,
    size_yx: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Apply one affine transform to numpy images and labels via SimpleITK."""
    ref = sitk.GetImageFromArray(images[0].astype(np.float32))
    out_imgs = [
        sitk.GetArrayFromImage(
            sitk.Resample(sitk.GetImageFromArray(img.astype(np.float32)), ref,
                          transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        )
        for img in images
    ]
    out_labs = [
        sitk.GetArrayFromImage(
            sitk.Resample(sitk.GetImageFromArray(lab.astype(np.float32)), ref,
                          transform, sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
        ).astype(np.uint8)
        for lab in labels
    ]
    return out_imgs, out_labs


def _apply_elastic_np(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    transform: sitk.Transform,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Apply one elastic displacement field to numpy images and labels."""
    ref = sitk.GetImageFromArray(images[0].astype(np.float32))
    out_imgs = [
        sitk.GetArrayFromImage(
            sitk.Resample(sitk.GetImageFromArray(img.astype(np.float32)), ref,
                          transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        )
        for img in images
    ]
    out_labs = [
        sitk.GetArrayFromImage(
            sitk.Resample(sitk.GetImageFromArray(lab.astype(np.float32)), ref,
                          transform, sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
        ).astype(np.uint8)
        for lab in labels
    ]
    return out_imgs, out_labs


class ChaosSliceDataset(Dataset):
    """Axial-slice dataset on the common template grid.

    Each sample is one (case, slice) pair. For MR cases the CT input is
    the corresponding template slice; for CT cases T1/T2 are zeroed.

    Args:
        preprocessed_root: Path to preprocessed/ directory.
        split: "train", "val", or "test".
        template_nii: Path to ct_template.nii.gz (loaded once, cached).
        augment: If True, apply rotation+scaling+elastic augmentation
                 identically to all channels and the label (train only).
        rng_seed: Seed for augmentation (one per epoch via set_epoch).
    """

    # Which CT/MR cases to include. None = all in split.
    def __init__(
        self,
        preprocessed_root: Path,
        split: str,
        template_nii: Path,
        augment: bool = False,
        rng_seed: int = 42,
    ) -> None:
        self.root = Path(preprocessed_root)
        self.split = split
        self.augment = augment
        self.rng_seed = rng_seed
        self._epoch = 0

        # Cache template volume (loaded once)
        self.template_vol = _load_template_volume(template_nii)  # (z, y, x)
        self.tpl_z = self.template_vol.shape[0]

        # Build flat index: list of (modality, case_dir, retained_idx)
        self.samples: list[tuple[str, Path, int]] = []
        self._build_index()

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to get different augmentation per epoch."""
        self._epoch = epoch

    def _build_index(self) -> None:
        # MR cases
        mr_splits = load_splits(self.root, "MR")
        for cid in mr_splits[self.split]:
            case_dir = self.root / "mr" / self.split / f"case_{cid:02d}"
            if not case_dir.is_dir():
                continue
            meta = json.loads((case_dir / "meta.json").read_text())
            retained = meta["slice_selection"]["retained_indices"]
            for idx in retained:
                self.samples.append(("MR", case_dir, idx))

        # CT cases
        ct_splits = load_splits(self.root, "CT")
        for cid in ct_splits[self.split]:
            case_dir = self.root / "ct" / self.split / f"case_{cid:02d}"
            if not case_dir.is_dir():
                continue
            meta = json.loads((case_dir / "meta.json").read_text())
            retained = meta["slice_selection"]["retained_indices"]
            for idx in retained:
                self.samples.append(("CT", case_dir, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_slice(self, case_dir: Path, prefix: str, idx: int) -> np.ndarray | None:
        f = case_dir / "slices" / f"{prefix}_{idx:04d}.npy"
        if not f.exists():
            return None
        return np.load(f)

    def _augment(
        self,
        imgs: list[np.ndarray],
        labs: list[np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Apply affine + elastic augmentation identically to all channels."""
        h, w = imgs[0].shape
        params = sample_augmentation_params(rng)

        # Affine (rotation + scaling)
        affine = random_rotate_scale_2d(
            params["angle_deg"], params["scale"], (w, h)
        )
        imgs, labs = _apply_affine_np(imgs, labs, affine, (h, w))

        # Elastic
        if params["apply_elastic"]:
            elastic = random_elastic_2d(
                (h, w), max_displacement=params["elastic_max"], rng=rng
            )
            imgs, labs = _apply_elastic_np(imgs, labs, elastic)

        return imgs, labs

    def __getitem__(self, i: int) -> dict:
        modality, case_dir, idx = self.samples[i]

        # Base augmentation RNG (per-sample, per-epoch)
        rng = np.random.default_rng(
            self.rng_seed + self._epoch * 100000 + i
        ) if self.augment else None

        if modality == "MR":
            # CT input = template slice at same index
            ct_img = self.template_vol[idx].copy()  # (y, x)

            t1 = self._load_slice(case_dir, "t1_inphase", idx)
            t2 = self._load_slice(case_dir, "t2_spir", idx)
            if t1 is None or t2 is None:
                # Should not happen after slice regeneration, but skip safely
                return self.__getitem__((i + 1) % len(self))

            lab1 = self._load_slice(case_dir, "labels_t1", idx)
            lab2 = self._load_slice(case_dir, "labels_t2", idx)
            # Use T2 label when available (5-class), else T1
            label = lab2 if lab2 is not None else lab1

            # CT patient ID: use a fixed template "patient" for contrastive
            # grouping. All MR patients share the same template CT.
            ct_patient = 0

            imgs = [ct_img, t1, t2]
            labs = [label]
            mask = [1.0, 1.0, 1.0]

        else:  # CT
            ct_img = self._load_slice(case_dir, "image", idx)
            label = self._load_slice(case_dir, "label", idx)
            if ct_img is None or label is None:
                return self.__getitem__((i + 1) % len(self))

            t1 = np.zeros_like(ct_img)
            t2 = np.zeros_like(ct_img)

            ct_patient = int(case_dir.name.split("_")[1]) + 100  # offset to separate from template

            imgs = [ct_img, t1, t2]
            labs = [label]
            mask = [1.0, 0.0, 0.0]

        # Augmentation: identical transform on all present channels + label
        if self.augment:
            imgs, labs = self._augment(imgs, labs, rng)

        ct = imgs[0].astype(np.float32)
        t1_out = imgs[1].astype(np.float32)
        t2_out = imgs[2].astype(np.float32)
        label_out = labs[0].astype(np.int64)

        return {
            "ct": torch.from_numpy(ct).unsqueeze(0),       # (1, H, W)
            "t1": torch.from_numpy(t1_out).unsqueeze(0),   # (1, H, W)
            "t2": torch.from_numpy(t2_out).unsqueeze(0),   # (1, H, W)
            "label": torch.from_numpy(label_out),           # (H, W)
            "mask": torch.tensor(mask, dtype=torch.float32),  # (3,)
            "ct_patient": ct_patient,
            "modality": modality,
            "case_id": int(case_dir.name.split("_")[1]),
            "slice_idx": idx,
        }


def make_dataloader(
    preprocessed_root: Path,
    split: str,
    template_nii: Path,
    batch_size: int = 8,
    augment: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for the given split."""
    ds = ChaosSliceDataset(
        preprocessed_root=preprocessed_root,
        split=split,
        template_nii=template_nii,
        augment=augment,
        rng_seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
