"""Label stacking and CHAOS MR grayscale remapping. Never interpolate as continuous."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from PIL import Image

from .config import PreprocessConfig
from .volume_utils import copy_image_with_array


class UnexpectedLabelError(RuntimeError):
    pass


def numeric_suffix(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        raise ValueError(f"No digits in {path.name}")
    return int(nums[-1])


def first_int(name: str) -> int:
    nums = re.findall(r"\d+", name)
    if not nums:
        raise ValueError(f"No digits in {name}")
    return int(nums[0])


def last_int(name: str) -> int:
    nums = re.findall(r"\d+", name)
    if not nums:
        raise ValueError(f"No digits in {name}")
    return int(nums[-1])


def load_png_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def _png_index_map(pngs: list[Path]) -> dict[int, Path]:
    mapping = {}
    for p in pngs:
        mapping[numeric_suffix(p)] = p
    return mapping


def match_ct_mask_files(dicom_files: list[str], mask_dir: Path, instance_numbers: list[int | None]) -> tuple[list[Path], str]:
    pngs = sorted(mask_dir.glob("*.png"), key=lambda p: (numeric_suffix(p), p.name))
    if not pngs:
        raise RuntimeError(f"No PNG masks in {mask_dir}")
    if len(pngs) != len(dicom_files):
        raise RuntimeError(
            f"CT mask count {len(pngs)} != DICOM count {len(dicom_files)} in {mask_dir}"
        )
    by_idx = _png_index_map(pngs)
    names = [Path(f).name for f in dicom_files]

    candidates: list[tuple[str, list[int]]] = []
    try:
        candidates.append(("filename_first_int", [first_int(n) for n in names]))
    except ValueError:
        pass
    try:
        candidates.append(("filename_last_int", [last_int(n) for n in names]))
    except ValueError:
        pass
    if all(v is not None for v in instance_numbers):
        inst = [int(v) for v in instance_numbers]
        candidates.append(("instance_number", inst))
        candidates.append(("instance_minus_1", [v - 1 for v in inst]))
    candidates.append(("sequential_0_n", list(range(len(dicom_files)))))

    for name, keys in candidates:
        if len(set(keys)) != len(keys):
            continue
        if all(k in by_idx for k in keys):
            return [by_idx[k] for k in keys], name

    # Last resort: sorted PNG zip GDCM order (documented as fallback).
    return pngs, "zip_sorted_png_with_gdcm_order"


def stack_masks_as_image(
    png_files: list[Path],
    reference: sitk.Image,
    mapper,
) -> tuple[sitk.Image, dict[str, Any]]:
    expected_yx = (int(reference.GetSize()[1]), int(reference.GetSize()[0]))
    slices = []
    uniques = set()
    for png in png_files:
        arr = load_png_uint8(png)
        if arr.shape != expected_yx:
            raise RuntimeError(
                f"Mask shape {arr.shape} != expected {expected_yx} for {png.name}"
            )
        mapped = mapper(arr)
        uniques.update(int(v) for v in np.unique(mapped))
        slices.append(mapped)
    vol = np.stack(slices, axis=0)
    img = copy_image_with_array(reference, vol.astype(np.uint8), sitk.sitkUInt8)
    return img, {"unique_after_map": sorted(uniques), "n_slices": len(slices)}


def ct_binary_mapper(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.uint8)


def make_mr_mapper(cfg: PreprocessConfig, case_id: int):
    ranges = cfg.mr_label_ranges
    mapping = cfg.mr_label_map

    def mapper(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(arr.shape, dtype=np.uint8)
        used = np.zeros(arr.shape, dtype=bool)
        for name, (lo, hi) in ranges.items():
            sel = (arr >= lo) & (arr <= hi)
            out[sel] = mapping[name]
            used |= sel
        bg = arr == 0
        used |= bg
        unexpected = np.unique(arr[~used])
        if unexpected.size:
            raise UnexpectedLabelError(
                f"MR case {case_id}: unexpected label values {unexpected.tolist()} "
                f"(unique all={np.unique(arr).tolist()})"
            )
        return out

    return mapper


def match_mr_mask_files(dicom_files: list[str], mask_dir: Path) -> tuple[list[Path], str]:
    pngs = list(mask_dir.glob("*.png"))
    by_stem = {p.stem: p for p in pngs}
    missing = []
    ordered = []
    for f in dicom_files:
        stem = Path(f).stem
        if stem not in by_stem:
            missing.append(stem)
        else:
            ordered.append(by_stem[stem])
    if missing:
        raise RuntimeError(
            f"Missing MR masks for {len(missing)} DICOM stems in {mask_dir}: {missing[:8]}"
        )
    if len(pngs) != len(dicom_files):
        extra = sorted(set(p.stem for p in pngs) - set(Path(f).stem for f in dicom_files))
        raise RuntimeError(
            f"MR mask count {len(pngs)} != DICOM {len(dicom_files)}; extra stems {extra[:8]}"
        )
    return ordered, "dicom_filename_stem"


def label_statistics(label: sitk.Image, class_ids: list[int]) -> dict[str, Any]:
    arr = sitk.GetArrayFromImage(label)
    unique = [int(v) for v in np.unique(arr)]
    counts = {str(c): int((arr == c).sum()) for c in class_ids}
    empty_slices = 0
    fg_slices = 0
    for sl in arr:
        if np.any(sl > 0):
            fg_slices += 1
        else:
            empty_slices += 1
    return {
        "unique": unique,
        "counts": counts,
        "foreground_voxels": int((arr > 0).sum()),
        "empty_slices": empty_slices,
        "foreground_slices": fg_slices,
        "n_slices": int(arr.shape[0]),
        "empty_volume": bool((arr > 0).sum() == 0),
    }


def retained_slice_indices(label: sitk.Image, method: str) -> list[int]:
    arr = sitk.GetArrayFromImage(label)
    n = arr.shape[0]
    if method == "all":
        return list(range(n))
    if method == "foreground_only":
        return [i for i in range(n) if np.any(arr[i] > 0)]
    raise ValueError(f"Unknown slice method {method}")


def dice_score(a: sitk.Image, b: sitk.Image, value: int) -> float | None:
    aa = sitk.GetArrayFromImage(a) == value
    bb = sitk.GetArrayFromImage(b) == value
    denom = int(aa.sum() + bb.sum())
    if denom == 0:
        return None
    inter = int(np.logical_and(aa, bb).sum())
    return 2.0 * inter / denom
