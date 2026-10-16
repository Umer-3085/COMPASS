"""Patient-level supervised splits. CT IDs and MR IDs are independent."""

from __future__ import annotations

import random
from pathlib import Path

from .config import PreprocessConfig
from .paths import ProjectPaths, list_numeric_case_dirs


def generate_split(ids: list[int], cfg: PreprocessConfig) -> dict[str, list[int]]:
    ids = sorted(int(i) for i in ids)
    n = len(ids)
    n_train, n_val, n_test = cfg.labeled_train, cfg.labeled_val, cfg.labeled_test
    if n_train + n_val + n_test != n:
        raise ValueError(
            f"Split sizes {n_train}/{n_val}/{n_test} do not sum to {n} labeled studies"
        )
    rng = random.Random(cfg.random_seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    split = {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }
    _validate_split(ids, split)
    return split


def _validate_split(all_ids: list[int], split: dict[str, list[int]]) -> None:
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    if train & val or train & test or val & test:
        raise RuntimeError("Split sets overlap")
    if train | val | test != set(all_ids):
        raise RuntimeError("Split sets do not cover all labeled IDs")


def write_split_files(split: dict[str, list[int]], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in split.items():
        (dest_dir / f"{name}_patients.txt").write_text(
            "\n".join(str(i) for i in ids) + ("\n" if ids else ""),
            encoding="utf-8",
        )


def read_split_files(dest_dir: Path) -> dict[str, list[int]]:
    out = {}
    for name in ("train", "val", "test"):
        text = (dest_dir / f"{name}_patients.txt").read_text(encoding="utf-8")
        out[name] = [int(x) for x in text.split() if x.strip()]
    return out


def split_lookup(split: dict[str, list[int]]) -> dict[int, str]:
    lookup = {}
    for name, ids in split.items():
        for i in ids:
            lookup[i] = name
    return lookup


def build_and_save_splits(paths: ProjectPaths, cfg: PreprocessConfig) -> dict[str, dict[str, list[int]]]:
    ct_ids = list_numeric_case_dirs(paths.ct_train)
    mr_ids = list_numeric_case_dirs(paths.mr_train)
    ct_split = generate_split(ct_ids, cfg)
    mr_split = generate_split(mr_ids, cfg)
    write_split_files(ct_split, paths.splits / "ct")
    write_split_files(mr_split, paths.splits / "mr")
    return {"ct": ct_split, "mr": mr_split}
