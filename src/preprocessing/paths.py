"""Dataset and output path layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def train_sets(self) -> Path:
        return self.raw / "CHAOS_Train_Sets" / "Train_Sets"

    @property
    def test_sets(self) -> Path:
        return self.raw / "CHAOS_Test_Sets" / "Test_Sets"

    @property
    def ct_train(self) -> Path:
        return self.train_sets / "CT"

    @property
    def mr_train(self) -> Path:
        return self.train_sets / "MR"

    @property
    def ct_official_test(self) -> Path:
        return self.test_sets / "CT"

    @property
    def mr_official_test(self) -> Path:
        return self.test_sets / "MR"

    @property
    def preprocessed(self) -> Path:
        return self.root / "preprocessed"

    @property
    def template(self) -> Path:
        return self.preprocessed / "template"

    @property
    def template_image(self) -> Path:
        return self.template / "ct_template.nii.gz"

    @property
    def template_liver_mask(self) -> Path:
        return self.template / "ct_template_liver_mask.nii.gz"

    @property
    def template_meta(self) -> Path:
        return self.template / "template_meta.json"

    @property
    def splits(self) -> Path:
        return self.preprocessed / "splits"

    @property
    def qc(self) -> Path:
        return self.preprocessed / "qc"

    def ct_case_raw(self, case_id: int, official_test: bool = False) -> Path:
        root = self.ct_official_test if official_test else self.ct_train
        return root / str(case_id)

    def mr_case_raw(self, case_id: int, official_test: bool = False) -> Path:
        root = self.mr_official_test if official_test else self.mr_train
        return root / str(case_id)

    def processed_case(self, modality: str, split: str, case_id: int) -> Path:
        return self.preprocessed / modality / split / f"case_{case_id:02d}"


def list_numeric_case_dirs(folder: Path) -> list[int]:
    if not folder.is_dir():
        return []
    ids = []
    for p in folder.iterdir():
        if p.is_dir() and p.name.isdigit():
            ids.append(int(p.name))
    return sorted(ids)
