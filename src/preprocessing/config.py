"""Central preprocessing configuration. Do not scatter these constants."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PreprocessConfig:
    random_seed: int = 42
    target_spacing: tuple[float, float, float] = (1.5, 1.5, 1.5)

    ct_window_min: float = -200.0
    ct_window_max: float = 250.0

    mri_n4: bool = True
    mri_zscore: bool = True
    n4_shrink_factor: int = 2
    n4_iterations: tuple[int, ...] = (50, 50, 30, 20)
    n4_convergence_threshold: float = 1e-6
    foreground_otsu_bins: int = 128

    register_t2_to_t1: bool = True
    registration_backend: str = "ants"  # "ants" or "sitk"
    registration_type: str = "affine"  # rigid then affine; no deformable by default
    registration_min_liver_dice: float = 0.40

    slice_selection: str = "foreground_train_all_eval"
    # "all" | "foreground_only" | "foreground_train_all_eval"

    ct_mr_paired: bool = False
    extract_slices: bool = True
    write_qc_images: bool = True

    labeled_train: int = 14
    labeled_val: int = 2
    labeled_test: int = 4

    mr_label_ranges: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "liver": (55, 70),
            "right_kidney": (110, 135),
            "left_kidney": (175, 200),
            "spleen": (240, 255),
        }
    )
    mr_label_map: dict[str, int] = field(
        default_factory=lambda: {
            "liver": 1,
            "right_kidney": 2,
            "left_kidney": 3,
            "spleen": 4,
        }
    )

    def slice_method_for_split(self, split: str) -> str:
        if self.slice_selection == "all":
            return "all"
        if self.slice_selection == "foreground_only":
            return "foreground_only"
        if self.slice_selection == "foreground_train_all_eval":
            return "foreground_only" if split == "train" else "all"
        raise ValueError(f"Unknown slice_selection: {self.slice_selection}")

    def to_dict(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["ct_mr_registration"] = False
        payload["unpaired_note"] = (
            "CT and MR are preprocessed independently because CHAOS CT and MR "
            "studies are unpaired. Intra-subject MR sequences (T1/T2) may be "
            "registered when required."
        )
        payload["software"] = software_versions()
        if extra:
            payload.update(extra)
        return payload


def software_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    for name, module in (
        ("simpleitk", "SimpleITK"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("matplotlib", "matplotlib"),
        ("ants", "ants"),
    ):
        try:
            mod = __import__(module)
            versions[name] = str(getattr(mod, "__version__", "ok"))
        except Exception:
            versions[name] = "unavailable"
    return versions


def save_config(cfg: PreprocessConfig, path: Path, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(extra), indent=2), encoding="utf-8")


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_CONFIG = PreprocessConfig()
