"""Intra-subject T2 → T1 registration. CT↔MR registration is forbidden."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .config import PreprocessConfig
from .dicom_utils import geometry_dict
from .label_utils import dice_score
from .volume_utils import resample_to_reference


class RegistrationError(RuntimeError):
    pass


def register_t2_to_t1(
    t1: sitk.Image,
    t2: sitk.Image,
    t2_label: sitk.Image | None,
    cfg: PreprocessConfig,
) -> dict[str, Any]:
    """Register T2 (moving) to T1 InPhase (fixed). Never register CT to MR."""
    backend = cfg.registration_backend
    ants_ok = False
    if backend == "ants":
        try:
            import ants  # noqa: F401

            ants_ok = True
        except Exception:
            ants_ok = False
            backend = "sitk"

    if backend == "ants" and ants_ok:
        result = _register_ants(t1, t2, t2_label)
    else:
        result = _register_sitk(t1, t2, t2_label)

    result["backend_requested"] = cfg.registration_backend
    result["backend_used"] = result.get("backend", backend)
    result["transform_type"] = cfg.registration_type
    result["fixed"] = "t1_inphase"
    result["moving"] = "t2_spir"
    result["ct_mr_registration"] = False

    warped = result["warped_image"]
    if tuple(warped.GetSize()) != tuple(t1.GetSize()):
        warped = resample_to_reference(warped, t1, is_label=False)
        result["warped_image"] = warped
    if t2_label is not None and result.get("warped_label") is not None:
        wlab = result["warped_label"]
        if tuple(wlab.GetSize()) != tuple(t1.GetSize()):
            wlab = resample_to_reference(wlab, t1, is_label=True)
            result["warped_label"] = wlab

    result["warped_geometry"] = geometry_dict(result["warped_image"])
    result["fixed_geometry"] = geometry_dict(t1)
    return result


def _register_ants(
    fixed: sitk.Image, moving: sitk.Image, moving_label: sitk.Image | None
) -> dict[str, Any]:
    import ants

    with tempfile.TemporaryDirectory(prefix="chaos_ants_") as td:
        td = Path(td)
        f_path = td / "fixed.nii.gz"
        m_path = td / "moving.nii.gz"
        sitk.WriteImage(sitk.Cast(fixed, sitk.sitkFloat32), str(f_path))
        sitk.WriteImage(sitk.Cast(moving, sitk.sitkFloat32), str(m_path))
        fa = ants.image_read(str(f_path))
        ma = ants.image_read(str(m_path))
        reg = ants.registration(
            fixed=fa,
            moving=ma,
            type_of_transform="Affine",
            verbose=False,
        )
        warped_a = ants.apply_transforms(
            fixed=fa,
            moving=ma,
            transformlist=reg["fwdtransforms"],
            interpolator="linear",
        )
        w_path = td / "warped.nii.gz"
        ants.image_write(warped_a, str(w_path))
        warped = sitk.ReadImage(str(w_path))
        warped.CopyInformation(fixed)

        warped_label = None
        if moving_label is not None:
            l_path = td / "moving_label.nii.gz"
            sitk.WriteImage(sitk.Cast(moving_label, sitk.sitkUInt8), str(l_path))
            la = ants.image_read(str(l_path))
            wl = ants.apply_transforms(
                fixed=fa,
                moving=la,
                transformlist=reg["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            wl_path = td / "warped_label.nii.gz"
            ants.image_write(wl, str(wl_path))
            warped_label = sitk.ReadImage(str(wl_path))
            warped_label = sitk.Cast(warped_label, sitk.sitkUInt8)
            warped_label.CopyInformation(fixed)

        return {
            "backend": "ants",
            "performed": True,
            "type": "affine",
            "transforms": list(reg.get("fwdtransforms", [])),
            "warped_image": sitk.Cast(warped, sitk.sitkFloat32),
            "warped_label": warped_label,
        }


def _register_sitk(
    fixed: sitk.Image, moving: sitk.Image, moving_label: sitk.Image | None
) -> dict[str, Any]:
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    rigid = sitk.CenteredTransformInitializer(
        fixed_f,
        moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rigid = _run_mi_registration(fixed_f, moving_f, rigid, scales=[8, 4, 2])

    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid.GetFixedParameters()[:3] if hasattr(rigid, "GetFixedParameters") else (0, 0, 0))
    try:
        affine.SetMatrix(sitk.Euler3DTransform(rigid).GetMatrix())
        affine.SetTranslation(sitk.Euler3DTransform(rigid).GetTranslation())
        affine.SetCenter(sitk.Euler3DTransform(rigid).GetCenter())
    except Exception:
        composite = sitk.CompositeTransform(rigid)
        affine = sitk.CenteredTransformInitializer(
            fixed_f,
            moving_f,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    affine = _run_mi_registration(fixed_f, moving_f, affine, scales=[4, 2, 1])

    warped = sitk.Resample(
        moving_f,
        fixed_f,
        affine,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    warped_label = None
    if moving_label is not None:
        warped_label = sitk.Resample(
            sitk.Cast(moving_label, sitk.sitkUInt8),
            fixed_f,
            affine,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    return {
        "backend": "sitk",
        "performed": True,
        "type": "rigid_then_affine",
        "warped_image": warped,
        "warped_label": warped_label,
    }


def _run_mi_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
    scales: list[int],
) -> sitk.Transform:
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=80,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    shrink = scales
    sigmas = [float(s) / 2.0 for s in scales]
    reg.SetShrinkFactorsPerLevel(shrink)
    reg.SetSmoothingSigmasPerLevel(sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(transform, inPlace=False)
    try:
        out = reg.Execute(fixed, moving)
    except Exception:
        return transform
    return out


def evaluate_registration(
    t1_label: sitk.Image | None,
    t2_label_warped: sitk.Image | None,
    min_liver_dice: float,
) -> dict[str, Any]:
    if t1_label is None or t2_label_warped is None:
        return {"checked": False, "failed": False, "reason": "no_labels"}
    dices = {}
    for name, value in (("liver", 1), ("right_kidney", 2), ("left_kidney", 3), ("spleen", 4)):
        dices[name] = dice_score(t1_label, t2_label_warped, value)
    liver = dices["liver"]
    failed = liver is not None and liver < min_liver_dice
    return {
        "checked": True,
        "failed": failed,
        "dice": dices,
        "min_liver_dice": min_liver_dice,
        "reason": "low_liver_dice" if failed else "ok",
    }
