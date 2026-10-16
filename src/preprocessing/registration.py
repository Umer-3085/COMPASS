"""Intra-subject T2 → T1 registration (legacy) + template registration.

CT↔MR per-patient registration is impossible (CHAOS CT/MR are unpaired
patients). Cross-modality alignment therefore goes through the common CT
mean template: every CT/T1/T2 volume is registered to the template with ANTs.
"""

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


def ants_available() -> bool:
    try:
        import ants  # noqa: F401

        return True
    except Exception:
        return False


def ants_register_to_fixed(
    fixed: sitk.Image,
    moving: sitk.Image,
    moving_label: sitk.Image | None = None,
    type_of_transform: str = "Affine",
    reg_kwargs: dict[str, Any] | None = None,
    extra_movings: dict[str, sitk.Image] | None = None,
) -> dict[str, Any]:
    """ANTs registration of moving -> fixed grid; warps label/extras too.

    extra_movings: additional images on the *moving* grid (e.g. T1 out-phase
    sharing T1 in-phase geometry) warped with the same transform.
    """
    import ants

    with tempfile.TemporaryDirectory(prefix="compass_ants_") as td:
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
            type_of_transform=type_of_transform,
            verbose=False,
            **(reg_kwargs or {}),
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

        warped_extras: dict[str, sitk.Image] = {}
        for name, extra in (extra_movings or {}).items():
            e_path = td / f"extra_{name}.nii.gz"
            sitk.WriteImage(sitk.Cast(extra, sitk.sitkFloat32), str(e_path))
            ea = ants.image_read(str(e_path))
            we = ants.apply_transforms(
                fixed=fa,
                moving=ea,
                transformlist=reg["fwdtransforms"],
                interpolator="linear",
            )
            we_path = td / f"warped_extra_{name}.nii.gz"
            ants.image_write(we, str(we_path))
            w_extra = sitk.ReadImage(str(we_path))
            w_extra.CopyInformation(fixed)
            warped_extras[name] = sitk.Cast(w_extra, sitk.sitkFloat32)

        return {
            "backend": "ants",
            "type": type_of_transform,
            "warped_image": sitk.Cast(warped, sitk.sitkFloat32),
            "warped_label": warped_label,
            "warped_extras": warped_extras,
        }


def template_registration_plan(cfg: PreprocessConfig) -> list[dict[str, Any]]:
    """Escalation ladder for template registration attempts."""
    plan: list[dict[str, Any]] = [
        {"name": "affine", "type_of_transform": "Affine", "reg_kwargs": {}},
    ]
    if cfg.registration_escalate_to_syn:
        plan.append({"name": "syn", "type_of_transform": "SyN", "reg_kwargs": {}})
        plan.append(
            {
                "name": "affine_extended",
                "type_of_transform": "Affine",
                "reg_kwargs": {"reg_iterations": [40, 20, 0]},
            }
        )
    return plan


def register_to_template(
    moving: sitk.Image,
    template: sitk.Image,
    template_liver_mask: sitk.Image | None,
    moving_label: sitk.Image | None,
    cfg: PreprocessConfig,
    moving_name: str = "volume",
    case_name: str = "case",
    extra_movings: dict[str, sitk.Image] | None = None,
) -> dict[str, Any]:
    """Register moving -> CT template with Affine→SyN escalation.

    QC gate: Dice of the warped liver label (value 1) against the template
    liver mask. Accepts the first attempt with Dice >= cfg threshold; if all
    attempts fail QC, keeps the best-Dice attempt and flags `failed=True`
    (never silently falls back to the native grid).
    """
    if not ants_available():
        raise RegistrationError(
            "Template registration requires ANTs (antspyx), which is not "
            "importable in this environment."
        )

    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_dice: float = -1.0

    for step in template_registration_plan(cfg):
        try:
            reg = ants_register_to_fixed(
                fixed=template,
                moving=moving,
                moving_label=moving_label,
                type_of_transform=step["type_of_transform"],
                reg_kwargs=step.get("reg_kwargs"),
                extra_movings=extra_movings,
            )
        except Exception as exc:
            attempts.append(
                {
                    "step": step["name"],
                    "type": step["type_of_transform"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        dice = None
        if template_liver_mask is not None and reg["warped_label"] is not None:
            d = dice_score(reg["warped_label"], template_liver_mask, 1)
            dice = round(d, 4) if d is not None else None
        qc_pass = dice is not None and dice >= cfg.registration_min_liver_dice
        # No labels at all -> accept blindly but mark unchecked.
        if dice is None and template_liver_mask is None:
            qc_pass = True

        attempt = {
            "step": step["name"],
            "type": step["type_of_transform"],
            "liver_dice_vs_template": dice,
            "qc_pass": qc_pass,
        }
        attempts.append(attempt)

        if dice is not None and dice > best_dice:
            best_dice = dice
            best = reg
        elif best is None:
            best = reg

        if qc_pass:
            return {
                "warped_image": reg["warped_image"],
                "warped_label": reg["warped_label"],
                "warped_extras": reg.get("warped_extras", {}),
                "accepted_step": step["name"],
                "type": step["type_of_transform"],
                "backend": "ants",
                "liver_dice_vs_template": dice,
                "failed": False,
                "attempts": attempts,
                "moving": moving_name,
                "case": case_name,
            }

    # All attempts failed QC (or errored): keep the best-Dice warp, flag it.
    return {
        "warped_image": best["warped_image"] if best else moving,
        "warped_label": best["warped_label"] if best else moving_label,
        "warped_extras": (best.get("warped_extras", {}) if best else {}),
        "accepted_step": None,
        "type": best["type"] if best else None,
        "backend": "ants",
        "liver_dice_vs_template": best_dice if best_dice >= 0 else None,
        "failed": True,
        "attempts": attempts,
        "moving": moving_name,
        "case": case_name,
    }


def ants_register_via_fixed_to_template(
    moving: sitk.Image,
    via_fixed: sitk.Image,
    template: sitk.Image,
    moving_label: sitk.Image | None,
    via_label: sitk.Image | None,
    template_liver_mask: sitk.Image | None,
    type_of_transform: str = "Affine",
    reg_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register moving -> template chained through an intermediate volume.

    Fallback for hard cross-modality cases (e.g. T2 -> CT template) where
    direct registration fails QC: register moving -> via (same patient,
    well-conditioned) and via -> template separately, then concatenate the
    two ANTs transform chains (via transform applied first) so `moving`
    lands on the template grid in one resampling. The output is still on
    the template grid; only the transform path differs.
    """
    import ants

    with tempfile.TemporaryDirectory(prefix="compass_ants_chain_") as td:
        td = Path(td)

        def _as_ants(name: str, img: sitk.Image):
            p = td / name
            sitk.WriteImage(sitk.Cast(img, sitk.sitkFloat32), str(p))
            return ants.image_read(str(p))

        fa = _as_ants("template.nii.gz", template)
        va = _as_ants("via.nii.gz", via_fixed)
        ma = _as_ants("moving.nii.gz", moving)

        # Step 1: moving -> via (within patient). Step 2: via -> template.
        reg_via = ants.registration(
            fixed=va, moving=ma, type_of_transform=type_of_transform,
            verbose=False, **(reg_kwargs or {}),
        )
        reg_tpl = ants.registration(
            fixed=fa, moving=va, type_of_transform=type_of_transform,
            verbose=False, **(reg_kwargs or {}),
        )
        # ANTs applies the LAST transform in the list first: [via->template
        # transforms..., moving->via transforms...] = via-transform first.
        composite = list(reg_tpl["fwdtransforms"]) + list(reg_via["fwdtransforms"])

        warped_a = ants.apply_transforms(
            fixed=fa, moving=ma, transformlist=composite, interpolator="linear"
        )
        wp = td / "warped.nii.gz"
        ants.image_write(warped_a, str(wp))
        warped = sitk.ReadImage(str(wp))
        warped.CopyInformation(template)

        warped_label = None
        warped_label_on_via = None
        if moving_label is not None:
            la = _as_ants("moving_label.nii.gz", sitk.Cast(moving_label, sitk.sitkUInt8))
            wl = ants.apply_transforms(
                fixed=fa, moving=la, transformlist=composite, interpolator="nearestNeighbor"
            )
            wlp = td / "warped_label.nii.gz"
            ants.image_write(wl, str(wlp))
            warped_label = sitk.Cast(sitk.ReadImage(str(wlp)), sitk.sitkUInt8)
            warped_label.CopyInformation(template)
            if via_label is not None:
                wl2 = ants.apply_transforms(
                    fixed=va, moving=la,
                    transformlist=list(reg_via["fwdtransforms"]),
                    interpolator="nearestNeighbor",
                )
                wlp2 = td / "warped_label_on_via.nii.gz"
                ants.image_write(wl2, str(wlp2))
                warped_label_on_via = sitk.Cast(sitk.ReadImage(str(wlp2)), sitk.sitkUInt8)
                warped_label_on_via.CopyInformation(via_fixed)

        return {
            "backend": "ants",
            "warped_image": sitk.Cast(warped, sitk.sitkFloat32),
            "warped_label": warped_label,
            "warped_label_on_via": warped_label_on_via,
        }


def register_to_template_via_t1(
    moving: sitk.Image,
    t1: sitk.Image,
    template: sitk.Image,
    template_liver_mask: sitk.Image | None,
    moving_label: sitk.Image | None,
    t1_label: sitk.Image | None,
    cfg: PreprocessConfig,
    moving_name: str = "volume",
    case_name: str = "case",
) -> dict[str, Any]:
    """Fallback: moving -> T1 (same patient) -> CT template, QC-gated.

    Same escalation ladder and template-liver-Dice gate as
    register_to_template; additionally records the within-patient
    (moving -> T1) liver Dice for diagnostics.
    """
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_type: str | None = None
    best_dice: float = -1.0

    for step in template_registration_plan(cfg):
        try:
            reg = ants_register_via_fixed_to_template(
                moving=moving,
                via_fixed=t1,
                template=template,
                moving_label=moving_label,
                via_label=t1_label,
                template_liver_mask=template_liver_mask,
                type_of_transform=step["type_of_transform"],
                reg_kwargs=step.get("reg_kwargs"),
            )
        except Exception as exc:
            attempts.append(
                {
                    "step": step["name"],
                    "type": step["type_of_transform"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        t_dice = None
        if template_liver_mask is not None and reg["warped_label"] is not None:
            d = dice_score(reg["warped_label"], template_liver_mask, 1)
            t_dice = round(d, 4) if d is not None else None
        v_dice = None
        if t1_label is not None and reg["warped_label_on_via"] is not None:
            d = dice_score(reg["warped_label_on_via"], t1_label, 1)
            v_dice = round(d, 4) if d is not None else None
        qc_pass = t_dice is not None and t_dice >= cfg.registration_min_liver_dice

        attempts.append(
            {
                "step": step["name"],
                "type": step["type_of_transform"],
                "liver_dice_vs_template": t_dice,
                "liver_dice_vs_t1": v_dice,
                "qc_pass": qc_pass,
            }
        )
        if t_dice is not None and t_dice > best_dice:
            best_dice = t_dice
            best = reg
            best_type = step["type_of_transform"]
        elif best is None:
            best = reg
            best_type = step["type_of_transform"]

        if qc_pass:
            return {
                "warped_image": reg["warped_image"],
                "warped_label": reg["warped_label"],
                "warped_extras": {},
                "accepted_step": f"chain_{step['name']}",
                "type": step["type_of_transform"],
                "backend": "ants",
                "liver_dice_vs_template": t_dice,
                "liver_dice_vs_t1": v_dice,
                "failed": False,
                "attempts": attempts,
                "moving": moving_name,
                "case": case_name,
            }

    return {
        "warped_image": best["warped_image"] if best else None,
        "warped_label": best["warped_label"] if best else None,
        "warped_extras": {},
        "accepted_step": None,
        "type": best_type,
        "backend": "ants",
        "liver_dice_vs_template": best_dice if best_dice >= 0 else None,
        "failed": True,
        "attempts": attempts,
        "moving": moving_name,
        "case": case_name,
    }


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
