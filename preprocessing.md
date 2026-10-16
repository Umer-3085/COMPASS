# CHAOS Preprocessing Guide

This document explains the local CHAOS preprocessing for all project members.
Read this before pairing CT and MRI or writing a dataloader.

## 1. Critical fact (do not ignore)

**CHAOS CT and MR are unpaired different patients.**

Folder ID does **not** mean the same person:

- `data/raw/.../CT/1` is not `data/raw/.../MR/1`
- Never register MRI to CT using matching folder numbers
- Never build one multimodal sample from CT case N and MR case N

The paper states “ANTs aligns the MRI sequences to the CT reference volume.” No matched CT
patient exists for any MR case, so the CT reference volume is a **study-specific CT mean
template** built from the 14 training CTs (`src/preprocessing/template.py`). Every CT case,
T1, T2, and their labels are registered onto that single template grid with ANTs (Affine,
escalating to SyN if the liver-Dice QC gate fails).

Correct statement:

> All modalities (CT, T1, T2) and labels are resampled onto one common CT-derived template
> grid (1.5 mm isotropic), so they share identical geometry and slice indices. Registration
> is study-level (each case → template), never CT patient N ↔ MR patient N.

## 2. What was implemented

Code lives in `src/preprocessing/`.

Run from the project root:

```bash
python -m src.preprocessing.pipeline
python preprocess.py --only-ct 1 --only-mr 1
python -m unittest tests.test_preprocessing
```

Pipeline (labeled data only):

```text
Raw CHAOS
→ dataset audit
→ patient splits (seed 42)
→ CT mean template build (14 training CTs, 2 ANTs Affine rounds, fused liver mask)
→ load DICOM with SimpleITK/GDCM (physical order, not filename order)
→ CT: HU check → clip [-200, 250] → scale to [0, 1] → 1.5 mm isotropic
     → ANTs register to template (liver-Dice QC gate) → liver labels on template grid
→ MR: N4 → 1.5 mm isotropic → ANTs register T1(+out-phase) and T2 to template
     → foreground z-score on template grid → 4-class labels on template grid
→ QC images (incl. red/cyan warped-liver vs template-liver overlays)
→ axial .npy slices (identical indices across CT/MR/labels)
```

Template-only / rebuild flags:

```bash
python preprocess.py --template-only      # build/load template, no cases
python preprocess.py --rebuild-template   # force rebuild
python preprocess.py --no-template        # legacy per-case path (tests)
python preprocess.py --template-cases 1,2,3   # subset of training CTs for the template
```

Official unlabeled CHAOS test is **not** in the supervised splits. Optional:

```bash
python -m src.preprocessing.pipeline --include-official-test
```

Rebuild the QC JSON from existing `meta.json` files (does not re-run preprocessing):

```bash
python -m src.preprocessing.rebuild_report
```

`preprocessed/qc/preprocessing_report.json` is **overwritten at the end of every pipeline run**. A later `--only-ct` / `--only-mr` command can make it look like only a few cases exist. Count `meta.json` files or run `rebuild_report` before trusting that file.

## 3. Configuration

Saved in `preprocessed/config.json`:

| Setting | Value |
|---|---|
| Random seed | 42 |
| Target spacing | 1.5 × 1.5 × 1.5 mm |
| CT window | [-200, 250] HU → [0, 1] |
| MRI | N4 + foreground z-score (per volume, not global) |
| Common grid | CT mean template (14 train CTs, seed CT 1, 2 rounds) |
| Case → template | ANTs Affine → SyN escalation; min warped-liver Dice 0.40 |
| CT↔MR patient registration | **false** |
| Slice policy | train: foreground only; val/test: all slices |
| Supervised split | 14 train / 2 val / 4 test **per modality** |

Software used when last run: Python 3.13, SimpleITK 2.5.6, ANTsPy (antspyx) 0.6.3.

## 4. Patient splits

Independent CT and MR splits (same seed, different ID lists):

**CT train:** 1, 2, 5, 6, 14, 16, 18, 21, 22, 23, 26, 27, 28, 29
**CT val:** 19, 24
**CT test:** 8, 10, 25, 30

**MR train:** 1, 2, 3, 5, 13, 15, 19, 21, 22, 31, 34, 36, 37, 38
**MR val:** 20, 32
**MR test:** 8, 10, 33, 39

Files: `preprocessed/splits/ct/` and `preprocessed/splits/mr/`.

A patient (study ID) belongs to exactly one split **within its modality**. CT ID 8 and MR ID 8 are still different people.

## 5. Labels

**CT** ground truth is liver only:

- 0 = background
- 1 = liver
- No kidneys or spleen on CT

**MR** PNG gray values (from CHAOS `NOTES_PLEASE_READ.txt`), remapped to:

| PNG range | Class |
|---|---|
| 0 | 0 background |
| 55–70 (typ. 63) | 1 liver |
| 110–135 (typ. 126) | 2 right kidney |
| 175–200 (typ. 189) | 3 left kidney |
| 240–255 (typ. 252) | 4 spleen |

Unexpected PNG values fail the case; they are not silently mapped.

## 6. Output layout

```text
preprocessed/
  config.json
  template/
    template.nii.gz          # CT mean template (reference volume)
    template_liver_mask.nii.gz
    template_meta.json
  splits/{ct,mr}/{train,val,test}_patients.txt
  ct/{train,val,test}/case_XX/
    ct.nii.gz
    labels.nii.gz
    meta.json
    slices/image_XXXX.npy
    slices/label_XXXX.npy
  mr/{train,val,test}/case_XX/
    t1_inphase.nii.gz
    t1_outphase.nii.gz
    t2_spir.nii.gz
    labels_t1.nii.gz
    labels_t2.nii.gz
    meta.json
    slices/...
  qc/ct/  qc/mr/  qc/template/  qc/template_registration/  qc/preprocessing_report.json
```

NIfTI volumes are the geometric source of truth (size, spacing, origin, direction).

`.npy` files are **2D axial slices** for a 2D network. They are binary, not text. Load with:

```python
import numpy as np
img = np.load("preprocessed/ct/train/case_01/slices/image_0085.npy")
lab = np.load("preprocessed/ct/train/case_01/slices/label_0085.npy")
print(img.shape, img.dtype, img.min(), img.max())
print(lab.shape, np.unique(lab))
```

Same index means the same resampled slice **on the common template grid**. Because every case
(CT and MR) is registered to the same template, slice index k refers to the same axial plane
in every CT volume, every MR volume, and every label volume — this is the paper’s “identical
slice indices across CT, MRI, and labels” property. Do not confuse this with original
(acquisition-space) slice numbers, which still differ per study.

All MR cases now have `on_template_grid: true`: T1, T2, and both label volumes share one
grid. Cases that fail the registration QC gate are flagged (`registration_failed: true` and
a warning in the report), never silently left on their native grid.

## 7. How to check success

1. Count `preprocessed/**/meta.json` → should be 40 labeled cases (20 CT + 20 MR).
2. Rebuild the report without reprocessing:

```bash
python -m src.preprocessing.rebuild_report
```

3. Inspect `preprocessed/qc/ct/*.png`, `preprocessed/qc/mr/*.png`, and the registration overlays
   in `preprocessed/qc/template_registration/*.png` (warped liver in **red** vs template liver
   contour in **cyan** — they should overlap closely).
4. In each `meta.json` confirm:
   - `processed_spacing` = `[1.5, 1.5, 1.5]`
   - `on_template_grid`: true (CT and MR)
   - `template_registration` (CT) / `template_registration.t1|t2` (MR) report `performed: true`
     and a warped-liver Dice ≥ 0.40
   - CT labels unique ⊆ `{0, 1}`
   - MR labels unique ⊆ `{0, 1, 2, 3, 4}`
   - `ct_mr_registration`: false
5. Confirm every case NIfTI has the **same size/origin/spacing/direction** as
   `preprocessed/template/template.nii.gz`.

## 8. Known issues

| Case | Issue | What the pipeline did |
|---|---|---|
| Any case | Template registration QC (warped liver Dice &lt; 0.40) after Affine | Escalates Affine → SyN → extended Affine; if all fail, case is flagged `failed` + report warning (never silently kept on native grid) |
| MR 36, 37 (legacy path only) | T2→T1 liver Dice &lt; 0.40 | Only applies with `--no-template`; on the template path T2 registers directly to the template |
| CT-1 | Filename order ≠ GDCM order | Masks matched by DICOM index / instance, not `liver_GT_000` = array slice 0 |
| CT-21 | `IMG-*.dcm`, 257 slices, ~1 mm z | Resampled to 1.5 mm (shape changes) |
| Report JSON | Can be overwritten by `--only-ct` / `--only-mr` | Rebuild with `rebuild_report` |

SimpleITK already applied DICOM rescale on checked CT cases. Do not apply slope/intercept twice.

## 9. What members should do next

1. Rebuild and review a complete `preprocessing_report.json`.
2. Visually QC a few PNGs (include CT-21, MR-10, and MR-36).
3. Write a dataloader using `preprocessed/splits/` and NIfTI or `.npy`.
4. Apply spatial augmentation **only in training**, same transform on image + label (linear vs nearest-neighbor).
   `src/preprocessing/augmentation.py` provides the paper’s three transforms: `random_rotate_scale_2d`
   (rotation + scaling), `random_elastic_2d` + `apply_same_elastic` (elastic stretching), sampled
   together by `sample_augmentation_params`.
5. Missing-modality experiments: mask channels in the dataloader; do not delete files.
6. Do not push `data/raw/` or large `*.nii.gz` / `slices/` to a public GitHub repo (license + size). Push code, splits, config, and this document.

## 10. Rules for other members

- Do not pair CT/N with MR/N.
- Do not linearly interpolate labels.
- Do not normalize masks.
- Do not augment val/test.
- Equal array shapes do not prove spatial alignment; check origin, spacing, and direction.
- Training augmentation is not part of offline preprocessing (`src/preprocessing/augmentation.py` is for the dataloader only).
