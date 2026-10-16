# COMPASS — CHAOS Preprocessing (template-aligned)

Preprocessing pipeline for the CHAOS dataset (CT + MR abdomen scans), rebuilt so it
matches the paper's preprocessing claims: ANTs registration to a **CT reference
volume**, one **common anatomical grid**, **identical slice indices** across CT, MRI,
and labels, and rotation + scaling + **elastic** training augmentation.

Deep-dive docs: [`preprocessing.md`](preprocessing.md) (read this before writing a dataloader).

---

## What was implemented

| Addition | Where | What it does |
|---|---|---|
| **CT mean template** | `src/preprocessing/template.py` | The paper's "CT reference volume": built from the **14 training CTs only** (no test leakage), 2 ANTs Affine rounds, seed = CT 1, fused majority-vote liver mask. Cached in `preprocessed/template/` |
| **ANTs registration to the common grid** | `src/preprocessing/registration.py` | Every CT, T1 (out-phase rides the same transform), and T2 is registered → template. Escalation ladder **Affine → SyN → extended Affine**, QC gate = warped-liver Dice vs template ≥ 0.40. Never silently falls back to the native grid — failures are flagged in meta + report |
| **T2→T1→template chain fallback** | `registration.py` | For hard cross-modality cases where direct T2→template fails QC: register T2→T1 (same patient) + T1→template and concatenate the transforms. Output still lands on the template grid |
| **Identical slice indices** | `ct_preprocessing.py`, `mr_preprocessing.py` | All 40 cases share one exact grid (235×235×96 @ 1.5 mm isotropic). Slice k means the same axial plane in every CT, MR, and label volume |
| **Elastic augmentation** | `src/preprocessing/augmentation.py` | `random_elastic_2d` + `apply_same_elastic` + `sample_augmentation_params` (train-only; identical transform on image + label) |
| **Registration QC overlays** | `src/preprocessing/qc.py` | Per-case red (warped liver) vs cyan (template liver contour) PNGs in `preprocessed/qc/template_registration/` |

CHAOS CT and MR are **unpaired different patients** — folder ID 1 (CT) ≠ folder ID 1 (MR).
Alignment therefore goes through the template (study-level), never CT patient N ↔ MR patient N.

---

## Environment

Python **3.13** is required (antspyx ships cp313 Windows wheels; cp314 does not exist yet).

```powershell
cd Project\COMPASS
py -3.13 -m venv venv313          # or use the existing venv313/
.\venv313\Scripts\pip install -r requirements.txt
```

Key deps: `antspyx==0.6.3` (ANTs), `SimpleITK`, `numpy`, `scipy`, `matplotlib`.

---

## How to run

All commands from the COMPASS root.

```powershell
# Full run: build/load template, then all 20 CT + 20 MR cases (~45 min)
.\venv313\Scripts\python.exe -u preprocess.py

# Template only (~13 min; cached afterwards, auto-loaded on later runs)
.\venv313\Scripts\python.exe -u preprocess.py --template-only

# Subset of cases / modalities
.\venv313\Scripts\python.exe -u preprocess.py --only-ct 1
.\venv313\Scripts\python.exe -u preprocess.py --only-mr 3,15,36

# Other flags
#   --rebuild-template    force a fresh template build
#   --template-cases 1,2  use a custom training-CT subset for the template
#   --no-template         legacy per-case path (used by tests)
#   --skip-slices         skip .npy slice export
#   --include-official-test   also preprocess the unlabeled official test set

# Rebuild the QC report from existing meta.json files (no reprocessing)
.\venv313\Scripts\python.exe -m src.preprocessing.rebuild_report

# Tests (12, incl. live DICOM tests)
.\venv313\Scripts\python.exe -m unittest tests.test_preprocessing
```

Note: `preprocessed/qc/preprocessing_report.json` is **overwritten at the end of every
run**. After a partial run (`--only-ct`/`--only-mr`), run `rebuild_report` before
trusting it.

Training-time augmentation is **not** part of offline preprocessing — call
`src/preprocessing/augmentation.py` from your dataloader on the training split only.

---

## Outputs

```text
preprocessed/
  template/
    ct_template.nii.gz            # the CT reference volume (git-ignored, reproducible)
    ct_template_liver_mask.nii.gz
    template_meta.json            # build provenance: cases, rounds, per-case Dice (tracked)
  splits/{ct,mr}/{train,val,test}_patients.txt
  ct/{train,val,test}/case_XX/    # ct.nii.gz, labels.nii.gz, meta.json, slices/*.npy
  mr/{train,val,test}/case_XX/    # t1_inphase, t1_outphase, t2_spir, labels_t1/2, meta.json, slices/
  qc/
    ct/  mr/                      # per-case montage PNGs
    template/                     # template + fused liver mask figure
    template_registration/        # red-vs-cyan registration overlays (46 figures)
    preprocessing_report.json
```

Every `meta.json` records `on_template_grid`, `template_registration` (per sequence:
accepted step, transform type, liver Dice, all attempts), and slice-selection info.
NIfTI volumes are the geometric source of truth; `.npy` files are 2D axial slices.

---

## Results (full run, branch `preprocessing-alignment`)

- **40/40 cases** (20 CT + 20 MR) preprocessed on one grid — 0 grid mismatches across 200 volumes
- **60 registrations**: mean warped-liver Dice **0.75**, max **0.92**; 57/60 accepted at plain Affine
- Template build: 13 min · full run: ~45 min · 12/12 tests pass
- MR 3 and MR 15 fixed via retry/chain (Dice 0.61 / 0.47); MR 36 is a known hard case —
  flagged in meta/report if the QC gate keeps failing (output still on the template grid)

Splits (seed 42, patient-level, per modality):

- **CT**: train 1,2,5,6,14,16,18,21,22,23,26,27,28,29 · val 19,24 · test 8,10,25,30
- **MR**: train 1,2,3,5,13,15,19,21,22,31,34,36,37,38 · val 20,32 · test 8,10,33,39

---

## Notes & rules for teammates

- **Never** pair CT/N with MR/N — they are different people.
- CT ground truth is **liver only** (classes 0/1); the paper's 5 classes = model output
  space. Train CT loss with a liver mask.
- MR labels: liver 1, right kidney 2, left kidney 3, spleen 4 (grayscale ranges from
  CHAOS `NOTES_PLEASE_READ.txt`).
- Do not push `data/`, `*.nii.gz`, or `slices/` to GitHub (license + size) — the
  `.gitignore` enforces this; push code, splits, config, QC figures, and docs.
- Do not linearly interpolate labels, do not augment val/test.
