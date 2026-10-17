# COMPASS — CHAOS pipeline (preprocessing + model + training)

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

## Model (branch `model-architecture`, `src/model/`)

COMPASS per the paper's Model Architecture section + architecture figure:

| Component | Implementation |
|---|---|
| **CT encoder** | ResNet34, ImageNet weights, 1→3 channel replication |
| **MRI encoder** | one shared ResNet34 applied to T1 and T2 (paper: weights shared between sequences) |
| **Feature pyramid** | stride-1 maxpool stem → 1/2 (64), 1/4 (128), 1/8 (256), 1/16 (512); each encoder's taps are 1×1-projected to **256 ch**, the 1/4 level being the paper's "256-channel features at one-quarter resolution" (also the 256-d pooled contrastive representation) |
| **Gatekeeper + fusion** | bidirectional cross-attention, 4 heads, d=256, residual + LayerNorm; K/V of a missing modality are **zeroed** (NaN-free even when a whole stream is absent); sigmoid-gated 1×1 reliability gates combine available modalities; the same module is applied at 1/16 + 1/8 + 1/4 → fused pyramid ("skip connections from the fused features") |
| **Decoder** | 4 up-blocks (GN + ReLU), 1/16 → 1/1 exact recovery on the padded 240 grid (235 center-padded, predictions cropped back), skips: fused 1/8, fused 1/4, CT-encoder 1/2 detail |
| **Heads** | seg: 1×1 → 5-class logits; evidential: 1×1 on fused 1/4 → Softplus+1 → 5 Dirichlet alphas |
| **Stage 2** | encoders + fusion frozen, **new** decoder + evidential head trained; uncertainty-guided per-modality gating w = 1/(1+ū_m) before fusion (text) and output fusion p′ = w·p + (1−w)/5 (figure) |

Interpretation note: the paper's "4 upsampling blocks" + exact resolution recovery
mathematically forces a 1/16 decoder start; the "1/4 feature resolution" spec is
realized at the branch/fusion level and the contrastive pooling level.

Losses (`src/losses.py`): Stage 1 `L_seg + 0.5·L_NT-Xent(T=0.07) + 0.3·L_CPS`
(bidirectional KL between auxiliary branch predictions); Stage 2 `L_seg +
L_Dirichlet + λ·L_KL`, λ annealed 0 → 0.1 over 20 epochs. MR samples use full
CE + soft-Dice; CT samples are **liver-masked** (CE on GT-liver pixels + class-1
Dice) because CHAOS CT labels cover the liver only.

---

## Training & evaluation

```powershell
# Stage 1: 150 ep, AdamW 1e-4, modality dropout p=0.2 (T1/T2; CT never dropped)
python -m src.train --stage 1 --data-root <preprocessed/> --out runs/stage1 \
    --batch-size 8 --amp [--resume]

# Stage 2: 75 ep, AdamW 5e-5 on frozen encoders+fusion + new decoder/evidential
python -m src.train --stage 2 --init-from runs/stage1/ckpt_best.pt \
    --data-root <preprocessed/> --out runs/stage2 --batch-size 8 --amp

# Six-condition test evaluation (Dice/IoU/precision/recall/pixel-acc + HD95/ECE/Brier/vacuity)
python -m src.evaluate --ckpt runs/stage2/ckpt_best.pt --data-root <preprocessed/> --out runs/eval
```

On Kaggle use the thin notebook [`train_kaggle.ipynb`](train_kaggle.ipynb)
(clone repo → mount dataset → run; cells auto-resume from `ckpt_last.pt`).
Smoke tests (CPU, seconds): `tests/test_model.py`, `tests/test_losses.py`,
`tests/test_metrics.py`, `tests/test_evaluate.py`, `tests/test_data.py`.

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
