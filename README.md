# COMPASS: Cross-mOdality Missing-mPattern Adaptive Segmentation System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evidential deep learning and multi-scale cross-attention framework for multi-organ abdominal segmentation under arbitrary missing modality conditions, benchmarked on the **CHAOS** challenge (CT, MRI T1-DUAL in-phase/out-phase, and MRI T2-SPIR).

---

## 📌 Project Overview & Submission Guide

In medical perception and clinical deployment, multi-modal imaging (CT + MRI T1 + MRI T2) is frequently incomplete due to scanner availability, patient contraindications, or emergency protocols. **COMPASS** addresses this via:

1. **Anatomical Co-Registration**: Multi-sequence ANTs registration to a unified 1.5 mm isotropic CT reference volume, establishing a shared axial slice coordinate system across unpaired cohorts.
2. **Multi-Scale Bidirectional Cross-Attention**: Dynamically exchanges cross-modal representations while cleanly masking absent modalities without numerical instability.
3. **Contrastive & Cross-Pseudo Supervision (Stage 1)**: Jointly aligns latent feature distributions across modalities using NT-Xent contrastive loss and cross-pseudo supervision.
4. **Evidential Uncertainty-Guided Gating (Stage 2)**: Replaces heuristic fusion with Dirichlet-based evidential deep learning ($u = K/S$), providing voxel-level and modality-level uncertainty estimates for trustworthy clinical decision support.

### 🎓 Submission Artifacts
This project follows professional research and engineering standards with a **modular codebase**:
* **Core Library**: Implemented under [`src/`](src/) (`src/model/`, `src/preprocessing/`, `src/losses.py`, `src/evaluate.py`, `src/metrics.py`).
* **Primary Runner Notebook**: [`compass-architecture.ipynb`](compass-architecture.ipynb) — Executed end-to-end on Kaggle GPU (Tesla T4). Handles repository synchronization, data mounting, Stage-1 training (150 epochs), Stage-2 evidential training (75 epochs), test evaluation across all 6 modality conditions, and calibration curve generation.
* **Ablation & Verification Notebook**: [`brier-and-ua-ablation.ipynb`](../brier-and-ua-ablation.ipynb) — Evaluates the held-out test set with the verified multi-class Brier score broadcast and runs the `--no-ug-gate` uniform-averaging (UA) fusion ablation.
* **Manuscript**: Comprehensive IEEE conference paper (`main.tex` and compiled PDF).

---

## 🗂️ Repository Structure

```text
COMPASS/
├── compass-architecture.ipynb   # Master Kaggle GPU runner (training + eval + calibration)
├── train_kaggle.ipynb           # Lightweight Kaggle runner script
├── preprocess.py                # CLI entry point for CHAOS dataset preprocessing
├── preprocessing.md             # Deep-dive documentation on registration & template build
├── requirements.txt             # Environment dependencies (antspyx, torch, SimpleITK, etc.)
│
├── src/
│   ├── data.py                  # PyTorch Dataset & DataLoader with slice-level paired masks
│   ├── evaluate.py              # 6-condition test evaluator (Dice, IoU, HD95, Brier, ECE)
│   ├── kaggle_setup.py          # Auto-discovery & mounting of Kaggle datasets + hotfixes
│   ├── losses.py                # Stage-1 (CE+Dice+NTXent+CPS) & Stage-2 (Dirichlet+KL) losses
│   ├── metrics.py               # Robust multi-class Dice, IoU, HD95, ECE, and Brier score
│   ├── train.py                 # Two-stage training pipeline with AMP and learning rate scheduling
│   │
│   ├── model/
│   │   ├── compass.py           # Top-level COMPASS model (dual ResNet-34 encoders + decoder)
│   │   ├── encoders.py          # CT and shared-MRI feature pyramid extractors (1/2, 1/4, 1/8, 1/16)
│   │   ├── fusion.py            # Multi-scale bidirectional cross-attention with missing-modality zeroing
│   │   └── decoder.py           # 4-stage convolutional upsampler + evidential Dirichlet head
│   │
│   └── preprocessing/
│       ├── template.py          # Iterative Affine CT mean template generation (train-only)
│       ├── registration.py      # ANTs Affine/SyN multi-modal registration ladder with QC gate
│       ├── ct_preprocessing.py  # CT windowing, isotropic resampling, and slice export
│       ├── mr_preprocessing.py  # MRI sequence co-registration and slice alignment
│       ├── augmentation.py      # Elastic deformation, scaling, and rotation transforms
│       └── qc.py                # Visual registration verification overlays (warped vs template)
│
└── tests/                       # Unit and integration test suite
    ├── test_data.py
    ├── test_evaluate.py
    ├── test_losses.py
    ├── test_metrics.py
    ├── test_model.py
    └── test_preprocessing.py
```

---

## 📊 Benchmark Results (CHAOS Challenge)

Evaluated on the held-out CHAOS test set (4 CT patients, 4 MR patients; 768 total test slices, 5 classes: Background, Liver, Right Kidney, Left Kidney, Spleen).

### 1. Six-Condition Missing Modality Performance
| Modality Condition | Inputs Present | Mean Dice ↑ | Mean IoU ↑ | HD95 (mm) ↓ | Brier Score ↓ | ECE ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Multi-Modal** | CT + T1 + T2 | **0.743** | **0.638** | **24.38** | 0.144 | 0.165 |
| **CT + T1** | CT + T1 | 0.398 | 0.301 | 48.71 | 0.205 | 0.207 |
| **CT + T2** | CT + T2 | 0.739 | 0.633 | 24.87 | **0.144** | 0.166 |
| **T1-Only** | T1 only | 0.246 | 0.176 | 62.77 | 0.252 | 0.278 |
| **T2-Only** | T2 only | 0.631 | 0.529 | 32.74 | 0.195 | 0.231 |
| **CT-Only** | CT only | 0.169* | 0.155 | 19.34 | **0.087** | **0.033** |

*\*Note on CT-Only: CHAOS CT annotations contain liver ground truth only (Liver Dice: **0.846**). The 4-organ mean averages in 0.0 for unannotated kidneys and spleen.*

### 2. Ablation Study: Evidential Gating (UG) vs. Uniform Averaging (UA)
| Modality Condition | Uncertainty-Guided (UG) | Uniform Averaging (UA) | $\Delta$ Dice |
|:---|:---:|:---:|:---:|
| **CT + T1** | **0.398** | 0.413 | -0.015 |
| **CT + T2** | **0.739** | 0.742 | -0.003 |
| **T1-Only** | 0.246 | 0.246 | +0.000 |
| **T2-Only** | 0.631 | 0.631 | +0.000 |

Evidential gating maintains segmentation fidelity comparable to uniform fusion while dynamically outputting voxel-level vacuity uncertainty ($u \in [0, 1]$), enabling clinician abstention on corrupted modalities.

---

## ⚙️ Environment & Installation

### Local Setup
Python **3.13** is recommended (`antspyx` provides prebuilt cp313 Windows wheels).

```powershell
# Clone the repository
git clone https://github.com/Umer-3085/COMPASS.git
cd COMPASS

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows (or source venv/bin/activate on Linux)

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution & Reproduction Guide

### 1. Data Preprocessing & Registration
```powershell
# Build CT reference template and preprocess all 40 CHAOS cases onto common grid
python -u preprocess.py

# Rebuild registration quality control report
python -m src.preprocessing.rebuild_report
```

### 2. Model Training (Two-Stage Pipeline)
```powershell
# Stage 1: Representation Learning (150 epochs, AdamW 1e-4, Modality Dropout p=0.2)
# Multi-scale Cross-Attention + NT-Xent Contrastive Loss + Cross-Pseudo Supervision (CPS)
python -m src.train --stage 1 --data-root <path_to_preprocessed> --out runs/stage1 --batch-size 8 --amp

# Stage 2: Evidential Uncertainty Calibration (75 epochs, AdamW 5e-5)
# Encoders & fusion frozen; trains new decoder + evidential Dirichlet head with annealed KL loss
python -m src.train --stage 2 --init-from runs/stage1/ckpt_best.pt --data-root <path_to_preprocessed> --out runs/stage2 --batch-size 8 --amp
```

### 3. Evaluation & Benchmarking
```powershell
# Standard 6-condition missing-modality evaluation (Uncertainty-Guided)
python -m src.evaluate --ckpt runs/stage2/ckpt_best.pt --data-root <path_to_preprocessed> --out runs/eval

# Ablation study: Uniform Averaging (UA) fusion without evidential gating
python -m src.evaluate --ckpt runs/stage2/ckpt_best.pt --data-root <path_to_preprocessed> --out runs/eval_ua --no-ug-gate
```

### 4. Running Unit Tests
```powershell
# Execute all smoke and mathematical correctness tests
python -m unittest discover tests
```

---

## 📓 Kaggle Cloud Execution

For training and evaluation on Kaggle GPU (Tesla T4, 16 GB):
1. Create a Kaggle notebook and set **Accelerator: GPU (T4 x 1)** and **Internet: ON**.
2. Open [`compass-architecture.ipynb`](compass-architecture.ipynb).
3. The notebook will automatically clone this repository, discover the preprocessed CHAOS dataset, run the two-stage training schedule, compute full multi-metric evaluations, and generate calibration curves.

---

## 👥 Authors & Acknowledgments
* **Muhammad Umer** — Department of Computer Science / Artificial Intelligence
* Benchmark Dataset: [CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation](https://chaos.grand-challenge.org/)
