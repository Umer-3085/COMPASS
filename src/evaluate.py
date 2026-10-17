"""COMPASS evaluation: six modality-availability conditions x metrics.

Condition realization on unpaired CHAOS (honest limitation, documented):
- MR test patients run Full / CT+T1 / CT+T2 / T1-only / T2-only with the
  template-CT standing in for a real CT stream (Option A of the data
  pipeline). CT-only is NOT reported for MR patients (it would only
  measure segmentation of the template anatomy, not a real scenario).
- CT test patients run CT-only (real CT, liver-only labels, so organ
  columns beyond liver are excluded by the GT-presence filter).

Metrics per patient per organ (classes 1-4): Dice, IoU, precision,
recall, HD95 (paper: Dice, HD95; supervisor: IoU/pixel-acc/precision/
recall) plus per-patient ECE, Brier, vacuity (paper). Stage-2 models use
uncertainty-guided probabilities for ECE/Brier (argmax is unaffected).

Usage:
    python -m src.evaluate --ckpt runs/stage2/ckpt_best.pt \
        --data-root preprocessed --out runs/eval [--limit-batches N]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.data import make_dataloader
from src.losses import uncertainty_guided_probs
from src.metrics import (ECE_BINS, brier_score, confusion, dice_from,
                         expected_calibration_error, hd95, iou_from,
                         pixel_accuracy, precision_from, recall_from,
                         vacuity_mean)
from src.model import COMPASS

NUM_CLASSES = 5

# (CT, T1, T2) availability per condition (paper Table: six scenarios).
CONDITIONS: dict[str, tuple[int, int, int]] = {
    "Full": (1, 1, 1),
    "CT+T1": (1, 1, 0),
    "CT+T2": (1, 0, 1),
    "T1-only": (0, 1, 0),
    "T2-only": (0, 0, 1),
    "CT-only": (1, 0, 0),
}


class CaseAccumulator:
    """Incremental per-patient stats over its slices (one condition)."""

    def __init__(self) -> None:
        # per-organ confusion + HD95 samples
        self.tp = [0] * NUM_CLASSES
        self.fp = [0] * NUM_CLASSES
        self.fn = [0] * NUM_CLASSES
        self.hd95: list[list[float]] = [[] for _ in range(NUM_CLASSES)]
        # calibration bins
        self.bin_cnt = np.zeros(ECE_BINS)
        self.bin_conf = np.zeros(ECE_BINS)
        self.bin_corr = np.zeros(ECE_BINS)
        self.brier_sum = 0.0
        self.vac_sum = 0.0
        self.pix = 0
        self.acc_correct = 0

    def update(self, pred: np.ndarray, probs: np.ndarray,
               alpha: np.ndarray, gt: np.ndarray) -> None:
        for cls in range(1, NUM_CLASSES):
            tp, fp, fn = confusion(pred, gt, cls)
            self.tp[cls] += tp
            self.fp[cls] += fp
            self.fn[cls] += fn
            if (tp + fn) or (tp + fp):          # organ present somewhere
                self.hd95[cls].append(hd95(pred == cls, gt == cls))

        conf = probs.max(axis=0).astype(np.float64).ravel()
        correct = (pred == gt).astype(np.float64).ravel()
        bins = np.minimum((conf * ECE_BINS).astype(int), ECE_BINS - 1)
        self.bin_cnt += np.bincount(bins, minlength=ECE_BINS)
        self.bin_conf += np.bincount(bins, weights=conf, minlength=ECE_BINS)
        self.bin_corr += np.bincount(bins, weights=correct, minlength=ECE_BINS)

        self.brier_sum += brier_score(probs, gt) * gt.size
        self.vac_sum += vacuity_mean(alpha) * gt.size
        self.pix += gt.size
        self.acc_correct += int((pred == gt).sum())

    # ---- derived per-organ metrics ------------------------------------
    def organ_metrics(self, cls: int) -> dict:
        tp, fp, fn = self.tp[cls], self.fp[cls], self.fn[cls]
        vals = self.hd95[cls]
        return {
            "gt_present": int(tp + fn > 0),
            "dice": dice_from(tp, fp, fn),
            "iou": iou_from(tp, fp, fn),
            "precision": precision_from(tp, fp, fn),
            "recall": recall_from(tp, fp, fn),
            "hd95": float(np.nanmean(vals)) if vals else float("nan"),
        }

    def overall_metrics(self) -> dict:
        conf = self.bin_conf / np.maximum(self.bin_cnt, 1)
        corr = self.bin_corr / np.maximum(self.bin_cnt, 1)
        m = self.bin_cnt > 0
        ece = float((self.bin_cnt[m] / self.bin_cnt.sum()
                     * np.abs(corr[m] - conf[m])).sum())
        return {
            "pixel_accuracy": self.acc_correct / max(self.pix, 1),
            "ece": ece,
            "brier": self.brier_sum / max(self.pix, 1),
            "vacuity": self.vac_sum / max(self.pix, 1),
        }


@torch.no_grad()
def run_condition(model: COMPASS, loader, name: str,
                  cond: tuple[int, int, int], device: torch.device,
                  use_ug: bool, limit_batches: int = 0,
                  ) -> dict[str, CaseAccumulator]:
    """Evaluate one availability condition; returns per-case accumulators.

    Eligibility: MR patients run the five MRI-bearing conditions (their
    CT stream is the template-CT proxy); CT patients run CT-only (their
    T1/T2 inputs are zero — nothing real to evaluate).
    """
    model.eval()
    cases: dict[str, CaseAccumulator] = {}
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        ct = batch["ct"].to(device)
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        lab = batch["label"]
        B = ct.shape[0]

        mask = torch.tensor([cond] * B, dtype=torch.float32, device=device)
        out = model(ct, t1, t2, mask)
        probs = uncertainty_guided_probs(out["seg_logits"], out["evidence"],
                                         enabled=use_ug).cpu()
        alpha = out["evidence"].cpu()
        pred = probs.argmax(dim=1)

        for s in range(B):
            mod = batch["modality"][s]
            eligible = ((mod == "CT" and name == "CT-only")
                        or (mod == "MR" and name != "CT-only"))
            if not eligible:
                continue
            # MR and CT patient numbers overlap (both have 8, 10, ...) —
            # key by modality + id to keep patients distinct.
            case = f"{mod}_{int(batch['case_id'][s])}"
            acc = cases.setdefault(case, CaseAccumulator())
            acc.update(pred[s].numpy(), probs[s].numpy(),
                       alpha[s].numpy(), lab[s].numpy())
    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COMPASS six-condition evaluation")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/eval"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-batches", type=int, default=0,
                   help="smoke test: cap test batches per condition")
    p.add_argument("--no-ug", action="store_true",
                   help="disable uncertainty-guided probs for ECE/Brier")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    stage = ckpt.get("stage", 1)
    use_ug = (stage == 2) and not args.no_ug
    print(f"checkpoint: epoch {ckpt['epoch']}, stage {stage}, "
          f"best val dice {ckpt.get('best_dice', float('nan')):.4f}, "
          f"UG probs for calibration: {use_ug}")

    model = COMPASS(pretrained=False).to(device)
    if stage == 2:
        model.init_stage2()
    model.load_state_dict(ckpt["model"])

    template = args.data_root / "template" / "ct_template.nii.gz"
    loader = make_dataloader(
        args.data_root, "test", template, batch_size=args.batch_size,
        augment=False, num_workers=args.num_workers)
    n_test = len(loader.dataset)
    print(f"test set: {n_test} slices")

    per_class_rows: list[dict] = []
    overall_rows: list[dict] = []
    summary_rows: list[dict] = []

    for name, cond in CONDITIONS.items():
        print(f"\n=== condition: {name} {cond} ===", flush=True)
        cases = run_condition(model, loader, name, cond, device, use_ug,
                              args.limit_batches)
        if not cases:
            print("  no eligible patients, skipped")
            continue

        for case, acc in sorted(cases.items()):
            ov = acc.overall_metrics()
            overall_rows.append({"condition": name, "case": case, **ov})
            print(f"  {case}: pix_acc={ov['pixel_accuracy']:.4f} "
                  f"ece={ov['ece']:.4f} brier={ov['brier']:.4f} "
                  f"vacuity={ov['vacuity']:.4f}")
            for cls in range(1, NUM_CLASSES):
                m = acc.organ_metrics(cls)
                per_class_rows.append({"condition": name, "case": case,
                                       "class": cls, **m})
                if m["gt_present"]:
                    print(f"    class {cls}: dice={m['dice']:.4f} "
                          f"iou={m['iou']:.4f} prec={m['precision']:.4f} "
                          f"rec={m['recall']:.4f} hd95={m['hd95']:.2f}")

        # summary = mean over patients with the organ present
        for cls in range(1, NUM_CLASSES):
            rows = [r for r in per_class_rows
                    if r["condition"] == name and r["class"] == cls
                    and r["gt_present"]]
            if not rows:
                continue
            summary_rows.append({
                "condition": name, "class": cls, "n_patients": len(rows),
                **{k: float(np.mean([r[k] for r in rows]))
                   for k in ("dice", "iou", "precision", "recall", "hd95")},
            })
        ov = [r for r in overall_rows if r["condition"] == name]
        summary_rows.append({
            "condition": name, "class": "ALL", "n_patients": len(ov),
            **{k: float(np.mean([r[k] for r in ov]))
               for k in ("pixel_accuracy", "ece", "brier", "vacuity")},
        })

    # persist (summary rows mix per-organ and overall fields -> union)
    def write(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)

    write(args.out / "per_patient_organ.csv", per_class_rows)
    write(args.out / "per_patient_overall.csv", overall_rows)
    write(args.out / "summary.csv", summary_rows)
    with open(args.out / "run.json", "w") as f:
        json.dump({"ckpt": str(args.ckpt), "stage": stage,
                   "use_ug": use_ug,
                   "n_test_slices": len(loader.dataset)}, f, indent=2)

    print("\n================ SUMMARY (mean over patients) ================")
    for r in summary_rows:
        if r["class"] == "ALL":
            print(f"{r['condition']:>8} | ALL: pix_acc={r['pixel_accuracy']:.4f} "
                  f"ece={r['ece']:.4f} brier={r['brier']:.4f} "
                  f"vacuity={r['vacuity']:.4f}")
        else:
            print(f"{r['condition']:>8} | c{r['class']}: "
                  f"dice={r['dice']:.4f} iou={r['iou']:.4f} "
                  f"prec={r['precision']:.4f} rec={r['recall']:.4f} "
                  f"hd95={r['hd95']:.2f} (n={r['n_patients']})")
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
