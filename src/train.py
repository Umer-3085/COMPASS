"""COMPASS two-stage training (paper, Training Strategy).

Stage 1 (150 epochs, AdamW 1e-4):
    L = L_seg + 0.5 * L_contra (NT-Xent, T=0.07) + 0.3 * L_cps
    trains both encoders, fusion, decoder; modality dropout p=0.2 (T1/T2).

Stage 2 (75 epochs, AdamW 5e-5):
    L = L_seg + L_evidential,  L_evidential = L_Dirichlet + lam * L_KL
    lam annealed linearly 0 -> 0.1 over the first 20 stage-2 epochs.
    Encoders + fusion are FROZEN; a NEW decoder and the evidential head
    are trained on top (model.init_stage2()), with uncertainty-guided
    per-modality gating active.

Usage (Kaggle GPU):
    python -m src.train --stage 1 \
        --data-root /kaggle/input/datasets/muhammadumerbsse/compass-preprocessed \
        --out /kaggle/working/runs/stage1 --batch-size 8 --num-workers 2 --amp

    python -m src.train --stage 2 \
        --init-from /kaggle/working/runs/stage1/ckpt_best.pt \
        --data-root /kaggle/input/datasets/muhammadumerbsse/compass-preprocessed \
        --out /kaggle/working/runs/stage2 --batch-size 8 --num-workers 2 --amp

    Add --resume to continue from <out>/ckpt_last.pt after an interruption.

Smoke test (CPU, tiny):
    python -m src.train --stage 1 --data-root preprocessed --out runs/smoke \
        --epochs 2 --limit-train-batches 3 --limit-val-batches 2 --no-pretrained
"""

from __future__ import annotations

import os

# Must be set before torch initializes CUDA: reduces allocator
# fragmentation on the T4 (16 GB is tight for 1/4-scale attention).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data import make_dataloader
from src.losses import stage1_loss, stage2_loss
from src.model import COMPASS

NUM_CLASSES = 5
DROPOUT_P = 0.2          # paper: modality dropout probability (T1/T2)
STAGE1_EPOCHS = 150      # paper
STAGE1_LR = 1e-4         # paper: AdamW initial learning rate
STAGE2_EPOCHS = 75       # paper
STAGE2_LR = 5e-5         # paper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COMPASS two-stage training")
    p.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p.add_argument("--data-root", type=Path, required=True,
                   help="preprocessed/ directory (e.g. Kaggle dataset mount)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: runs/stage<n>")
    p.add_argument("--init-from", type=Path, default=None,
                   help="stage-1 checkpoint (required for --stage 2)")
    p.add_argument("--epochs", type=int, default=None,
                   help="default: 150 (stage 1) / 75 (stage 2)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-4 (stage 1) / 5e-5 (stage 2)")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dropout-p", type=float, default=DROPOUT_P)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true",
                   help="mixed precision (GPU only)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="skip ImageNet weights (offline smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="resume from <out>/ckpt_last.pt")
    p.add_argument("--limit-train-batches", type=int, default=0,
                   help="smoke test: cap batches per epoch (0 = all)")
    p.add_argument("--limit-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    if args.out is None:
        args.out = Path(f"runs/stage{args.stage}")
    if args.epochs is None:
        args.epochs = STAGE1_EPOCHS if args.stage == 1 else STAGE2_EPOCHS
    if args.lr is None:
        args.lr = STAGE1_LR if args.stage == 1 else STAGE2_LR
    if args.stage == 2 and args.init_from is None and not args.resume:
        p.error("--stage 2 requires --init-from <stage-1 ckpt_best.pt>")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _jsonable(args: argparse.Namespace) -> dict:
    """vars(args) with Paths stringified (JSON/checkpoint-safe)."""
    return {k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()}


def modality_dropout(mask: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly zero T1/T2 availability per sample (CT never dropped).

    Applied in the training loop (not the dataset) so the SAME augmented
    slice tensors feed the model while the gatekeeper sees the modality
    as missing — exactly the paper's "zero the signal in the
    cross-attention module" behavior.
    """
    m = mask.clone()
    drop = torch.rand_like(m[:, 1]) < p
    m[:, 1] = m[:, 1] * (~drop).float()
    drop = torch.rand_like(m[:, 2]) < p
    m[:, 2] = m[:, 2] * (~drop).float()
    return m


@torch.no_grad()
def evaluate(model: COMPASS, loader, device: torch.device,
             limit_batches: int = 0) -> float:
    """Validation mean foreground Dice (classes 1-4, per-sample available
    modalities = the "Full" condition for that patient).

    Dice is computed per sample per organ class and averaged over
    classes actually present in that sample's ground truth (CT slices
    contribute liver only — their labels cover class 1).
    """
    model.eval()
    per_entry: list[torch.Tensor] = []
    for i, batch in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        ct = batch["ct"].to(device)
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        mask = batch["mask"].to(device)
        lab = batch["label"].to(device)

        out = model(ct, t1, t2, mask)
        pred = out["seg_logits"].argmax(dim=1)   # UG weighting preserves argmax

        for cls in range(1, NUM_CLASSES):
            p = pred == cls
            g = lab == cls
            inter = (p & g).sum(dim=(1, 2)).float()
            denom = p.sum(dim=(1, 2)).float() + g.sum(dim=(1, 2)).float()
            dice = (2 * inter + 1e-7) / (denom + 1e-7)
            present = g.any(dim=(1, 2))
            if present.any():
                per_entry.append(dice[present])
    model.train()
    if not per_entry:
        return 0.0
    return float(torch.cat(per_entry).mean())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    stage2 = args.stage == 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("[warn] --amp ignored: no CUDA device")

    template = args.data_root / "template" / "ct_template.nii.gz"
    train_loader = make_dataloader(
        args.data_root, "train", template, batch_size=args.batch_size,
        augment=True, num_workers=args.num_workers, seed=args.seed)
    val_loader = make_dataloader(
        args.data_root, "val", template, batch_size=args.batch_size,
        augment=False, num_workers=args.num_workers, seed=args.seed)
    print(f"[stage {args.stage}] train={len(train_loader.dataset)} slices "
          f"val={len(val_loader.dataset)} slices device={device.type} "
          f"amp={use_amp} epochs={args.epochs} lr={args.lr}")

    model = COMPASS(pretrained=not args.no_pretrained).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [init] not in stage-1 ckpt (expected: decoder2): "
                  f"{[k for k in missing if not k.startswith('decoder2')] or missing}")
        if unexpected:
            print(f"  [init] unexpected keys: {unexpected}")
    if stage2:
        model.init_stage2()          # freeze encoders+fusion, fresh decoder
        print("  [stage2] encoders+fusion frozen; new decoder attached")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_dice = 0, 0.0
    if args.resume:
        ckpt_path = args.out / "ckpt_last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume: {ckpt_path} not found")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"resumed at epoch {start_epoch} (best val dice {best_dice:.4f})")

    # Per-stage CSV columns (paper loss terms).
    loss_cols = (["total", "seg", "contra", "cps"] if not stage2 else
                 ["total", "seg", "evidential", "dir_nll", "kl"])
    header = ["epoch", "lr"] + loss_cols + ["val_dice", "seconds", "is_best"]
    log_path = args.out / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
    with open(args.out / "args.json", "w") as f:
        json.dump(_jsonable(args), f, indent=2)

    def save(path: Path, epoch: int) -> None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "epoch": epoch,
            "best_dice": best_dice,
            "stage": args.stage,
            "args": _jsonable(args),
        }, path)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loader.dataset.set_epoch(epoch)   # fresh augmentation per epoch
        sums: dict[str, float] = {"total": 0.0}
        steps = 0

        for it, batch in enumerate(train_loader):
            if args.limit_train_batches and it >= args.limit_train_batches:
                break
            ct = batch["ct"].to(device)
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            is_ct = torch.tensor([m == "CT" for m in batch["modality"]],
                                 device=device)

            if args.dropout_p > 0:
                mask = modality_dropout(mask, args.dropout_p)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(ct, t1, t2, mask)
                if stage2:
                    loss, parts = stage2_loss(out, label, is_ct, epoch)
                else:
                    loss, parts = stage1_loss(out, label, is_ct)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            sums["total"] += float(loss.detach())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
            if (it + 1) % args.log_every == 0:
                print(f"  ep {epoch} it {it + 1}: "
                      + " ".join(f"{k}={sums[k] / steps:.4f}"
                                 for k in sums), flush=True)

        val_dice = evaluate(model, val_loader, device, args.limit_val_batches)
        seconds = time.time() - t0
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            save(args.out / "ckpt_best.pt", epoch)
        save(args.out / "ckpt_last.pt", epoch)

        lr_now = optimizer.param_groups[0]["lr"]
        row = ([epoch, lr_now]
               + [sums.get(k, 0.0) / max(steps, 1) for k in loss_cols]
               + [f"{val_dice:.4f}", f"{seconds:.1f}", int(is_best)])
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"epoch {epoch}: total={sums['total'] / max(steps, 1):.4f} "
              f"val_dice={val_dice:.4f} best={best_dice:.4f} "
              f"({seconds:.0f}s){' *BEST*' if is_best else ''}", flush=True)

    print(f"done. best val dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
