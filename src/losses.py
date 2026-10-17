"""Loss functions for COMPASS training (paper equations).

Stage 1:  L = L_seg + 0.5 * L_contra (NT-Xent, T=0.07) + 0.3 * L_cps
Stage 2:  L = L_seg + L_evidential,  L_evidential = L_Dirichlet + lam * L_KL
          with lam annealed linearly 0 -> 0.1 over 20 epochs.

Segmentation loss conventions (documented deviation resolution for the
unpaired CHAOS data):
- MR samples: full CE + soft-Dice over the 4 organ classes.
- CT samples: labels are LIVER ONLY, so the loss is liver-masked —
  CE restricted to GT-liver pixels + class-1 soft Dice (which also
  penalizes over-prediction on empty slices). Background/other-organ
  supervision comes from the MR samples.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_CLASSES = 5
TEMPERATURE = 0.07       # NT-Xent temperature (paper)
W_CONTRA = 0.5           # paper: 0.5 * L_contra
W_CPS = 0.3              # paper: 0.3 * L_cps
KL_LAMBDA_MAX = 0.1      # paper: lam annealed to 0.1
KL_ANNEAL_EPOCHS = 20    # paper: over 20 epochs


# --------------------------------------------------------------------- seg

def _onehot(target: torch.Tensor) -> torch.Tensor:
    """(B,H,W) int -> (B,C,H,W) float one-hot."""
    return F.one_hot(target.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()


def soft_dice_loss(probs: torch.Tensor, target: torch.Tensor,
                   classes: range = range(1, NUM_CLASSES)) -> torch.Tensor:
    """1 - mean Dice over `classes` (batch-aggregated)."""
    tgt = _onehot(target)
    dims = (0, 2, 3)
    inter = (probs * tgt).sum(dims)
    denom = probs.sum(dims) + tgt.sum(dims)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    idx = list(classes)
    return 1.0 - dice[idx].mean()


def seg_loss_mr(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MR samples: CE over all pixels + soft Dice over organ classes."""
    ce = F.cross_entropy(logits, target.long())
    dice = soft_dice_loss(logits.softmax(dim=1), target)
    return ce + dice


def seg_loss_ct(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CT samples: liver-masked loss (labels cover class 1 only).

    - CE restricted to GT-liver pixels (no false background supervision
      for unlabeled organs on CT).
    - Class-1 soft Dice over the whole image bounds over-prediction and
      supervises empty-liver slices.
    """
    tgt = target.long()
    gt_liver = tgt == 1
    if gt_liver.any():
        pix_logits = logits.permute(0, 2, 3, 1)[gt_liver]   # (N, 5)
        ce = F.cross_entropy(pix_logits, tgt[gt_liver])
    else:
        ce = logits.sum() * 0.0
    probs = logits.softmax(dim=1)
    p1 = probs[:, 1]
    g1 = gt_liver.float()
    inter = (p1 * g1).sum()
    denom = p1.sum() + g1.sum()
    dice1 = (2 * inter + 1e-6) / (denom + 1e-6)
    return ce + (1.0 - dice1)


def seg_loss(logits: torch.Tensor, target: torch.Tensor,
             is_ct: torch.Tensor) -> torch.Tensor:
    """Per-sample routed segmentation loss (is_ct: (B,) bool)."""
    if is_ct.dim() == 0:
        is_ct = is_ct.expand(logits.shape[0])
    is_ct = is_ct.bool()
    losses = []
    if (~is_ct).any():
        losses.append(seg_loss_mr(logits[~is_ct], target[~is_ct]))
    if is_ct.any():
        losses.append(seg_loss_ct(logits[is_ct], target[is_ct]))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


# ------------------------------------------------------------- stage-1 aux

def nt_xent_loss(z_ct: torch.Tensor, z_mri: torch.Tensor,
                 temp: float = TEMPERATURE) -> torch.Tensor:
    """NT-Xent over in-batch CT/MRI pairs (row i <-> row i positive).

    Both inputs are (B, 256) L2-normalized pooled branch representations.
    Requires B >= 2 for negatives; returns 0 otherwise.
    """
    n = z_ct.shape[0]
    if n < 2:
        return z_ct.sum() * 0.0
    sim = z_ct @ z_mri.t() / temp                     # (B, B)
    labels = torch.arange(n, device=z_ct.device)
    return (F.cross_entropy(sim, labels)
            + F.cross_entropy(sim.t(), labels)) / 2.0


def cps_loss(aux_ct: torch.Tensor, aux_mri: torch.Tensor,
             valid: torch.Tensor) -> torch.Tensor:
    """Cross-pseudo supervision: bidirectional KL between branch predictions.

    valid: (B,) 1.0 where both branches exist (>=1 MRI sequence present).
    Each branch guides the other -> stop-gradient on the target side of
    each direction (symmetric distillation).
    """
    sel = valid > 0.5
    if not sel.any():
        return aux_ct.sum() * 0.0
    a, b = aux_ct[sel], aux_mri[sel]
    kl_ab = F.kl_div(F.log_softmax(a, dim=1), b.softmax(dim=1).detach(),
                     reduction="batchmean")
    kl_ba = F.kl_div(F.log_softmax(b, dim=1), a.softmax(dim=1).detach(),
                     reduction="batchmean")
    return (kl_ab + kl_ba) / 2.0


def stage1_loss(out: dict, target: torch.Tensor, is_ct: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, float]]:
    """L_stage1 = L_seg + 0.5 L_contra + 0.3 L_cps (paper Eq. stage1)."""
    l_seg = seg_loss(out["seg_logits"], target, is_ct)

    # Contrastive: only samples where BOTH branches exist (CT always
    # available in training data; MRI needs >=1 sequence after dropout).
    valid = out["mri_available"]
    sel = valid > 0.5
    if sel.sum() >= 2:
        l_contra = nt_xent_loss(out["pooled_ct"][sel], out["pooled_mri"][sel])
    else:
        l_contra = out["seg_logits"].sum() * 0.0

    l_cps = cps_loss(out["aux_ct_logits"], out["aux_mri_logits"], valid)

    total = l_seg + W_CONTRA * l_contra + W_CPS * l_cps
    return total, {"seg": float(l_seg), "contra": float(l_contra),
                   "cps": float(l_cps)}


# ---------------------------------------------------------------- stage-2

def kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL[Dir(alpha) || Dir(1,...,1)] per sample, mean over batch."""
    S = alpha.sum(dim=1)                                # (B,H,W)
    K = alpha.shape[1]
    lgamma = torch.lgamma
    kl = (lgamma(S)
          - torch.lgamma(torch.tensor(float(K), device=alpha.device))
          - lgamma(alpha).sum(dim=1)
          + ((alpha - 1.0) * (torch.digamma(alpha)
                              - torch.digamma(S.unsqueeze(1)))).sum(dim=1))
    return kl.mean()


def evidential_loss(alpha: torch.Tensor, target: torch.Tensor,
                    lam: float) -> tuple[torch.Tensor, dict[str, float]]:
    """L_evidential = Dirichlet risk + lam * KL  (paper Eq. evidential).

    Dirichlet risk = expected squared error of the softmax probabilities
    under Dir(alpha) (Sensoy et al. 2018 — the standard evidential
    segmentation objective, i.e. the paper's "Dirichlet NLL").
    """
    y = _onehot(target)                                  # (B,C,H,W)
    S = alpha.sum(dim=1, keepdim=True)                   # (B,1,H,W)
    p = alpha / S
    term1 = (y - p) ** 2
    term2 = alpha * (S - alpha) / (S ** 2 * (S + 1.0))
    nll = (term1 + term2).sum(dim=1).mean()
    kl = kl_to_uniform(alpha)
    return nll + lam * kl, {"dir_nll": float(nll), "kl": float(kl),
                            "lam": lam}


def kl_lambda(epoch: int) -> float:
    """Anneal 0 -> 0.1 over the first KL_ANNEAL_EPOCHS epochs (paper)."""
    return min(KL_LAMBDA_MAX, KL_LAMBDA_MAX * epoch / KL_ANNEAL_EPOCHS)


def stage2_loss(out: dict, target: torch.Tensor, is_ct: torch.Tensor,
                epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    """L_stage2 = L_seg + L_evidential (paper Eq. stage2)."""
    l_seg = seg_loss(out["seg_logits"], target, is_ct)
    lam = kl_lambda(epoch)
    l_ev, ev_parts = evidential_loss(out["evidence"], target, lam)
    total = l_seg + l_ev
    parts = {"seg": float(l_seg), "evidential": float(l_ev), **ev_parts}
    return total, parts


# -------------------------------------------------------- UG prediction

def uncertainty_guided_probs(logits: torch.Tensor, alpha: torch.Tensor,
                             enabled: bool = True) -> torch.Tensor:
    """Uncertainty-guided fusion of the final prediction (figure):

        u  = C / alpha_0   (vacuity)
        w  = 1 / (1 + mean_u)
        p' = w * softmax(logits) + (1 - w) * (1 / C)

    High uncertainty pulls the prediction toward the uniform prior
    (Dir(1) mean), low uncertainty keeps the model's distribution.
    """
    p = logits.softmax(dim=1)
    if not enabled:
        return p
    C = alpha.shape[1]
    vacuity = C / alpha.sum(dim=1)                       # (B,H,W)
    w = 1.0 / (1.0 + vacuity.mean(dim=(1, 2)))           # (B,)
    w = w.view(-1, 1, 1, 1)
    return w * p + (1.0 - w) * (1.0 / C)
