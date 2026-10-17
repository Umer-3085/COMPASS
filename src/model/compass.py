"""COMPASS: Cross-Modality Missing-Modality Adaptive Segmentation System.

Full model assembly per the paper's Model Architecture section and the
COMPASS architecture figure:

    CT ----> CT encoder (ResNet34, ImageNet)
    T1,T2 -> shared MRI encoder (ResNet34, shared weights)
        |         (pyramid projections -> 256 ch, incl. native 1/4 level)
        v
    Missing-Modality Gatekeeper + Bidirectional Cross-Attention Fusion
        (4 heads, d=256, residual + LayerNorm, reliability 1x1 gates,
         shared module applied at 1/16, 1/8, 1/4 scales)
        |
        +--> Segmentation Decoder (4 up-blocks, GN + ReLU) --> seg logits
        +--> Evidential Head (1x1 conv, Softplus + 1) ------> Dirichlet alpha
        |
    (Stage 2) Uncertainty-Guided Fusion: w = 1/(1+u)

Also exposes the auxiliary outputs required by the training strategy:
- 256-d pooled branch representations (NT-Xent contrastive loss)
- auxiliary branch predictions (cross-pseudo supervision, bidirectional KL)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.decoder import EvidentialHead, SegmentationDecoder
from src.model.encoders import ResNet34Encoder
from src.model.fusion import SCALE_ORDER, CrossAttentionFusion

NUM_CLASSES = 5


def _empty_pyramid(B: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Zero pyramid for a batch-level-absent modality (saves encoder passes)."""
    return {
        "1/2": torch.zeros(B, 64, 120, 120, device=device),
        "1/4": torch.zeros(B, 128, 60, 60, device=device),
        "1/8": torch.zeros(B, 256, 30, 30, device=device),
        "1/16": torch.zeros(B, 512, 15, 15, device=device),
    }


class _Projections(nn.Module):
    """Per-encoder 1x1 projections: pyramid -> 256 channels per scale.

    The 1/4 level (from layer2, 128 ch) is the paper's stated feature
    resolution: "256-channel feature representations at one-quarter of
    the original spatial resolution" for both branches.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.p16 = nn.Conv2d(512, dim, 1)   # 1/16
        self.p8 = nn.Conv2d(256, dim, 1)    # 1/8
        self.p4 = nn.Conv2d(128, dim, 1)    # 1/4

    def forward(self, pyr: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "1/16": self.p16(pyr["1/16"]),
            "1/8": self.p8(pyr["1/8"]),
            "1/4": self.p4(pyr["1/4"]),
        }


class COMPASS(nn.Module):
    def __init__(self, pretrained: bool = True, dim: int = 256, heads: int = 4):
        super().__init__()
        # Encoders (MRI encoder shared between T1 and T2 by construction:
        # the same module is applied to both inputs).
        self.ct_encoder = ResNet34Encoder(pretrained)
        self.mri_encoder = ResNet34Encoder(pretrained)

        self.ct_proj = _Projections(dim)
        self.mri_proj = _Projections(dim)

        self.fusion = CrossAttentionFusion(dim, heads)
        self.decoder = SegmentationDecoder(dim, NUM_CLASSES)
        self.evidential = EvidentialHead(dim, NUM_CLASSES)   # shared head
        # Aux predictions for cross-pseudo supervision (shared CT/MRI).
        self.aux_head = nn.Conv2d(dim, NUM_CLASSES, kernel_size=1)

        # Stage 2 artifacts (created lazily by init_stage2()).
        self.decoder2: SegmentationDecoder | None = None
        self.stage = 1
        self.uncertainty_guided = False   # Stage-2 output-level UG fusion

    # ------------------------- stage management -------------------------

    def init_stage2(self) -> None:
        """Stage 2: freeze encoders + fusion; attach a NEW decoder.

        Per paper: "the trained CT encoder, MRI encoder and fusion module
        are frozen. The segmentation decoder and evidential head are then
        trained on top of the frozen feature representations."
        """
        self.decoder2 = SegmentationDecoder()   # fresh initialization
        self.freeze_encoders_and_fusion()
        self.stage = 2
        self.uncertainty_guided = True

    def freeze_encoders_and_fusion(self) -> None:
        for module in (self.ct_encoder, self.mri_encoder,
                       self.ct_proj, self.mri_proj, self.fusion):
            for p in module.parameters():
                p.requires_grad = False

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    # ------------------------- helpers ----------------------------------

    def _pool256(self, feat4: torch.Tensor) -> torch.Tensor:
        """256-d global average pooled, L2-normalized branch representation."""
        z = feat4.mean(dim=(2, 3))
        return F.normalize(z, dim=1)

    # ------------------------- forward ----------------------------------

    def forward(
        self,
        ct: torch.Tensor,    # (B, 1, H, W)
        t1: torch.Tensor,    # (B, 1, H, W)
        t2: torch.Tensor,    # (B, 1, H, W)
        mask: torch.Tensor,  # (B, 3) availability of [CT, T1, T2], float
    ) -> dict[str, torch.Tensor]:
        B = ct.shape[0]
        device = ct.device
        H, W = ct.shape[-2:]
        ct_v, t1_v, t2_v = mask[:, 0], mask[:, 1], mask[:, 2]
        mri_available = ((t1_v + t2_v) > 0).float()

        # Pad 235 -> 240 (exact /16 pyramid), grayscale -> 3ch for ImageNet.
        ct_p = F.pad(ct, (2, 3, 2, 3)).repeat(1, 3, 1, 1)
        t1_p = F.pad(t1, (2, 3, 2, 3)).repeat(1, 3, 1, 1)
        t2_p = F.pad(t2, (2, 3, 2, 3)).repeat(1, 3, 1, 1)

        # Encode (batch-level short-circuit when a modality is fully absent).
        ct_pyr = (self.ct_encoder(ct_p) if ct_v.any()
                  else _empty_pyramid(B, device))
        t1_pyr = (self.mri_encoder(t1_p) if t1_v.any()
                  else _empty_pyramid(B, device))
        t2_pyr = (self.mri_encoder(t2_p) if t2_v.any()
                  else _empty_pyramid(B, device))

        ct_f = self.ct_proj(ct_pyr)
        t1_f = self.mri_proj(t1_pyr)
        t2_f = self.mri_proj(t2_pyr)

        # Pooled branch representations (pre-fusion, for NT-Xent).
        pooled_ct = self._pool256(ct_f["1/4"])
        mri_pool_num = (t1_v.view(B, 1) * self._pool256(t1_f["1/4"])
                        + t2_v.view(B, 1) * self._pool256(t2_f["1/4"]))
        mri_pool_den = (t1_v + t2_v).view(B, 1).clamp_min(1.0)
        pooled_mri = F.normalize(mri_pool_num / mri_pool_den, dim=1)

        # Auxiliary branch predictions for cross-pseudo supervision.
        aux_ct = self.aux_head(ct_f["1/4"])
        mri4_num = t1_v.view(B, 1, 1, 1) * t1_f["1/4"] + \
            t2_v.view(B, 1, 1, 1) * t2_f["1/4"]
        mri4 = mri4_num / mri_pool_den.view(B, 1, 1, 1)
        aux_mri = self.aux_head(mri4)

        # Stage 2: uncertainty-guided per-modality gating of branch
        # features (paper: w_m = 1/(1 + mean_uncertainty_m), weights
        # normalized over available modalities). Stage 1: uniform (off).
        if self.stage == 2:
            ct_f, t1_f, t2_f = self._uncertainty_gate(
                ct_f, t1_f, t2_f, ct_v, t1_v, t2_v
            )

        # Fusion with gatekeeper (shared module across scales).
        fused = self.fusion(ct_f, t1_f, t2_f, ct_v, t1_v, t2_v)

        # Detail skip: raw CT encoder 1/2 features (64 ch, decoder up3;
        # zeroed when CT is unavailable so no dead-stream leak).
        detail = ct_pyr["1/2"] * ct_v.view(B, 1, 1, 1)

        # Decoder (stage 2 uses the NEW decoder).
        decoder = self.decoder2 if (self.stage == 2 and self.decoder2) else self.decoder
        seg_logits = decoder(fused, detail)

        # Evidential head on the finest fused scale -> upsample -> crop.
        evidence_low = self.evidential(fused["1/4"])
        evidence = F.interpolate(
            evidence_low, size=(H, W), mode="bilinear", align_corners=False
        )

        out = {
            "seg_logits": seg_logits[:, :, :H, :W],
            "evidence": evidence,                       # Dirichlet alphas
            "pooled_ct": pooled_ct,
            "pooled_mri": pooled_mri,
            "mri_available": mri_available,
            "aux_ct_logits": aux_ct,
            "aux_mri_logits": aux_mri,
        }
        return out

    # ------------------------- stage-2 gating ----------------------------

    def _uncertainty_gate(
        self,
        ct_f: dict[str, torch.Tensor],
        t1_f: dict[str, torch.Tensor],
        t2_f: dict[str, torch.Tensor],
        ct_v: torch.Tensor,
        t1_v: torch.Tensor,
        t2_v: torch.Tensor,
    ) -> tuple[dict, dict, dict]:
        """Uncertainty-guided reliability of each modality (Stage 2).

        For each modality m: u_m = mean of its vacuity map
        (C / alpha_0, evidence from the shared evidential head applied to
        that branch's 1/4 features), w_m = 1 / (1 + u_m), weights
        normalized over available modalities; branch features scaled.
        """
        B = ct_v.shape[0]

        def branch_weight(feat4: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            alpha = self.evidential(feat4)                 # (B, 5, h, w)
            vacuity = NUM_CLASSES / alpha.sum(dim=1)       # (B, h, w)
            u = vacuity.mean(dim=(1, 2))                   # (B,)
            w = 1.0 / (1.0 + u) * valid                    # mask out missing
            return w

        w_ct = branch_weight(ct_f["1/4"], ct_v)
        w_t1 = branch_weight(t1_f["1/4"], t1_v)
        w_t2 = branch_weight(t2_f["1/4"], t2_v)
        denom = (w_ct + w_t1 + w_t2).clamp_min(1e-6).view(B, 1, 1, 1)
        w_ct, w_t1, w_t2 = (w.view(B, 1, 1, 1) / denom
                            for w in (w_ct, w_t1, w_t2))

        ct_f = {s: ct_f[s] * w_ct for s in SCALE_ORDER}
        t1_f = {s: t1_f[s] * w_t1 for s in SCALE_ORDER}
        t2_f = {s: t2_f[s] * w_t2 for s in SCALE_ORDER}
        return ct_f, t1_f, t2_f
