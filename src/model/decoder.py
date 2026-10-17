"""Segmentation decoder for COMPASS.

Per paper: "The fused feature representation is passed through four
upsampling blocks, each of which progressively increases the spatial
resolution until the original image resolution is recovered. Skip
connections from the fused representation supply the spatial information
needed for accurate organ boundaries. Each decoder block applies Group
Normalization followed by ReLU activation."

Resolution bookkeeping (input padded 235 -> 240):
    fused 1/16 (15) --up1 + fused 1/8 skip-->  1/8 (30)
                          --up2 + fused 1/4-->  1/4 (60)
                          --up3 + encoder 1/2--> 1/2 (120)
                          --up4-------------->  1/1 (240)  -> crop 235

Four doubling blocks recover the original resolution exactly, which is
only possible from a 1/16 start — resolving the paper's internal
tension between "feature resolution 1/4" (satisfied: the fusion module
operates AT 1/4 and its features skip into block 2) and "4 upsampling
blocks" (satisfied: 1/16 -> 1/1). Block 3 uses the always-present CT
encoder's 1/2 features for boundary detail (the CT stream is the
primary reference modality per the paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    groups = 8
    while ch % groups != 0:
        groups //= 2
    return nn.GroupNorm(max(groups, 1), ch)


class UpBlock(nn.Module):
    """Bilinear x2 upsample -> GN -> ReLU -> 3x3 conv (-> concat skip)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _gn(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.post = nn.Sequential(
            _gn(out_ch + skip_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.body(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(
                    skip, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.post(x)


class SegmentationDecoder(nn.Module):
    """4 up-blocks over multi-scale fused features + encoder detail skip."""

    def __init__(self, dim: int = 256, num_classes: int = 5) -> None:
        super().__init__()
        self.up1 = UpBlock(dim, dim, 128)          # 1/16 -> 1/8  (+ fused 1/8)
        self.up2 = UpBlock(128, dim, 64)           # 1/8  -> 1/4  (+ fused 1/4)
        self.up3 = UpBlock(64, 64, 32)             # 1/4  -> 1/2  (+ enc 1/2)
        self.up4 = UpBlock(32, 0, 16)              # 1/2  -> 1/1
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(
        self,
        fused: dict[str, torch.Tensor],
        detail_skip: torch.Tensor,   # CT encoder 1/2 features (B, 64, H/2, W/2)
    ) -> torch.Tensor:
        x = self.up1(fused["1/16"], fused["1/8"])
        x = self.up2(x, fused["1/4"])
        x = self.up3(x, detail_skip)
        x = self.up4(x, None)
        return self.head(x)


class EvidentialHead(nn.Module):
    """1x1 conv -> Softplus + 1 -> Dirichlet alphas (paper Eq. evidential_alpha).

    Applied to the finest fused scale (1/4 — "the fused feature
    representation"), then upsampled to input resolution by the caller.
    """

    def __init__(self, dim: int = 256, num_classes: int = 5) -> None:
        super().__init__()
        self.evidence = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.evidence(feat)) + 1.0
