"""ResNet34 encoders for COMPASS.

Per paper: CT branch uses ResNet34 with ImageNet-pretrained weights; the
MRI branch uses the SAME ResNet34 architecture with weights shared
between the T1 and T2 inputs (one module applied to both).

Feature pyramid taps (conv1 stride 2, standard ResNet34):
    1/2  -> layer1  (64 ch)
    1/4  -> layer2  (128 ch)
    1/8  -> layer3  (256 ch)
    1/16 -> layer4  (512 ch)

The paper's "both branches produce 256-channel feature representations
at one-quarter of the original spatial resolution" is realized as a
per-encoder 1x1 projection of the pyramid to 256 channels, including a
native 1/4-resolution level (see projections in COMPASS).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet34_Weights, resnet34

# Padded input size (235 -> 240) yields exact pyramid sizes divisible by 16:
# 240 -> 120 (1/2) -> 60 (1/4) -> 30 (1/8) -> 15 (1/16)
PADDED = 240


def build_resnet34(pretrained: bool = True) -> nn.Module:
    """Create a ResNet34, loading ImageNet weights when available."""
    if pretrained:
        try:
            return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except Exception as exc:  # offline / blocked download -> random init
            warnings.warn(
                f"ImageNet weights unavailable ({exc}); using random init."
            )
    return resnet34(weights=None)


class ResNet34Encoder(nn.Module):
    """ResNet34 feature extractor returning a multi-scale pyramid.

    Input: (B, 3, H, W) — grayscale replicated to 3 channels by the
    caller (keeps ImageNet first-conv weights valid without surgery).
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net = build_resnet34(pretrained)
        # Stem keeps conv1 stride 2 but the stride-2 maxpool is replaced by
        # a stride-1 maxpool (standard segmentation adaptation). Without it
        # the pyramid would shift to 1/4..1/32 and neither the paper's
        # "feature resolution 1/4" tap (layer2, 128 ch) nor the exact
        # 1/16 -> 1/1 recovery of the 4 decoder blocks would hold.
        # ImageNet weights are unaffected (maxpool has no parameters).
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = net.layer1   # 1/2,  64
        self.layer2 = net.layer2   # 1/4,  128
        self.layer3 = net.layer3   # 1/8,  256
        self.layer4 = net.layer4   # 1/16, 512

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.maxpool(self.stem(x))
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return {"1/2": l1, "1/4": l2, "1/8": l3, "1/16": l4}
