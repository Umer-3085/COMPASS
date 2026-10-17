"""COMPASS model package.

Architecture per paper Section "Model Architecture" + COMPASS architecture figure:
- CT encoder: ResNet34, ImageNet pretrained
- MRI encoder: shared ResNet34 for T1/T2
- Missing-Modality Gatekeeper: zero K/V of missing modalities in cross-attention
- Bidirectional cross-attention fusion: 4 heads, d=256, residual + LayerNorm
- Reliability weights: sigmoid-gated 1x1 conv
- Decoder: 4 up-blocks, GroupNorm + ReLU, skips from fused features
- Segmentation head: 1x1 conv -> 5-class logits
- Evidential head: 1x1 conv -> Softplus + 1 -> 5 Dirichlet alphas
- Uncertainty-guided fusion: w = 1 / (1 + u)
"""

from src.model.compass import COMPASS

__all__ = ["COMPASS"]
