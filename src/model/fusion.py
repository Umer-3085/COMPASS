"""Cross-attention fusion with Missing-Modality Gatekeeper.

Per paper + architecture figure:
- Bidirectional cross-attention between CT and MRI streams:
    CT-to-MRI: CT features are Q, MRI features are K/V.
    MRI-to-CT: MRI features are Q, CT features are K/V.
- 4 heads, attention dimension 256, residual connection + LayerNorm.
- Gatekeeper: the K/V representations of a missing modality are ZEROED
  before attention ("Zero K/V of missing MRI" in the figure), so a
  missing input cannot influence the fused representation. Zeroed K/V is
  numerically safe: softmax over all-zero scores is uniform and values
  are zero, so the attention output is exactly 0 (no NaN even when an
  entire modality stream is absent).
- Reliability weights: a sigmoid-gated 1x1 convolution dynamically
  adjusts how strongly each available modality contributes; weights are
  masked and renormalized over available modalities.

The SAME fusion module (shared weights) is applied at three scales
(1/16, 1/8, 1/4) of the encoder pyramids, producing multi-scale fused
features. The 1/4 level is the paper's stated "feature resolution"
(256 channels at 1/4 of input resolution) and supplies the decoder's
fine-grained skips ("skip connections from the fused features").
"""

from __future__ import annotations

import torch
import torch.nn as nn

SCALE_ORDER = ("1/16", "1/8", "1/4")


class ZeroKVAttention(nn.Module):
    """Multi-head attention whose K/V can be zeroed per token (gatekeeper)."""

    def __init__(self, dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.d_head = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        # bias=False: when all K/V are zeroed the attention output must be
        # exactly 0 so residuals stay clean (no constant bias leak).
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        q_src: torch.Tensor,       # (B, N, C) queries
        kv_src: torch.Tensor,      # (B, M, C) keys/values
        kv_valid: torch.Tensor,    # (B, M) 1 = valid, 0 = missing -> K/V zeroed
    ) -> torch.Tensor:
        B, N, C = q_src.shape
        M = kv_src.shape[1]
        h, d = self.heads, self.d_head

        q = self.q(q_src).view(B, N, h, d).transpose(1, 2)   # (B,h,N,d)
        k = self.k(kv_src).view(B, M, h, d).transpose(1, 2)   # (B,h,M,d)
        v = self.v(kv_src).view(B, M, h, d).transpose(1, 2)

        # Gatekeeper: zero keys and values of missing tokens (paper/figure).
        kv_valid = kv_valid.view(B, 1, M, 1).to(k.dtype)
        k = k * kv_valid
        v = v * kv_valid

        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)       # (B,h,N,M)
        attn = scores.softmax(dim=-1)
        y = attn @ v                                           # (B,h,N,d)
        y = y.transpose(1, 2).reshape(B, N, C)
        return self.out(y)


class CrossAttentionFusion(nn.Module):
    """Bidirectional CT<->MRI fusion + reliability gates, shared across scales."""

    def __init__(self, dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.ct_to_mri = ZeroKVAttention(dim, heads)   # CT queries MRI
        self.mri_to_ct = ZeroKVAttention(dim, heads)   # MRI queries CT
        self.norm_ct = nn.LayerNorm(dim)
        self.norm_mri = nn.LayerNorm(dim)
        # Reliability weights: sigmoid-gated 1x1 conv (one conv shared
        # across modalities, applied to each modality's feature map).
        self.reliability = nn.Conv2d(dim, 1, kernel_size=1)

    @staticmethod
    def _tokens(feat: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)."""
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2)

    @staticmethod
    def _maps(tok: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """(B, H*W, C) -> (B, C, H, W)."""
        B, N, C = tok.shape
        H, W = hw
        return tok.transpose(1, 2).view(B, C, H, W)

    def forward_scale(
        self,
        ct: torch.Tensor,        # (B, C, H, W)
        t1: torch.Tensor,        # (B, C, H, W)
        t2: torch.Tensor,        # (B, C, H, W)
        ct_valid: torch.Tensor,  # (B,)
        t1_valid: torch.Tensor,  # (B,)
        t2_valid: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Fuse one scale; returns fused (B, C, H, W)."""
        B, C, H, W = ct.shape
        device = ct.device

        # Belt: zero features of missing modalities so residual/queries
        # of dead streams carry no information at all.
        ct_v = ct_valid.view(B, 1, 1, 1)
        t1_v = t1_valid.view(B, 1, 1, 1)
        t2_v = t2_valid.view(B, 1, 1, 1)
        ct0, t1_0, t2_0 = ct * ct_v, t1 * t1_v, t2 * t2_v

        # MRI token sequence = [T1 tokens ; T2 tokens] with per-token
        # validity so the gatekeeper can zero K/V of each missing one.
        mri_map = torch.stack([t1_0, t2_0], dim=2)             # (B, C, 2, H, W)
        mri_map = mri_map.flatten(3)                           # (B, C, 2*H*W)
        mri_valid = torch.stack([t1_valid, t2_valid], dim=1)    # (B, 2)

        ct_tok = self._tokens(ct0)                             # (B, N, C)
        mri_tok = self._tokens(mri_map)                        # (B, 2N, C)
        mri_tok_v = mri_valid.repeat_interleave(H * W, dim=1)  # (B, 2N)
        ct_tok_v = torch.ones(B, H * W, device=device, dtype=ct.dtype)

        # Parallel bidirectional attention (both directions read the inputs).
        ct_att = self.ct_to_mri(ct_tok, mri_tok, mri_tok_v)
        mri_att = self.mri_to_ct(mri_tok, ct_tok, ct_tok_v)

        ct_out = self.norm_ct(ct_tok + ct_att)
        mri_out = self.norm_mri(mri_tok + mri_att)

        ct_f = self._maps(ct_out, (H, W))
        # Split MRI stream back into T1/T2 feature maps.
        t1_f = self._maps(mri_out[:, : H * W], (H, W))
        t2_f = self._maps(mri_out[:, H * W:], (H, W))

        # Reliability gates (sigmoid-gated 1x1 conv), masked + renormalized.
        g_ct = torch.sigmoid(self.reliability(ct_f)) * ct_v
        g_t1 = torch.sigmoid(self.reliability(t1_f)) * t1_v
        g_t2 = torch.sigmoid(self.reliability(t2_f)) * t2_v
        denom = (g_ct + g_t1 + g_t2).clamp_min(1e-6)
        fused = (g_ct * ct_f + g_t1 * t1_f + g_t2 * t2_f) / denom
        return fused

    def forward(
        self,
        ct_feats: dict[str, torch.Tensor],
        t1_feats: dict[str, torch.Tensor],
        t2_feats: dict[str, torch.Tensor],
        ct_valid: torch.Tensor,
        t1_valid: torch.Tensor,
        t2_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused: dict[str, torch.Tensor] = {}
        for s in SCALE_ORDER:
            fused[s] = self.forward_scale(
                ct_feats[s], t1_feats[s], t2_feats[s],
                ct_valid, t1_valid, t2_valid,
            )
        return fused
