"""Tests for COMPASS loss functions (shapes, values, annealing)."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses import (cps_loss, evidential_loss, kl_lambda, nt_xent_loss,
                        seg_loss, stage1_loss, stage2_loss,
                        uncertainty_guided_probs)

torch.manual_seed(0)


def test_seg_loss_routing():
    """MR samples get full CE+Dice; CT samples get liver-masked loss."""
    B, H, W = 4, 32, 32
    logits = torch.randn(B, 5, H, W, requires_grad=True)
    # samples 0-1 MR (labels 0..4), samples 2-3 CT (labels 0/1 only)
    target = torch.randint(0, 5, (B, H, W))
    target[2:] = (torch.rand(B // 2, H, W) > 0.7).long()
    is_ct = torch.tensor([False, False, True, True])

    loss = seg_loss(logits, target, is_ct)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print("  [seg routing] OK")

    # CT-only batch with NO liver pixels must still be finite
    logits2 = torch.randn(2, 5, H, W)
    tgt0 = torch.zeros(2, H, W).long()
    l0 = seg_loss(logits2, tgt0, torch.tensor([True, True]))
    assert torch.isfinite(l0)
    print("  [seg empty-liver CT] OK")


def test_nt_xent():
    """Paired (identical) embeddings must beat shuffled/random ones."""
    z = torch.nn.functional.normalize(torch.randn(8, 256), dim=1)
    same = nt_xent_loss(z, z.clone())               # perfect pairs
    rand = nt_xent_loss(z, torch.nn.functional.normalize(torch.randn(8, 256), dim=1))
    assert same < rand, (same, rand)
    assert same < 0.5                                # log(8)=2.08 lower bound-ish
    print(f"  [nt-xent] same={same:.3f} < random={rand:.3f} OK")

    # batch of 1 -> 0
    assert float(nt_xent_loss(z[:1], z[:1])) == 0.0
    print("  [nt-xent degenerate] OK")


def test_cps():
    """Identical branch predictions -> KL ~ 0; valid mask respected."""
    aux = torch.randn(4, 5, 16, 16)
    zero = cps_loss(aux, aux.clone(), torch.ones(4))
    assert abs(float(zero)) < 1e-5
    diff = cps_loss(aux, torch.randn(4, 5, 16, 16), torch.ones(4))
    assert float(diff) > 0
    none = cps_loss(aux, aux, torch.zeros(4))       # no valid samples -> 0
    assert float(none) == 0.0
    print("  [cps] OK")


def test_evidential_and_anneal():
    """Dirichlet risk + KL finite; kl_lambda: 0 -> 0.05 -> 0.1 over 20 ep."""
    alpha = torch.rand(2, 5, 16, 16) + 1.0
    target = torch.randint(0, 5, (2, 16, 16))
    loss, parts = evidential_loss(alpha, target, lam=0.1)
    assert torch.isfinite(loss)
    assert set(parts) == {"dir_nll", "kl", "lam"}
    assert parts["kl"] >= 0
    print(f"  [evidential] nll={parts['dir_nll']:.4f} kl={parts['kl']:.4f} OK")

    assert kl_lambda(0) == 0.0
    assert abs(kl_lambda(10) - 0.05) < 1e-9
    assert kl_lambda(20) == 0.1 and kl_lambda(75) == 0.1
    print("  [kl anneal] OK")


def test_stage_losses():
    """Stage-1 and stage-2 compose finite totals with logged parts."""
    B, H, W = 4, 32, 32
    out = {
        "seg_logits": torch.randn(B, 5, H, W),
        "evidence": torch.rand(B, 5, H, W) + 1.0,
        "pooled_ct": torch.nn.functional.normalize(torch.randn(B, 256), dim=1),
        "pooled_mri": torch.nn.functional.normalize(torch.randn(B, 256), dim=1),
        "mri_available": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "aux_ct_logits": torch.randn(B, 5, 8, 8),
        "aux_mri_logits": torch.randn(B, 5, 8, 8),
    }
    target = torch.randint(0, 5, (B, H, W))
    is_ct = torch.tensor([False, True, False, True])

    total, parts = stage1_loss(out, target, is_ct)
    assert torch.isfinite(total)
    assert set(parts) == {"seg", "contra", "cps"}
    print(f"  [stage1] seg={parts['seg']:.3f} contra={parts['contra']:.3f} "
          f"cps={parts['cps']:.3f} OK")

    total2, parts2 = stage2_loss(out, target, is_ct, epoch=5)
    assert torch.isfinite(total2)
    assert parts2["lam"] == 0.025                     # 0.1 * 5/20
    print(f"  [stage2] seg={parts2['seg']:.3f} nll={parts2['dir_nll']:.3f} "
          f"lam={parts2['lam']} OK")


def test_ug_probs():
    """enabled=True pulls toward uniform by uncertainty; disabled = softmax."""
    logits = torch.randn(2, 5, 16, 16)
    low_u = torch.full((2, 5, 16, 16), 0.01) + 1.0   # alpha ~1 -> vacuity ~5/2ish
    p_off = uncertainty_guided_probs(logits, low_u, enabled=False)
    p_on = uncertainty_guided_probs(logits, low_u, enabled=True)
    assert torch.allclose(p_off, logits.softmax(1), atol=1e-6)
    assert torch.allclose(p_on.sum(1), torch.ones(2, 16, 16), atol=1e-4)
    # high evidence -> w near 1 -> close to softmax
    hi = torch.full((2, 5, 16, 16), 200.0)
    p_hi = uncertainty_guided_probs(logits, hi, enabled=True)
    assert torch.allclose(p_hi, p_off, atol=0.01)
    print("  [ug probs] OK")


if __name__ == "__main__":
    print("=== test_seg_loss_routing ===")
    test_seg_loss_routing()
    print("=== test_nt_xent ===")
    test_nt_xent()
    print("=== test_cps ===")
    test_cps()
    print("=== test_evidential_and_anneal ===")
    test_evidential_and_anneal()
    print("=== test_stage_losses ===")
    test_stage_losses()
    print("=== test_ug_probs ===")
    test_ug_probs()
    print("\nALL LOSS TESTS PASSED")
