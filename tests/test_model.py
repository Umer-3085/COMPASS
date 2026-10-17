"""Shape/behavior tests for the COMPASS model (CPU, random weights)."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import COMPASS

B, H, W = 2, 235, 235

# Six evaluation conditions from the paper:
# Full (CT+T1+T2), CT+T1, CT+T2, T1-only, T2-only, CT-only
CONDITIONS = {
    "Full":     [1.0, 1.0, 1.0],
    "CT+T1":    [1.0, 1.0, 0.0],
    "CT+T2":    [1.0, 0.0, 1.0],
    "T1-only":  [0.0, 1.0, 0.0],
    "T2-only":  [0.0, 0.0, 1.0],
    "CT-only":  [1.0, 0.0, 0.0],
}


def make_model() -> COMPASS:
    # pretrained=False: no weight download in tests (training will use True)
    return COMPASS(pretrained=False)


def test_stage1_conditions():
    model = make_model()
    model.eval()  # BN in eval mode; structure test only
    for name, m in CONDITIONS.items():
        mask = torch.tensor([m, m])
        out = model(torch.rand(B, 1, H, W), torch.rand(B, 1, H, W),
                    torch.rand(B, 1, H, W), mask)
        assert out["seg_logits"].shape == (B, 5, H, W), f"{name}: seg {out['seg_logits'].shape}"
        assert out["evidence"].shape == (B, 5, H, W), f"{name}: evid {out['evidence'].shape}"
        assert (out["evidence"] >= 1.0).all(), f"{name}: alpha must be >= 1 (Softplus+1)"
        assert out["pooled_ct"].shape == (B, 256)
        assert out["pooled_mri"].shape == (B, 256)
        assert out["aux_ct_logits"].shape == (B, 5, 60, 60)
        assert out["aux_mri_logits"].shape == (B, 5, 60, 60)
        assert torch.isfinite(out["seg_logits"]).all(), f"{name}: non-finite seg logits"
        print(f"  [{name}] seg={tuple(out['seg_logits'].shape)} "
              f"evid min={out['evidence'].min():.2f} OK")


def test_mixed_batch():
    """One batch with different masks per sample (train-time dropout)."""
    model = make_model()
    model.eval()
    mask = torch.tensor([
        [1.0, 1.0, 1.0],   # Full
        [1.0, 0.0, 0.0],   # CT-only
    ])
    out = model(torch.rand(B, 1, H, W), torch.rand(B, 1, H, W),
                torch.rand(B, 1, H, W), mask)
    assert out["seg_logits"].shape == (B, 5, H, W)
    assert out["mri_available"].tolist() == [1.0, 0.0]
    assert torch.isfinite(out["seg_logits"]).all()
    print("  [mixed batch] OK")


def test_gradients_stage1():
    """Backward pass: all stage-1 modules receive gradients."""
    model = make_model()
    model.train()
    mask = torch.ones(B, 3)
    out = model(torch.rand(B, 1, H, W), torch.rand(B, 1, H, W),
                torch.rand(B, 1, H, W), mask)
    loss = (out["seg_logits"].mean() + out["pooled_ct"].mean()
            + out["evidence"].mean() + out["aux_ct_logits"].mean()
            + out["aux_mri_logits"].mean() + out["pooled_mri"].mean())
    loss.backward()
    for name, mod in [("ct_enc", model.ct_encoder), ("mri_enc", model.mri_encoder),
                      ("fusion", model.fusion), ("decoder", model.decoder),
                      ("evidential", model.evidential)]:
        grads = [p.grad for p in mod.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), \
            f"{name} received no gradient"
    print("  [stage1 grads] OK")


def test_stage2():
    """init_stage2: encoders+fusion frozen, new decoder + evidential trainable."""
    model = make_model()
    model.init_stage2()
    assert model.stage == 2 and model.decoder2 is not None

    frozen = ["ct_encoder", "mri_encoder", "ct_proj", "mri_proj", "fusion"]
    for name in frozen:
        for p in getattr(model, name).parameters():
            assert not p.requires_grad, f"{name} should be frozen"
    for name in ["decoder2", "evidential", "aux_head"]:
        for p in getattr(model, name).parameters():
            assert p.requires_grad, f"{name} should be trainable"

    model.train()
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    out = model(torch.rand(B, 1, H, W), torch.rand(B, 1, H, W),
                torch.rand(B, 1, H, W), mask)
    assert out["seg_logits"].shape == (B, 5, H, W)
    assert torch.isfinite(out["seg_logits"]).all()

    loss = out["seg_logits"].mean()
    loss.backward()
    # New decoder gets grads; frozen encoder must NOT
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.decoder2.parameters())
    assert all(p.grad is None for p in model.ct_encoder.parameters())
    print("  [stage2 freeze+grads] OK")


def test_uncertainty_guided_probs():
    """UG fusion output: p = w*p + (1-w)/5 with w = 1/(1+u)."""
    from src.losses import uncertainty_guided_probs
    logits = torch.randn(2, 5, 32, 32)
    alpha = torch.rand(2, 5, 32, 32) + 1.0
    p = uncertainty_guided_probs(logits, alpha, enabled=True)
    assert p.shape == (2, 5, 32, 32)
    assert torch.allclose(p.sum(dim=1), torch.ones(2, 32, 32), atol=1e-4)
    assert (p >= 0).all()
    print("  [UG probs] OK")


if __name__ == "__main__":
    print("=== test_stage1_conditions ===")
    test_stage1_conditions()
    print("=== test_mixed_batch ===")
    test_mixed_batch()
    print("=== test_gradients_stage1 ===")
    test_gradients_stage1()
    print("=== test_stage2 ===")
    test_stage2()
    print("=== test_uncertainty_guided_probs ===")
    test_uncertainty_guided_probs()
    print("\nALL MODEL TESTS PASSED")
