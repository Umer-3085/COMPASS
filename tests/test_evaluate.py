"""Tests for the evaluation runner: condition eligibility and patient keys."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import CONDITIONS, CaseAccumulator, run_condition


class MockModel:
    """Returns constant outputs (background prediction, uniform evidence)."""

    def eval(self):
        pass

    def __call__(self, ct, t1, t2, mask):
        B = ct.shape[0]
        logits = torch.zeros(B, 5, 235, 235)
        logits[:, 0] = 5.0
        return {"seg_logits": logits,
                "evidence": torch.full((B, 5, 235, 235), 2.0)}


def _batch():
    """One MR patient 8 slice + one CT patient 8 slice (IDs collide)."""
    return {
        "ct": torch.rand(2, 1, 235, 235),
        "t1": torch.rand(2, 1, 235, 235),
        "t2": torch.rand(2, 1, 235, 235),
        "label": torch.randint(0, 5, (2, 235, 235)),
        "mask": torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
        "modality": ["MR", "CT"],
        "case_id": [8, 8],
    }


def test_eligibility_and_keys():
    model = MockModel()
    loader = [_batch()]

    # Full: only the MR patient is eligible
    cases = run_condition(model, loader, "Full", CONDITIONS["Full"],
                          torch.device("cpu"), use_ug=True)
    assert list(cases.keys()) == ["MR_8"]

    # CT-only: only the CT patient is eligible
    cases = run_condition(model, loader, "CT-only", CONDITIONS["CT-only"],
                          torch.device("cpu"), use_ug=True)
    assert list(cases.keys()) == ["CT_8"]
    print("  [eligibility + collision-free keys] OK")


def test_accumulator_metrics():
    """Perfect background prediction on background-only labels."""
    acc = CaseAccumulator()
    gt = torch.zeros(235, 235, dtype=torch.int64)
    pred = torch.zeros(235, 235, dtype=torch.int64)
    probs = torch.zeros(5, 235, 235)
    probs[0] = 1.0
    acc.update(pred.numpy(), probs.numpy(), torch.full((5, 235, 235), 2.0).numpy(),
               gt.numpy())

    ov = acc.overall_metrics()
    assert ov["pixel_accuracy"] == 1.0
    assert ov["ece"] == 0.0                    # conf 1.0, 100% correct
    assert ov["brier"] == 0.0
    assert abs(ov["vacuity"] - 0.5) < 1e-6     # C/alpha0 = 5/10
    for cls in range(1, 5):
        m = acc.organ_metrics(cls)
        assert m["gt_present"] == 0            # organ absent -> filtered out
        assert m["dice"] == 1.0                # empty-vs-empty convention
        assert m["hd95"] != m["hd95"]          # NaN, excluded from averages
    print("  [accumulator perfect-bg metrics] OK")


if __name__ == "__main__":
    print("=== test_eligibility_and_keys ===")
    test_eligibility_and_keys()
    print("=== test_accumulator_metrics ===")
    test_accumulator_metrics()
    print("\nALL EVALUATE TESTS PASSED")
