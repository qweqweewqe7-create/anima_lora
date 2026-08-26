"""Invariant tests for the stratified tagger-eval metrics.

Pins the AP/F1 math against hand-computed values and — the load-bearing one —
that :func:`predict_with_inference_rule` mirrors ``AnimaTagger.predict``'s
group refinement: solo/escape gating from *predicted* tags, group tags
cleared, argmax winner always emitted when the gate applies.
"""

from __future__ import annotations

import math

import torch

from scripts.anima_tagger.eval_metrics import (
    aggregate_slices,
    assign_slices,
    micro_f1,
    per_tag_average_precision,
    per_tag_prf,
    predict_with_inference_rule,
)
from scripts.anima_tagger.train_common import GroupRouter, _SoftmaxGroup


def _logits(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


# ── Metric math ───────────────────────────────────────────────────────────


def test_per_tag_prf_hand_values():
    pred = torch.tensor([[1, 0], [1, 1], [0, 0]], dtype=torch.float32)
    tgt = torch.tensor([[1, 1], [0, 1], [0, 0]], dtype=torch.float32)
    prec, rec, f1, support = per_tag_prf(pred, tgt)
    # tag0: tp=1 fp=1 fn=0 → p=0.5 r=1 f1=2/3 ; tag1: tp=1 fp=0 fn=1 → p=1 r=0.5 f1=2/3
    assert torch.allclose(prec, torch.tensor([0.5, 1.0]))
    assert torch.allclose(rec, torch.tensor([1.0, 0.5]))
    assert torch.allclose(f1, torch.tensor([2 / 3, 2 / 3]))
    assert torch.equal(support, torch.tensor([1.0, 2.0]))


def test_average_precision_hand_values():
    # scores rank samples [0.9, 0.7, 0.4, 0.2]; positives at ranks 1 and 3
    # → AP = (1/1 + 2/3) / 2 = 5/6. Second tag has no positives → NaN.
    scores = torch.tensor([[0.9, 0.1], [0.7, 0.2], [0.4, 0.3], [0.2, 0.4]])
    tgt = torch.tensor([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=torch.float32)
    ap = per_tag_average_precision(scores, tgt)
    assert abs(ap[0].item() - 5 / 6) < 1e-6
    assert math.isnan(ap[1].item())


def test_micro_f1_pools_cells():
    pred = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    tgt = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    # tp=1 fp=1 fn=1 → micro F1 = 2*1 / (2*1 + 1 + 1) = 0.5
    assert abs(micro_f1(pred, tgt) - 0.5) < 1e-6


# ── Inference-rule prediction ─────────────────────────────────────────────
#
# Vocab layout (n_tags=8):
#   0 solo   1 2girls   2..4 eye colors (group, escape=5)   5 heterochromia
#   6 7 residual tags


def _router(mode: str = "softmax_when_solo") -> GroupRouter:
    return GroupRouter(
        n_tags=8,
        bce_pos_weight=torch.ones(8),
        softmax_groups=[
            _SoftmaxGroup(
                name="eye_color",
                mode=mode,
                tag_indices=torch.tensor([2, 3, 4]),
                escape_indices=torch.tensor([5]),
            )
        ],
        softmax_member_indices=torch.tensor([2, 3, 4]),
        solo_indices=torch.tensor([0]),
        multi_indices=torch.tensor([1]),
    )


def test_group_argmax_replaces_threshold_output_on_solo():
    # Solo predicted; two eye colors pass threshold; argmax (idx 3) must win
    # alone. Residual tag 6 passes threshold untouched.
    probs = torch.tensor([[0.9, 0.1, 0.7, 0.8, 0.1, 0.1, 0.9, 0.1]])
    pred = predict_with_inference_rule(_logits(probs), torch.full((8,), 0.5), _router())
    assert pred[0].tolist() == [True, False, False, True, False, False, True, False]


def test_winner_emitted_even_below_threshold():
    # Solo, no eye color passes threshold — inference still emits the argmax.
    probs = torch.tensor([[0.9, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1]])
    pred = predict_with_inference_rule(_logits(probs), torch.full((8,), 0.5), _router())
    assert pred[0, 3].item() is True
    assert pred[0, [2, 4]].sum().item() == 0


def test_multi_subject_and_escape_leave_threshold_path():
    # Row 0: 2girls predicted → not solo → threshold decides (both colors kept).
    # Row 1: escape tag predicted → gate off even though solo.
    probs = torch.tensor(
        [
            [0.9, 0.9, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.7, 0.8, 0.1, 0.9, 0.1, 0.1],
        ]
    )
    pred = predict_with_inference_rule(_logits(probs), torch.full((8,), 0.5), _router())
    assert pred[0, [2, 3]].tolist() == [True, True]
    assert pred[1, [2, 3]].tolist() == [True, True]
    assert pred[1, 5].item() is True


def test_solo_gating_uses_predictions_not_targets():
    # Nothing but eye colors predicted → solo tag NOT predicted → gate off
    # (mirrors inference, where GT is unavailable).
    probs = torch.tensor([[0.1, 0.1, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1]])
    pred = predict_with_inference_rule(_logits(probs), torch.full((8,), 0.5), _router())
    assert pred[0, [2, 3]].tolist() == [True, True]  # threshold path kept both


def test_plain_softmax_mode_ignores_solo():
    probs = torch.tensor([[0.1, 0.9, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1]])
    pred = predict_with_inference_rule(
        _logits(probs), torch.full((8,), 0.5), _router(mode="softmax")
    )
    assert pred[0, [2, 3, 4]].tolist() == [False, True, False]


# ── Slices ────────────────────────────────────────────────────────────────


class _FakeInfo:
    def __init__(self, path: str):
        self.category_path = path
        self.description = "d"


class _FakeKB:
    def __init__(self, paths: dict):
        self.paths = paths

    def describe(self, name: str):
        p = self.paths.get(name)
        return _FakeInfo(p) if p else None


def test_assign_slices_kb_and_freq():
    vocab = {
        "tags": [
            {"name": "blue eyes", "index": 0, "category": "general", "freq": 5000},
            {"name": "weirdtag", "index": 1, "category": "general", "freq": 300},
            {"name": "some_artist", "index": 2, "category": "artist", "freq": 60},
            {"name": "renamed tag", "index": 3, "category": "general", "freq": 100},
        ]
    }
    kb = _FakeKB({"blue eyes": "얼굴/눈 > 눈 색상", "danbooru tag": "포즈/구도 > 포즈"})
    s = assign_slices(vocab, kb, rename_recovery={"renamed tag": "danbooru tag"})
    assert s.kb_slice == ["eye_color", "general:unmatched", "cat:artist", "pose"]
    assert s.freq_tier == ["head", "mid", "tail", "tail"]


def test_aggregate_slices_masks_unsupported():
    pred = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    tgt = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    _, _, f1, support = per_tag_prf(pred, tgt)
    ap = per_tag_average_precision(pred, tgt)
    rows = aggregate_slices({"a": [0, 1], "b": [2]}, pred, tgt, f1, ap, support)
    by = {r["slice"]: r for r in rows}
    # slice a: tag0 perfect, tag1 unsupported → macro over supported = 1.0
    assert by["a"]["n_supported"] == 1
    assert abs(by["a"]["macro_f1"] - 1.0) < 1e-6
    # slice b: nothing supported → NaN macro, defined micro (no cells fired)
    assert math.isnan(by["b"]["macro_f1"])


def test_freq_sliced_metrics_bins_and_nan_handling():
    import math

    import torch

    from scripts.anima_tagger.train_common import FREQ_BINS, freq_sliced_metrics

    per_tag = torch.tensor([0.5, float("nan"), 0.2, 0.8, 0.9, 0.7])
    train_pos = torch.tensor([10, 30, 100, 500, 5000, 1000])
    out = freq_sliced_metrics(per_tag, train_pos, "f1")
    # Stable key set: every bin reports a mean and a scored-tag count.
    for name, _lo, _hi in FREQ_BINS:
        assert f"f1_{name}" in out and f"n_f1_{name}" in out
    # NaN rows are excluded from both the mean and the count.
    assert out["n_f1_lt50"] == 1 and out["f1_lt50"] == 0.5
    assert out["n_f1_50_199"] == 1 and math.isclose(out["f1_50_199"], 0.2, abs_tol=1e-6)
    # Bins are [lo, hi): 1000 lands in ge1000, not 200_999.
    assert out["n_f1_200_999"] == 1 and math.isclose(
        out["f1_200_999"], 0.8, abs_tol=1e-6
    )
    assert out["n_f1_ge1000"] == 2 and math.isclose(out["f1_ge1000"], 0.8, abs_tol=1e-6)
    # An empty bin is NaN with n=0, never a silent zero.
    empty = freq_sliced_metrics(torch.tensor([0.3]), torch.tensor([5]), "x")
    assert empty["n_x_ge1000"] == 0 and math.isnan(empty["x_ge1000"])
