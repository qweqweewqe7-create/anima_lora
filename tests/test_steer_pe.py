"""SteerPE invariants: zero-init identity, gate gradient, frozen tower, targets."""

from __future__ import annotations

import torch

from library.models.pe import PEVisionTransformer
from networks.methods.steer_pe import SteerPE, patch_targets, pr_auc


def _tiny_pe(**kw) -> PEVisionTransformer:
    torch.manual_seed(0)
    pe = PEVisionTransformer(
        patch_size=8,
        width=32,
        layers=4,
        heads=4,
        mlp_ratio=2.0,
        output_dim=None,
        use_cls_token=True,
        pool_type="none",
        use_ln_post=False,
        image_size=32,
        **kw,
    )
    torch.manual_seed(0)
    for p in pe.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.02)
    return pe.eval()


def test_zero_gate_is_bit_exact_base_tower():
    pe = _tiny_pe()
    model = SteerPE(pe, text_dim=16, heads=4).eval()
    x = torch.randn(2, 3, 32, 32)
    text = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.long)
    base = pe.forward_features(x, norm=True, strip_cls_token=True)
    with torch.no_grad():
        unsteered = model(x)
        steered = model(x, text, mask)
    assert torch.equal(unsteered, base)
    assert torch.equal(steered, base), "tanh(0) gate must leave the tower untouched"
    assert model.cross_attn_layers == [1, 3]


def test_gate_receives_gradient_and_tower_stays_frozen():
    pe = _tiny_pe()
    model = SteerPE(pe, text_dim=16, heads=4)
    x = torch.randn(2, 3, 32, 32)
    text = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.long)
    tokens = model(x, text, mask)
    logits = model.heat_logits(tokens, model.grid(x))
    assert logits.shape == (2, 4, 4)
    # seg head is zero-init → push through a non-zero head so the gate sees signal
    torch.nn.init.normal_(model.seg_head.weight)
    tokens = model(x, text, mask)
    model.heat_logits(tokens, model.grid(x)).sum().backward()
    for ca in model.cross_attn.values():
        assert ca.gate.grad is not None and float(ca.gate.grad.abs()) > 0
    assert all(not p.requires_grad for p in pe.parameters())
    names = {n for n, _ in model.named_parameters() if not n.startswith("pe.")}
    assert {k for k in model.adapter_state_dict()} == names


def test_gate_scale_interpolates_and_roundtrips():
    pe = _tiny_pe()
    model = SteerPE(pe, text_dim=16, heads=4).eval()
    with torch.no_grad():
        for ca in model.cross_attn.values():
            ca.gate.fill_(1.0)
    x = torch.randn(1, 3, 32, 32)
    text = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.long)
    with torch.no_grad():
        base = model(x)
        full = model(x, text, mask)
        model.set_gate_scale(0.0)
        zero = model(x, text, mask)
        model.set_gate_scale(1.0)
    assert torch.allclose(zero, base, atol=1e-6)
    assert not torch.allclose(full, base)
    clone = SteerPE(_tiny_pe(), text_dim=16, heads=4).eval()
    clone.load_adapter_state_dict(model.adapter_state_dict())
    with torch.no_grad():
        assert torch.allclose(clone(x, text, mask), full)


def test_patch_targets_and_pr_auc():
    m = torch.zeros(1, 1, 16, 16)
    m[..., :8, :8] = 1
    t = patch_targets(m, (2, 2))
    assert torch.allclose(t, torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]))
    scores = torch.tensor([[0.9, 0.1], [0.2, 0.0]])
    assert pr_auc(scores, t[0] > 0.5) == 1.0
