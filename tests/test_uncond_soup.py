"""Invariant tests for the uncond-soup ΔW-exact soup builder.

The load-bearing claim of bench/uncond_soup/soup.py: the rank-concatenated
soup's delta equals the weighted average of the ingredients' deltas
*exactly* (no SVD truncation), including alpha scaling and the channel-scaling
``inv_scale`` fold — and it refuses non-plain checkpoints loudly.
"""

import pytest
import torch

from bench.uncond_soup.soup import soup_state_dicts


def _delta(sd: dict, module: str) -> torch.Tensor:
    """ΔW as LoRAModule.get_weight computes it (multiplier=1)."""
    down = sd[f"{module}.lora_down.weight"].to(torch.float32)
    up = sd[f"{module}.lora_up.weight"].to(torch.float32)
    inv = sd.get(f"{module}.inv_scale")
    if inv is not None:
        down = down * inv.to(torch.float32).unsqueeze(0)
    alpha = sd.get(f"{module}.alpha")
    scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
    return (up @ down) * scale


def _make_ingredient(seed: int, *, rank=4, din=16, dout=24, with_inv=False) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for module in ("lora_unet_a", "lora_unet_b"):
        sd[f"{module}.lora_down.weight"] = torch.randn(rank, din, generator=g)
        sd[f"{module}.lora_up.weight"] = torch.randn(dout, rank, generator=g)
        sd[f"{module}.alpha"] = torch.tensor(2.0)
        if with_inv:
            sd[f"{module}.inv_scale"] = torch.rand(din, generator=g) + 0.5
    return sd


@pytest.mark.parametrize("with_inv", [False, True])
def test_soup_delta_is_exact_weighted_average(with_inv):
    sds = [_make_ingredient(s, with_inv=with_inv) for s in (0, 1, 2)]
    weights = [0.5, 0.3, 0.2]
    soup = soup_state_dicts(sds, weights, out_dtype=torch.float32)
    for module in ("lora_unet_a", "lora_unet_b"):
        expected = sum(w * _delta(sd, module) for w, sd in zip(weights, sds))
        torch.testing.assert_close(_delta(soup, module), expected, rtol=1e-5, atol=1e-5)
        # Soup itself is normalized: alpha == rank (scale 1), no inv_scale.
        assert (
            float(soup[f"{module}.alpha"])
            == soup[f"{module}.lora_down.weight"].shape[0]
        )
        assert f"{module}.inv_scale" not in soup


def test_uniform_default_and_midpoint():
    sds = [_make_ingredient(s) for s in (0, 1)]
    uniform = soup_state_dicts(sds, out_dtype=torch.float32)
    midpoint = soup_state_dicts(sds, [0.5, 0.5], out_dtype=torch.float32)
    for module in ("lora_unet_a", "lora_unet_b"):
        torch.testing.assert_close(_delta(uniform, module), _delta(midpoint, module))


def test_refuses_non_plain_keys():
    sds = [_make_ingredient(s) for s in (0, 1)]
    sds[1]["lora_unet_a.lora_up_weight"] = torch.randn(3, 24, 4)  # Hydra-style
    with pytest.raises(ValueError, match="unrecognized key suffix"):
        soup_state_dicts(sds)


def test_refuses_register_tokens():
    sds = [_make_ingredient(s) for s in (0, 1)]
    sds[0]["register_tokens"] = torch.randn(4, 8)
    with pytest.raises(ValueError, match="not a plain-LoRA module key"):
        soup_state_dicts(sds)


def test_refuses_mismatched_module_sets():
    sds = [_make_ingredient(0), _make_ingredient(1)]
    sds[1]["lora_unet_c.lora_down.weight"] = torch.randn(4, 16)
    sds[1]["lora_unet_c.lora_up.weight"] = torch.randn(24, 4)
    with pytest.raises(ValueError, match="module sets differ"):
        soup_state_dicts(sds)
