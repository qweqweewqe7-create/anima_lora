"""SteerPE — text-steerable PE vision tower (SteerViT recipe on PE-Core / PE-Spatial).

Borrowed from *Steerable Visual Representations* (Ruthardt & Gaur et al.,
arXiv 2604.02327): interleave zero-initialised tanh-gated cross-attention
layers into a **frozen** ViT so its patch tokens can attend to a text prompt
(vision → language, the inverse of Flamingo). ``tanh(0) = 0`` so a fresh
adapter is bit-exact the base tower; ``texts=None`` keeps that path forever.

Two deliberate deviations from the paper, both driven by what the shipped
SteerViT checkpoint could not do on this dataset (memory: steervit-probe):

* **Keys/values come from the Anima Qwen3 text encoder**, not RoBERTa — it
  already speaks booru tags (``black hair``, ``1boy``, character names).
* **The proxy head is per-patch sigmoid/BCE**, not the paper's softmax over
  patches. Softmax mass is winner-take-all, which is exactly why the DINOv2
  checkpoint lit one view of a two-view sheet; binary per-instance SAM3
  masks are the supervision here, so BCE is the honest likelihood and lets
  ``the girl`` light every girl.

The tower is driven by re-implementing ``PEVisionTransformer.forward_features``
as a per-block loop (``SteerPE.forward``) — the PE module itself is untouched,
so the frozen features stay cacheable by every existing consumer.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.models.pe import PEVisionTransformer


class TextConnector(nn.Module):
    """ℓ2-normalised text tokens → vision width, two-layer MLP (paper Tab. 3 row 5)."""

    def __init__(self, text_dim: int, vision_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, vision_dim),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.mlp(F.normalize(text.float(), dim=-1).to(self.mlp[0].weight.dtype))


class GatedCrossAttention(nn.Module):
    """``x + tanh(α) · CA(LN(x), H_t)`` with ``α`` zero-initialised."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor | None
    ) -> torch.Tensor:
        B, Tq, _ = x.shape
        q = (
            self.to_q(self.norm(x))
            .view(B, Tq, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = (
            self.to_kv(text)
            .view(B, -1, 2, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        mask = None if text_mask is None else text_mask[:, None, None, :].bool()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, Tq, -1)
        return x + torch.tanh(self.gate) * self.to_out(out)


class SteerPE(nn.Module):
    """Frozen PE tower + gated cross-attention every other block + patch seg head.

    ``forward(pixels, text, text_mask)`` returns post-``ln_post`` tokens with the
    CLS stripped (``(B, N, width)``); ``text=None`` is the unsteered tower.
    ``heat_logits`` maps tokens to a ``(B, gh, gw)`` per-patch logit grid.
    """

    def __init__(
        self,
        pe: PEVisionTransformer,
        *,
        text_dim: int,
        cross_attn_layers: Sequence[int] | None = None,
        heads: int = 8,
    ):
        super().__init__()
        self.pe = pe
        self.pe.requires_grad_(False)
        width = pe.width
        n = pe.transformer.layers
        layers = (
            list(cross_attn_layers)
            if cross_attn_layers is not None
            else list(range(1, n, 2))
        )
        self.cross_attn_layers = sorted(int(i) for i in layers)
        self.connector = TextConnector(text_dim, width)
        self.cross_attn = nn.ModuleDict(
            {str(i): GatedCrossAttention(width, heads) for i in self.cross_attn_layers}
        )
        self.seg_head = nn.Linear(width, 1)
        nn.init.zeros_(self.seg_head.weight)
        nn.init.zeros_(self.seg_head.bias)
        self._gate_scale = 1.0

    # -- trainable surface -------------------------------------------------
    def adapter_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("pe."):
                yield p

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("pe.")}

    def load_adapter_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith("pe.")]
        if bad or unexpected:
            raise RuntimeError(
                f"SteerPE adapter mismatch: missing={bad} unexpected={list(unexpected)}"
            )

    def set_gate_scale(self, scale: float) -> None:
        """Inference knob ω: scale every gate (paper Fig. 7; 0 = base tower)."""
        self._gate_scale = float(scale)

    # -- forward -----------------------------------------------------------
    def grid(self, pixels: torch.Tensor) -> tuple[int, int]:
        return pixels.shape[-2] // self.pe.patch_size, pixels.shape[
            -1
        ] // self.pe.patch_size

    def forward(
        self,
        pixels: torch.Tensor,
        text: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pe = self.pe
        B, _, h, w = pixels.shape
        gh, gw = h // pe.patch_size, w // pe.patch_size
        x = pe.conv1(pixels)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, pe.width)
        if pe.use_cls_token:
            x = torch.cat(
                [pe.class_embedding.view(1, 1, -1).expand(B, -1, -1), x], dim=1
            )
        if pe.use_abs_posemb:
            x = x + pe._sample_abs_posemb(gh, gw)
        if pe.use_rope2d:
            pe.rope.update_grid(x.device, gh, gw)
        x = pe.ln_pre(x)

        H = None
        if text is not None:
            H = self.connector(text).to(x.dtype)
        for i, block in enumerate(pe.transformer.resblocks):
            key = str(i)
            if H is not None and key in self.cross_attn:
                ca = self.cross_attn[key]
                if self._gate_scale == 1.0:
                    x = ca(x, H, text_mask)
                else:
                    x = x + self._gate_scale * (ca(x, H, text_mask) - x)
            x = block(x)
        x = pe.ln_post(x)
        if pe.use_cls_token:
            x = x[:, 1:, :]
        return x

    def heat_logits(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        gh, gw = grid
        return self.seg_head(tokens).squeeze(-1).view(tokens.shape[0], gh, gw)


@torch.no_grad()
def encode_prompts(
    text_encoder,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_length: int = 32,
    device="cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3 last hidden state for short steering prompts (padded, mask returned).

    Mirrors ``AnimaTextEncodingStrategy.encode_tokens`` (last_hidden_state,
    padding zeroed) but at a short ``max_length`` — steering phrases are a few
    tokens and the cross-attention masks padding itself.
    """
    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    out = text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    out = out.masked_fill(~mask.bool()[..., None], 0)
    return out, mask


def patch_targets(mask: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Binary pixel mask ``(B, 1, H, W)`` → per-patch foreground fraction ``(B, gh, gw)``."""
    return F.adaptive_avg_pool2d(mask.float(), grid).squeeze(1)


def soft_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), target.float())


def pr_auc(scores: torch.Tensor, positives: torch.Tensor) -> float:
    """Area under the precision-recall curve over flattened patches."""
    s = scores.flatten().float()
    y = positives.flatten().float()
    if y.sum() == 0:
        return math.nan
    order = torch.argsort(s, descending=True)
    tp = torch.cumsum(y[order], 0)
    prec = tp / torch.arange(1, len(s) + 1, device=s.device)
    rec = tp / y.sum()
    rec = torch.cat([torch.zeros(1, device=s.device), rec])
    prec = torch.cat([torch.ones(1, device=s.device), prec])
    return float(torch.trapezoid(prec, rec))
