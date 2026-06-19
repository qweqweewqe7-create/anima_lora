"""Bench: 2D-folded Qwen-Image VAE vs the stock 3D causal-Conv3d VAE.

Mirrors kohya-ss sd-scripts' ``--qwen_image_vae_2d``. For single images
(T=1) the causal temporal padding is zero-pad, so only the *last* temporal
tap of every ``QwenImageCausalConv3d`` sees real data. The exact 2D
equivalent is therefore ``weight[:, :, -1, :, :]`` (bias unchanged) run as a
plain ``nn.Conv2d`` on the squeezed (B,C,H,W) frame.

This bench:
  1. proves equivalence (fp32 algebra is exact; bf16 differs only by
     accumulation order) for both encode and decode,
  2. times encode + decode for 3D vs 2D,
  3. records peak CUDA memory for each.

Usage::
    uv run python bench/qwen_vae_2d/run_bench.py --res 1024 --batch 10
    uv run python bench/qwen_vae_2d/run_bench.py --res 1024 --image <path>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.models.qwen_vae import (  # noqa: E402
    AutoencoderKLQwenImage,
    QwenImageCausalConv3d,
    load_vae,
)


class Folded2DConv(nn.Module):
    """2D-equivalent of a T=1 ``QwenImageCausalConv3d`` (zero-pad causal).

    Keeps the 5D (B,C,1,H,W) plumbing intact: squeeze the singleton time
    axis, run a Conv2d, unsqueeze back. Exact in real arithmetic.
    """

    def __init__(self, c3d: QwenImageCausalConv3d) -> None:
        super().__init__()
        # _padding = (pad_w, pad_w, pad_h, pad_h, 2*pad_t, 0); spatial pad is
        # symmetric, so recover (pad_h, pad_w) directly.
        pad_w = c3d._padding[0]
        pad_h = c3d._padding[2]
        self.conv = nn.Conv2d(
            c3d.in_channels,
            c3d.out_channels,
            kernel_size=(c3d.kernel_size[1], c3d.kernel_size[2]),
            stride=(c3d.stride[1], c3d.stride[2]),
            padding=(pad_h, pad_w),
            bias=c3d.bias is not None,
        )
        self.conv = self.conv.to(device=c3d.weight.device, dtype=c3d.weight.dtype)
        with torch.no_grad():
            # last temporal tap is the only one that sees the real frame
            self.conv.weight.copy_(c3d.weight[:, :, -1, :, :])
            if c3d.bias is not None:
                self.conv.bias.copy_(c3d.bias)

    def forward(self, x: torch.Tensor, cache_x=None) -> torch.Tensor:
        x = x[:, :, 0]  # (B,C,1,H,W) -> (B,C,H,W)
        x = self.conv(x)
        return x[:, :, None]  # -> (B,C,1,H',W')


def fold_to_2d(vae: AutoencoderKLQwenImage) -> int:
    """Replace every QwenImageCausalConv3d in-place with a Folded2DConv."""
    n = 0
    for parent in vae.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, QwenImageCausalConv3d):
                setattr(parent, name, Folded2DConv(child))
                n += 1
    return n


def _load_image(path: str, res: int, device, dtype) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
    return t.unsqueeze(0).to(device=device, dtype=dtype)


@torch.no_grad()
def _timed(fn, iters: int, device) -> tuple[float, int]:
    """Return (median ms per call, peak bytes)."""
    # warmup
    for _ in range(2):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated(device)
    times.sort()
    return times[len(times) // 2], peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--image", type=str, default=None, help="optional real image")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    import anima_lora

    vae_path = str(anima_lora.ROOT / anima_lora.default_checkpoints().vae)

    # ---- build pixels (real image broadcast to batch, or random) ----
    if args.image:
        px1 = _load_image(args.image, args.res, device, dtype)
        pixels = px1.repeat(args.batch, 1, 1, 1).contiguous()
        src = args.image
    else:
        gen = torch.Generator(device=device).manual_seed(0)
        pixels = (
            torch.rand(
                args.batch, 3, args.res, args.res,
                generator=gen, device=device, dtype=torch.float32,
            )
            * 2.0
            - 1.0
        ).to(dtype)
        src = "random[-1,1]"

    # ============ EQUIVALENCE (fp32, exact algebra) ============
    vae32 = load_vae(vae_path, device=device, dtype=torch.float32, eval=True)
    px32 = pixels.float()
    lat3d = vae32.encode_pixels_to_latents(px32)
    img3d = vae32.decode_to_pixels(lat3d)

    n_folded = fold_to_2d(vae32)
    lat2d = vae32.encode_pixels_to_latents(px32)
    img2d = vae32.decode_to_pixels(lat2d)

    enc_max = (lat3d - lat2d).abs().max().item()
    enc_rel = enc_max / (lat3d.abs().max().item() + 1e-12)
    dec_max = (img3d - img2d).abs().max().item()
    del vae32, px32, lat3d, img3d, lat2d, img2d
    torch.cuda.empty_cache()

    # ============ TIMING + VRAM (production dtype) ============
    def run_block(fold: bool):
        vae = load_vae(vae_path, device=device, dtype=dtype, eval=True)
        if fold:
            fold_to_2d(vae)
        lat = vae.encode_pixels_to_latents(pixels)
        enc_ms, enc_peak = _timed(
            lambda: vae.encode_pixels_to_latents(pixels), args.iters, device
        )
        dec_ms, dec_peak = _timed(
            lambda: vae.decode_to_pixels(lat), args.iters, device
        )
        del vae, lat
        torch.cuda.empty_cache()
        return {
            "encode_ms": enc_ms,
            "decode_ms": dec_ms,
            "encode_peak_mb": enc_peak / 1e6,
            "decode_peak_mb": dec_peak / 1e6,
        }

    m3d = run_block(fold=False)
    m2d = run_block(fold=True)

    # bf16/production-dtype numerical diff (accumulation-order only)
    vaeP = load_vae(vae_path, device=device, dtype=dtype, eval=True)
    latP_3d = vaeP.encode_pixels_to_latents(pixels)
    fold_to_2d(vaeP)
    latP_2d = vaeP.encode_pixels_to_latents(pixels)
    enc_max_prod = (latP_3d.float() - latP_2d.float()).abs().max().item()
    del vaeP, latP_3d, latP_2d
    torch.cuda.empty_cache()

    metrics = {
        "config": {
            "res": args.res,
            "batch": args.batch,
            "iters": args.iters,
            "dtype": args.dtype,
            "source": src,
            "n_conv3d_folded": n_folded,
        },
        "equivalence_fp32": {
            "encode_max_abs_diff": enc_max,
            "encode_rel_diff": enc_rel,
            "decode_max_abs_diff": dec_max,
        },
        "equivalence_prod_dtype": {
            "encode_max_abs_diff": enc_max_prod,
        },
        "vae_3d": m3d,
        "vae_2d": m2d,
        "speedup": {
            "encode_x": m3d["encode_ms"] / m2d["encode_ms"],
            "decode_x": m3d["decode_ms"] / m2d["decode_ms"],
            "encode_peak_ratio_2d_over_3d": m2d["encode_peak_mb"]
            / m3d["encode_peak_mb"],
            "decode_peak_ratio_2d_over_3d": m2d["decode_peak_mb"]
            / m3d["decode_peak_mb"],
        },
    }

    run_dir = make_run_dir("qwen_vae_2d", label=args.label)
    write_result(
        run_dir, script=__file__, args=args, metrics=metrics,
        label=args.label, device=device,
    )

    # ---- console summary ----
    print(f"\n=== Qwen VAE 2D-fold bench ({args.res}px x{args.batch}, {args.dtype}) ===")
    print(f"folded {n_folded} Conv3d -> Conv2d   source={src}")
    print("\n[equivalence]")
    print(f"  fp32  encode max|Δ| = {enc_max:.3e}  (rel {enc_rel:.2e})")
    print(f"  fp32  decode max|Δ| = {dec_max:.3e}")
    print(f"  {args.dtype}  encode max|Δ| = {enc_max_prod:.3e}  (accum-order only)")
    print("\n[encode]")
    print(f"  3D: {m3d['encode_ms']:7.1f} ms  peak {m3d['encode_peak_mb']:7.0f} MB")
    print(f"  2D: {m2d['encode_ms']:7.1f} ms  peak {m2d['encode_peak_mb']:7.0f} MB")
    print(f"  -> {metrics['speedup']['encode_x']:.2f}x faster, "
          f"{metrics['speedup']['encode_peak_ratio_2d_over_3d']:.2f}x peak mem")
    print("\n[decode]")
    print(f"  3D: {m3d['decode_ms']:7.1f} ms  peak {m3d['decode_peak_mb']:7.0f} MB")
    print(f"  2D: {m2d['decode_ms']:7.1f} ms  peak {m2d['decode_peak_mb']:7.0f} MB")
    print(f"  -> {metrics['speedup']['decode_x']:.2f}x faster, "
          f"{metrics['speedup']['decode_peak_ratio_2d_over_3d']:.2f}x peak mem")
    print(f"\nresult.json -> {run_dir}")


if __name__ == "__main__":
    main()
