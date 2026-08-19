#!/usr/bin/env python3
"""Small post-install check for the supported Windows ROCm path.

Two entry points on purpose: ``install.ps1`` / ``docs/guidelines/rocm.md`` run
this as a script (``python tests/rocm_smoke_test.py``) right after installing
the ROCm wheels, and pytest picks up ``test_rocm_smoke`` so ``make test-unit``
exercises the same body on real AMD hardware. Everywhere else (CUDA, CPU) the
test skips — it needs a live ROCm GPU, not a mock.
"""

from __future__ import annotations

import pytest
import torch


def run_smoke() -> None:
    """Raise unless a real ROCm GPU passes the tensor/compile/SDPA/backward path."""
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch cannot access an AMD GPU")
    if torch.version.hip is None:
        raise RuntimeError("installed PyTorch is not a ROCm build")

    # Exercise device allocation, compile, SDPA, and backward rather than
    # accepting an import-only success, which misses runtime/device/Triton
    # failures and the attention path used by Anima on ROCm.
    @torch.compile
    def compiled_loss(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean()

    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    compiled_loss(x).backward()

    q = torch.randn(
        2, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    attention.float().square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(attention).all() or not torch.isfinite(q.grad).all():
        raise RuntimeError("ROCm PyTorch SDPA produced non-finite values")


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs a live ROCm GPU",
)
def test_rocm_smoke() -> None:
    run_smoke()


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}"
    )

    run_smoke()

    print("ROCm tensor/compile/SDPA/backward smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
