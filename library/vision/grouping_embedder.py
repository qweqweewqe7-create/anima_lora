"""PE-Spatial embedder for dataset grouping / near-twin mining (trainer side).

The curation-side feature cache (``library.vision.pe_features.embed_members``)
takes any :class:`~library.vision.pe_features.Embedder`; this module is the
trainer's PE-Spatial-B16-512 implementation of that protocol. It is the only
place grouping code touches the PE loader — the curation package must not
import it (see ``docs/proposal/curation_repo_split.md`` §seam).

``pe_spatial_embedder`` is a dotted-path factory (``module:callable``) so the
``build_groups.py`` CLI can be pointed at it with ``--embedder`` without
importing the trainer at module import time.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from library.vision.encoder import VisionEncoderBundle, load_pe_encoder
from library.vision.pe_features import GRID_CACHE, GRID_NATIVE


class PESpatialEmbedder:
    """Wraps a PE-Spatial bundle as a grouping ``Embedder``: ``[B,3,512,512]``
    device batch → ``(cls [B,768] L2-normed f32, grid16 [B,16,16,768] f16)``."""

    name = "pe_spatial"

    def __init__(self, bundle: VisionEncoderBundle):
        self.bundle = bundle
        self.device = bundle.device
        self.dtype = bundle.dtype

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        out = self.bundle.encoder(batch)
        lhs = out.last_hidden_state.float()  # [B, 1+1024, 768]
        cls = F.normalize(lhs[:, 0], dim=-1)  # global descriptor
        grid = lhs[:, 1:].reshape(lhs.shape[0], GRID_NATIVE, GRID_NATIVE, -1)
        g = grid.permute(0, 3, 1, 2)  # [B, 768, 32, 32]
        g16 = F.adaptive_avg_pool2d(g, GRID_CACHE).permute(0, 2, 3, 1)
        return cls.cpu().numpy(), g16.cpu().numpy().astype(np.float16)


def pe_spatial_embedder(
    device: torch.device | str | None = None, name: str = "pe_spatial"
) -> PESpatialEmbedder:
    """Factory: load the PE encoder ``name`` on ``device`` (auto when None)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return PESpatialEmbedder(load_pe_encoder(dev, name=name))
