"""Captioning — ``anima_lora.captioning``.

``AnimaTagger`` (canonical home ``library.captioning``) maps an image to an
Anima-format caption; resolving it is what first drags in torch.
"""

from __future__ import annotations

from anima_lora._lazy import attach

attach(
    globals(),
    {
        "AnimaTagger": "library.captioning",
    },
)
