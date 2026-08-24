"""Model loading + device helpers — ``anima_lora.models``.

| export | canonical home |
|--------|----------------|
| ``load_dit_model`` | ``library.inference.models`` |
| ``load_anima_model`` | ``library.anima.weights`` |
| ``load_vae`` | ``library.models.qwen_vae`` |
| ``default_checkpoints`` / ``DefaultCheckpoints`` | ``library.env`` |
| ``str_to_dtype`` | ``library.runtime.device`` |
"""

from __future__ import annotations

from anima_lora._lazy import attach

attach(
    globals(),
    {
        "load_dit_model": "library.inference.models",
        "load_anima_model": "library.anima.weights",
        "load_vae": "library.models.qwen_vae",
        "default_checkpoints": "library.env",
        "DefaultCheckpoints": "library.env",
        "str_to_dtype": "library.runtime.device",
    },
)
