"""Curation → trainer dependency-direction guard.

The curation half of the tree (tagger, masking, grouping, caption polishing) is
being split into ``sorryhyun/anime_tools`` (``docs/proposal/curation_repo_split.md``).
The one rule of that split is::

    anima_lora (trainer) ──depends on──▶ anime_tools        never the reverse

This test pins the boundary *before* the move so it cannot regress: every file
in the move manifest must not import the DiT/VAE/adapter side. If a curation
feature needs something from the trainer, move that leaf into the curation set
(or duplicate a ≤50-line pure helper) — do not add an exemption here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Move manifest (proposal §"Move manifest"). Directories are recursive; globs
# are relative to the repo root.
CURATION_PATHS: tuple[str, ...] = (
    # Phase 1 (2026-08-30) moved captions / tagger / stages into the
    # ``anime_tools`` package; what remains here is the Phase 2 set
    # (masking + grouping + the task wrappers). ``anime_tools`` imports are
    # allowed — that is the dependency direction.
    "library/vision/pe_features.py",
    "library/vision/pe_matching.py",
    "library/datasets/grouping.py",
    "scripts/curate/**/*.py",
    "scripts/preprocess/generate_masks.py",
    "scripts/preprocess/generate_masks_mit.py",
    "scripts/preprocess/merge_masks.py",
    "scripts/preprocess/probe_nms_pairs.py",
    "scripts/preprocess/probe_sam_masks.py",
    "scripts/tasks/tagger.py",
    "scripts/tasks/masking.py",
    "scripts/tasks/curate.py",
)

# Anything under ``library.`` that is not itself in the manifest is trainer-only
# (``library.env`` / ``library.log`` / ``library.io`` / ``library.datasets`` /
# ``library.runtime`` included — the curation side uses ``anime_tools.{_env,_walk,_hf}``),
# plus ``networks`` and ``train``. The Phase 1 shims (``library.captioning.*``,
# ``library.preprocess.caption_variants`` …) are trainer-side forwarding modules
# and are forbidden here too: curation code imports ``anime_tools`` directly.
FORBIDDEN_ROOTS: tuple[str, ...] = ("library", "networks", "train")


_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?P<from>[\w.]+)\s+import|import\s+(?P<mod>[\w.]+))",
    re.MULTILINE,
)


def _curation_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in CURATION_PATHS:
        files.update(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    return sorted(files)


def _in_manifest(module: str, manifest: set[str]) -> bool:
    rel = module.replace(".", "/")
    return rel + ".py" in manifest or any(m.startswith(rel + "/") for m in manifest)


def _forbidden(module: str, manifest: set[str]) -> bool:
    if not any(module == r or module.startswith(r + ".") for r in FORBIDDEN_ROOTS):
        return False
    return not _in_manifest(module, manifest)


def _violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    manifest = {str(p.relative_to(REPO_ROOT)) for p in _curation_files()}
    out: list[str] = []
    for m in _IMPORT_RE.finditer(text):
        mod = m.group("from") or m.group("mod")
        if _forbidden(mod, manifest):
            line = text.count("\n", 0, m.start()) + 1
            out.append(f"{path.relative_to(REPO_ROOT)}:{line}: {m.group(0).strip()}")
    return out


def test_manifest_globs_resolve():
    """A renamed/moved file must update the manifest, not silently drop out."""
    for pattern in CURATION_PATHS:
        assert list(REPO_ROOT.glob(pattern)), (
            f"manifest entry matches nothing: {pattern}"
        )


@pytest.mark.parametrize(
    "path", _curation_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_curation_module_does_not_import_trainer(path: Path):
    violations = _violations(path)
    assert not violations, (
        "curation-side module imports the trainer (dependency direction is "
        "trainer → anime_tools, never the reverse):\n  " + "\n  ".join(violations)
    )
