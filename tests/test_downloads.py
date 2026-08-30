from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import pytest

from scripts.tasks import downloads


def test_danbooru_tags_download_url_points_to_source_repo():
    assert downloads.DANBOORU_TAGS_URLS == (
        "https://raw.githubusercontent.com/Localsmile/danbooru_KR_wiki_tag_search/main/danbooru_tags_classified.csv",
    )


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return self._payload


def test_download_danbooru_tags_writes_models_file(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(
        downloads, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",)
    )
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda _req, timeout=60: _FakeResponse(
            b"name,category,post_count,description\n1girl,0,1,test\n"
        ),
    )

    downloads.cmd_download_danbooru_tags([])

    assert dest.read_text(encoding="utf-8").startswith("name,category")


def test_download_danbooru_tags_skips_existing_without_force(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    dest.parent.mkdir(parents=True)
    dest.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    called = False

    def _fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        return _FakeResponse(BytesIO().read())

    monkeypatch.setattr(downloads.urllib.request, "urlopen", _fail_if_called)

    downloads.cmd_download_danbooru_tags([])

    assert not called
    assert dest.read_text(encoding="utf-8") == "existing"


def test_download_danbooru_tags_failure_names_source_repo(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(
        downloads, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",)
    )
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("nope")),
    )

    with pytest.raises(SystemExit) as exc:
        downloads.cmd_download_danbooru_tags([])

    assert "Localsmile/danbooru_KR_wiki_tag_search" in str(exc.value)


# --------------------------------------------------------------------------- #
# Anima Tagger: our checkpoint dir + the gated external backbone
# --------------------------------------------------------------------------- #


def _install_tagger_ckpt(tmp_path, monkeypatch, repo: str | None = None) -> None:
    """Point the tagger target at a tmp checkpoint dir holding every required file."""
    ckpt = tmp_path / "anima-tagger-dbv4"
    ckpt.mkdir(parents=True)
    for name in downloads.TAGGER_CKPT_REQUIRED:
        (ckpt / name).write_text("{}", encoding="utf-8")
    if repo is not None:
        (ckpt / "config.json").write_text(
            json.dumps({"backend": "dbv4", "dbv4": {"repo": repo}}), encoding="utf-8"
        )
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "TAGGER_CKPT_REL", "anima-tagger-dbv4")


def test_download_tagger_model_skips_both_halves_when_present(tmp_path, monkeypatch):
    """Idempotency contract (GH #21): a re-run verifies, it doesn't re-fetch 500MB."""
    _install_tagger_ckpt(tmp_path, monkeypatch)
    monkeypatch.setattr("anime_tools._hf.hf_file_cached", lambda *_a, **_k: True)
    calls = []
    monkeypatch.setattr(downloads, "run", lambda cmd, **kw: calls.append(cmd))

    downloads.cmd_download_tagger_model([])

    assert calls == []


def test_download_tagger_model_fetches_backbone_when_uncached(tmp_path, monkeypatch):
    """Checkpoint on disk but backbone missing from the HF cache -> fetch only it."""
    _install_tagger_ckpt(tmp_path, monkeypatch)
    monkeypatch.setattr("anime_tools._hf.hf_file_cached", lambda *_a, **_k: False)
    calls = []
    monkeypatch.setattr(downloads, "run", lambda cmd, **kw: calls.append(cmd))

    downloads.cmd_download_tagger_model([])

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:3] == ["hf", "download", downloads.TAGGER_BACKBONE_REPO]
    # No --local-dir: the loader reads the backbone straight out of the hub cache.
    assert "--local-dir" not in cmd
    assert set(downloads.TAGGER_BACKBONE_FILES) <= set(cmd)


def test_tagger_backbone_repo_follows_the_installed_checkpoint(tmp_path, monkeypatch):
    """config.json's dbv4.repo wins — a checkpoint built against another
    animetimm variant must download *that* backbone, not the default."""
    _install_tagger_ckpt(tmp_path, monkeypatch, repo="animetimm/caformer_s36.dbv4-full")
    assert downloads._tagger_backbone_repo() == "animetimm/caformer_s36.dbv4-full"


def test_tagger_backbone_repo_falls_back_without_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "TAGGER_CKPT_REL", "missing")
    assert downloads._tagger_backbone_repo() == downloads.TAGGER_BACKBONE_REPO


def test_flatten_subfolder_overwrites_existing(tmp_path):
    """``--force`` re-download lands the file again; the flatten must overwrite
    rather than raise (``shutil.move`` onto an existing path fails on Windows)."""
    dst = tmp_path / "ckpt"
    (dst / "dbv4").mkdir(parents=True)
    (dst / "vocab.json").write_text("stale", encoding="utf-8")
    (dst / "dbv4" / "vocab.json").write_text("fresh", encoding="utf-8")

    downloads._flatten_subfolder(dst, "dbv4")

    assert (dst / "vocab.json").read_text(encoding="utf-8") == "fresh"
    assert not (dst / "dbv4").exists()


def test_models_dialog_rows_map_to_registered_tasks():
    """The GUI runs ``download-<key>`` for every row — keep the keys in sync with
    the task registry so a button can't silently point at a missing target."""
    pytest.importorskip("PySide6")
    import tasks
    from gui.system_dialog import _MODEL_GROUPS

    for key, _label, _paths, _extra in _MODEL_GROUPS:
        assert f"download-{key}" in tasks.COMMANDS, key
    assert "tagger-model" in {g[0] for g in _MODEL_GROUPS}
