import os
import stat
from pathlib import Path

import server


def test_configured_thumbnails_dir_prefers_new_env(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    monkeypatch.setenv(server.ENV_LEGACY_IMGSBACKUP_DIR, str(legacy))
    monkeypatch.setenv(server.ENV_THUMBNAILS_DIR, str(current))
    assert server._configured_thumbnails_dir() == str(current)


def test_configured_thumbnails_dir_accepts_legacy_env(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    monkeypatch.delenv(server.ENV_THUMBNAILS_DIR, raising=False)
    monkeypatch.setenv(server.ENV_LEGACY_IMGSBACKUP_DIR, str(legacy))
    assert server._configured_thumbnails_dir() == str(legacy)


def test_promote_to_thumbnails_creates_webp(tmp_path):
    thumbnails = tmp_path / "thumbnails"
    thumbnails.mkdir()
    src = tmp_path / "src.webp"
    src.write_bytes(b"webp-bytes")

    ok = server.promote_to_thumbnails(str(src), str(thumbnails), "demo", "image.psd")
    assert ok is True
    out = thumbnails / "demo" / "image.webp"
    assert out.is_file() and out.read_bytes() == b"webp-bytes"
    leftovers = [p for p in (thumbnails / "demo").iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_promote_to_thumbnails_returns_false_on_readonly(tmp_path):
    thumbnails = tmp_path / "thumbnails"
    thumbnails.mkdir()
    (thumbnails / "demo").mkdir()
    # remove write bit on the target dir
    os.chmod(thumbnails / "demo", stat.S_IRUSR | stat.S_IXUSR)
    src = tmp_path / "src.webp"
    src.write_bytes(b"webp")

    try:
        ok = server.promote_to_thumbnails(str(src), str(thumbnails), "demo", "img.psd")
        assert ok is False
        assert not (thumbnails / "demo" / "img.webp").exists()
    finally:
        os.chmod(thumbnails / "demo", stat.S_IRWXU)


def test_check_thumbnails_write_happy(tmp_path):
    status = server._check_thumbnails_write(str(tmp_path), min_free_bytes=0)
    assert status["ok"] is True
    assert status["path"] == str(tmp_path)


def test_check_thumbnails_write_missing_dir(tmp_path):
    status = server._check_thumbnails_write(str(tmp_path / "nope"), min_free_bytes=0)
    assert status["ok"] is False
    assert "error" in status
