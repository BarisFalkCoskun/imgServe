import os
import stat
from pathlib import Path

import server


def test_promote_to_imgsbackup_creates_webp(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    src = tmp_path / "src.webp"
    src.write_bytes(b"webp-bytes")

    ok = server.promote_to_imgsbackup(str(src), str(backup), "demo", "image.psd")
    assert ok is True
    out = backup / "demo" / "image.webp"
    assert out.is_file() and out.read_bytes() == b"webp-bytes"
    leftovers = [p for p in (backup / "demo").iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_promote_to_imgsbackup_returns_false_on_readonly(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "demo").mkdir()
    # remove write bit on the target dir
    os.chmod(backup / "demo", stat.S_IRUSR | stat.S_IXUSR)
    src = tmp_path / "src.webp"
    src.write_bytes(b"webp")

    try:
        ok = server.promote_to_imgsbackup(str(src), str(backup), "demo", "img.psd")
        assert ok is False
        assert not (backup / "demo" / "img.webp").exists()
    finally:
        os.chmod(backup / "demo", stat.S_IRWXU)


def test_check_imgsbackup_write_happy(tmp_path):
    status = server._check_imgsbackup_write(str(tmp_path), min_free_bytes=0)
    assert status["ok"] is True
    assert status["path"] == str(tmp_path)


def test_check_imgsbackup_write_missing_dir(tmp_path):
    status = server._check_imgsbackup_write(str(tmp_path / "nope"), min_free_bytes=0)
    assert status["ok"] is False
    assert "error" in status
