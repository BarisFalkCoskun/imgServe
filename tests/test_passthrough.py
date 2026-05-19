from pathlib import Path

import pytest


@pytest.mark.parametrize("fixture,ext,expected_ct", [
    ("doc.pdf", ".pdf", "application/pdf"),
    ("page.html", ".html", "text/html; charset=utf-8"),
])
def test_passthrough_documents(client, put_source, fixture, ext, expected_ct, env_dirs):
    src = put_source(fixture, folder="docs")
    r = client.get(f"/imgs/docs/{src.name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == expected_ct
    assert r.content == src.read_bytes()
    # nothing written to thumbnails
    assert list((env_dirs["thumbnails"]).rglob("*.webp")) == []


@pytest.mark.parametrize("fixture,expected_ct", [
    ("clip.mp4", "video/mp4"),
    ("clip.mov", "video/quicktime"),
    ("clip.m4v", "video/x-m4v"),
])
def test_passthrough_video(client, put_source, fixture, expected_ct, env_dirs):
    src_path = Path(__file__).parent / "fixtures" / fixture
    if not src_path.is_file():
        pytest.skip(f"{fixture} not generated (ffmpeg missing during fixture gen)")
    src = put_source(fixture, folder="vid")
    r = client.get(f"/imgs/vid/{src.name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == expected_ct
    assert r.content == src.read_bytes()
    assert list((env_dirs["thumbnails"]).rglob("*.webp")) == []
