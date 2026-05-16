import pytest

from converter import get_content_type, PASSTHROUGH_EXTS, RAW_EXTS, is_passthrough


@pytest.mark.parametrize("ext,expected", [
    (".jpg", "image/jpeg"),
    (".webp", "image/webp"),
    (".mp4", "video/mp4"),
    (".mov", "video/quicktime"),
    (".m4v", "video/x-m4v"),
    (".html", "text/html; charset=utf-8"),
    (".pdf", "application/pdf"),
    (".JPG", "image/jpeg"),         # case-insensitive
    (".bogus", "application/octet-stream"),
])
def test_get_content_type(ext, expected):
    assert get_content_type(ext) == expected


def test_passthrough_constants():
    assert ".mp4" in PASSTHROUGH_EXTS
    assert ".mov" in PASSTHROUGH_EXTS
    assert ".m4v" in PASSTHROUGH_EXTS
    assert ".html" in PASSTHROUGH_EXTS
    assert ".pdf" in PASSTHROUGH_EXTS


def test_raw_constants():
    for ext in (".nef", ".arw", ".dng", ".cr2", ".cr3"):
        assert ext in RAW_EXTS


@pytest.mark.parametrize("ext,expected", [
    (".mp4", True),                # explicit passthrough
    (".HTML", True),               # case-insensitive passthrough
    (".pdf", True),
    (".jpg", False),               # web image — needs WebP conversion
    (".png", False),
    (".psd", False),               # convertible image
    (".tif", False),
    (".nef", False),               # RAW — handled by rawpy backend
    (".xyz", True),                # unknown — fall through to passthrough
    (".", True),                   # unknown — fall through to passthrough
    ("", True),                    # unknown — fall through to passthrough
])
def test_is_passthrough(ext, expected):
    assert is_passthrough(ext) is expected
