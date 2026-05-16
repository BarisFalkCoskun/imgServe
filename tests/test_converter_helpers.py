import pytest

from converter import get_content_type


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
