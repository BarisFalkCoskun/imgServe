import pytest
from PIL import Image

import converter
from converter import get_content_type, PASSTHROUGH_EXTS, RAW_EXTS, is_passthrough, choose_output_format


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


def test_choose_output_format_defaults_to_webp_for_opaque():
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    assert choose_output_format(img, None) == "webp"


def test_choose_output_format_respects_request():
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    assert choose_output_format(img, "PNG") == "png"
    assert choose_output_format(img, "jpg") == "jpg"


from converter import convert_image


def test_convert_image_handles_jpg_to_webp(tmp_path):
    src = tmp_path / "src.jpg"
    Image.new("RGB", (8, 8), (200, 100, 50)).save(src, format="JPEG")
    dst = tmp_path / "out.webp"
    assert convert_image(str(src), str(dst), "webp") is True
    assert Image.open(dst).format == "WEBP"


def test_convert_image_prefers_imagemagick_for_psd(tmp_path, monkeypatch):
    calls = []
    src = tmp_path / "source.psd"
    dst = tmp_path / "out.webp"
    src.write_bytes(b"psd")

    def fake_magick(_src, _dst, _fmt):
        calls.append("ImageMagick")
        dst.write_bytes(b"webp")
        return True

    def fake_pillow(_src, _dst, _fmt):
        calls.append("Pillow")
        return True

    monkeypatch.setattr(converter, "convert_with_magick", fake_magick)
    monkeypatch.setattr(converter, "convert_with_pillow", fake_pillow)

    assert converter.convert_image(str(src), str(dst), "webp") is True
    assert calls == ["ImageMagick"]


def test_pillow_converts_png_without_icc(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (8, 8), (10, 200, 10)).save(src, format="PNG")
    dst = tmp_path / "out.webp"
    from converter import convert_with_pillow
    assert convert_with_pillow(str(src), str(dst), "webp") is True
    img = Image.open(dst)
    assert img.format == "WEBP" and img.size == (8, 8)
