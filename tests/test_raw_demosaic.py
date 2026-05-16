"""RAW demosaic sanity check.

Skipped when rawpy is not installed or when no DNG fixture is present.
Asserts the converted WebP is not the channel-separation artifact we saw
with the generic backends.
"""
import importlib.util
from pathlib import Path

import pytest

from PIL import Image

from converter import convert_with_rawpy

RAWPY_AVAILABLE = importlib.util.find_spec("rawpy") is not None
FIXTURE = Path(__file__).parent / "fixtures" / "sample.dng"
HAS_FIXTURE = FIXTURE.is_file()

pytestmark = [
    pytest.mark.skipif(not RAWPY_AVAILABLE, reason="rawpy not installed"),
    pytest.mark.skipif(not HAS_FIXTURE, reason="tests/fixtures/sample.dng missing — add a small DNG to enable"),
]


def test_rawpy_returns_false_for_non_raw(tmp_path):
    src = tmp_path / "not_a_raw.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(src, format="JPEG")
    out = tmp_path / "out.webp"
    assert convert_with_rawpy(str(src), str(out), "webp") is False
    assert not out.exists()


def test_rawpy_converts_dng_to_webp(tmp_path):
    out = tmp_path / "out.webp"
    assert convert_with_rawpy(str(FIXTURE), str(out), "webp") is True
    assert out.is_file() and out.stat().st_size > 0
    img = Image.open(out).convert("RGB")
    r, g, b = img.split()

    def stddev(channel):
        hist = channel.histogram()
        n = sum(hist)
        mean = sum(i * c for i, c in enumerate(hist)) / n
        var = sum(c * (i - mean) ** 2 for i, c in enumerate(hist)) / n
        return var ** 0.5

    for channel in (r, g, b):
        assert stddev(channel) > 5.0, "channel looks flat; demosaic may have failed"
