"""Generate small binary fixtures for the test suite.

Run once from the repo root:
    python tests/fixtures/make_fixtures.py

Generates ~10 KB of files used by tests. Idempotent.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image

FIXTURES_DIR = Path(__file__).parent


def write_solid_image(path: Path, size: tuple[int, int], color, fmt: str) -> None:
    img = Image.new("RGB", size, color)
    img.save(path, format=fmt, quality=90)


def make_jpg() -> None:
    write_solid_image(FIXTURES_DIR / "solid.jpg", (32, 32), (200, 50, 50), "JPEG")


def make_png() -> None:
    write_solid_image(FIXTURES_DIR / "solid.png", (32, 32), (50, 200, 50), "PNG")


def make_webp_source() -> None:
    write_solid_image(FIXTURES_DIR / "solid_source.webp", (32, 32), (50, 50, 200), "WEBP")


def make_gif() -> None:
    img = Image.new("P", (16, 16), 0)
    img.putpalette([0, 0, 0] + [255, 255, 255] * 255)
    img.save(FIXTURES_DIR / "solid.gif", format="GIF")


def make_bmp() -> None:
    write_solid_image(FIXTURES_DIR / "solid.bmp", (16, 16), (123, 45, 67), "BMP")


def make_psd() -> None:
    # Pillow can write PSD via the PSDImagePlugin only for read; we ship a hand-rolled
    # minimal PSD that contains a 4x4 RGB image. PSDs converted by Pillow on the
    # server side use the flattened composite, which is exactly this image.
    path = FIXTURES_DIR / "solid.psd"
    # 4x4 RGB, 8-bit, no layers, single channel block of solid red.
    width, height = 4, 4
    header = b"8BPS" + struct.pack(">H", 1) + b"\x00" * 6
    header += struct.pack(">H", 3)              # channels
    header += struct.pack(">I", height)
    header += struct.pack(">I", width)
    header += struct.pack(">H", 8)              # depth
    header += struct.pack(">H", 3)              # color mode RGB
    color_mode_block = struct.pack(">I", 0)
    image_resources = struct.pack(">I", 0)
    layer_and_mask = struct.pack(">I", 0)
    # Image data section: compression=0 (raw), then per-channel rows.
    pixel_count = width * height
    image_data = struct.pack(">H", 0) + (b"\xff" * pixel_count) + (b"\x10" * pixel_count) + (b"\x10" * pixel_count)
    path.write_bytes(header + color_mode_block + image_resources + layer_and_mask + image_data)


def make_tif() -> None:
    write_solid_image(FIXTURES_DIR / "solid.tif", (16, 16), (10, 220, 30), "TIFF")


def make_pdf() -> None:
    # Tiny one-page PDF generated via Pillow.
    img = Image.new("RGB", (32, 32), (180, 180, 180))
    img.save(FIXTURES_DIR / "doc.pdf", format="PDF")


def make_html() -> None:
    (FIXTURES_DIR / "page.html").write_text(
        "<!doctype html><meta charset=utf-8><title>fx</title><p>hi", encoding="utf-8"
    )


def make_mp4() -> None:
    # ffmpeg is a system dep used by the server already; if missing, skip with a note.
    out = FIXTURES_DIR / "clip.mp4"
    if out.exists():
        return
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=32x32:d=0.2",
                "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-loglevel", "error",
                str(out),
            ],
            check=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"ffmpeg unavailable; skipping {out.name}: {exc}", file=sys.stderr)


def make_mov() -> None:
    out = FIXTURES_DIR / "clip.mov"
    if out.exists():
        return
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=32x32:d=0.2",
                "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-loglevel", "error",
                str(out),
            ],
            check=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"ffmpeg unavailable; skipping {out.name}: {exc}", file=sys.stderr)


def make_m4v() -> None:
    src = FIXTURES_DIR / "clip.mp4"
    dst = FIXTURES_DIR / "clip.m4v"
    if dst.exists() or not src.exists():
        return
    dst.write_bytes(src.read_bytes())


def main() -> int:
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    make_jpg(); make_png(); make_webp_source(); make_gif(); make_bmp()
    make_psd(); make_tif(); make_pdf(); make_html()
    make_mp4(); make_mov(); make_m4v()
    print(f"Fixtures written to {FIXTURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
