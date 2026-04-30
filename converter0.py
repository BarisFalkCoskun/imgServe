"""
Image conversion module.

Converts non-web-friendly image formats to PNG (preserves transparency)
or JPG/WebP on request. Uses multiple backends as fallbacks:
  1. Pillow (most formats including PSD first composite layer)
  2. ImageMagick (broader format support)
  3. ffmpeg (handles TIFFs that crash ImageMagick)
  4. tiffcp + Pillow (for TIFFs with broken metadata)
  5. JxrDecApp (JPEG XR)
"""

import os
import subprocess
import tempfile
import logging
from PIL import Image

logger = logging.getLogger("imgserve.converter")

Image.MAX_IMAGE_PIXELS = None


def has_transparency(img: Image.Image) -> bool:
    """Check if a Pillow image has meaningful transparency."""
    if img.mode in ("RGBA", "LA", "PA"):
        alpha = img.getchannel("A")
        extrema = alpha.getextrema()
        # If min alpha < 255, there's some transparency
        return extrema[0] < 255
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False


def choose_output_format(img: Image.Image, requested_format: str | None) -> str:
    """Choose output format. Respects request, but defaults to PNG if transparent."""
    if requested_format:
        return requested_format.lower()
    if has_transparency(img):
        return "png"
    return "png"  # Default to PNG for lossless quality


def convert_with_pillow(src: str, dst: str, fmt: str) -> bool:
    """Convert using Pillow. Handles PSD (composite), most standard formats."""
    try:
        with Image.open(src) as img:
            img.load()

            out_fmt = fmt.upper()
            if out_fmt == "JPG":
                out_fmt = "JPEG"

            if out_fmt == "JPEG" and img.mode in ("RGBA", "LA", "PA", "P"):
                # JPEG doesn't support transparency — composite onto white
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA", "PA") else None)
                img = bg
            elif out_fmt == "JPEG" and img.mode == "CMYK":
                img = img.convert("RGB")
            elif img.mode == "CMYK":
                img = img.convert("RGBA" if has_transparency(img) else "RGB")

            img.save(dst, format=out_fmt, quality=95)
            return True
    except Exception as e:
        logger.debug(f"Pillow failed for {src}: {e}")
        return False


def convert_with_magick(src: str, dst: str, fmt: str) -> bool:
    """Convert using ImageMagick."""
    try:
        out_path = dst
        cmd = ["convert", src + "[0]", "-quality", "95", out_path]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        logger.debug(f"ImageMagick failed for {src}: {r.stderr.decode(errors='replace')[:200]}")
        return False
    except Exception as e:
        logger.debug(f"ImageMagick exception for {src}: {e}")
        return False


def convert_with_ffmpeg(src: str, dst: str, fmt: str) -> bool:
    """Convert using ffmpeg. Good for TIFFs that crash ImageMagick."""
    try:
        cmd = ["ffmpeg", "-y", "-i", src, "-frames:v", "1", dst]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        logger.debug(f"ffmpeg failed for {src}: {r.stderr.decode(errors='replace')[:200]}")
        return False
    except Exception as e:
        logger.debug(f"ffmpeg exception for {src}: {e}")
        return False


def convert_with_tiffcp(src: str, dst: str, fmt: str) -> bool:
    """Fix broken TIFF metadata with tiffcp, then convert with Pillow."""
    ext = os.path.splitext(src)[1].lower()
    if ext not in (".tif", ".tiff"):
        return False
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        r = subprocess.run(["tiffcp", src, tmp_path], capture_output=True, timeout=30)
        if r.returncode != 0:
            os.unlink(tmp_path)
            return False
        result = convert_with_pillow(tmp_path, dst, fmt)
        os.unlink(tmp_path)
        return result
    except Exception as e:
        logger.debug(f"tiffcp failed for {src}: {e}")
        return False


def convert_jxr(src: str, dst: str, fmt: str) -> bool:
    """Convert JPEG XR using JxrDecApp -> intermediate TIF -> final format."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        r = subprocess.run(
            ["JxrDecApp", "-i", src, "-o", tmp_path],
            capture_output=True, timeout=30
        )
        if r.returncode != 0:
            os.unlink(tmp_path)
            return False
        result = convert_with_pillow(tmp_path, dst, fmt)
        os.unlink(tmp_path)
        return result
    except FileNotFoundError:
        logger.debug("JxrDecApp not installed")
        return False
    except Exception as e:
        logger.debug(f"JxrDecApp failed for {src}: {e}")
        return False


# Formats that browsers/apps can display directly
WEB_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".apng", ".bmp", ".svg"}

# Formats needing conversion
CONVERT_FORMATS = {".psd", ".tif", ".tiff", ".dng", ".nef", ".arw", ".jxr", ".cr2", ".cr3"}

# Map output format string to file extension
FORMAT_TO_EXT = {
    "png": ".png",
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "webp": ".webp",
}


def needs_conversion(ext: str) -> bool:
    """Check if a file extension needs conversion."""
    return ext.lower() not in WEB_FORMATS


def get_content_type(ext: str) -> str:
    """Get MIME type for a file extension."""
    types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".avif": "image/avif",
        ".apng": "image/apng",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    return types.get(ext.lower(), "application/octet-stream")


def convert_image(src_path: str, dst_path: str, fmt: str = "png") -> bool:
    """
    Convert an image to the specified format using the best available backend.
    Tries multiple backends in order of preference.
    Returns True if conversion succeeded.
    """
    ext = os.path.splitext(src_path)[1].lower()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # JPEG XR has its own pipeline
    if ext == ".jxr":
        if convert_jxr(src_path, dst_path, fmt):
            return True
        return False

    # Try backends in order
    backends = [
        ("Pillow", convert_with_pillow),
        ("ImageMagick", convert_with_magick),
        ("ffmpeg", convert_with_ffmpeg),
        ("tiffcp", convert_with_tiffcp),
    ]

    for name, backend in backends:
        if backend(src_path, dst_path, fmt):
            logger.info(f"Converted {src_path} with {name}")
            return True

    logger.error(f"All backends failed for {src_path}")
    return False
