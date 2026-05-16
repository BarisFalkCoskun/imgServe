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
    """Choose output format. Respects request; defaults to WebP."""
    if requested_format:
        return requested_format.lower()
    return "webp"


def convert_with_rawpy(src: str, dst: str, fmt: str) -> bool:
    """Camera RAW via libraw — proper demosaic + camera color matrix → sRGB.

    Returns False (without touching dst) for non-RAW inputs so it can sit
    safely at the head of the backend chain.
    """
    if os.path.splitext(src)[1].lower() not in RAW_EXTS:
        return False
    try:
        import rawpy  # soft dependency
    except ImportError:
        logger.warning("rawpy not installed; cannot decode RAW %s", src)
        return False

    try:
        with rawpy.imread(src) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
                no_auto_bright=False,
            )
        img = Image.fromarray(rgb)  # numpy → PIL, mode='RGB'
        out_fmt = fmt.upper()
        if out_fmt == "JPG":
            out_fmt = "JPEG"
        save_kwargs = {"format": out_fmt}
        if out_fmt == "WEBP":
            save_kwargs.update(quality=92, method=6)
        elif out_fmt == "JPEG":
            save_kwargs.update(quality=95)
        img.save(dst, **save_kwargs)
        return True
    except Exception as exc:
        logger.debug("rawpy failed for %s: %s", src, exc)
        return False


def convert_with_pillow(src: str, dst: str, fmt: str) -> bool:
    """Convert using Pillow. Handles PSD (composite), most standard formats."""
    try:
        with Image.open(src) as img:
            img.load()

            # Apply embedded ICC profile → sRGB for color-accurate output.
            # Best-effort: failures fall through to today's behavior.
            icc = img.info.get("icc_profile")
            if icc and img.mode in ("RGB", "RGBA", "L", "LA"):
                try:
                    import io
                    from PIL import ImageCms
                    src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                    dst_profile = ImageCms.createProfile("sRGB")
                    out_mode = "RGBA" if has_transparency(img) else "RGB"
                    img = ImageCms.profileToProfile(
                        img,
                        inputProfile=src_profile,
                        outputProfile=dst_profile,
                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                        outputMode=out_mode,
                    )
                except Exception as exc:
                    logger.debug("ICC convert failed for %s: %s", src, exc)

            out_fmt = fmt.upper()
            if out_fmt == "JPG":
                out_fmt = "JPEG"

            if out_fmt == "JPEG" and img.mode in ("RGBA", "LA", "PA", "P"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA", "PA") else None)
                img = bg
            elif out_fmt == "JPEG" and img.mode == "CMYK":
                img = img.convert("RGB")
            elif img.mode == "CMYK":
                img = img.convert("RGBA" if has_transparency(img) else "RGB")

            save_kwargs = {"format": out_fmt, "quality": 95}
            if out_fmt == "WEBP":
                save_kwargs["method"] = 6
            img.save(dst, **save_kwargs)
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

# Formats needing conversion (non-RAW)
CONVERT_FORMATS = {".psd", ".tif", ".tiff", ".jxr"}

# Camera RAW formats — routed to rawpy first for proper demosaicing.
RAW_EXTS = {".nef", ".arw", ".dng", ".cr2", ".cr3", ".raf", ".rw2", ".orf", ".pef", ".srw"}

# Types we deliberately stream from the source without any conversion.
PASSTHROUGH_EXTS = {".mp4", ".mov", ".m4v", ".html", ".pdf"}


def is_passthrough(ext: str) -> bool:
    """True if we should stream the source as-is rather than convert it.

    Triggers for: explicit passthrough types, and any extension we have no
    converter for (defensive default — never block an unknown extension).
    """
    e = ext.lower()
    if e in PASSTHROUGH_EXTS:
        return True
    return e not in WEB_FORMATS and e not in CONVERT_FORMATS and e not in RAW_EXTS


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
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".m4v": "video/x-m4v",
        ".html": "text/html; charset=utf-8",
        ".pdf": "application/pdf",
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
        ("rawpy", convert_with_rawpy),       # short-circuits False for non-RAW
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
