"""
Image conversion module.

Converts non-web-friendly image formats to PNG (preserves transparency)
or JPG/WebP on request. Uses multiple backends as fallbacks:
  1. rawpy/libraw for camera RAW
  2. ImageMagick first for PSD flattened composites
  3. Pillow for standard image formats
  4. ffmpeg (handles TIFFs that crash ImageMagick)
  5. tiffcp + Pillow (for TIFFs with broken metadata)
  6. JxrDecApp (JPEG XR)
"""

import os
import subprocess
import tempfile
import logging
import contextvars
import math
from PIL import Image

logger = logging.getLogger("imgserve.converter")

Image.MAX_IMAGE_PIXELS = None
WEBP_MAX_DIMENSION = 16383
WEBP_MAX_PIXELS = 100_000_000
RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

_backend_failures_var: contextvars.ContextVar[list[dict[str, str]] | None] = contextvars.ContextVar(
    "backend_failures",
    default=None,
)


def _short_detail(value: object, limit: int = 500) -> str:
    detail = str(value).replace("\n", " ").replace("\r", " ").strip()
    if len(detail) > limit:
        return detail[:limit] + "..."
    return detail


def _record_backend_failure(backend: str, src: str, detail: object) -> None:
    formatted = _short_detail(detail)
    failures = _backend_failures_var.get()
    if failures is not None:
        failures.append({"backend": backend, "detail": formatted})
    logger.debug("%s failed for %s: %s", backend, src, formatted)


def _file_diagnostics(path: str) -> dict[str, object]:
    diagnostics: dict[str, object] = {"path": path}
    try:
        stat = os.stat(path)
        diagnostics["size"] = stat.st_size
    except OSError as exc:
        diagnostics["stat_error"] = repr(exc)
        return diagnostics

    try:
        with open(path, "rb") as handle:
            diagnostics["header_hex"] = handle.read(32).hex()
    except OSError as exc:
        diagnostics["read_error"] = repr(exc)
    return diagnostics


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


def resize_for_webp_limit(img: Image.Image, src: str, fmt: str) -> Image.Image:
    if fmt.lower() != "webp":
        return img

    width, height = img.size
    max_side = max(width, height)
    pixel_count = width * height
    if max_side <= WEBP_MAX_DIMENSION and pixel_count <= WEBP_MAX_PIXELS:
        return img

    scale = min(
        WEBP_MAX_DIMENSION / max_side,
        math.sqrt(WEBP_MAX_PIXELS / pixel_count),
    )
    resized_size = (
        max(1, min(WEBP_MAX_DIMENSION, int(width * scale))),
        max(1, min(WEBP_MAX_DIMENSION, int(height * scale))),
    )
    logger.info(
        "Resizing image for WebP limit: src=%s original_size=%sx%s resized_size=%sx%s "
        "max_dimension=%s max_pixels=%s scale=%.6f",
        src,
        width,
        height,
        resized_size[0],
        resized_size[1],
        WEBP_MAX_DIMENSION,
        WEBP_MAX_PIXELS,
        scale,
    )
    return img.resize(resized_size, RESAMPLE_LANCZOS)


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
        img = resize_for_webp_limit(img, src, fmt)
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
        _record_backend_failure("rawpy", src, f"{type(exc).__name__}: {exc}")
        return False


def convert_with_pillow(src: str, dst: str, fmt: str) -> bool:
    """Convert using Pillow. Handles PSD (composite), most standard formats."""
    try:
        with Image.open(src) as img:
            logger.debug(
                "Pillow opened image: src=%s format=%s mode=%s size=%sx%s target_format=%s",
                src,
                img.format,
                img.mode,
                img.width,
                img.height,
                fmt,
            )
            if fmt.lower() == "webp" and (
                img.width > WEBP_MAX_DIMENSION or img.height > WEBP_MAX_DIMENSION
            ):
                logger.info(
                    "Pillow image exceeds WebP encoder dimension limit: src=%s size=%sx%s limit=%s",
                    src,
                    img.width,
                    img.height,
                    WEBP_MAX_DIMENSION,
                )
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

            img = resize_for_webp_limit(img, src, fmt)

            save_kwargs = {"format": out_fmt, "quality": 95}
            if out_fmt == "WEBP":
                save_kwargs["method"] = 6
            img.save(dst, **save_kwargs)
            return True
    except Exception as e:
        _record_backend_failure("Pillow", src, f"{type(e).__name__}: {e}")
        return False


def convert_with_magick(src: str, dst: str, fmt: str) -> bool:
    """Convert using ImageMagick."""
    try:
        out_path = dst
        cmd = ["convert", src + "[0]", "-quality", "95", out_path]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        stderr = r.stderr.decode(errors="replace")
        _record_backend_failure("ImageMagick", src, f"returncode={r.returncode} stderr={stderr}")
        return False
    except Exception as e:
        _record_backend_failure("ImageMagick", src, f"{type(e).__name__}: {e}")
        return False


def convert_with_ffmpeg(src: str, dst: str, fmt: str) -> bool:
    """Convert using ffmpeg. Good for TIFFs that crash ImageMagick."""
    try:
        cmd = ["ffmpeg", "-y", "-i", src, "-frames:v", "1", dst]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        stderr = r.stderr.decode(errors="replace")
        _record_backend_failure("ffmpeg", src, f"returncode={r.returncode} stderr={stderr}")
        return False
    except Exception as e:
        _record_backend_failure("ffmpeg", src, f"{type(e).__name__}: {e}")
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
        _record_backend_failure("tiffcp", src, f"{type(e).__name__}: {e}")
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
        _record_backend_failure("JxrDecApp", src, "not installed")
        return False
    except Exception as e:
        _record_backend_failure("JxrDecApp", src, f"{type(e).__name__}: {e}")
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

    # Try backends in order. Complex PSDs can expose per-layer/channel data
    # through Pillow; ImageMagick's [0] scene gives the flattened composite.
    if ext == ".psd":
        backends = [
            ("ImageMagick", convert_with_magick),
            ("Pillow", convert_with_pillow),
            ("ffmpeg", convert_with_ffmpeg),
            ("tiffcp", convert_with_tiffcp),
        ]
        logger.info("PSD conversion will prefer ImageMagick flattened composite: %s", src_path)
    else:
        backends = [
            ("rawpy", convert_with_rawpy),       # short-circuits False for non-RAW
            ("Pillow", convert_with_pillow),
            ("ImageMagick", convert_with_magick),
            ("ffmpeg", convert_with_ffmpeg),
            ("tiffcp", convert_with_tiffcp),
        ]

    backend_failures: list[dict[str, str]] = []
    token = _backend_failures_var.set(backend_failures)
    try:
        for name, backend in backends:
            if backend(src_path, dst_path, fmt):
                logger.info(f"Converted {src_path} with {name}")
                return True
            if ext == ".psd" and name == "ImageMagick":
                logger.warning(
                    "PSD ImageMagick conversion failed; falling back to other backends "
                    "may expose layer/channel artifacts: %s",
                    src_path,
                )
    finally:
        _backend_failures_var.reset(token)

    logger.error(
        "All backends failed for %s; diagnostics=%s; backend_failures=%s",
        src_path,
        _file_diagnostics(src_path),
        backend_failures,
    )
    return False
