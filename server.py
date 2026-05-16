#!/usr/bin/env python3
"""
imgServe — Internal image server with on-the-fly format conversion.

Serves optimized WebP images from /mnt/imgsbackup/imgs3/ when available,
mapping incoming legacy file extensions to .webp while keeping the original
folder structure. Falls back to /mnt/storagebox/imgs/ with automatic
conversion of non-web-friendly formats (PSD, TIFF, DNG, NEF, ARW, JXR, etc.)
to PNG/JPG/WebP. Converted images are cached on the local SSD.

Binds to 127.0.0.1 only — not accessible from the internet.

Usage:
  python3 server.py
  python3 server.py --port 8100
  python3 server.py --cache-dir /path/to/cache

Request examples:
  GET /imgs/fillop/48e04f71...d956b4.jpg              → serves optimized WebP
  GET /imgs/fillop/48e04f71...d956b4.psd              → auto-converts to PNG
  GET /imgs/fillop/48e04f71...d956b4.psd?format=webp  → converts to WebP
  GET /imgs/bilka/abc123.jpg                          → serves optimized WebP when present
  GET /health                                         → health check
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager

import fcntl
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from converter import (
    FORMAT_TO_EXT,
    PASSTHROUGH_EXTS,
    convert_image,
    get_content_type,
    is_passthrough,
    needs_conversion,
)

DEFAULT_IMGS_BASE = "/mnt/storagebox/imgs"
DEFAULT_IMGSBACKUP_PRIMARY = "/mnt/imgsbackup/imgs3"
DEFAULT_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
DEFAULT_HEALTH_MIN_FREE_BYTES = 1 * 1024 * 1024 * 1024
DEFAULT_CONVERSION_SLOTS = 1
DEFAULT_CONVERSION_SLOT_TIMEOUT_SECONDS = 30
DEFAULT_WORKERS = 2
IMGSBACKUP_READ_DIRS = [
    "/mnt/imgsbackup/imgs3",
]
ENV_IMGS_DIR = "IMGSERVE_IMGS_DIR"
ENV_HEALTH_MIN_FREE_BYTES = "IMGSERVE_HEALTH_MIN_FREE_BYTES"
ENV_CONVERSION_SLOTS = "IMGSERVE_CONVERSION_SLOTS"
ENV_CONVERSION_SLOT_TIMEOUT_SECONDS = "IMGSERVE_CONVERSION_SLOT_TIMEOUT_SECONDS"
ENV_IMGSBACKUP_PRIMARY = "IMGSERVE_IMGSBACKUP_DIR"
ENV_STATE_DIR = "IMGSERVE_STATE_DIR"
CACHE_CONVERSION_SLOTS_DIR_NAME = ".conversion-slots"
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"
DIRECT_SOURCE_CACHE_CONTROL = "public, max-age=3600"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("imgserve")


def _configured_path(env_name: str, default: str) -> str:
    return os.path.abspath(os.environ.get(env_name, default))


def _configured_nonnegative_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", env_name, raw, default)
        return default

    if value < 0:
        logger.warning("Negative %s=%r; using default %s", env_name, raw, default)
        return default

    return value


def _configured_positive_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", env_name, raw, default)
        return default

    if value <= 0:
        logger.warning("Non-positive %s=%r; using default %s", env_name, raw, default)
        return default

    return value


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"

    amount = float(value)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        amount /= 1024.0
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"

    return f"{value} B"


def _validate_path_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or ".." in value or "\x00" in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")

    separators = {os.sep}
    if os.path.altsep:
        separators.add(os.path.altsep)

    if any(sep in value for sep in separators):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")

    return value


def _check_readable_dir(path: str) -> dict[str, object]:
    status: dict[str, object] = {"path": path}
    try:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a directory")
        if not os.access(path, os.R_OK | os.X_OK):
            raise PermissionError(f"{path} is not readable")

        with os.scandir(path):
            pass
    except Exception as exc:
        status["ok"] = False
        status["error"] = str(exc)
        return status

    status["ok"] = True
    return status


def _check_optimized_webp_dirs() -> dict[str, object]:
    checks = [_check_readable_dir(path) for path in IMGSBACKUP_READ_DIRS]
    return {
        "ok": any(bool(check.get("ok")) for check in checks),
        "paths": checks,
    }


def _check_imgsbackup_write(imgsbackup_dir: str, min_free_bytes: int) -> dict[str, object]:
    status: dict[str, object] = {"path": imgsbackup_dir}
    try:
        if not os.path.isdir(imgsbackup_dir):
            raise FileNotFoundError(f"{imgsbackup_dir} is not a directory")
        if not os.access(imgsbackup_dir, os.W_OK | os.X_OK):
            raise PermissionError(f"{imgsbackup_dir} is not writable")
        with tempfile.NamedTemporaryFile(dir=imgsbackup_dir, prefix=".health-", delete=True) as tmp:
            tmp.write(b"ok")
            tmp.flush()
        usage = shutil.disk_usage(imgsbackup_dir)
    except Exception as exc:
        status["ok"] = False
        status["error"] = str(exc)
        return status

    status["free_bytes"] = usage.free
    status["total_bytes"] = usage.total
    status["min_free_bytes"] = min_free_bytes
    status["free_ok"] = usage.free >= min_free_bytes
    status["ok"] = bool(status["free_ok"])
    if not status["ok"]:
        status["error"] = (
            f"free disk below threshold: {_format_bytes(usage.free)} "
            f"< {_format_bytes(min_free_bytes)}"
        )
    return status


def _health_payload(imgs_dir: str, imgsbackup_dir: str, health_min_free_bytes: int) -> tuple[dict[str, object], int]:
    optimized_webp = _check_optimized_webp_dirs()
    fallback_source = _check_readable_dir(imgs_dir)
    imgsbackup = _check_imgsbackup_write(imgsbackup_dir, health_min_free_bytes)
    source_available = bool(optimized_webp.get("ok")) or bool(fallback_source.get("ok"))
    healthy = source_available and bool(imgsbackup.get("ok"))

    return {
        "status": "ok" if healthy else "error",
        "source_available": source_available,
        "checks": {
            "optimized_webp": optimized_webp,
            "fallback_source": fallback_source,
            "imgsbackup": imgsbackup,
        },
    }, (200 if healthy else 503)


@contextmanager
def _conversion_slot(state_dir: str, slot_count: int, timeout_seconds: int):
    slots_dir = os.path.join(state_dir, CACHE_CONVERSION_SLOTS_DIR_NAME)
    os.makedirs(slots_dir, exist_ok=True)

    started_at = time.monotonic()
    logged_wait = False
    while True:
        for slot_index in range(slot_count):
            lock_path = os.path.join(slots_dir, f"slot-{slot_index}.lock")
            lock_file = open(lock_path, "a+b")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_file.close()
                continue
            except OSError:
                lock_file.close()
                raise

            wait_seconds = time.monotonic() - started_at
            try:
                yield slot_index, wait_seconds
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            return

        elapsed_seconds = time.monotonic() - started_at
        if elapsed_seconds >= timeout_seconds:
            logger.warning(
                "Conversion slot timeout: pid=%s slots=%s wait_seconds=%.3f",
                os.getpid(),
                slot_count,
                elapsed_seconds,
            )
            raise HTTPException(status_code=503, detail="Conversion capacity unavailable")

        if not logged_wait and elapsed_seconds >= 1.0:
            logger.info(
                "Waiting for conversion slot: pid=%s slots=%s wait_seconds=%.3f",
                os.getpid(),
                slot_count,
                elapsed_seconds,
            )
            logged_wait = True

        time.sleep(0.1)


def _file_response(path: str, media_type: str, cache_control: str) -> FileResponse:
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": cache_control})


def promote_to_imgsbackup(local_path: str, imgsbackup_dir: str, folder: str, filename: str) -> bool:
    """Copy a freshly-converted WebP into imgsbackup atomically.

    Returns True on success, False on any OSError. Failure is logged at WARNING
    but does not raise — the caller still serves the local bytes.
    """
    basename = os.path.splitext(filename)[0]
    dest_dir = os.path.join(imgsbackup_dir, folder)
    dest_path = os.path.join(dest_dir, f"{basename}.webp")
    tmp_name = f".{basename}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    tmp_path = os.path.join(dest_dir, tmp_name)
    try:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(local_path, tmp_path)
        os.rename(tmp_path, dest_path)
        logger.info(
            "Promoted to imgsbackup: pid=%s path=%s bytes=%s",
            os.getpid(),
            dest_path,
            os.path.getsize(dest_path),
        )
        return True
    except OSError as exc:
        logger.warning(
            "Could not promote to imgsbackup (serving anyway): pid=%s dest=%s error=%s",
            os.getpid(),
            dest_path,
            exc,
        )
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def find_optimized_webp(folder: str, filename: str) -> str | None:
    basename = os.path.splitext(filename)[0]
    webp_name = f"{basename}.webp"
    for base in IMGSBACKUP_READ_DIRS:
        candidate = os.path.join(base, folder, webp_name)
        try:
            if os.path.isfile(candidate):
                return candidate
        except OSError:
            continue
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="imgServe", docs_url=None, redoc_url=None)
    app.state.imgs_dir = _configured_path(ENV_IMGS_DIR, DEFAULT_IMGS_BASE)
    app.state.health_min_free_bytes = _configured_nonnegative_int(
        ENV_HEALTH_MIN_FREE_BYTES,
        DEFAULT_HEALTH_MIN_FREE_BYTES,
    )
    app.state.conversion_slots = _configured_positive_int(
        ENV_CONVERSION_SLOTS,
        DEFAULT_CONVERSION_SLOTS,
    )
    app.state.conversion_slot_timeout_seconds = _configured_nonnegative_int(
        ENV_CONVERSION_SLOT_TIMEOUT_SECONDS,
        DEFAULT_CONVERSION_SLOT_TIMEOUT_SECONDS,
    )
    app.state.imgsbackup_dir = _configured_path(ENV_IMGSBACKUP_PRIMARY, DEFAULT_IMGSBACKUP_PRIMARY)
    app.state.state_dir = _configured_path(ENV_STATE_DIR, DEFAULT_STATE_DIR)
    os.makedirs(app.state.state_dir, exist_ok=True)

    @app.get("/health")
    def health(request: Request):
        payload, status_code = _health_payload(
            request.app.state.imgs_dir,
            request.app.state.imgsbackup_dir,
            request.app.state.health_min_free_bytes,
        )
        return JSONResponse(payload, status_code=status_code)

    @app.get("/imgs/{folder}/{filename}")
    def serve_image(
        request: Request,
        folder: str,
        filename: str,
        format: str | None = Query(None, description="Output format: png, jpg, webp"),
    ):
        folder = _validate_path_component(folder, "folder")
        filename = _validate_path_component(filename, "filename")

        # 1. Cached canonical WebP — fast path.
        if format is None or format.lower() == "webp":
            optimized = find_optimized_webp(folder, filename)
            if optimized is not None:
                logger.debug(
                    "Serving optimized WebP: pid=%s image=%s/%s path=%s",
                    os.getpid(), folder, filename, optimized,
                )
                return _file_response(optimized, media_type="image/webp",
                                      cache_control=IMMUTABLE_CACHE_CONTROL)

        # 2. Source must exist.
        src_path = os.path.join(request.app.state.imgs_dir, folder, filename)
        if not os.path.isfile(src_path):
            raise HTTPException(status_code=404, detail="Image not found")

        ext = os.path.splitext(filename)[1].lower()

        # 3. Passthrough — video, html, pdf, or anything we can't convert.
        if is_passthrough(ext):
            logger.debug(
                "Passthrough source: pid=%s image=%s/%s ext=%s",
                os.getpid(), folder, filename, ext,
            )
            return _file_response(
                src_path,
                media_type=get_content_type(ext),
                cache_control=DIRECT_SOURCE_CACHE_CONTROL,
            )

        # 4. Decide output format.
        if format:
            fmt_lower = format.lower()
            if fmt_lower not in FORMAT_TO_EXT:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            out_fmt = fmt_lower
        else:
            out_fmt = "webp"

        out_ext = FORMAT_TO_EXT[out_fmt]
        content_type = get_content_type(out_ext)

        # 5. Convert in /tmp under a conversion slot.
        src_size = os.path.getsize(src_path)
        conversion_started_at = time.monotonic()
        logger.info(
            "Conversion queued: pid=%s image=%s/%s format=%s src_bytes=%s",
            os.getpid(), folder, filename, out_fmt, src_size,
        )

        tmp_fd, tmp_out = tempfile.mkstemp(prefix="imgserve-", suffix=out_ext)
        os.close(tmp_fd)
        try:
            with _conversion_slot(
                request.app.state.state_dir,
                slot_count=request.app.state.conversion_slots,
                timeout_seconds=request.app.state.conversion_slot_timeout_seconds,
            ) as (slot_index, slot_wait_seconds):
                logger.info(
                    "Conversion started: pid=%s slot=%s image=%s/%s format=%s slot_wait_seconds=%.3f",
                    os.getpid(), slot_index, folder, filename, out_fmt, slot_wait_seconds,
                )
                success = convert_image(src_path, tmp_out, out_fmt)

            if not success:
                raise HTTPException(status_code=500, detail="Conversion failed")

            output_bytes = os.path.getsize(tmp_out)
            duration_seconds = time.monotonic() - conversion_started_at

            # 6. Write-back only for canonical WebP. Fail-soft.
            if out_fmt == "webp":
                promote_to_imgsbackup(
                    tmp_out,
                    request.app.state.imgsbackup_dir,
                    folder,
                    filename,
                )

            logger.info(
                "Conversion finished: pid=%s image=%s/%s format=%s duration_seconds=%.3f "
                "src_bytes=%s output_bytes=%s",
                os.getpid(), folder, filename, out_fmt, duration_seconds, src_size, output_bytes,
            )

            return FileResponse(
                tmp_out,
                media_type=content_type,
                headers={"Cache-Control": IMMUTABLE_CACHE_CONTROL},
                background=BackgroundTask(_safe_unlink, tmp_out),
            )
        except Exception:
            # If we never returned a FileResponse, clean up the temp file now.
            _safe_unlink(tmp_out)
            raise

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="imgServe — Internal image server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    parser.add_argument(
        "--imgs-dir",
        default=DEFAULT_IMGS_BASE,
        help=f"Source images directory (default: {DEFAULT_IMGS_BASE})",
    )
    parser.add_argument(
        "--imgsbackup-dir",
        default=DEFAULT_IMGSBACKUP_PRIMARY,
        help=f"Writable canonical WebP store (default: {DEFAULT_IMGSBACKUP_PRIMARY})",
    )
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        help=f"Directory for conversion slot lock files (default: {DEFAULT_STATE_DIR})",
    )
    parser.add_argument(
        "--health-min-free-bytes",
        type=int,
        default=DEFAULT_HEALTH_MIN_FREE_BYTES,
        help=f"Mark /health unhealthy below this imgsbackup free-space floor (default: {DEFAULT_HEALTH_MIN_FREE_BYTES})",
    )
    parser.add_argument(
        "--conversion-slots",
        type=int,
        default=DEFAULT_CONVERSION_SLOTS,
        help=f"Cross-worker conversion slots on this host (default: {DEFAULT_CONVERSION_SLOTS})",
    )
    parser.add_argument(
        "--conversion-slot-timeout-seconds",
        type=int,
        default=DEFAULT_CONVERSION_SLOT_TIMEOUT_SECONDS,
        help=(
            "Maximum time a request waits for a conversion slot before returning 503 "
            f"(default: {DEFAULT_CONVERSION_SLOT_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of workers (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    if args.health_min_free_bytes < 0:
        parser.error("--health-min-free-bytes must be >= 0")
    if args.conversion_slots <= 0:
        parser.error("--conversion-slots must be > 0")
    if args.conversion_slot_timeout_seconds < 0:
        parser.error("--conversion-slot-timeout-seconds must be >= 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")

    imgs_dir = os.path.abspath(args.imgs_dir)
    imgsbackup_dir = os.path.abspath(args.imgsbackup_dir)
    state_dir = os.path.abspath(args.state_dir)
    os.environ[ENV_IMGS_DIR] = imgs_dir
    os.environ[ENV_IMGSBACKUP_PRIMARY] = imgsbackup_dir
    os.environ[ENV_STATE_DIR] = state_dir
    os.environ[ENV_HEALTH_MIN_FREE_BYTES] = str(args.health_min_free_bytes)
    os.environ[ENV_CONVERSION_SLOTS] = str(args.conversion_slots)
    os.environ[ENV_CONVERSION_SLOT_TIMEOUT_SECONDS] = str(args.conversion_slot_timeout_seconds)
    os.makedirs(state_dir, exist_ok=True)

    logger.info("Optimized WebP read dirs: %s", ", ".join(IMGSBACKUP_READ_DIRS))
    logger.info("Source images: %s", imgs_dir)
    logger.info("imgsbackup write target: %s", imgsbackup_dir)
    logger.info("State dir (conversion slot locks): %s", state_dir)
    logger.info("Health min free bytes: %s", _format_bytes(args.health_min_free_bytes))
    logger.info("Conversion slots: %s", args.conversion_slots)
    logger.info("Conversion slot timeout seconds: %s", args.conversion_slot_timeout_seconds)
    logger.info("Listening on: %s:%s", args.host, args.port)

    uvicorn.run(
        "server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
