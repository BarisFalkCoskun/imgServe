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
import hashlib
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

from converter import FORMAT_TO_EXT, convert_image, get_content_type, needs_conversion

DEFAULT_IMGS_BASE = "/mnt/storagebox/imgs"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_CACHE_MAX_AGE_HOURS = 24 * 30
DEFAULT_STALE_LOCK_AGE_HOURS = 6
IMGSBACKUP_DIRS = [
    "/mnt/imgsbackup/imgs3",
]
ENV_IMGS_DIR = "IMGSERVE_IMGS_DIR"
ENV_CACHE_DIR = "IMGSERVE_CACHE_DIR"
ENV_CACHE_MAX_AGE_HOURS = "IMGSERVE_CACHE_MAX_AGE_HOURS"
ENV_STALE_LOCK_AGE_HOURS = "IMGSERVE_STALE_LOCK_AGE_HOURS"
CACHE_CLEANUP_LOCK_NAME = ".cache-cleanup.lock"

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


def _validate_path_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or ".." in value or "\x00" in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")

    separators = {os.sep}
    if os.path.altsep:
        separators.add(os.path.altsep)

    if any(sep in value for sep in separators):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")

    return value


def _safe_cache_stem(filename: str) -> str:
    stem = os.path.splitext(filename)[0] or "image"
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return safe_stem[:80] or "image"


def _build_cache_paths(
    cache_dir: str,
    folder: str,
    filename: str,
    out_ext: str,
    src_size: int,
    src_mtime_ns: int,
) -> tuple[str, str]:
    cache_key_material = f"{folder}/{filename}|{src_size}|{src_mtime_ns}|{out_ext}"
    cache_key = hashlib.sha256(cache_key_material.encode("utf-8")).hexdigest()[:24]
    cache_filename = f"{_safe_cache_stem(filename)}-{cache_key}{out_ext}"
    cache_path = os.path.join(cache_dir, folder, cache_filename)
    lock_path = f"{cache_path}.lock"
    return cache_path, lock_path


def _cleanup_cache_dir(cache_dir: str, cache_max_age_seconds: int, stale_lock_age_seconds: int) -> dict[str, int]:
    now = time.time()
    removed_cache_files = 0
    removed_transient_files = 0
    removed_bytes = 0
    removed_dirs = 0
    errors = 0

    for root, dirs, files in os.walk(cache_dir, topdown=False):
        for filename in files:
            if filename == CACHE_CLEANUP_LOCK_NAME:
                continue

            path = os.path.join(root, filename)

            try:
                stat_result = os.stat(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Could not stat cache artifact %s: %s", path, exc)
                errors += 1
                continue

            age_seconds = max(0, now - stat_result.st_mtime)
            is_transient = filename.endswith(".lock") or filename.startswith(".tmp-")
            max_age_seconds = stale_lock_age_seconds if is_transient else cache_max_age_seconds
            if age_seconds < max_age_seconds:
                continue

            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Could not remove cache artifact %s: %s", path, exc)
                errors += 1
                continue

            removed_bytes += stat_result.st_size
            if is_transient:
                removed_transient_files += 1
            else:
                removed_cache_files += 1

        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            try:
                os.rmdir(dir_path)
            except OSError:
                continue
            else:
                removed_dirs += 1

    return {
        "removed_cache_files": removed_cache_files,
        "removed_transient_files": removed_transient_files,
        "removed_dirs": removed_dirs,
        "removed_bytes": removed_bytes,
        "errors": errors,
    }


def _run_cache_cleanup(cache_dir: str, cache_max_age_hours: int, stale_lock_age_hours: int) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    cleanup_lock_path = os.path.join(cache_dir, CACHE_CLEANUP_LOCK_NAME)
    with open(cleanup_lock_path, "a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("Skipping cache cleanup in %s; another worker is already cleaning", cache_dir)
            return

        try:
            stats = _cleanup_cache_dir(
                cache_dir,
                cache_max_age_seconds=cache_max_age_hours * 3600,
                stale_lock_age_seconds=stale_lock_age_hours * 3600,
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    logger.info(
        "Cache cleanup finished: removed %s cache files, %s transient files, %s dirs, freed %s bytes, %s errors",
        stats["removed_cache_files"],
        stats["removed_transient_files"],
        stats["removed_dirs"],
        stats["removed_bytes"],
        stats["errors"],
    )


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
    checks = [_check_readable_dir(path) for path in IMGSBACKUP_DIRS]
    return {
        "ok": any(bool(check.get("ok")) for check in checks),
        "paths": checks,
    }


def _check_cache_dir(cache_dir: str) -> dict[str, object]:
    status: dict[str, object] = {"path": cache_dir}
    try:
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"{cache_dir} is not a directory")
        if not os.access(cache_dir, os.W_OK | os.X_OK):
            raise PermissionError(f"{cache_dir} is not writable")

        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".health-", delete=True) as tmp:
            tmp.write(b"ok")
            tmp.flush()

        usage = shutil.disk_usage(cache_dir)
    except Exception as exc:
        status["ok"] = False
        status["error"] = str(exc)
        return status

    status["ok"] = True
    status["free_bytes"] = usage.free
    status["total_bytes"] = usage.total
    return status


def _health_payload(imgs_dir: str, cache_dir: str) -> tuple[dict[str, object], int]:
    optimized_webp = _check_optimized_webp_dirs()
    fallback_source = _check_readable_dir(imgs_dir)
    cache = _check_cache_dir(cache_dir)
    healthy = bool(optimized_webp.get("ok")) and bool(cache.get("ok"))

    return {
        "status": "ok" if healthy else "error",
        "checks": {
            "optimized_webp": optimized_webp,
            "fallback_source": fallback_source,
            "cache": cache,
        },
    }, (200 if healthy else 503)


@contextmanager
def _cache_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _convert_into_cache(src_path: str, cache_path: str, out_fmt: str) -> bool:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=os.path.splitext(cache_path)[1],
        dir=os.path.dirname(cache_path),
    )
    os.close(fd)

    try:
        if not convert_image(src_path, tmp_path, out_fmt):
            return False

        os.replace(tmp_path, cache_path)
        return True
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def find_optimized_webp(folder: str, filename: str) -> str | None:
    basename = os.path.splitext(filename)[0]
    webp_name = f"{basename}.webp"
    for base in IMGSBACKUP_DIRS:
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
    app.state.cache_dir = _configured_path(ENV_CACHE_DIR, DEFAULT_CACHE_DIR)
    app.state.cache_max_age_hours = _configured_nonnegative_int(
        ENV_CACHE_MAX_AGE_HOURS,
        DEFAULT_CACHE_MAX_AGE_HOURS,
    )
    app.state.stale_lock_age_hours = _configured_nonnegative_int(
        ENV_STALE_LOCK_AGE_HOURS,
        DEFAULT_STALE_LOCK_AGE_HOURS,
    )

    @app.on_event("startup")
    def startup():
        os.makedirs(app.state.cache_dir, exist_ok=True)
        _run_cache_cleanup(
            app.state.cache_dir,
            cache_max_age_hours=app.state.cache_max_age_hours,
            stale_lock_age_hours=app.state.stale_lock_age_hours,
        )

    @app.get("/health")
    def health(request: Request):
        payload, status_code = _health_payload(request.app.state.imgs_dir, request.app.state.cache_dir)
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

        if format is None or format.lower() == "webp":
            optimized = find_optimized_webp(folder, filename)
            if optimized is not None:
                return FileResponse(optimized, media_type="image/webp")

        src_path = os.path.join(request.app.state.imgs_dir, folder, filename)
        if not os.path.isfile(src_path):
            raise HTTPException(status_code=404, detail="Image not found")

        ext = os.path.splitext(filename)[1].lower()
        out_fmt = None
        if format:
            format = format.lower()
            if format not in FORMAT_TO_EXT:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            out_fmt = format

        if not needs_conversion(ext) and not out_fmt:
            return FileResponse(src_path, media_type=get_content_type(ext))

        if not out_fmt:
            out_fmt = "png"

        src_stat = os.stat(src_path)
        out_ext = FORMAT_TO_EXT[out_fmt]
        cache_path, lock_path = _build_cache_paths(
            request.app.state.cache_dir,
            folder,
            filename,
            out_ext,
            src_stat.st_size,
            src_stat.st_mtime_ns,
        )
        content_type = get_content_type(out_ext)

        if os.path.isfile(cache_path):
            return FileResponse(cache_path, media_type=content_type)

        with _cache_lock(lock_path):
            if os.path.isfile(cache_path):
                return FileResponse(cache_path, media_type=content_type)

            logger.info("Converting %s/%s -> %s", folder, filename, out_fmt)
            success = _convert_into_cache(src_path, cache_path, out_fmt)

        if not success:
            raise HTTPException(status_code=500, detail="Conversion failed")

        return FileResponse(cache_path, media_type=content_type)

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="imgServe — Internal image server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    parser.add_argument(
        "--imgs-dir",
        default=DEFAULT_IMGS_BASE,
        help=f"Images directory (default: {DEFAULT_IMGS_BASE})",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--cache-max-age-hours",
        type=int,
        default=DEFAULT_CACHE_MAX_AGE_HOURS,
        help=f"Delete converted cache files older than this many hours at startup (default: {DEFAULT_CACHE_MAX_AGE_HOURS})",
    )
    parser.add_argument(
        "--stale-lock-age-hours",
        type=int,
        default=DEFAULT_STALE_LOCK_AGE_HOURS,
        help=f"Delete stale lock/tmp files older than this many hours at startup (default: {DEFAULT_STALE_LOCK_AGE_HOURS})",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of workers (default: 16)")
    args = parser.parse_args()

    if args.cache_max_age_hours < 0:
        parser.error("--cache-max-age-hours must be >= 0")
    if args.stale_lock_age_hours < 0:
        parser.error("--stale-lock-age-hours must be >= 0")

    imgs_dir = os.path.abspath(args.imgs_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    os.environ[ENV_IMGS_DIR] = imgs_dir
    os.environ[ENV_CACHE_DIR] = cache_dir
    os.environ[ENV_CACHE_MAX_AGE_HOURS] = str(args.cache_max_age_hours)
    os.environ[ENV_STALE_LOCK_AGE_HOURS] = str(args.stale_lock_age_hours)
    os.makedirs(cache_dir, exist_ok=True)

    logger.info("Optimized WebP directories: %s", ", ".join(IMGSBACKUP_DIRS))
    logger.info("Serving images from: %s", imgs_dir)
    logger.info("Cache directory: %s", cache_dir)
    logger.info("Cache max age (hours): %s", args.cache_max_age_hours)
    logger.info("Stale lock max age (hours): %s", args.stale_lock_age_hours)
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
