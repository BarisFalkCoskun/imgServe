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
from contextlib import asynccontextmanager, contextmanager

import fcntl
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from converter import FORMAT_TO_EXT, convert_image, get_content_type, needs_conversion

DEFAULT_IMGS_BASE = "/mnt/storagebox/imgs"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_CACHE_MAX_AGE_HOURS = 24 * 30
DEFAULT_CACHE_MAX_BYTES = 12 * 1024 * 1024 * 1024
DEFAULT_STALE_LOCK_AGE_HOURS = 6
DEFAULT_MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_HEALTH_MIN_FREE_BYTES = 1 * 1024 * 1024 * 1024
DEFAULT_CONVERSION_SLOTS = 1
DEFAULT_CONVERSION_SLOT_TIMEOUT_SECONDS = 30
DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS = 300
DEFAULT_WORKERS = 2
IMGSBACKUP_DIRS = [
    "/mnt/imgsbackup/imgs3",
]
ENV_IMGS_DIR = "IMGSERVE_IMGS_DIR"
ENV_CACHE_DIR = "IMGSERVE_CACHE_DIR"
ENV_CACHE_MAX_AGE_HOURS = "IMGSERVE_CACHE_MAX_AGE_HOURS"
ENV_CACHE_MAX_BYTES = "IMGSERVE_CACHE_MAX_BYTES"
ENV_STALE_LOCK_AGE_HOURS = "IMGSERVE_STALE_LOCK_AGE_HOURS"
ENV_MIN_FREE_BYTES = "IMGSERVE_MIN_FREE_BYTES"
ENV_HEALTH_MIN_FREE_BYTES = "IMGSERVE_HEALTH_MIN_FREE_BYTES"
ENV_CONVERSION_SLOTS = "IMGSERVE_CONVERSION_SLOTS"
ENV_CONVERSION_SLOT_TIMEOUT_SECONDS = "IMGSERVE_CONVERSION_SLOT_TIMEOUT_SECONDS"
ENV_CACHE_CLEANUP_INTERVAL_SECONDS = "IMGSERVE_CACHE_CLEANUP_INTERVAL_SECONDS"
CACHE_KEY_VERSION = "v2"
CACHE_CLEANUP_LOCK_NAME = ".cache-cleanup.lock"
CACHE_CLEANUP_LAST_NAME = ".cache-cleanup-last"
CACHE_CONVERSION_SLOTS_DIR_NAME = ".conversion-slots"
CACHE_INTERNAL_FILES = {CACHE_CLEANUP_LOCK_NAME, CACHE_CLEANUP_LAST_NAME}
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
    cache_key_material = f"{CACHE_KEY_VERSION}|{folder}/{filename}|{src_size}|{src_mtime_ns}|{out_ext}"
    cache_key = hashlib.sha256(cache_key_material.encode("utf-8")).hexdigest()[:24]
    cache_filename = f"{_safe_cache_stem(filename)}-{cache_key}{out_ext}"
    cache_path = os.path.join(cache_dir, folder, cache_filename)
    lock_path = f"{cache_path}.lock"
    return cache_path, lock_path


def _is_protected_cache_root(cache_dir: str, root: str) -> bool:
    rel_root = os.path.relpath(root, cache_dir)
    return rel_root == CACHE_CONVERSION_SLOTS_DIR_NAME or rel_root.startswith(
        CACHE_CONVERSION_SLOTS_DIR_NAME + os.sep
    )


def _cleanup_cache_dir(
    cache_dir: str,
    cache_max_age_seconds: int,
    stale_lock_age_seconds: int,
    cache_max_bytes: int,
) -> dict[str, int]:
    now = time.time()
    remaining_cache_files: list[tuple[float, str, int]] = []
    remaining_cache_bytes = 0
    removed_expired_cache_files = 0
    removed_budget_cache_files = 0
    removed_transient_files = 0
    removed_bytes = 0
    removed_dirs = 0
    errors = 0

    for root, dirs, files in os.walk(cache_dir, topdown=False):
        if _is_protected_cache_root(cache_dir, root):
            continue

        for filename in files:
            if filename in CACHE_INTERNAL_FILES:
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
                if not is_transient:
                    remaining_cache_files.append((stat_result.st_mtime, path, stat_result.st_size))
                    remaining_cache_bytes += stat_result.st_size
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
                removed_expired_cache_files += 1

    if cache_max_bytes > 0 and remaining_cache_bytes > cache_max_bytes:
        for _mtime, path, size in sorted(remaining_cache_files):
            if remaining_cache_bytes <= cache_max_bytes:
                break

            try:
                os.unlink(path)
            except FileNotFoundError:
                remaining_cache_bytes -= size
                continue
            except OSError as exc:
                logger.warning("Could not remove cache artifact %s: %s", path, exc)
                errors += 1
                continue

            remaining_cache_bytes -= size
            removed_bytes += size
            removed_budget_cache_files += 1

    for root, dirs, _files in os.walk(cache_dir, topdown=False):
        if _is_protected_cache_root(cache_dir, root):
            continue
        for dirname in dirs:
            if dirname == CACHE_CONVERSION_SLOTS_DIR_NAME:
                continue

            dir_path = os.path.join(root, dirname)
            try:
                os.rmdir(dir_path)
            except OSError:
                continue
            else:
                removed_dirs += 1

    return {
        "removed_expired_cache_files": removed_expired_cache_files,
        "removed_budget_cache_files": removed_budget_cache_files,
        "removed_transient_files": removed_transient_files,
        "removed_dirs": removed_dirs,
        "removed_bytes": removed_bytes,
        "remaining_cache_bytes": max(0, remaining_cache_bytes),
        "cache_max_bytes": cache_max_bytes,
        "errors": errors,
    }


def _touch_cache_cleanup_marker(cache_dir: str) -> None:
    marker_path = os.path.join(cache_dir, CACHE_CLEANUP_LAST_NAME)
    try:
        with open(marker_path, "a", encoding="utf-8"):
            pass
        os.utime(marker_path)
    except OSError as exc:
        logger.warning("Could not update cache cleanup marker %s: %s", marker_path, exc)


def _run_cache_cleanup(
    cache_dir: str,
    cache_max_age_hours: int,
    stale_lock_age_hours: int,
    cache_max_bytes: int,
) -> dict[str, int] | None:
    os.makedirs(cache_dir, exist_ok=True)
    cleanup_lock_path = os.path.join(cache_dir, CACHE_CLEANUP_LOCK_NAME)
    with open(cleanup_lock_path, "a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("Skipping cache cleanup in %s; another worker is already cleaning", cache_dir)
            return None

        try:
            stats = _cleanup_cache_dir(
                cache_dir,
                cache_max_age_seconds=cache_max_age_hours * 3600,
                stale_lock_age_seconds=stale_lock_age_hours * 3600,
                cache_max_bytes=cache_max_bytes,
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    logger.info(
        "Cache cleanup finished: removed %s expired cache files, %s budget cache files, "
        "%s transient files, %s dirs, freed %s, remaining cache %s, max cache %s, %s errors",
        stats["removed_expired_cache_files"],
        stats["removed_budget_cache_files"],
        stats["removed_transient_files"],
        stats["removed_dirs"],
        _format_bytes(stats["removed_bytes"]),
        _format_bytes(stats["remaining_cache_bytes"]),
        "disabled" if stats["cache_max_bytes"] == 0 else _format_bytes(stats["cache_max_bytes"]),
        stats["errors"],
    )
    _touch_cache_cleanup_marker(cache_dir)
    return stats


def _maybe_run_cache_cleanup(
    cache_dir: str,
    cache_cleanup_interval_seconds: int,
    cache_max_age_hours: int,
    stale_lock_age_hours: int,
    cache_max_bytes: int,
) -> None:
    if cache_cleanup_interval_seconds <= 0:
        return

    marker_path = os.path.join(cache_dir, CACHE_CLEANUP_LAST_NAME)
    try:
        marker_age_seconds = time.time() - os.stat(marker_path).st_mtime
    except FileNotFoundError:
        marker_age_seconds = cache_cleanup_interval_seconds
    except OSError as exc:
        logger.warning("Could not stat cache cleanup marker %s: %s", marker_path, exc)
        marker_age_seconds = cache_cleanup_interval_seconds

    if marker_age_seconds < cache_cleanup_interval_seconds:
        return

    logger.info(
        "Cache cleanup due before conversion: marker_age_seconds=%.1f interval_seconds=%s",
        marker_age_seconds,
        cache_cleanup_interval_seconds,
    )
    _run_cache_cleanup(
        cache_dir,
        cache_max_age_hours=cache_max_age_hours,
        stale_lock_age_hours=stale_lock_age_hours,
        cache_max_bytes=cache_max_bytes,
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


def _check_cache_dir(cache_dir: str, min_free_bytes: int) -> dict[str, object]:
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


def _health_payload(imgs_dir: str, cache_dir: str, health_min_free_bytes: int) -> tuple[dict[str, object], int]:
    optimized_webp = _check_optimized_webp_dirs()
    fallback_source = _check_readable_dir(imgs_dir)
    cache = _check_cache_dir(cache_dir, health_min_free_bytes)
    source_available = bool(optimized_webp.get("ok")) or bool(fallback_source.get("ok"))
    healthy = source_available and bool(cache.get("ok"))

    return {
        "status": "ok" if healthy else "error",
        "source_available": source_available,
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
        started_at = time.monotonic()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        wait_seconds = time.monotonic() - started_at
        try:
            yield wait_seconds
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _conversion_slot(cache_dir: str, slot_count: int, timeout_seconds: int):
    slots_dir = os.path.join(cache_dir, CACHE_CONVERSION_SLOTS_DIR_NAME)
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


def _ensure_cache_storage(
    cache_dir: str,
    min_free_bytes: int,
    cache_max_age_hours: int,
    stale_lock_age_hours: int,
    cache_max_bytes: int,
) -> int:
    usage = shutil.disk_usage(cache_dir)
    if usage.free >= min_free_bytes:
        return usage.free

    logger.warning(
        "Low cache disk before conversion: free=%s min_free=%s; running cleanup",
        _format_bytes(usage.free),
        _format_bytes(min_free_bytes),
    )
    _run_cache_cleanup(
        cache_dir,
        cache_max_age_hours=cache_max_age_hours,
        stale_lock_age_hours=stale_lock_age_hours,
        cache_max_bytes=cache_max_bytes,
    )

    usage = shutil.disk_usage(cache_dir)
    if usage.free < min_free_bytes:
        logger.error(
            "Rejecting conversion due to low cache disk: free=%s min_free=%s",
            _format_bytes(usage.free),
            _format_bytes(min_free_bytes),
        )
        raise HTTPException(status_code=503, detail="Insufficient cache disk space")

    return usage.free


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


def _file_response(path: str, media_type: str, cache_control: str) -> FileResponse:
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": cache_control})


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
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        os.makedirs(app.state.cache_dir, exist_ok=True)
        _run_cache_cleanup(
            app.state.cache_dir,
            cache_max_age_hours=app.state.cache_max_age_hours,
            stale_lock_age_hours=app.state.stale_lock_age_hours,
            cache_max_bytes=app.state.cache_max_bytes,
        )
        yield

    app = FastAPI(title="imgServe", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.imgs_dir = _configured_path(ENV_IMGS_DIR, DEFAULT_IMGS_BASE)
    app.state.cache_dir = _configured_path(ENV_CACHE_DIR, DEFAULT_CACHE_DIR)
    app.state.cache_max_age_hours = _configured_nonnegative_int(
        ENV_CACHE_MAX_AGE_HOURS,
        DEFAULT_CACHE_MAX_AGE_HOURS,
    )
    app.state.cache_max_bytes = _configured_nonnegative_int(
        ENV_CACHE_MAX_BYTES,
        DEFAULT_CACHE_MAX_BYTES,
    )
    app.state.stale_lock_age_hours = _configured_nonnegative_int(
        ENV_STALE_LOCK_AGE_HOURS,
        DEFAULT_STALE_LOCK_AGE_HOURS,
    )
    app.state.min_free_bytes = _configured_nonnegative_int(
        ENV_MIN_FREE_BYTES,
        DEFAULT_MIN_FREE_BYTES,
    )
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
    app.state.cache_cleanup_interval_seconds = _configured_nonnegative_int(
        ENV_CACHE_CLEANUP_INTERVAL_SECONDS,
        DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS,
    )

    @app.get("/health")
    def health(request: Request):
        payload, status_code = _health_payload(
            request.app.state.imgs_dir,
            request.app.state.cache_dir,
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

        if format is None or format.lower() == "webp":
            optimized = find_optimized_webp(folder, filename)
            if optimized is not None:
                logger.debug(
                    "Serving optimized WebP: pid=%s image=%s/%s path=%s",
                    os.getpid(),
                    folder,
                    filename,
                    optimized,
                )
                return _file_response(optimized, media_type="image/webp", cache_control=IMMUTABLE_CACHE_CONTROL)

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
            logger.debug("Serving source image directly: pid=%s image=%s/%s", os.getpid(), folder, filename)
            return _file_response(
                src_path,
                media_type=get_content_type(ext),
                cache_control=DIRECT_SOURCE_CACHE_CONTROL,
            )

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
            logger.debug(
                "Serving cached conversion: pid=%s image=%s/%s format=%s path=%s",
                os.getpid(),
                folder,
                filename,
                out_fmt,
                cache_path,
            )
            return _file_response(cache_path, media_type=content_type, cache_control=IMMUTABLE_CACHE_CONTROL)

        with _cache_lock(lock_path) as file_lock_wait_seconds:
            if os.path.isfile(cache_path):
                logger.info(
                    "Cache filled by another worker: pid=%s image=%s/%s format=%s file_lock_wait_seconds=%.3f",
                    os.getpid(),
                    folder,
                    filename,
                    out_fmt,
                    file_lock_wait_seconds,
                )
                return _file_response(cache_path, media_type=content_type, cache_control=IMMUTABLE_CACHE_CONTROL)

            _maybe_run_cache_cleanup(
                request.app.state.cache_dir,
                cache_cleanup_interval_seconds=request.app.state.cache_cleanup_interval_seconds,
                cache_max_age_hours=request.app.state.cache_max_age_hours,
                stale_lock_age_hours=request.app.state.stale_lock_age_hours,
                cache_max_bytes=request.app.state.cache_max_bytes,
            )
            free_before_bytes = _ensure_cache_storage(
                request.app.state.cache_dir,
                min_free_bytes=request.app.state.min_free_bytes,
                cache_max_age_hours=request.app.state.cache_max_age_hours,
                stale_lock_age_hours=request.app.state.stale_lock_age_hours,
                cache_max_bytes=request.app.state.cache_max_bytes,
            )

            conversion_started_at = time.monotonic()
            logger.info(
                "Conversion queued: pid=%s image=%s/%s format=%s src_bytes=%s free_before=%s "
                "file_lock_wait_seconds=%.3f",
                os.getpid(),
                folder,
                filename,
                out_fmt,
                src_stat.st_size,
                _format_bytes(free_before_bytes),
                file_lock_wait_seconds,
            )
            with _conversion_slot(
                request.app.state.cache_dir,
                slot_count=request.app.state.conversion_slots,
                timeout_seconds=request.app.state.conversion_slot_timeout_seconds,
            ) as (slot_index, slot_wait_seconds):
                logger.info(
                    "Conversion started: pid=%s slot=%s image=%s/%s format=%s "
                    "slot_wait_seconds=%.3f",
                    os.getpid(),
                    slot_index,
                    folder,
                    filename,
                    out_fmt,
                    slot_wait_seconds,
                )
                success = _convert_into_cache(src_path, cache_path, out_fmt)

            duration_seconds = time.monotonic() - conversion_started_at
            output_bytes = os.path.getsize(cache_path) if success and os.path.isfile(cache_path) else 0
            free_after_bytes = shutil.disk_usage(request.app.state.cache_dir).free
            logger.info(
                "Conversion finished: pid=%s image=%s/%s format=%s success=%s duration_seconds=%.3f "
                "slot_wait_seconds=%.3f file_lock_wait_seconds=%.3f src_bytes=%s output_bytes=%s "
                "free_before=%s free_after=%s",
                os.getpid(),
                folder,
                filename,
                out_fmt,
                success,
                duration_seconds,
                slot_wait_seconds,
                file_lock_wait_seconds,
                src_stat.st_size,
                output_bytes,
                _format_bytes(free_before_bytes),
                _format_bytes(free_after_bytes),
            )

        if not success:
            raise HTTPException(status_code=500, detail="Conversion failed")

        return _file_response(cache_path, media_type=content_type, cache_control=IMMUTABLE_CACHE_CONTROL)

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
        "--cache-max-bytes",
        type=int,
        default=DEFAULT_CACHE_MAX_BYTES,
        help=f"Keep converted cache near this byte budget; 0 disables size cleanup (default: {DEFAULT_CACHE_MAX_BYTES})",
    )
    parser.add_argument(
        "--stale-lock-age-hours",
        type=int,
        default=DEFAULT_STALE_LOCK_AGE_HOURS,
        help=f"Delete stale lock/tmp files older than this many hours at startup (default: {DEFAULT_STALE_LOCK_AGE_HOURS})",
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=DEFAULT_MIN_FREE_BYTES,
        help=f"Reject new conversions below this cache disk free-space floor (default: {DEFAULT_MIN_FREE_BYTES})",
    )
    parser.add_argument(
        "--health-min-free-bytes",
        type=int,
        default=DEFAULT_HEALTH_MIN_FREE_BYTES,
        help=f"Mark /health unhealthy below this cache disk free-space floor (default: {DEFAULT_HEALTH_MIN_FREE_BYTES})",
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
    parser.add_argument(
        "--cache-cleanup-interval-seconds",
        type=int,
        default=DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS,
        help=(
            "Minimum interval between pre-conversion cache cleanup passes; 0 disables periodic cleanup "
            f"(default: {DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS})"
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of workers (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    if args.cache_max_age_hours < 0:
        parser.error("--cache-max-age-hours must be >= 0")
    if args.cache_max_bytes < 0:
        parser.error("--cache-max-bytes must be >= 0")
    if args.stale_lock_age_hours < 0:
        parser.error("--stale-lock-age-hours must be >= 0")
    if args.min_free_bytes < 0:
        parser.error("--min-free-bytes must be >= 0")
    if args.health_min_free_bytes < 0:
        parser.error("--health-min-free-bytes must be >= 0")
    if args.conversion_slots <= 0:
        parser.error("--conversion-slots must be > 0")
    if args.conversion_slot_timeout_seconds < 0:
        parser.error("--conversion-slot-timeout-seconds must be >= 0")
    if args.cache_cleanup_interval_seconds < 0:
        parser.error("--cache-cleanup-interval-seconds must be >= 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")

    imgs_dir = os.path.abspath(args.imgs_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    os.environ[ENV_IMGS_DIR] = imgs_dir
    os.environ[ENV_CACHE_DIR] = cache_dir
    os.environ[ENV_CACHE_MAX_AGE_HOURS] = str(args.cache_max_age_hours)
    os.environ[ENV_CACHE_MAX_BYTES] = str(args.cache_max_bytes)
    os.environ[ENV_STALE_LOCK_AGE_HOURS] = str(args.stale_lock_age_hours)
    os.environ[ENV_MIN_FREE_BYTES] = str(args.min_free_bytes)
    os.environ[ENV_HEALTH_MIN_FREE_BYTES] = str(args.health_min_free_bytes)
    os.environ[ENV_CONVERSION_SLOTS] = str(args.conversion_slots)
    os.environ[ENV_CONVERSION_SLOT_TIMEOUT_SECONDS] = str(args.conversion_slot_timeout_seconds)
    os.environ[ENV_CACHE_CLEANUP_INTERVAL_SECONDS] = str(args.cache_cleanup_interval_seconds)
    os.makedirs(cache_dir, exist_ok=True)

    logger.info("Optimized WebP directories: %s", ", ".join(IMGSBACKUP_DIRS))
    logger.info("Serving images from: %s", imgs_dir)
    logger.info("Cache directory: %s", cache_dir)
    logger.info("Cache max age (hours): %s", args.cache_max_age_hours)
    logger.info(
        "Cache max bytes: %s",
        "disabled" if args.cache_max_bytes == 0 else _format_bytes(args.cache_max_bytes),
    )
    logger.info("Stale lock max age (hours): %s", args.stale_lock_age_hours)
    logger.info("Minimum free bytes for conversions: %s", _format_bytes(args.min_free_bytes))
    logger.info("Minimum free bytes for health: %s", _format_bytes(args.health_min_free_bytes))
    logger.info("Conversion slots: %s", args.conversion_slots)
    logger.info("Conversion slot timeout seconds: %s", args.conversion_slot_timeout_seconds)
    logger.info("Cache cleanup interval seconds: %s", args.cache_cleanup_interval_seconds)
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
