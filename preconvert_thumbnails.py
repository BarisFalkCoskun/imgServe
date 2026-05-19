#!/usr/bin/env python3
"""Preconvert source images into the canonical thumbnails WebP store.

This mirrors the default `/imgs/{folder}/{filename}` behavior from server.py:

1. Check `/mnt/storagebox/thumbnails/{folder}/{basename}.webp`.
2. If missing, prefetch the source from `/mnt/storagebox/imgs/{folder}/{filename}` to local SSD.
3. Skip passthrough/unknown types exactly like the server.
4. Convert image-like sources to WebP in /tmp.
5. Atomically promote the WebP into the thumbnails tree.

The source files are never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from converter import FORMAT_TO_EXT, convert_image, is_passthrough
from server import DEFAULT_IMGS_BASE, DEFAULT_THUMBNAILS_DIR, promote_to_thumbnails

DEFAULT_FOLDER = "salling"
DEFAULT_STATE_DIR = Path(__file__).resolve().parent / "state" / "preconvert"
DEFAULT_TMP_DIR = tempfile.gettempdir()
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2)))
DEFAULT_SCAN_LOG_INTERVAL = 1000
DEFAULT_SCAN_LOG_SECONDS = 10.0
EVENT_LOG_NAME = "events.jsonl"

logger = logging.getLogger("imgserve.preconvert")

_WORKER_CONTEXT: dict[str, Any] = {}
_WORKER_COUNTS: dict[str, int] = {
    "converted": 0,
    "failed": 0,
    "skipped_existing": 0,
    "skipped_passthrough": 0,
}


@dataclasses.dataclass(frozen=True)
class ConversionTask:
    source_path: str
    thumbnails_dir: str
    folder: str
    filename: str
    dest_path: str
    lock_path: str
    force: bool
    tmp_dir: str
    local_source_path: str | None = None
    prefetch_elapsed_seconds: float | None = None
    source_bytes: int | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def source_dir_for(source_root: Path, folder: str) -> Path:
    return source_root / folder


def thumbnail_path(thumbnails_dir: Path, folder: str, filename: str) -> Path:
    return thumbnails_dir / folder / f"{Path(filename).stem}.webp"


def lock_path_for(state_dir: Path, dest_path: Path) -> Path:
    digest = hashlib.sha256(str(dest_path).encode("utf-8")).hexdigest()
    return state_dir / "locks" / f"{digest}.lock"


def iter_source_files(source_dir: Path):
    for path in source_dir.iterdir():
        if path.is_file() and not path.name.startswith("."):
            yield path


def build_tasks(
    source_dir: Path,
    thumbnails_dir: Path,
    state_dir: Path,
    folder: str,
    force: bool,
    tmp_dir: Path,
    limit: int | None = None,
    scan_log_interval: int = DEFAULT_SCAN_LOG_INTERVAL,
    scan_log_seconds: float = DEFAULT_SCAN_LOG_SECONDS,
) -> tuple[list[ConversionTask], dict[str, Any]]:
    stats = {
        "seen": 0,
        "queued": 0,
        "skipped_existing": 0,
        "skipped_passthrough": 0,
        "skipped_collision": 0,
        "dest_exists_checks": 0,
        "dest_exists_elapsed_seconds": 0.0,
    }
    tasks: list[ConversionTask] = []
    queued_stems: set[str] = set()
    scan_started = time.monotonic()
    last_log_at = scan_started
    last_logged_seen = 0

    def emit_scan_progress(reason: str, source_path: Path | None = None, force_log: bool = False) -> None:
        nonlocal last_log_at, last_logged_seen
        now = time.monotonic()
        enough_files = stats["seen"] - last_logged_seen >= scan_log_interval
        enough_time = now - last_log_at >= scan_log_seconds
        if not force_log and not enough_files and not enough_time:
            return

        payload = {
            "phase": "scanning",
            "reason": reason,
            "updated_at": utc_now(),
            "source_dir": str(source_dir),
            "thumbnails_dir": str(thumbnails_dir),
            "folder": folder,
            "elapsed_seconds": round(now - scan_started, 3),
            "stats": dict(stats),
            "last_source_path": str(source_path) if source_path is not None else None,
        }
        write_json_atomic(state_dir / "scan.json", payload)
        append_event_to(state_dir / EVENT_LOG_NAME, {"event": "scan_progress", **payload})
        logger.info(
            "Scan progress: seen=%s queued=%s skipped_existing=%s skipped_passthrough=%s dest_exists_checks=%s dest_exists_seconds=%.3f last=%s",
            stats["seen"],
            stats["queued"],
            stats["skipped_existing"],
            stats["skipped_passthrough"],
            stats["dest_exists_checks"],
            stats["dest_exists_elapsed_seconds"],
            source_path,
        )
        last_log_at = now
        last_logged_seen = stats["seen"]

    logger.info("Scan started: source_dir=%s thumbnails_dir=%s folder=%s", source_dir, thumbnails_dir, folder)
    emit_scan_progress("start", force_log=True)

    for source_path in iter_source_files(source_dir):
        stats["seen"] += 1
        ext = source_path.suffix.lower()
        if is_passthrough(ext):
            stats["skipped_passthrough"] += 1
            emit_scan_progress("passthrough", source_path)
            continue

        stem = source_path.stem
        dest_path = thumbnail_path(thumbnails_dir, folder, source_path.name)
        exists_started = time.monotonic()
        dest_exists = dest_path.exists()
        stats["dest_exists_checks"] += 1
        stats["dest_exists_elapsed_seconds"] = round(
            stats["dest_exists_elapsed_seconds"] + time.monotonic() - exists_started,
            3,
        )
        if dest_exists and not force:
            stats["skipped_existing"] += 1
            emit_scan_progress("existing", source_path)
            continue

        if stem in queued_stems and not force:
            stats["skipped_collision"] += 1
            logger.warning(
                "Skipping basename collision for %s; destination would be %s",
                source_path,
                dest_path,
            )
            emit_scan_progress("collision", source_path)
            continue

        queued_stems.add(stem)
        tasks.append(
            ConversionTask(
                source_path=str(source_path),
                thumbnails_dir=str(thumbnails_dir),
                folder=folder,
                filename=source_path.name,
                dest_path=str(dest_path),
                lock_path=str(lock_path_for(state_dir, dest_path)),
                force=force,
                tmp_dir=str(tmp_dir),
            )
        )
        stats["queued"] += 1
        emit_scan_progress("queued", source_path)

        if limit is not None and len(tasks) >= limit:
            break

    emit_scan_progress("finish", force_log=True)
    return tasks, stats


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def append_event_to(event_log: str | Path | None, event: dict[str, Any]) -> None:
    if not event_log:
        return

    payload = {"ts": utc_now(), **event}
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with open(event_log, "a", encoding="utf-8") as handle:
        handle.write(line)


def append_event(event: dict[str, Any]) -> None:
    append_event_to(_WORKER_CONTEXT.get("event_log"), event)


def update_worker_status(status: str, task: ConversionTask | None = None, **extra: Any) -> None:
    status_dir = _WORKER_CONTEXT.get("status_dir")
    worker_id = _WORKER_CONTEXT.get("worker_id")
    if not status_dir or not worker_id:
        return

    payload: dict[str, Any] = {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "status": status,
        "updated_at": utc_now(),
        "counts": dict(_WORKER_COUNTS),
    }
    if task is not None:
        payload.update(
            {
                "source_path": task.source_path,
                "dest_path": task.dest_path,
                "folder": task.folder,
                "filename": task.filename,
            }
        )
    payload.update(extra)
    write_json_atomic(Path(status_dir) / f"{worker_id}.json", payload)


def init_worker(state_dir: str) -> None:
    worker_id = f"worker-{os.getpid()}"
    status_dir = str(Path(state_dir) / "workers")
    event_log = str(Path(state_dir) / EVENT_LOG_NAME)
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "worker_id": worker_id,
            "status_dir": status_dir,
            "event_log": event_log,
        }
    )
    for key in _WORKER_COUNTS:
        _WORKER_COUNTS[key] = 0
    update_worker_status("idle")


@contextmanager
def output_lock(lock_path: str):
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def process_task(task: ConversionTask) -> dict[str, Any]:
    started = time.monotonic()
    update_worker_status("converting", task, started_at=utc_now())
    append_event(
        {
            "event": "start",
            "worker_id": _WORKER_CONTEXT.get("worker_id"),
            "pid": os.getpid(),
            "source_path": task.source_path,
            "dest_path": task.dest_path,
        }
    )

    source_path = Path(task.source_path)
    conversion_source_path = Path(task.local_source_path) if task.local_source_path else source_path
    dest_path = Path(task.dest_path)

    try:
        lock_started = time.monotonic()
        with output_lock(task.lock_path):
            lock_wait_seconds = time.monotonic() - lock_started
            if dest_path.exists() and not task.force:
                _WORKER_COUNTS["skipped_existing"] += 1
                elapsed = time.monotonic() - started
                result = task_result(
                    task,
                    "skipped_existing",
                    elapsed,
                    lock_wait_seconds=round(lock_wait_seconds, 3),
                )
                append_event({"event": "finish", **result})
                update_worker_status("idle", task)
                return result

            tmp_fd, tmp_out = tempfile.mkstemp(prefix="preconvert-", suffix=FORMAT_TO_EXT["webp"], dir=task.tmp_dir)
            os.close(tmp_fd)
            try:
                convert_started = time.monotonic()
                success = convert_image(str(conversion_source_path), tmp_out, "webp")
                convert_elapsed_seconds = time.monotonic() - convert_started
                if not success:
                    _WORKER_COUNTS["failed"] += 1
                    elapsed = time.monotonic() - started
                    result = task_result(
                        task,
                        "failed",
                        elapsed,
                        lock_wait_seconds=round(lock_wait_seconds, 3),
                        convert_elapsed_seconds=round(convert_elapsed_seconds, 3),
                    )
                    append_event({"event": "finish", **result})
                    update_worker_status("idle", task)
                    return result

                promote_started = time.monotonic()
                promoted = promote_to_thumbnails(tmp_out, task.thumbnails_dir, task.folder, task.filename)
                promote_elapsed_seconds = time.monotonic() - promote_started
                elapsed = time.monotonic() - started
                if promoted:
                    _WORKER_COUNTS["converted"] += 1
                    result = task_result(
                        task,
                        "converted",
                        elapsed,
                        output_bytes=dest_path.stat().st_size,
                        lock_wait_seconds=round(lock_wait_seconds, 3),
                        convert_elapsed_seconds=round(convert_elapsed_seconds, 3),
                        promote_elapsed_seconds=round(promote_elapsed_seconds, 3),
                    )
                else:
                    _WORKER_COUNTS["failed"] += 1
                    result = task_result(
                        task,
                        "failed_promote",
                        elapsed,
                        lock_wait_seconds=round(lock_wait_seconds, 3),
                        convert_elapsed_seconds=round(convert_elapsed_seconds, 3),
                        promote_elapsed_seconds=round(promote_elapsed_seconds, 3),
                    )
                append_event({"event": "finish", **result})
                update_worker_status("idle", task)
                return result
            except Exception as exc:
                _WORKER_COUNTS["failed"] += 1
                elapsed = time.monotonic() - started
                result = task_result(task, "exception", elapsed, error=repr(exc))
                append_event({"event": "finish", **result})
                update_worker_status("idle", task, error=repr(exc))
                return result
            finally:
                try:
                    os.unlink(tmp_out)
                except OSError:
                    pass
    finally:
        if task.local_source_path:
            try:
                os.unlink(task.local_source_path)
            except OSError:
                pass


def task_result(task: ConversionTask, status: str, elapsed_seconds: float, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "source_path": task.source_path,
        "dest_path": task.dest_path,
        "filename": task.filename,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "pid": os.getpid(),
        "worker_id": _WORKER_CONTEXT.get("worker_id"),
        "prefetched": task.local_source_path is not None,
    }
    if task.local_source_path is not None:
        payload["local_source_path"] = task.local_source_path
    if task.prefetch_elapsed_seconds is not None:
        payload["prefetch_elapsed_seconds"] = task.prefetch_elapsed_seconds
    if task.source_bytes is not None:
        payload["source_bytes"] = task.source_bytes
    payload.update(extra)
    return payload


def write_summary(state_dir: Path, summary: dict[str, Any]) -> None:
    write_json_atomic(state_dir / "summary.json", summary)


def record_finished_task(summary: dict[str, Any], result: dict[str, Any]) -> None:
    status = result["status"]
    if status == "converted":
        summary["converted"] += 1
    elif status == "skipped_existing":
        summary["skipped_existing_after_lock"] += 1
    elif status == "prefetch_failed":
        summary["prefetch_failed"] += 1
        summary["failed"] += 1
        logger.warning("Source prefetch failed: %s", result)
    else:
        summary["failed"] += 1
        logger.warning("Task did not convert: %s", result)


def log_progress(completed: int, total: int, summary: dict[str, Any], state_dir: Path) -> None:
    logger.info(
        "Progress: %s/%s converted=%s failed=%s skipped_after_lock=%s prefetched=%s",
        completed,
        total,
        summary["converted"],
        summary["failed"],
        summary["skipped_existing_after_lock"],
        summary["prefetched"],
    )
    summary["updated_at"] = utc_now()
    write_summary(state_dir, summary)


def prefetch_task(task: ConversionTask, prefetch_dir: str, event_log: str) -> dict[str, Any]:
    started = time.monotonic()
    source_path = Path(task.source_path)
    digest = hashlib.sha256(task.source_path.encode("utf-8")).hexdigest()[:16]
    append_event_to(
        event_log,
        {
            "event": "prefetch_start",
            "source_path": task.source_path,
            "dest_path": task.dest_path,
        },
    )

    tmp_fd, tmp_source = tempfile.mkstemp(
        prefix=f"prefetch-{digest}-",
        suffix=source_path.suffix,
        dir=prefetch_dir,
    )
    os.close(tmp_fd)
    try:
        shutil.copyfile(source_path, tmp_source)
        source_bytes = Path(tmp_source).stat().st_size
        elapsed = time.monotonic() - started
        prefetched = dataclasses.replace(
            task,
            local_source_path=tmp_source,
            prefetch_elapsed_seconds=round(elapsed, 3),
            source_bytes=source_bytes,
        )
        result = {
            "status": "prefetched",
            "source_path": task.source_path,
            "dest_path": task.dest_path,
            "filename": task.filename,
            "prefetch_elapsed_seconds": round(elapsed, 3),
            "source_bytes": source_bytes,
        }
        append_event_to(event_log, {"event": "prefetch_finish", **result})
        return {"status": "prefetched", "task": prefetched, "result": result}
    except Exception as exc:
        try:
            os.unlink(tmp_source)
        except OSError:
            pass
        elapsed = time.monotonic() - started
        result = task_result(task, "prefetch_failed", elapsed, error=repr(exc))
        append_event_to(event_log, {"event": "prefetch_finish", **result})
        return {"status": "prefetch_failed", "task": task, "result": result}


def run_direct_conversion(
    tasks: list[ConversionTask],
    workers: int,
    state_dir: Path,
    summary: dict[str, Any],
) -> None:
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(str(state_dir),),
    ) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            completed += 1
            record_finished_task(summary, future.result())

            if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                log_progress(completed, len(tasks), summary, state_dir)


def run_prefetch_pipeline(
    tasks: list[ConversionTask],
    workers: int,
    state_dir: Path,
    summary: dict[str, Any],
    prefetch_dir: Path,
    prefetch_workers: int,
    prefetch_buffer: int,
) -> None:
    total = len(tasks)
    completed = 0
    task_iter = iter(tasks)
    ready_to_convert: deque[ConversionTask] = deque()
    pending_prefetch: dict[concurrent.futures.Future, ConversionTask] = {}
    pending_convert: dict[concurrent.futures.Future, ConversionTask] = {}
    event_log = str(state_dir / EVENT_LOG_NAME)

    def local_pipeline_size() -> int:
        return len(pending_prefetch) + len(ready_to_convert) + len(pending_convert)

    def submit_prefetches(prefetch_executor: concurrent.futures.ThreadPoolExecutor) -> None:
        while len(pending_prefetch) < prefetch_workers and local_pipeline_size() < prefetch_buffer:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            future = prefetch_executor.submit(prefetch_task, task, str(prefetch_dir), event_log)
            pending_prefetch[future] = task

    def submit_conversions(convert_executor: concurrent.futures.ProcessPoolExecutor) -> None:
        while ready_to_convert and len(pending_convert) < workers:
            task = ready_to_convert.popleft()
            future = convert_executor.submit(process_task, task)
            pending_convert[future] = task

    with concurrent.futures.ThreadPoolExecutor(max_workers=prefetch_workers) as prefetch_executor:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(str(state_dir),),
        ) as convert_executor:
            last_logged_completed = -1
            submit_prefetches(prefetch_executor)
            submit_conversions(convert_executor)

            while completed < total:
                wait_for = set(pending_prefetch) | set(pending_convert)
                if not wait_for:
                    break
                done, _ = concurrent.futures.wait(wait_for, return_when=concurrent.futures.FIRST_COMPLETED)

                for future in done:
                    if future in pending_prefetch:
                        pending_prefetch.pop(future)
                        prefetch_result = future.result()
                        result = prefetch_result["result"]
                        if prefetch_result["status"] == "prefetched":
                            summary["prefetched"] += 1
                            summary["prefetched_bytes"] += result.get("source_bytes", 0)
                            ready_to_convert.append(prefetch_result["task"])
                        else:
                            completed += 1
                            record_finished_task(summary, result)
                    else:
                        pending_convert.pop(future)
                        completed += 1
                        record_finished_task(summary, future.result())

                submit_conversions(convert_executor)
                submit_prefetches(prefetch_executor)

                should_log = completed > 0 and (completed == 1 or completed % 25 == 0 or completed == total)
                if should_log and completed != last_logged_completed:
                    log_progress(completed, total, summary, state_dir)
                    last_logged_completed = completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preconvert source images into the thumbnails WebP store")
    parser.add_argument("--source-root", default=DEFAULT_IMGS_BASE,
                        help=f"Source image root (default: {DEFAULT_IMGS_BASE})")
    parser.add_argument("--folder", default=DEFAULT_FOLDER,
                        help=f"Folder under source root and thumbnails dir (default: {DEFAULT_FOLDER})")
    parser.add_argument("--source-dir", default=None,
                        help="Explicit source directory; overrides --source-root/--folder")
    parser.add_argument("--thumbnails-dir", default=DEFAULT_THUMBNAILS_DIR,
                        help=f"Canonical WebP store (default: {DEFAULT_THUMBNAILS_DIR})")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR),
                        help=f"Status/log directory (default: {DEFAULT_STATE_DIR})")
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR,
                        help=f"Temporary conversion directory (default: {DEFAULT_TMP_DIR})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel conversion workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--prefetch-dir", default=None,
                        help="Local SSD directory for temporary source copies (default: <tmp-dir>/preconvert-prefetch)")
    parser.add_argument("--prefetch-workers", type=int, default=None,
                        help="Concurrent source-copy workers (default: min(4, --workers))")
    parser.add_argument("--prefetch-buffer", type=int, default=None,
                        help="Max source files in the local prefetch/conversion pipeline (default: max(workers * 2, workers + prefetch_workers))")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Convert directly from source paths instead of first copying sources to local SSD")
    parser.add_argument("--scan-log-interval", type=int, default=DEFAULT_SCAN_LOG_INTERVAL,
                        help=f"Log scan progress after this many source files (default: {DEFAULT_SCAN_LOG_INTERVAL})")
    parser.add_argument("--scan-log-seconds", type=float, default=DEFAULT_SCAN_LOG_SECONDS,
                        help=f"Log scan progress after this many seconds during scanning (default: {DEFAULT_SCAN_LOG_SECONDS:g})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only queue this many convertible missing files")
    parser.add_argument("--force", action="store_true",
                        help="Reconvert even when the destination WebP already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and print the plan without converting")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be > 0")
    if args.prefetch_workers is not None and args.prefetch_workers <= 0:
        parser.error("--prefetch-workers must be > 0")
    if args.prefetch_buffer is not None and args.prefetch_buffer <= 0:
        parser.error("--prefetch-buffer must be > 0")
    if args.scan_log_interval <= 0:
        parser.error("--scan-log-interval must be > 0")
    if args.scan_log_seconds <= 0:
        parser.error("--scan-log-seconds must be > 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")
    return args


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    source_root = Path(args.source_root).resolve()
    folder = args.folder.strip("/")
    source_dir = Path(args.source_dir).resolve() if args.source_dir else source_dir_for(source_root, folder)
    thumbnails_dir = Path(args.thumbnails_dir).resolve()
    state_dir = Path(args.state_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    prefetch_workers = args.prefetch_workers if args.prefetch_workers is not None else max(1, min(4, args.workers))
    prefetch_buffer = (
        args.prefetch_buffer
        if args.prefetch_buffer is not None
        else max(args.workers * 2, args.workers + prefetch_workers)
    )
    prefetch_dir = Path(args.prefetch_dir).resolve() if args.prefetch_dir else tmp_dir / "preconvert-prefetch"
    state_dir.mkdir(parents=True, exist_ok=True)
    workers_dir = state_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "locks").mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_prefetch:
        prefetch_dir.mkdir(parents=True, exist_ok=True)
    for stale_status in workers_dir.glob("*.json"):
        stale_status.unlink()
    event_log_path = state_dir / EVENT_LOG_NAME
    if event_log_path.exists():
        event_log_path.unlink()

    if not source_dir.is_dir():
        logger.error("Source directory does not exist: %s", source_dir)
        return 2

    tasks, scan_stats = build_tasks(
        source_dir=source_dir,
        thumbnails_dir=thumbnails_dir,
        state_dir=state_dir,
        folder=folder,
        force=args.force,
        tmp_dir=tmp_dir,
        limit=args.limit,
        scan_log_interval=args.scan_log_interval,
        scan_log_seconds=args.scan_log_seconds,
    )

    summary: dict[str, Any] = {
        "started_at": utc_now(),
        "source_dir": str(source_dir),
        "thumbnails_dir": str(thumbnails_dir),
        "folder": folder,
        "workers": args.workers,
        "force": args.force,
        "dry_run": args.dry_run,
        "prefetch": {
            "enabled": not args.no_prefetch,
            "dir": str(prefetch_dir) if not args.no_prefetch else None,
            "workers": prefetch_workers if not args.no_prefetch else 0,
            "buffer": prefetch_buffer if not args.no_prefetch else 0,
        },
        "scan": scan_stats,
        "prefetched": 0,
        "prefetch_failed": 0,
        "prefetched_bytes": 0,
        "converted": 0,
        "failed": 0,
        "skipped_existing_after_lock": 0,
    }
    write_summary(state_dir, summary)

    logger.info(
        "Scan complete: seen=%s queued=%s skipped_existing=%s skipped_passthrough=%s skipped_collision=%s",
        scan_stats["seen"],
        scan_stats["queued"],
        scan_stats["skipped_existing"],
        scan_stats["skipped_passthrough"],
        scan_stats["skipped_collision"],
    )
    logger.info("Worker status: %s", state_dir / "workers")
    logger.info("Event log: %s", state_dir / EVENT_LOG_NAME)
    if args.no_prefetch:
        logger.info("Prefetch disabled; conversion workers will read directly from source paths")
    else:
        logger.info(
            "Prefetch enabled: dir=%s workers=%s buffer=%s",
            prefetch_dir,
            prefetch_workers,
            prefetch_buffer,
        )

    if args.dry_run or not tasks:
        summary["finished_at"] = utc_now()
        write_summary(state_dir, summary)
        return 0

    if args.no_prefetch:
        run_direct_conversion(tasks, args.workers, state_dir, summary)
    else:
        run_prefetch_pipeline(
            tasks=tasks,
            workers=args.workers,
            state_dir=state_dir,
            summary=summary,
            prefetch_dir=prefetch_dir,
            prefetch_workers=prefetch_workers,
            prefetch_buffer=prefetch_buffer,
        )

    summary["finished_at"] = utc_now()
    write_summary(state_dir, summary)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
