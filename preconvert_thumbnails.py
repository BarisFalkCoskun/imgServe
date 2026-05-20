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
from collections.abc import Iterable
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
DEFAULT_SYNC_WORKERS = 2
DEFAULT_PRIORITY_LOG_INTERVAL = 1000
EVENT_LOG_NAME = "events.jsonl"

logger = logging.getLogger("imgserve.preconvert")

_WORKER_CONTEXT: dict[str, Any] = {}
_WORKER_COUNTS: dict[str, int] = {
    "converted": 0,
    "failed": 0,
    "skipped_existing": 0,
    "skipped_completed": 0,
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
    completed_paths: tuple[str, ...] = ()
    local_source_path: str | None = None
    prefetch_elapsed_seconds: float | None = None
    source_bytes: int | None = None
    source_stat_bytes_before: int | None = None
    source_stat_bytes_after: int | None = None
    prefetch_size_match: bool | None = None


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


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def first_existing_path(paths: Iterable[str | Path]) -> Path | None:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def priority_source_path(raw_entry: str, source_dir: Path, folder: str) -> Path | None:
    entry = raw_entry.strip().strip("\"'")
    if not entry or entry.startswith("#"):
        return None

    path = Path(entry)
    if path.is_absolute() and path.is_file():
        return path

    parts = [part for part in path.parts if part not in ("", os.sep)]
    if folder in parts:
        folder_index = parts.index(folder)
        tail = parts[folder_index + 1 :]
        if tail:
            return source_dir / Path(*tail)

    if path.name:
        return source_dir / path.name

    return None


def iter_source_files(source_dir: Path):
    for path in source_dir.iterdir():
        if path.is_file() and not path.name.startswith("."):
            yield path


class TaskScanner:
    def __init__(
        self,
        source_dir: Path,
        thumbnails_dir: Path,
        state_dir: Path,
        folder: str,
        force: bool,
        tmp_dir: Path,
        completed_thumbnails_dirs: Iterable[Path] = (),
        priority_list: Path | None = None,
        limit: int | None = None,
        scan_log_interval: int = DEFAULT_SCAN_LOG_INTERVAL,
        scan_log_seconds: float = DEFAULT_SCAN_LOG_SECONDS,
        priority_log_interval: int = DEFAULT_PRIORITY_LOG_INTERVAL,
    ) -> None:
        self.source_dir = source_dir
        self.thumbnails_dir = thumbnails_dir
        self.state_dir = state_dir
        self.folder = folder
        self.force = force
        self.tmp_dir = tmp_dir
        self.completed_thumbnails_dirs = unique_paths(
            path for path in completed_thumbnails_dirs if path != thumbnails_dir
        )
        self.priority_list = priority_list
        self.limit = limit
        self.scan_log_interval = scan_log_interval
        self.scan_log_seconds = scan_log_seconds
        self.priority_log_interval = priority_log_interval
        self.stats: dict[str, Any] = {
            "seen": 0,
            "queued": 0,
            "skipped_existing": 0,
            "skipped_completed": 0,
            "skipped_passthrough": 0,
            "skipped_collision": 0,
            "dest_exists_checks": 0,
            "dest_exists_elapsed_seconds": 0.0,
            "completed_exists_checks": 0,
            "completed_exists_elapsed_seconds": 0.0,
            "priority_entries": 0,
            "priority_unique_stems": 0,
            "priority_queued": 0,
            "priority_skipped_existing": 0,
            "priority_skipped_completed": 0,
            "priority_skipped_passthrough": 0,
            "priority_missing_source": 0,
            "priority_duplicate_stem": 0,
            "priority_collision": 0,
            "fallback_skipped_priority_stem": 0,
        }
        self.queued_stems: set[str] = set()
        self.priority_stems: set[str] = set()
        self.scan_started = time.monotonic()
        self.last_log_at = self.scan_started
        self.last_logged_seen = 0
        self.finished = False

    def emit_progress(self, reason: str, source_path: Path | None = None, force_log: bool = False) -> None:
        now = time.monotonic()
        enough_files = self.stats["seen"] - self.last_logged_seen >= self.scan_log_interval
        enough_time = now - self.last_log_at >= self.scan_log_seconds
        if not force_log and not enough_files and not enough_time:
            return

        payload = {
            "phase": "scanning",
            "reason": reason,
            "updated_at": utc_now(),
            "source_dir": str(self.source_dir),
            "thumbnails_dir": str(self.thumbnails_dir),
            "folder": self.folder,
            "elapsed_seconds": round(now - self.scan_started, 3),
            "stats": dict(self.stats),
            "last_source_path": str(source_path) if source_path is not None else None,
        }
        write_json_atomic(self.state_dir / "scan.json", payload)
        append_event_to(self.state_dir / EVENT_LOG_NAME, {"event": "scan_progress", **payload})
        logger.info(
            "Scan progress: seen=%s queued=%s priority_queued=%s skipped_existing=%s skipped_completed=%s skipped_passthrough=%s priority_missing=%s dest_exists_checks=%s completed_exists_checks=%s last=%s",
            self.stats["seen"],
            self.stats["queued"],
            self.stats["priority_queued"],
            self.stats["skipped_existing"],
            self.stats["skipped_completed"],
            self.stats["skipped_passthrough"],
            self.stats["priority_missing_source"],
            self.stats["dest_exists_checks"],
            self.stats["completed_exists_checks"],
            source_path,
        )
        self.last_log_at = now
        self.last_logged_seen = self.stats["seen"]

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.emit_progress("finish", force_log=True)
        logger.info(
            "Scan complete: seen=%s queued=%s priority_entries=%s priority_queued=%s skipped_existing=%s skipped_completed=%s skipped_passthrough=%s skipped_collision=%s priority_missing=%s",
            self.stats["seen"],
            self.stats["queued"],
            self.stats["priority_entries"],
            self.stats["priority_queued"],
            self.stats["skipped_existing"],
            self.stats["skipped_completed"],
            self.stats["skipped_passthrough"],
            self.stats["skipped_collision"],
            self.stats["priority_missing_source"],
        )

    def completed_paths_for(self, source_path: Path) -> tuple[str, ...]:
        return tuple(
            str(thumbnail_path(completed_dir, self.folder, source_path.name))
            for completed_dir in self.completed_thumbnails_dirs
        )

    def make_task(self, source_path: Path, dest_path: Path, completed_paths: tuple[str, ...]) -> ConversionTask:
        return ConversionTask(
            source_path=str(source_path),
            thumbnails_dir=str(self.thumbnails_dir),
            folder=self.folder,
            filename=source_path.name,
            dest_path=str(dest_path),
            lock_path=str(lock_path_for(self.state_dir, dest_path)),
            force=self.force,
            tmp_dir=str(self.tmp_dir),
            completed_paths=completed_paths,
        )

    def maybe_task_for_source(self, source_path: Path, phase: str):
        self.stats["seen"] += 1
        ext = source_path.suffix.lower()
        if is_passthrough(ext):
            self.stats["skipped_passthrough"] += 1
            if phase == "priority":
                self.stats["priority_skipped_passthrough"] += 1
            self.emit_progress(f"{phase}_passthrough", source_path)
            return None

        stem = source_path.stem
        dest_path = thumbnail_path(self.thumbnails_dir, self.folder, source_path.name)
        exists_started = time.monotonic()
        dest_exists = dest_path.exists()
        self.stats["dest_exists_checks"] += 1
        self.stats["dest_exists_elapsed_seconds"] = round(
            self.stats["dest_exists_elapsed_seconds"] + time.monotonic() - exists_started,
            3,
        )
        if dest_exists and not self.force:
            self.stats["skipped_existing"] += 1
            if phase == "priority":
                self.stats["priority_skipped_existing"] += 1
            self.emit_progress(f"{phase}_existing", source_path)
            return None

        completed_paths = self.completed_paths_for(source_path)
        completed_started = time.monotonic()
        completed_path = first_existing_path(completed_paths)
        self.stats["completed_exists_checks"] += len(completed_paths)
        self.stats["completed_exists_elapsed_seconds"] = round(
            self.stats["completed_exists_elapsed_seconds"] + time.monotonic() - completed_started,
            3,
        )
        if completed_path is not None and not self.force:
            self.stats["skipped_completed"] += 1
            if phase == "priority":
                self.stats["priority_skipped_completed"] += 1
            self.emit_progress(f"{phase}_completed", source_path)
            return None

        if stem in self.queued_stems and not self.force:
            self.stats["skipped_collision"] += 1
            if phase == "priority":
                self.stats["priority_collision"] += 1
            logger.warning(
                "Skipping basename collision for %s; destination would be %s",
                source_path,
                dest_path,
            )
            self.emit_progress(f"{phase}_collision", source_path)
            return None

        self.queued_stems.add(stem)
        self.stats["queued"] += 1
        if phase == "priority":
            self.stats["priority_queued"] += 1
        self.emit_progress(f"{phase}_queued", source_path)
        return self.make_task(source_path, dest_path, completed_paths)

    def iter_priority_tasks(self):
        if self.priority_list is None:
            return
        logger.info("Priority list started: %s", self.priority_list)
        append_event_to(
            self.state_dir / EVENT_LOG_NAME,
            {"event": "priority_start", "priority_list": str(self.priority_list)},
        )

        with open(self.priority_list, "r", encoding="utf-8") as handle:
            for line_number, raw_entry in enumerate(handle, 1):
                source_path = priority_source_path(raw_entry, self.source_dir, self.folder)
                if source_path is None:
                    continue

                self.stats["priority_entries"] += 1
                stem = source_path.stem
                if stem in self.priority_stems:
                    self.stats["priority_duplicate_stem"] += 1
                    continue

                self.priority_stems.add(stem)
                self.stats["priority_unique_stems"] = len(self.priority_stems)

                if self.stats["priority_entries"] % self.priority_log_interval == 0:
                    self.emit_progress("priority_progress", source_path, force_log=True)

                if not source_path.is_file():
                    self.stats["priority_missing_source"] += 1
                    append_event_to(
                        self.state_dir / EVENT_LOG_NAME,
                        {
                            "event": "priority_missing_source",
                            "line_number": line_number,
                            "entry": raw_entry.strip(),
                            "source_path": str(source_path),
                        },
                    )
                    continue

                task = self.maybe_task_for_source(source_path, "priority")
                if task is not None:
                    yield task
                    if self.limit is not None and self.stats["queued"] >= self.limit:
                        self.emit_progress("priority_limit", source_path, force_log=True)
                        return

        self.emit_progress("priority_finish", force_log=True)
        append_event_to(
            self.state_dir / EVENT_LOG_NAME,
            {"event": "priority_finish", "priority_list": str(self.priority_list), "stats": dict(self.stats)},
        )
        logger.info(
            "Priority list complete: entries=%s unique_stems=%s queued=%s skipped_existing=%s skipped_completed=%s missing=%s passthrough=%s duplicates=%s",
            self.stats["priority_entries"],
            self.stats["priority_unique_stems"],
            self.stats["priority_queued"],
            self.stats["priority_skipped_existing"],
            self.stats["priority_skipped_completed"],
            self.stats["priority_missing_source"],
            self.stats["priority_skipped_passthrough"],
            self.stats["priority_duplicate_stem"],
        )

    def iter_tasks(self):
        logger.info(
            "Scan started: source_dir=%s thumbnails_dir=%s folder=%s",
            self.source_dir,
            self.thumbnails_dir,
            self.folder,
        )
        self.emit_progress("start", force_log=True)

        yield from self.iter_priority_tasks()
        if self.limit is not None and self.stats["queued"] >= self.limit:
            self.finish()
            return

        for source_path in iter_source_files(self.source_dir):
            stem = source_path.stem
            if stem in self.priority_stems and not self.force:
                self.stats["fallback_skipped_priority_stem"] += 1
                continue

            task = self.maybe_task_for_source(source_path, "fallback")
            if task is not None:
                yield task

            if self.limit is not None and self.stats["queued"] >= self.limit:
                break

        self.finish()


def build_tasks(
    source_dir: Path,
    thumbnails_dir: Path,
    state_dir: Path,
    folder: str,
    force: bool,
    tmp_dir: Path,
    completed_thumbnails_dirs: Iterable[Path] = (),
    priority_list: Path | None = None,
    limit: int | None = None,
    scan_log_interval: int = DEFAULT_SCAN_LOG_INTERVAL,
    scan_log_seconds: float = DEFAULT_SCAN_LOG_SECONDS,
    priority_log_interval: int = DEFAULT_PRIORITY_LOG_INTERVAL,
) -> tuple[list[ConversionTask], dict[str, Any]]:
    scanner = TaskScanner(
        source_dir=source_dir,
        thumbnails_dir=thumbnails_dir,
        state_dir=state_dir,
        folder=folder,
        force=force,
        tmp_dir=tmp_dir,
        completed_thumbnails_dirs=completed_thumbnails_dirs,
        priority_list=priority_list,
        limit=limit,
        scan_log_interval=scan_log_interval,
        scan_log_seconds=scan_log_seconds,
        priority_log_interval=priority_log_interval,
    )
    tasks = list(scanner.iter_tasks())
    return tasks, dict(scanner.stats)


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


def copy_file_atomic(source_path: Path, dest_path: Path) -> int:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{dest_path.name}.", suffix=".tmp", dir=dest_path.parent)
    os.close(fd)
    try:
        shutil.copyfile(source_path, tmp_path)
        os.replace(tmp_path, dest_path)
        return dest_path.stat().st_size
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

            completed_path = first_existing_path(task.completed_paths)
            if completed_path is not None and not task.force:
                _WORKER_COUNTS["skipped_completed"] += 1
                elapsed = time.monotonic() - started
                result = task_result(
                    task,
                    "skipped_completed",
                    elapsed,
                    completed_path=str(completed_path),
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
        "folder": task.folder,
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
    if task.source_stat_bytes_before is not None:
        payload["source_stat_bytes_before"] = task.source_stat_bytes_before
    if task.source_stat_bytes_after is not None:
        payload["source_stat_bytes_after"] = task.source_stat_bytes_after
    if task.prefetch_size_match is not None:
        payload["prefetch_size_match"] = task.prefetch_size_match
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
    elif status == "skipped_completed":
        summary["skipped_completed_after_lock"] += 1
    elif status == "prefetch_failed":
        summary["prefetch_failed"] += 1
        summary["failed"] += 1
        logger.warning("Source prefetch failed: %s", result)
    else:
        summary["failed"] += 1
        logger.warning("Task did not convert: %s", result)


def record_sync_result(summary: dict[str, Any], result: dict[str, Any]) -> None:
    status = result["status"]
    if status == "synced":
        summary["synced"] += 1
    elif status == "already_synced":
        summary["already_synced"] += 1
    else:
        summary["sync_failed"] += 1
        summary["failed"] += 1
        logger.warning("Sync failed: %s", result)

    if result.get("local_deleted"):
        summary["local_deleted_after_sync"] += 1
    elif result.get("delete_error"):
        summary["local_delete_failed"] += 1
        summary["failed"] += 1
        logger.warning("Local delete after sync failed: %s", result)


def log_progress(completed: int, total: int | None, summary: dict[str, Any], state_dir: Path) -> None:
    total_label = str(total) if total is not None else "scanning"
    logger.info(
        "Progress: completed=%s total=%s queued=%s converted=%s synced=%s failed=%s skipped_after_lock=%s skipped_completed_after_lock=%s prefetched=%s pending_sync=%s",
        completed,
        total_label,
        summary["scan"]["queued"],
        summary["converted"],
        summary["synced"] + summary["already_synced"],
        summary["failed"],
        summary["skipped_existing_after_lock"],
        summary["skipped_completed_after_lock"],
        summary["prefetched"],
        summary.get("pending_sync", 0),
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
        try:
            source_stat_bytes_before = source_path.stat().st_size
        except OSError:
            source_stat_bytes_before = None
        shutil.copyfile(source_path, tmp_source)
        source_bytes = Path(tmp_source).stat().st_size
        try:
            source_stat_bytes_after = source_path.stat().st_size
        except OSError:
            source_stat_bytes_after = None
        source_sizes = [
            size
            for size in (source_stat_bytes_before, source_stat_bytes_after)
            if size is not None
        ]
        prefetch_size_match = not source_sizes or all(size == source_bytes for size in source_sizes)
        if not prefetch_size_match:
            logger.warning(
                "Prefetch size mismatch: source=%s before=%s after=%s local=%s",
                task.source_path,
                source_stat_bytes_before,
                source_stat_bytes_after,
                source_bytes,
            )
        elapsed = time.monotonic() - started
        prefetched = dataclasses.replace(
            task,
            local_source_path=tmp_source,
            prefetch_elapsed_seconds=round(elapsed, 3),
            source_bytes=source_bytes,
            source_stat_bytes_before=source_stat_bytes_before,
            source_stat_bytes_after=source_stat_bytes_after,
            prefetch_size_match=prefetch_size_match,
        )
        result = {
            "status": "prefetched",
            "source_path": task.source_path,
            "dest_path": task.dest_path,
            "filename": task.filename,
            "prefetch_elapsed_seconds": round(elapsed, 3),
            "source_bytes": source_bytes,
            "source_stat_bytes_before": source_stat_bytes_before,
            "source_stat_bytes_after": source_stat_bytes_after,
            "prefetch_size_match": prefetch_size_match,
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


def sync_converted_result(
    result: dict[str, Any],
    sync_to_thumbnails_dir: str,
    delete_local_after_sync: bool,
    event_log: str,
) -> dict[str, Any]:
    started = time.monotonic()
    local_path = Path(result["dest_path"])
    final_path = thumbnail_path(Path(sync_to_thumbnails_dir), result["folder"], result["filename"])
    append_event_to(
        event_log,
        {
            "event": "sync_start",
            "source_path": result["source_path"],
            "local_path": str(local_path),
            "final_path": str(final_path),
            "filename": result["filename"],
        },
    )

    try:
        if final_path.exists():
            status = "already_synced"
            final_bytes = final_path.stat().st_size
        else:
            final_bytes = copy_file_atomic(local_path, final_path)
            status = "synced"

        local_deleted = False
        delete_error = None
        if delete_local_after_sync:
            try:
                local_path.unlink()
                local_deleted = True
            except FileNotFoundError:
                local_deleted = True
            except OSError as exc:
                delete_error = repr(exc)

        payload = {
            "status": status,
            "source_path": result["source_path"],
            "filename": result["filename"],
            "local_path": str(local_path),
            "final_path": str(final_path),
            "final_bytes": final_bytes,
            "local_deleted": local_deleted,
            "sync_elapsed_seconds": round(time.monotonic() - started, 3),
        }
        if delete_error is not None:
            payload["delete_error"] = delete_error
        append_event_to(event_log, {"event": "sync_finish", **payload})
        return payload
    except Exception as exc:
        payload = {
            "status": "sync_failed",
            "source_path": result["source_path"],
            "filename": result["filename"],
            "local_path": str(local_path),
            "final_path": str(final_path),
            "sync_elapsed_seconds": round(time.monotonic() - started, 3),
            "error": repr(exc),
        }
        append_event_to(event_log, {"event": "sync_finish", **payload})
        return payload


def submit_sync_if_needed(
    result: dict[str, Any],
    sync_executor: concurrent.futures.ThreadPoolExecutor | None,
    pending_sync: dict[concurrent.futures.Future, dict[str, Any]],
    sync_to_thumbnails_dir: Path | None,
    delete_local_after_sync: bool,
    event_log: str,
) -> None:
    if sync_executor is None or sync_to_thumbnails_dir is None or result.get("status") != "converted":
        return

    future = sync_executor.submit(
        sync_converted_result,
        result,
        str(sync_to_thumbnails_dir),
        delete_local_after_sync,
        event_log,
    )
    pending_sync[future] = result


def run_direct_conversion(
    tasks: Iterable[ConversionTask],
    workers: int,
    state_dir: Path,
    summary: dict[str, Any],
    sync_to_thumbnails_dir: Path | None,
    delete_local_after_sync: bool,
    sync_workers: int,
    sync_buffer: int,
) -> None:
    completed = 0
    source_exhausted = False
    task_iter = iter(tasks)
    pending: dict[concurrent.futures.Future, ConversionTask] = {}
    pending_sync: dict[concurrent.futures.Future, dict[str, Any]] = {}
    event_log = str(state_dir / EVENT_LOG_NAME)
    sync_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=sync_workers)
        if sync_to_thumbnails_dir is not None
        else None
    )

    def submit_more(executor: concurrent.futures.ProcessPoolExecutor) -> None:
        nonlocal source_exhausted
        if sync_executor is not None and len(pending_sync) >= sync_buffer:
            return
        while not source_exhausted and len(pending) < max(workers * 2, workers):
            if sync_executor is not None and len(pending_sync) >= sync_buffer:
                return
            try:
                task = next(task_iter)
            except StopIteration:
                source_exhausted = True
                return
            pending[executor.submit(process_task, task)] = task

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(str(state_dir),),
        ) as executor:
            submit_more(executor)
            while pending or pending_sync or not source_exhausted:
                wait_for = set(pending) | set(pending_sync)
                if not wait_for:
                    submit_more(executor)
                    wait_for = set(pending) | set(pending_sync)
                if not wait_for:
                    break

                done, _ = concurrent.futures.wait(wait_for, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    if future in pending:
                        pending.pop(future)
                        completed += 1
                        result = future.result()
                        record_finished_task(summary, result)
                        submit_sync_if_needed(
                            result,
                            sync_executor,
                            pending_sync,
                            sync_to_thumbnails_dir,
                            delete_local_after_sync,
                            event_log,
                        )
                    else:
                        pending_sync.pop(future)
                        record_sync_result(summary, future.result())

                summary["pending_sync"] = len(pending_sync)
                submit_more(executor)

                total = summary["scan"]["queued"] if source_exhausted else None
                if completed == 1 or completed % 25 == 0 or (source_exhausted and completed == total and not pending_sync):
                    log_progress(completed, total, summary, state_dir)
    finally:
        if sync_executor is not None:
            sync_executor.shutdown()


def run_prefetch_pipeline(
    tasks: Iterable[ConversionTask],
    workers: int,
    state_dir: Path,
    summary: dict[str, Any],
    prefetch_dir: Path,
    prefetch_workers: int,
    prefetch_buffer: int,
    sync_to_thumbnails_dir: Path | None,
    delete_local_after_sync: bool,
    sync_workers: int,
    sync_buffer: int,
) -> None:
    completed = 0
    task_iter = iter(tasks)
    source_exhausted = False
    ready_to_convert: deque[ConversionTask] = deque()
    pending_prefetch: dict[concurrent.futures.Future, ConversionTask] = {}
    pending_convert: dict[concurrent.futures.Future, ConversionTask] = {}
    pending_sync: dict[concurrent.futures.Future, dict[str, Any]] = {}
    event_log = str(state_dir / EVENT_LOG_NAME)
    sync_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=sync_workers)
        if sync_to_thumbnails_dir is not None
        else None
    )

    def local_pipeline_size() -> int:
        return len(pending_prefetch) + len(ready_to_convert) + len(pending_convert) + len(pending_sync)

    def submit_prefetches(prefetch_executor: concurrent.futures.ThreadPoolExecutor) -> None:
        nonlocal source_exhausted
        if sync_executor is not None and len(pending_sync) >= sync_buffer:
            return
        while not source_exhausted and len(pending_prefetch) < prefetch_workers and local_pipeline_size() < prefetch_buffer:
            if sync_executor is not None and len(pending_sync) >= sync_buffer:
                return
            try:
                task = next(task_iter)
            except StopIteration:
                source_exhausted = True
                return
            future = prefetch_executor.submit(prefetch_task, task, str(prefetch_dir), event_log)
            pending_prefetch[future] = task

    def submit_conversions(convert_executor: concurrent.futures.ProcessPoolExecutor) -> None:
        if sync_executor is not None and len(pending_sync) >= sync_buffer:
            return
        while ready_to_convert and len(pending_convert) < workers:
            if sync_executor is not None and len(pending_sync) >= sync_buffer:
                return
            task = ready_to_convert.popleft()
            future = convert_executor.submit(process_task, task)
            pending_convert[future] = task

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=prefetch_workers) as prefetch_executor:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=(str(state_dir),),
            ) as convert_executor:
                last_logged_completed = -1
                submit_prefetches(prefetch_executor)
                submit_conversions(convert_executor)

                while not source_exhausted or pending_prefetch or ready_to_convert or pending_convert or pending_sync:
                    wait_for = set(pending_prefetch) | set(pending_convert) | set(pending_sync)
                    if not wait_for:
                        submit_prefetches(prefetch_executor)
                        submit_conversions(convert_executor)
                        wait_for = set(pending_prefetch) | set(pending_convert) | set(pending_sync)
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
                        elif future in pending_convert:
                            pending_convert.pop(future)
                            completed += 1
                            result = future.result()
                            record_finished_task(summary, result)
                            submit_sync_if_needed(
                                result,
                                sync_executor,
                                pending_sync,
                                sync_to_thumbnails_dir,
                                delete_local_after_sync,
                                event_log,
                            )
                        else:
                            pending_sync.pop(future)
                            record_sync_result(summary, future.result())

                    summary["pending_sync"] = len(pending_sync)
                    submit_conversions(convert_executor)
                    submit_prefetches(prefetch_executor)

                    total = summary["scan"]["queued"] if source_exhausted else None
                    should_log = completed > 0 and (
                        completed == 1
                        or completed % 25 == 0
                        or (source_exhausted and completed == total and not pending_sync)
                    )
                    if should_log and completed != last_logged_completed:
                        log_progress(completed, total, summary, state_dir)
                        last_logged_completed = completed
    finally:
        if sync_executor is not None:
            sync_executor.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preconvert source images into the thumbnails WebP store")
    parser.add_argument("--source-root", default=DEFAULT_IMGS_BASE,
                        help=f"Source image root (default: {DEFAULT_IMGS_BASE})")
    parser.add_argument("--folder", default=DEFAULT_FOLDER,
                        help=f"Folder under source root and thumbnails dir (default: {DEFAULT_FOLDER})")
    parser.add_argument("--source-dir", default=None,
                        help="Explicit source directory; overrides --source-root/--folder")
    parser.add_argument("--priority-list", default=None,
                        help="Optional newline-separated source path list to convert before the remaining folder")
    parser.add_argument("--thumbnails-dir", default=DEFAULT_THUMBNAILS_DIR,
                        help=f"Canonical WebP store (default: {DEFAULT_THUMBNAILS_DIR})")
    parser.add_argument("--completed-thumbnails-dir", action="append", default=[],
                        help="Additional WebP store to treat as already completed during scan; can be repeated")
    parser.add_argument("--sync-to-thumbnails-dir", default=None,
                        help="After local conversion, copy each WebP to this final thumbnails store")
    parser.add_argument("--delete-local-after-sync", action="store_true",
                        help="Delete the local converted WebP only after it exists in --sync-to-thumbnails-dir")
    parser.add_argument("--sync-workers", type=int, default=DEFAULT_SYNC_WORKERS,
                        help=f"Concurrent final-copy workers when --sync-to-thumbnails-dir is set (default: {DEFAULT_SYNC_WORKERS})")
    parser.add_argument("--sync-buffer", type=int, default=None,
                        help="Max converted files waiting for final sync before throttling new work (default: max(workers * 2, sync_workers * 4))")
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
    parser.add_argument("--priority-log-interval", type=int, default=DEFAULT_PRIORITY_LOG_INTERVAL,
                        help=f"Log priority-list progress after this many list entries (default: {DEFAULT_PRIORITY_LOG_INTERVAL})")
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
    if args.sync_workers <= 0:
        parser.error("--sync-workers must be > 0")
    if args.sync_buffer is not None and args.sync_buffer <= 0:
        parser.error("--sync-buffer must be > 0")
    if args.delete_local_after_sync and not args.sync_to_thumbnails_dir:
        parser.error("--delete-local-after-sync requires --sync-to-thumbnails-dir")
    if args.scan_log_interval <= 0:
        parser.error("--scan-log-interval must be > 0")
    if args.scan_log_seconds <= 0:
        parser.error("--scan-log-seconds must be > 0")
    if args.priority_log_interval <= 0:
        parser.error("--priority-log-interval must be > 0")
    if args.priority_list is not None and not Path(args.priority_list).is_file():
        parser.error(f"--priority-list does not exist or is not a file: {args.priority_list}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")
    return args


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    source_root = Path(args.source_root).resolve()
    folder = args.folder.strip("/")
    source_dir = Path(args.source_dir).resolve() if args.source_dir else source_dir_for(source_root, folder)
    priority_list = Path(args.priority_list).resolve() if args.priority_list else None
    thumbnails_dir = Path(args.thumbnails_dir).resolve()
    sync_to_thumbnails_dir = Path(args.sync_to_thumbnails_dir).resolve() if args.sync_to_thumbnails_dir else None
    completed_thumbnails_dirs = [Path(path).resolve() for path in args.completed_thumbnails_dir]
    if sync_to_thumbnails_dir is not None:
        completed_thumbnails_dirs.append(sync_to_thumbnails_dir)
    completed_thumbnails_dirs = unique_paths(completed_thumbnails_dirs)
    state_dir = Path(args.state_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    prefetch_workers = args.prefetch_workers if args.prefetch_workers is not None else max(1, min(4, args.workers))
    prefetch_buffer = (
        args.prefetch_buffer
        if args.prefetch_buffer is not None
        else max(args.workers * 2, args.workers + prefetch_workers)
    )
    sync_buffer = (
        args.sync_buffer
        if args.sync_buffer is not None
        else max(args.workers * 2, args.sync_workers * 4)
    )
    prefetch_dir = Path(args.prefetch_dir).resolve() if args.prefetch_dir else tmp_dir / "preconvert-prefetch"
    state_dir.mkdir(parents=True, exist_ok=True)
    workers_dir = state_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "locks").mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    if sync_to_thumbnails_dir is not None:
        sync_to_thumbnails_dir.mkdir(parents=True, exist_ok=True)
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

    scanner = TaskScanner(
        source_dir=source_dir,
        thumbnails_dir=thumbnails_dir,
        state_dir=state_dir,
        folder=folder,
        force=args.force,
        tmp_dir=tmp_dir,
        completed_thumbnails_dirs=completed_thumbnails_dirs,
        priority_list=priority_list,
        limit=args.limit,
        scan_log_interval=args.scan_log_interval,
        scan_log_seconds=args.scan_log_seconds,
        priority_log_interval=args.priority_log_interval,
    )

    summary: dict[str, Any] = {
        "started_at": utc_now(),
        "source_dir": str(source_dir),
        "priority_list": str(priority_list) if priority_list is not None else None,
        "thumbnails_dir": str(thumbnails_dir),
        "completed_thumbnails_dirs": [str(path) for path in completed_thumbnails_dirs],
        "folder": folder,
        "workers": args.workers,
        "force": args.force,
        "dry_run": args.dry_run,
        "sync": {
            "enabled": sync_to_thumbnails_dir is not None,
            "to_dir": str(sync_to_thumbnails_dir) if sync_to_thumbnails_dir is not None else None,
            "delete_local_after_sync": args.delete_local_after_sync,
            "workers": args.sync_workers if sync_to_thumbnails_dir is not None else 0,
            "buffer": sync_buffer if sync_to_thumbnails_dir is not None else 0,
        },
        "prefetch": {
            "enabled": not args.no_prefetch,
            "dir": str(prefetch_dir) if not args.no_prefetch else None,
            "workers": prefetch_workers if not args.no_prefetch else 0,
            "buffer": prefetch_buffer if not args.no_prefetch else 0,
        },
        "scan": scanner.stats,
        "prefetched": 0,
        "prefetch_failed": 0,
        "prefetched_bytes": 0,
        "converted": 0,
        "synced": 0,
        "already_synced": 0,
        "sync_failed": 0,
        "local_deleted_after_sync": 0,
        "local_delete_failed": 0,
        "pending_sync": 0,
        "failed": 0,
        "skipped_existing_after_lock": 0,
        "skipped_completed_after_lock": 0,
    }
    write_summary(state_dir, summary)
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
    if sync_to_thumbnails_dir is not None:
        logger.info(
            "Final sync enabled: to_dir=%s workers=%s buffer=%s delete_local_after_sync=%s",
            sync_to_thumbnails_dir,
            args.sync_workers,
            sync_buffer,
            args.delete_local_after_sync,
        )
    if priority_list is not None:
        logger.info("Priority list enabled: %s", priority_list)

    tasks = scanner.iter_tasks()

    if args.dry_run:
        list(tasks)
        summary["finished_at"] = utc_now()
        write_summary(state_dir, summary)
        return 0

    if args.no_prefetch:
        run_direct_conversion(
            tasks,
            args.workers,
            state_dir,
            summary,
            sync_to_thumbnails_dir,
            args.delete_local_after_sync,
            args.sync_workers,
            sync_buffer,
        )
    else:
        run_prefetch_pipeline(
            tasks=tasks,
            workers=args.workers,
            state_dir=state_dir,
            summary=summary,
            prefetch_dir=prefetch_dir,
            prefetch_workers=prefetch_workers,
            prefetch_buffer=prefetch_buffer,
            sync_to_thumbnails_dir=sync_to_thumbnails_dir,
            delete_local_after_sync=args.delete_local_after_sync,
            sync_workers=args.sync_workers,
            sync_buffer=sync_buffer,
        )

    summary["finished_at"] = utc_now()
    write_summary(state_dir, summary)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
