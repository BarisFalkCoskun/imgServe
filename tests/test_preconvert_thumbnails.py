import multiprocessing
from pathlib import Path

import preconvert_thumbnails as preconvert


def test_choose_process_start_method_auto_avoids_fork_when_possible():
    method = preconvert.choose_process_start_method("auto")
    available = multiprocessing.get_all_start_methods()
    if "forkserver" in available:
        assert method == "forkserver"
    elif "spawn" in available:
        assert method == "spawn"
    else:
        assert method in available


def test_choose_process_start_method_respects_requested_spawn_when_available():
    if "spawn" not in multiprocessing.get_all_start_methods():
        return

    assert preconvert.choose_process_start_method("spawn") == "spawn"


def test_build_tasks_skips_existing_and_passthrough(tmp_path):
    source_dir = tmp_path / "imgs" / "salling"
    thumbnails_dir = tmp_path / "thumbnails"
    state_dir = tmp_path / "state"
    source_dir.mkdir(parents=True)
    (thumbnails_dir / "salling").mkdir(parents=True)

    (source_dir / "image.jpg").write_bytes(b"jpg")
    (source_dir / "clip.mp4").write_bytes(b"mp4")
    (source_dir / "already.png").write_bytes(b"png")
    (thumbnails_dir / "salling" / "already.webp").write_bytes(b"existing")

    tasks, stats = preconvert.build_tasks(
        source_dir=source_dir,
        thumbnails_dir=thumbnails_dir,
        state_dir=state_dir,
        folder="salling",
        force=False,
        tmp_dir=tmp_path,
    )

    assert {
        key: stats[key]
        for key in (
            "seen",
            "queued",
            "skipped_existing",
            "skipped_passthrough",
            "skipped_collision",
            "dest_exists_checks",
        )
    } == {
        "seen": 3,
        "queued": 1,
        "skipped_existing": 1,
        "skipped_passthrough": 1,
        "skipped_collision": 0,
        "dest_exists_checks": 2,
    }
    assert stats["dest_exists_elapsed_seconds"] >= 0
    assert [Path(task.source_path).name for task in tasks] == ["image.jpg"]
    assert Path(tasks[0].dest_path) == thumbnails_dir / "salling" / "image.webp"


def test_task_scanner_yields_before_scanning_full_source_dir(tmp_path):
    source_dir = tmp_path / "imgs" / "salling"
    thumbnails_dir = tmp_path / "thumbnails"
    state_dir = tmp_path / "state"
    source_dir.mkdir(parents=True)
    thumbnails_dir.mkdir()
    state_dir.mkdir()

    (source_dir / "first.jpg").write_bytes(b"jpg")
    (source_dir / "second.png").write_bytes(b"png")

    scanner = preconvert.TaskScanner(
        source_dir=source_dir,
        thumbnails_dir=thumbnails_dir,
        state_dir=state_dir,
        folder="salling",
        force=False,
        tmp_dir=tmp_path,
    )

    first_task = next(scanner.iter_tasks())

    assert Path(first_task.source_path).name in {"first.jpg", "second.png"}
    assert scanner.stats["seen"] == 1
    assert scanner.stats["queued"] == 1


def test_build_tasks_skips_when_completed_thumbnail_exists_elsewhere(tmp_path):
    source_dir = tmp_path / "imgs" / "salling"
    local_thumbnails_dir = tmp_path / "local-thumbnails"
    completed_thumbnails_dir = tmp_path / "final-thumbnails"
    state_dir = tmp_path / "state"
    source_dir.mkdir(parents=True)
    (completed_thumbnails_dir / "salling").mkdir(parents=True)

    (source_dir / "already.jpg").write_bytes(b"jpg")
    (completed_thumbnails_dir / "salling" / "already.webp").write_bytes(b"done")

    tasks, stats = preconvert.build_tasks(
        source_dir=source_dir,
        thumbnails_dir=local_thumbnails_dir,
        completed_thumbnails_dirs=[completed_thumbnails_dir],
        state_dir=state_dir,
        folder="salling",
        force=False,
        tmp_dir=tmp_path,
    )

    assert tasks == []
    assert stats["queued"] == 0
    assert stats["skipped_completed"] == 1
    assert stats["completed_exists_checks"] == 1


def test_priority_list_tasks_are_queued_before_fallback(tmp_path):
    source_dir = tmp_path / "imgs" / "salling"
    thumbnails_dir = tmp_path / "thumbnails"
    state_dir = tmp_path / "state"
    priority_list = tmp_path / "list.txt"
    source_dir.mkdir(parents=True)

    (source_dir / "fallback.jpg").write_bytes(b"jpg")
    (source_dir / "first.jpg").write_bytes(b"jpg")
    (source_dir / "second.png").write_bytes(b"png")
    priority_list.write_text("/salling/second.png\n/salling/first.jpg\n", encoding="utf-8")

    tasks, stats = preconvert.build_tasks(
        source_dir=source_dir,
        thumbnails_dir=thumbnails_dir,
        priority_list=priority_list,
        state_dir=state_dir,
        folder="salling",
        force=False,
        tmp_dir=tmp_path,
    )

    assert [Path(task.source_path).name for task in tasks[:2]] == ["second.png", "first.jpg"]
    assert {Path(task.source_path).name for task in tasks} == {"second.png", "first.jpg", "fallback.jpg"}
    assert stats["priority_entries"] == 2
    assert stats["priority_queued"] == 2
    assert stats["fallback_skipped_priority_stem"] == 2


def test_priority_list_skips_final_thumbnail_even_when_local_missing(tmp_path):
    source_dir = tmp_path / "imgs" / "salling"
    local_thumbnails_dir = tmp_path / "local-thumbnails"
    final_thumbnails_dir = tmp_path / "final-thumbnails"
    state_dir = tmp_path / "state"
    priority_list = tmp_path / "list.txt"
    source_dir.mkdir(parents=True)
    (final_thumbnails_dir / "salling").mkdir(parents=True)

    (source_dir / "already.jpg").write_bytes(b"jpg")
    (final_thumbnails_dir / "salling" / "already.webp").write_bytes(b"done")
    priority_list.write_text("/salling/already.jpg\n", encoding="utf-8")

    tasks, stats = preconvert.build_tasks(
        source_dir=source_dir,
        thumbnails_dir=local_thumbnails_dir,
        completed_thumbnails_dirs=[final_thumbnails_dir],
        priority_list=priority_list,
        state_dir=state_dir,
        folder="salling",
        force=False,
        tmp_dir=tmp_path,
    )

    assert tasks == []
    assert stats["priority_entries"] == 1
    assert stats["priority_skipped_completed"] == 1
    assert stats["skipped_completed"] == 1
    assert stats["fallback_skipped_priority_stem"] == 1


def test_process_task_writes_thumbnail_without_modifying_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "imgs" / "salling"
    thumbnails_dir = tmp_path / "thumbnails"
    state_dir = tmp_path / "state"
    source_dir.mkdir(parents=True)
    thumbnails_dir.mkdir()
    state_dir.mkdir()
    source = source_dir / "image.jpg"
    original_bytes = b"source bytes"
    source.write_bytes(original_bytes)

    def fake_convert(src_path, dst_path, fmt):
        assert src_path == str(source)
        assert fmt == "webp"
        Path(dst_path).write_bytes(b"converted webp")
        return True

    monkeypatch.setattr(preconvert, "convert_image", fake_convert)
    preconvert.init_worker(str(state_dir))

    task = preconvert.ConversionTask(
        source_path=str(source),
        thumbnails_dir=str(thumbnails_dir),
        folder="salling",
        filename=source.name,
        dest_path=str(thumbnails_dir / "salling" / "image.webp"),
        lock_path=str(state_dir / "locks" / "image.lock"),
        force=False,
        tmp_dir=str(tmp_path),
    )

    result = preconvert.process_task(task)

    assert result["status"] == "converted"
    assert source.read_bytes() == original_bytes
    assert (thumbnails_dir / "salling" / "image.webp").read_bytes() == b"converted webp"
    assert list((thumbnails_dir / "salling").glob(".*.tmp")) == []


def test_prefetch_task_uses_local_copy_and_cleans_it_after_conversion(tmp_path, monkeypatch):
    source_dir = tmp_path / "imgs" / "salling"
    thumbnails_dir = tmp_path / "thumbnails"
    state_dir = tmp_path / "state"
    prefetch_dir = tmp_path / "prefetch"
    source_dir.mkdir(parents=True)
    thumbnails_dir.mkdir()
    state_dir.mkdir()
    prefetch_dir.mkdir()
    source = source_dir / "image.jpg"
    original_bytes = b"source bytes"
    source.write_bytes(original_bytes)

    task = preconvert.ConversionTask(
        source_path=str(source),
        thumbnails_dir=str(thumbnails_dir),
        folder="salling",
        filename=source.name,
        dest_path=str(thumbnails_dir / "salling" / "image.webp"),
        lock_path=str(state_dir / "locks" / "image.lock"),
        force=False,
        tmp_dir=str(tmp_path),
    )

    prefetch_result = preconvert.prefetch_task(task, str(prefetch_dir), str(state_dir / "events.jsonl"))

    assert prefetch_result["status"] == "prefetched"
    prefetched_task = prefetch_result["task"]
    local_source = Path(prefetched_task.local_source_path)
    assert local_source.read_bytes() == original_bytes
    assert local_source.parent == prefetch_dir
    assert prefetched_task.source_bytes == len(original_bytes)
    assert prefetched_task.source_stat_bytes_before == len(original_bytes)
    assert prefetched_task.source_stat_bytes_after == len(original_bytes)
    assert prefetched_task.prefetch_size_match is True

    def fake_convert(src_path, dst_path, fmt):
        assert src_path == str(local_source)
        assert fmt == "webp"
        Path(dst_path).write_bytes(b"converted webp")
        return True

    monkeypatch.setattr(preconvert, "convert_image", fake_convert)
    preconvert.init_worker(str(state_dir))

    result = preconvert.process_task(prefetched_task)

    assert result["status"] == "converted"
    assert result["prefetched"] is True
    assert result["prefetch_elapsed_seconds"] >= 0
    assert source.read_bytes() == original_bytes
    assert not local_source.exists()
    assert (thumbnails_dir / "salling" / "image.webp").read_bytes() == b"converted webp"


def test_sync_converted_result_copies_to_final_and_deletes_local(tmp_path):
    local_thumbnails_dir = tmp_path / "local-thumbnails"
    final_thumbnails_dir = tmp_path / "final-thumbnails"
    state_dir = tmp_path / "state"
    local_path = local_thumbnails_dir / "salling" / "image.webp"
    local_path.parent.mkdir(parents=True)
    state_dir.mkdir()
    local_path.write_bytes(b"converted webp")

    result = preconvert.sync_converted_result(
        {
            "status": "converted",
            "source_path": str(tmp_path / "imgs" / "salling" / "image.jpg"),
            "folder": "salling",
            "filename": "image.jpg",
            "dest_path": str(local_path),
        },
        str(final_thumbnails_dir),
        True,
        str(state_dir / "events.jsonl"),
    )

    assert result["status"] == "synced"
    assert result["local_deleted"] is True
    assert not local_path.exists()
    assert (final_thumbnails_dir / "salling" / "image.webp").read_bytes() == b"converted webp"
