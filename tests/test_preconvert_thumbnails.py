from pathlib import Path

import preconvert_thumbnails as preconvert


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

    assert stats == {
        "seen": 3,
        "queued": 1,
        "skipped_existing": 1,
        "skipped_passthrough": 1,
        "skipped_collision": 0,
    }
    assert [Path(task.source_path).name for task in tasks] == ["image.jpg"]
    assert Path(tasks[0].dest_path) == thumbnails_dir / "salling" / "image.webp"


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
