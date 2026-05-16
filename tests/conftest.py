"""Shared pytest fixtures.

Builds an isolated FastAPI app per test, pointing imgs/imgsbackup/state dirs
at tmp_path so tests never touch real mounts.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server

FIXTURES = Path(__file__).parent / "fixtures"


def _put(src_fixture: str, dst_dir: Path, dst_name: str | None = None) -> Path:
    src = FIXTURES / src_fixture
    if not src.is_file():
        pytest.skip(f"missing fixture {src}")
    dst = dst_dir / (dst_name or src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


@pytest.fixture
def env_dirs(tmp_path):
    imgs = tmp_path / "imgs"
    backup = tmp_path / "imgsbackup"
    state = tmp_path / "state"
    for d in (imgs, backup, state):
        d.mkdir(parents=True, exist_ok=True)
    return {"imgs": imgs, "backup": backup, "state": state}


@pytest.fixture
def app(env_dirs, monkeypatch):
    monkeypatch.setenv(server.ENV_IMGS_DIR, str(env_dirs["imgs"]))
    monkeypatch.setenv(server.ENV_IMGSBACKUP_PRIMARY, str(env_dirs["backup"]))
    monkeypatch.setenv(server.ENV_STATE_DIR, str(env_dirs["state"]))
    monkeypatch.setenv(server.ENV_CONVERSION_SLOTS, "2")
    monkeypatch.setenv(server.ENV_CONVERSION_SLOT_TIMEOUT_SECONDS, "5")
    monkeypatch.setattr(server, "IMGSBACKUP_READ_DIRS", [str(env_dirs["backup"])])
    return server.create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def put_source(env_dirs):
    """Place a fixture file into the source tree at imgs/<folder>/<name>."""
    def _put_source(fixture_name: str, folder: str = "demo", as_name: str | None = None) -> Path:
        folder_path = env_dirs["imgs"] / folder
        return _put(fixture_name, folder_path, as_name)
    return _put_source


@pytest.fixture
def put_backup(env_dirs):
    """Pre-place a WebP into imgsbackup/<folder>/<name>.webp."""
    def _put_backup(fixture_name: str, folder: str = "demo", as_name: str | None = None) -> Path:
        folder_path = env_dirs["backup"] / folder
        return _put(fixture_name, folder_path, as_name)
    return _put_backup
