"""Shared fixtures for the temporary groceryImgsOnly server branch."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def grocery_root(tmp_path: Path) -> Path:
    root = tmp_path / "groceryImgsOnly" / "thumbnails"
    for folder in server.ALLOWED_FOLDERS:
        (root / folder).mkdir(parents=True)
    return root


@pytest.fixture
def app(grocery_root: Path):
    return server.create_app(images_dir=str(grocery_root))


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def put_image(grocery_root: Path):
    def _put_image(folder: str, filename: str, content: bytes = b"image-bytes") -> Path:
        path = grocery_root / folder / filename
        path.write_bytes(content)
        return path

    return _put_image
