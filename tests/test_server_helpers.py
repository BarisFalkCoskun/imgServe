import logging
from pathlib import Path

import pytest

import server


def test_grocery_branch_uses_isolated_root_and_port():
    assert server.DEFAULT_IMAGES_DIR == "/mnt/groceryImgsOnly/thumbnails"
    assert server.DEFAULT_PORT == 8101
    assert server.DEFAULT_PORT != 8100


@pytest.mark.parametrize("folder", sorted(server.ALLOWED_FOLDERS))
def test_serves_exact_image_from_each_allowed_folder(client, put_image, folder):
    image = put_image(folder, "product.jpg", content=f"{folder}-image".encode())

    response = client.get(f"/imgs/{folder}/{image.name}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == server.IMMUTABLE_CACHE_CONTROL
    assert response.content == image.read_bytes()


def test_missing_image_returns_404_and_does_not_create_a_file(client, grocery_root, caplog):
    before = sorted(path.relative_to(grocery_root) for path in grocery_root.rglob("*"))

    with caplog.at_level(logging.INFO, logger="imgserve"):
        response = client.get("/imgs/coop/missing.jpg")

    after = sorted(path.relative_to(grocery_root) for path in grocery_root.rglob("*"))
    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}
    assert after == before
    assert "Image request miss" in caplog.text


def test_lookup_is_exact_and_does_not_substitute_webp(client, put_image):
    put_image("salling", "product.webp")

    response = client.get("/imgs/salling/product.jpg")

    assert response.status_code == 404


def test_rejects_folder_outside_grocery_allowlist(client, grocery_root):
    other = grocery_root / "other"
    other.mkdir()
    (other / "product.jpg").write_bytes(b"other-image")

    response = client.get("/imgs/other/product.jpg")

    assert response.status_code == 404


def test_rejects_non_image_file_even_inside_allowed_folder(client, grocery_root):
    document = grocery_root / "coop" / "notes.pdf"
    document.write_bytes(b"not-an-image")

    response = client.get("/imgs/coop/notes.pdf")

    assert response.status_code == 404


def test_rejects_symlink_to_image_outside_allowed_folder(client, grocery_root, tmp_path):
    external = tmp_path / "external.jpg"
    external.write_bytes(b"external-image")
    (grocery_root / "coop" / "external.jpg").symlink_to(external)

    response = client.get("/imgs/coop/external.jpg")

    assert response.status_code == 404


def test_health_reports_read_only_grocery_mode(client, grocery_root):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == "grocery-images-only"
    assert body["read_only"] is True
    assert body["checks"]["root"]["path"] == str(grocery_root)
    assert set(body["checks"]["folders"]) == server.ALLOWED_FOLDERS


def test_health_fails_when_required_folder_is_missing(grocery_root: Path):
    (grocery_root / "dagrofa").rmdir()
    app = server.create_app(images_dir=str(grocery_root))

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["checks"]["folders"]["dagrofa"]["ok"] is False
