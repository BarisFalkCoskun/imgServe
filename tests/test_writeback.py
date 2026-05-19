from unittest.mock import patch
from fastapi.testclient import TestClient

import server


def test_first_request_writes_webp_to_thumbnails(client, put_source, env_dirs):
    src = put_source("solid.psd", folder="demo")
    r = client.get(f"/imgs/demo/{src.name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"

    out = env_dirs["thumbnails"] / "demo" / "solid.webp"
    assert out.is_file()
    assert out.stat().st_size > 0
    # body equals what was promoted
    assert r.content == out.read_bytes()


def test_second_request_skips_conversion(client, put_source, env_dirs):
    src = put_source("solid.psd", folder="demo")
    # prime cache
    r1 = client.get(f"/imgs/demo/{src.name}")
    assert r1.status_code == 200

    # second request must not invoke convert_image
    with patch.object(server, "convert_image") as mock_convert:
        r2 = client.get(f"/imgs/demo/{src.name}")
    assert r2.status_code == 200
    assert mock_convert.call_count == 0
    assert r2.headers["content-type"] == "image/webp"


def test_webp_source_with_existing_backup_serves_backup(client, put_source, put_backup, env_dirs):
    # source exists, but a WebP is already in the backup — backup wins.
    src = put_source("solid.psd", folder="demo")
    backup_file = put_backup("solid_source.webp", folder="demo", as_name="solid.webp")

    with patch.object(server, "convert_image") as mock_convert:
        r = client.get(f"/imgs/demo/{src.name}")
    assert r.status_code == 200
    assert r.content == backup_file.read_bytes()
    assert mock_convert.call_count == 0


def test_existing_backup_uses_runtime_thumbnails_dir_without_source(env_dirs, put_backup, monkeypatch):
    backup_file = put_backup("solid_source.webp", folder="demo", as_name="solid.webp")
    monkeypatch.setenv(server.ENV_IMGS_DIR, str(env_dirs["imgs"]))
    monkeypatch.setenv(server.ENV_THUMBNAILS_DIR, str(env_dirs["thumbnails"]))
    monkeypatch.setenv(server.ENV_STATE_DIR, str(env_dirs["state"]))

    app = server.create_app()
    with TestClient(app) as test_client:
        with patch.object(server, "convert_image") as mock_convert:
            r = test_client.get("/imgs/demo/solid.psd")

    assert r.status_code == 200
    assert r.content == backup_file.read_bytes()
    assert mock_convert.call_count == 0


def test_web_source_is_backfilled(client, put_source, env_dirs):
    # A .jpg with no existing backup should be converted to WebP and written back.
    src = put_source("solid.jpg", folder="demo")
    r = client.get(f"/imgs/demo/{src.name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"
    assert (env_dirs["thumbnails"] / "demo" / "solid.webp").is_file()
