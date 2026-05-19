import os
import stat
import logging


def test_readonly_thumbnails_still_serves_bytes(client, put_source, env_dirs, caplog):
    src = put_source("solid.psd", folder="demo")
    # Make the thumbnails root read-only so write-back fails.
    os.chmod(env_dirs["thumbnails"], stat.S_IRUSR | stat.S_IXUSR)
    try:
        with caplog.at_level(logging.WARNING, logger="imgserve"):
            r = client.get(f"/imgs/demo/{src.name}")
    finally:
        os.chmod(env_dirs["thumbnails"], stat.S_IRWXU)

    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"
    assert b"WEBP" in r.content or len(r.content) > 0
    assert any("Could not promote to thumbnails" in rec.message for rec in caplog.records)
    # And no WebP made it in
    assert not (env_dirs["thumbnails"] / "demo" / "solid.webp").exists()


def test_next_request_retries_conversion(client, put_source, env_dirs, monkeypatch):
    src = put_source("solid.psd", folder="demo")
    # Force a write-back failure for the first request only.
    real_promote = __import__("server").promote_to_thumbnails
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return real_promote(*args, **kwargs)

    monkeypatch.setattr("server.promote_to_thumbnails", flaky)

    r1 = client.get(f"/imgs/demo/{src.name}")
    assert r1.status_code == 200
    assert not (env_dirs["thumbnails"] / "demo" / "solid.webp").exists()

    r2 = client.get(f"/imgs/demo/{src.name}")
    assert r2.status_code == 200
    assert (env_dirs["thumbnails"] / "demo" / "solid.webp").is_file()
    assert calls["n"] == 2  # promotion attempted both times
