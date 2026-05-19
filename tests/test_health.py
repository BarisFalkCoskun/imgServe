import os
import stat


def test_health_ok(client, env_dirs):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["thumbnails"]["ok"] is True
    assert body["checks"]["thumbnails"]["path"] == str(env_dirs["thumbnails"])


def test_health_503_when_thumbnails_readonly(client, env_dirs):
    os.chmod(env_dirs["thumbnails"], stat.S_IRUSR | stat.S_IXUSR)
    try:
        r = client.get("/health")
    finally:
        os.chmod(env_dirs["thumbnails"], stat.S_IRWXU)
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "error"
    assert body["checks"]["thumbnails"]["ok"] is False
