def test_format_png_does_not_write_back(client, put_source, env_dirs):
    src = put_source("solid.psd", folder="demo")
    r = client.get(f"/imgs/demo/{src.name}?format=png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    # No WebP written to imgsbackup
    assert list((env_dirs["backup"]).rglob("*.webp")) == []
    # And no spurious PNG either
    assert list((env_dirs["backup"]).rglob("*.png")) == []


def test_format_jpg_does_not_write_back(client, put_source, env_dirs):
    src = put_source("solid.psd", folder="demo")
    r = client.get(f"/imgs/demo/{src.name}?format=jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert list((env_dirs["backup"]).rglob("*")) == []


def test_unsupported_format_400(client, put_source):
    src = put_source("solid.psd", folder="demo")
    r = client.get(f"/imgs/demo/{src.name}?format=xyz")
    assert r.status_code == 400
