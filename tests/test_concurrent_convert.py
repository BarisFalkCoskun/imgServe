from concurrent.futures import ThreadPoolExecutor, as_completed


def test_concurrent_requests_for_same_source(client, put_source, env_dirs):
    src = put_source("solid.psd", folder="demo")

    def hit():
        r = client.get(f"/imgs/demo/{src.name}")
        return r.status_code, len(r.content), r.headers["content-type"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [f.result() for f in as_completed([pool.submit(hit) for _ in range(4)])]

    assert all(status == 200 for status, _, _ in results)
    sizes = {sz for _, sz, _ in results}
    cts = {ct for _, _, ct in results}
    assert sizes and len(sizes) == 1, f"divergent payload sizes: {sizes}"
    assert cts == {"image/webp"}

    # Exactly one canonical WebP in thumbnails, no stray .tmp files.
    thumbnails_demo = env_dirs["thumbnails"] / "demo"
    files = sorted(p.name for p in thumbnails_demo.iterdir())
    assert files == ["solid.webp"], files
