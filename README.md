# imgServe — `groceryImgsOnly` test branch

This branch is a temporary, isolated image server for the grocery-only test dataset.
It intentionally differs from `master`:

- reads only from `/mnt/groceryImgsOnly/thumbnails/`;
- only accepts the folders `coop`, `salling`, `dagrofa`, and `rema1000`;
- serves only an exact image filename that already exists;
- never falls back to another image tree;
- never converts, writes, or creates thumbnails;
- defaults to `127.0.0.1:8101`, separate from the regular server on port `8100`.

If `/mnt/groceryImgsOnly/thumbnails/coop/example.jpg` exists, request it as:

```text
GET http://127.0.0.1:8101/imgs/coop/example.jpg
```

If that exact file does not exist, the response is `404 Image not found`. A similarly
named `.webp` file is not substituted for a missing `.jpg` file.

## Expected directory layout

```text
/mnt/groceryImgsOnly/thumbnails/
├── coop/
├── dagrofa/
├── rema1000/
└── salling/
```

The health endpoint reports `503` if the root or any required folder is missing or
unreadable:

```text
GET http://127.0.0.1:8101/health
```

## Setup and run

Python 3.10+ is required.

```bash
uv venv
uv pip install -r requirements.txt
uv run python server.py
```

The port can be overridden for local debugging, but do not use `8100` for this test
instance because that would remove the intended separation from the regular server.

## Tests

```bash
python3 -m pytest -q
```

The server tests are in `tests/test_server_helpers.py`.
