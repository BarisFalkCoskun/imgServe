# imgServe

Internal image server with write-through conversion. Serves canonical WebP from `/mnt/imgsbackup/imgs3/` when available; on miss, converts the source from `/mnt/storagebox/imgs/` to WebP in `/tmp`, streams it, and atomically promotes the result back into `/mnt/imgsbackup/imgs3/` so future requests skip conversion.

Camera RAW (`.nef/.arw/.dng/.cr2/.cr3/...`) is decoded via libraw (`rawpy`) for correct demosaicing — no channel-separation artifacts. Non-image types (`.mp4/.mov/.m4v/.html/.pdf` and any unrecognized extension) stream through with the right `Content-Type`.

Binds to `127.0.0.1` only.

## Requirements

**System packages** (Homebrew / apt names):
- `ffmpeg` — TIFF fallback + optional video fixture generation
- `imagemagick` — broader format fallback
- `libtiff` (`tiffcp` binary) — repairs broken TIFFs before Pillow
- `jxrlib` (`JxrDecApp`, optional) — JPEG-XR support

macOS:

```bash
brew install ffmpeg imagemagick libtiff jxrlib
```

Debian/Ubuntu:

```bash
sudo apt install ffmpeg imagemagick libtiff-tools
sudo add-apt-repository universe
sudo apt update
sudo apt install libjxr-tools
```

**Python:** 3.10+ (uses union types like `str | None`).

## Setup with uv

[uv](https://docs.astral.sh/uv/) handles Python + venv + deps in one tool.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create a project venv and install deps

From the repo root:

```bash
uv venv                          # creates .venv/ using a managed Python
uv pip install -r requirements.txt
```

`rawpy` is listed in `requirements.txt`. It pulls a libraw wheel — no system package needed. If the wheel fails for your platform, drop the line; the server runs without it (RAW decoding will fall through to the generic backends and may show channel separation, but everything else still works).

### 3. Activate or prefix every command

Either activate:

```bash
source .venv/bin/activate
```

…or prefix each call with `uv run`:

```bash
uv run python server.py --help
```

The examples below assume `uv run`.

## Running the server

Defaults bind `127.0.0.1:8100` and use `/mnt/storagebox/imgs/` + `/mnt/imgsbackup/imgs3/`:

```bash
uv run python server.py
```

Local dev with temp directories:

```bash
mkdir -p /tmp/imgserve/imgs/demo /tmp/imgserve/backup /tmp/imgserve/state
cp some-image.psd /tmp/imgserve/imgs/demo/

uv run python server.py \
  --imgs-dir /tmp/imgserve/imgs \
  --imgsbackup-dir /tmp/imgserve/backup \
  --state-dir /tmp/imgserve/state \
  --workers 1
```

Then:

```bash
curl -I http://127.0.0.1:8100/imgs/demo/some-image.psd
curl    http://127.0.0.1:8100/health
```

The first request to a source converts + writes back to `--imgsbackup-dir`. The second request reads straight from there with no conversion.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8100` | Port |
| `--imgs-dir` | `/mnt/storagebox/imgs` | Source images (read-only) |
| `--imgsbackup-dir` | `/mnt/imgsbackup/imgs3` | Canonical WebP store (read + write) |
| `--state-dir` | `./state` | Holds conversion-slot fcntl locks |
| `--health-min-free-bytes` | `1073741824` (1 GiB) | Threshold below which `/health` reports unhealthy |
| `--conversion-slots` | `1` | Max concurrent conversions across all workers on this host |
| `--conversion-slot-timeout-seconds` | `30` | Per-request wait for a slot before returning 503 |
| `--workers` | `2` | uvicorn worker count |

Every flag also has an env-var equivalent (`IMGSERVE_*`). See `--help`.

## Running the tests

```bash
uv run python tests/fixtures/make_fixtures.py    # generates small binary fixtures (once)
uv run python -m pytest -v
```

47 tests should pass. 2 RAW tests skip unless both `rawpy` is installed and a `tests/fixtures/sample.dng` is present.

The `ffmpeg` fixtures (`clip.mp4/.mov/.m4v`) are only generated if `ffmpeg` is on `PATH`; the corresponding `test_passthrough.py` tests skip when those fixtures are absent.

## Endpoints

| Endpoint | Behavior |
|---|---|
| `GET /imgs/{folder}/{filename}` | Serves WebP if cached in imgsbackup, else converts the source and writes back. `?format=png` and `?format=jpg` re-convert every time (not cached). |
| `GET /health` | 200 with check details on success; 503 if either source or imgsbackup write-probe fails. |

## Adding more file extensions

Edit `converter.py`:

- **Image format that needs conversion** → add to `CONVERT_FORMATS`. If it has its own decoder, write a `convert_with_<name>` function and insert it into the `backends` list in `convert_image`.
- **Streamed-as-is type** (video, document, anything we can't or shouldn't convert) → add to `PASSTHROUGH_EXTS` and `get_content_type`.
- **RAW format** → add to `RAW_EXTS`; the existing `convert_with_rawpy` will pick it up automatically.

`is_passthrough(ext)` falls back to "stream as-is" for any extension that isn't in `WEB_FORMATS`, `CONVERT_FORMATS`, or `RAW_EXTS`, so unknown types degrade gracefully.
