# imgServe

Internal image server with write-through conversion. Serves canonical WebP from `/mnt/storagebox/thumbnails/` when available; on miss, converts the source from `/mnt/storagebox/imgs/` to WebP in `/tmp`, streams it, and atomically promotes the result back into `/mnt/storagebox/thumbnails/` so future requests skip conversion.

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

Defaults bind `127.0.0.1:8100` and use `/mnt/storagebox/imgs/` + `/mnt/storagebox/thumbnails/`:

```bash
uv run python server.py
```

Local dev with temp directories:

```bash
mkdir -p /tmp/imgserve/imgs/demo /tmp/imgserve/thumbnails /tmp/imgserve/state
cp some-image.psd /tmp/imgserve/imgs/demo/

uv run python server.py \
  --imgs-dir /tmp/imgserve/imgs \
  --thumbnails-dir /tmp/imgserve/thumbnails \
  --state-dir /tmp/imgserve/state \
  --workers 1
```

Then:

```bash
curl -I http://127.0.0.1:8100/imgs/demo/some-image.psd
curl    http://127.0.0.1:8100/health
```

The first request to a source converts + writes back to `--thumbnails-dir`. The second request reads straight from there with no conversion.

## Preconverting thumbnails

For a large one-time backfill, run the preconverter on a stronger machine against the same mounted storage. It uses the same conversion rules as the server, never modifies source files, skips passthrough types such as videos/PDF/HTML, and writes canonical WebP files into the thumbnails tree.

By default the preconverter streams the scan into the worker pipeline: while it is still walking Storage Box and checking which WebPs already exist, a small thread pool copies upcoming source files into a bounded local SSD prefetch directory and separate process workers convert already-prefetched files. This keeps the CPUs busy without leaving a large permanent cache on the server; prefetched source copies are deleted after their conversion task finishes.

For low SSD usage, write converted WebPs to local SSD first, sync each completed WebP to Storage Box in the background, and delete the local WebP only after the final copy exists:

```bash
uv run python preconvert_thumbnails.py \
  --source-root /mnt/storagebox/imgs \
  --folder salling \
  --priority-list list.txt \
  --thumbnails-dir /local-ssd/thumbnails \
  --sync-to-thumbnails-dir /mnt/storagebox/thumbnails \
  --delete-local-after-sync \
  --tmp-dir /local-ssd/tmp \
  --prefetch-dir /local-ssd/prefetch \
  --workers 16 \
  --prefetch-workers 4 \
  --prefetch-buffer 40 \
  --sync-workers 2 \
  --sync-buffer 24
```

Progress and worker state are written under `state/preconvert/` by default:

```bash
cat state/preconvert/scan.json
cat state/preconvert/summary.json
ls state/preconvert/workers/
tail -f state/preconvert/events.jsonl
```

Use `--priority-list list.txt` to process likely-needed files first. The list can contain paths such as `/salling/<hash>.jpg`; each line is resolved against `--source-root/--folder`, and completion is checked by basename/stem, so `/salling/foo.jpg` is skipped if `foo.webp` already exists in the local or final thumbnails directory. After all priority entries are handled, the script continues with the remaining source files.

Use `--dry-run` to see how many files would be queued without converting. Existing `{basename}.webp` files are skipped unless `--force` is passed. When `--sync-to-thumbnails-dir` is set, that final directory is also treated as completed during scan/resume, so files already synced to Storage Box are skipped even if the local SSD copy has been deleted. Use `--no-prefetch` to disable local source prefetching and convert directly from the mounted source paths. During a large Storage Box scan, `scan.json` and `events.jsonl` are updated continuously; tune this with `--scan-log-interval`, `--scan-log-seconds`, and `--priority-log-interval`.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8100` | Port |
| `--imgs-dir` | `/mnt/storagebox/imgs` | Source images (read-only) |
| `--thumbnails-dir` | `/mnt/storagebox/thumbnails` | Canonical WebP store (read + write) |
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
| `GET /imgs/{folder}/{filename}` | Serves WebP if cached in thumbnails, else converts the source and writes back. `?format=png` and `?format=jpg` re-convert every time (not cached). |
| `GET /health` | 200 with check details on success; 503 if either source or thumbnails write-probe fails. |

## Adding more file extensions

Edit `converter.py`:

- **Image format that needs conversion** → add to `CONVERT_FORMATS`. If it has its own decoder, write a `convert_with_<name>` function and insert it into the `backends` list in `convert_image`.
- **Streamed-as-is type** (video, document, anything we can't or shouldn't convert) → add to `PASSTHROUGH_EXTS` and `get_content_type`.
- **RAW format** → add to `RAW_EXTS`; the existing `convert_with_rawpy` will pick it up automatically.

`is_passthrough(ext)` falls back to "stream as-is" for any extension that isn't in `WEB_FORMATS`, `CONVERT_FORMATS`, or `RAW_EXTS`, so unknown types degrade gracefully.
