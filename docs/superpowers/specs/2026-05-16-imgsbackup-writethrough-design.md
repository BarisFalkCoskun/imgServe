# imgServe — write-through to `/mnt/imgsbackup/imgs3/` and RAW color fix

**Date:** 2026-05-16
**Status:** Approved by user (pending spec review)
**Scope:** `server.py`, `converter.py`, new `tests/`

## Problem

1. Every on-the-fly conversion is cached only on the local SSD. The same source file is re-converted indefinitely if its local cache entry is evicted by the size/age budgets. Local disk is the expensive resource; `/mnt/imgsbackup/imgs3/` is cheap and large.
2. Some conversions (especially camera RAW: `.nef`, `.arw`, `.dng`, `.cr2`, `.cr3`) come out as a tiled red/green/blue pattern. Root cause: the current backends (Pillow, ImageMagick `convert`, ffmpeg) read the Bayer-pattern sensor data as a flat 2D bitmap and never demosaic it.
3. The current request list now includes video and document types (`.mp4`, `.mov`, `.m4v`, `.html`, `.pdf`) that the server should serve as-is.

## Goals

- Make `/mnt/imgsbackup/imgs3/` the durable store for converted WebP. Once a source has been converted, future requests for any extension of that basename hit imgsbackup without conversion.
- Shrink the local cache footprint to ~zero by removing the persistent local cache entirely.
- Eliminate the RAW channel-separation artifact by adding a `rawpy`/libraw backend.
- Add explicit passthrough handling for `.mp4`, `.mov`, `.m4v`, `.html`, `.pdf` (and any other unknown ext) so the server streams them with the correct `Content-Type`.

## Non-goals

- No async/background write-back queue.
- No multi-format write-back: only canonical WebP lands in imgsbackup. Explicit `?format=png`/`?format=jpg` requests re-convert on every hit.
- No staleness detection: imgsbackup is authoritative. If a source file truly changes, the operator deletes the matching `imgsbackup/.../X.webp` manually.
- No thumbnails for video/PDF.
- No cross-worker per-key deduplication beyond the existing global `conversion_slot` cap.

## Decisions log

| # | Question | Decision |
|---|---|---|
| Q1 | Write-back format policy | Canonical WebP only |
| Q2 | `.mp4`/`.mov`/`.m4v`/`.html`/`.pdf` | Serve source as-is, no conversion or write-back |
| Q3 | Already web-friendly sources (`.jpg`/`.png`/`.gif`/`.bmp`/`.avif`/`.apng`) | Convert to WebP and write back |
| Q4 | Local cache role | None — convert in `/tmp`, serve from imgsbackup |
| Q5 | Source-file change detection | Ignore; imgsbackup is authoritative |
| Q6 | imgsbackup write failure | Serve the converted bytes, log a warning |
| F1 | State dir default | `./state` (next to `server.py`) |
| F2 | `rawpy` dependency | Soft (import-on-demand; warn-and-fallback) |
| F3 | Removing old `--cache-*` flags | Silent removal (no back-compat shim) |

---

## Architecture

```
                         /imgs/{folder}/{filename}
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ 1. WebP hit in /mnt/imgsbackup/imgs3/{folder}/X.webp?  │
       │    → stream it, Cache-Control: immutable               │
       └────────────────────────────────────────────────────────┘
                                    │ no
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ 2. Source exists in /mnt/storagebox/imgs/{folder}/X.Y? │
       │    → 404 if not                                        │
       └────────────────────────────────────────────────────────┘
                                    │ yes
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ 3. Is Y a passthrough type (mp4/mov/m4v/pdf/html/etc)? │
       │    → stream source with correct Content-Type, done.    │
       └────────────────────────────────────────────────────────┘
                                    │ no — image
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ 4. Convert to WebP in /tmp under conversion_slot       │
       │    backends: rawpy → Pillow → IM → ffmpeg → tiffcp/JXR │
       └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ 5. Promote /tmp file to imgsbackup via .tmp + rename   │
       │    (fail-soft: warn, still serve)                      │
       └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                  stream WebP, Cache-Control: immutable
```

### Paths

| Role | Path |
|---|---|
| Source | `/mnt/storagebox/imgs/{folder}/{filename}` |
| Backup primary (read+write) | `/mnt/imgsbackup/imgs3/{folder}/{basename}.webp` |
| Backup additional (read-only) | future entries in `IMGSBACKUP_READ_DIRS` |
| Backup temp during write | `/mnt/imgsbackup/imgs3/{folder}/.{basename}.{pid}.{rand}.tmp` |
| Conversion scratch | `/tmp/imgserve-{rand}.webp` |
| State (fcntl slot locks) | `./state/.conversion-slots/slot-{n}.lock` |

`{basename} = splitext({filename})[0]`. Filenames are already content-hashed in practice (`48e04f71...d956b4.psd`), so collisions across different source extensions sharing a basename are not a concern.

### Concurrency model

- Existing `conversion_slot` fcntl-based slot limiter is kept verbatim (caps simultaneous conversions across all uvicorn workers on the host).
- The old per-cache-key `_cache_lock` is removed. Two workers converting the same source simultaneously each produce identical WebP bytes, each rename atomically into the final imgsbackup path, last writer wins, both serve correct bytes. Bounded duplication of CPU is acceptable.
- `os.rename` within a single directory is atomic on POSIX and on NFS (single metadata op). Cross-FS moves are avoided by `shutil.copyfile` → same-dir `rename`.

---

## Component changes

### `server.py`

**Remove:**
- `DEFAULT_CACHE_DIR`, `DEFAULT_CACHE_MAX_AGE_HOURS`, `DEFAULT_CACHE_MAX_BYTES`, `DEFAULT_STALE_LOCK_AGE_HOURS`, `DEFAULT_MIN_FREE_BYTES`, `DEFAULT_CACHE_CLEANUP_INTERVAL_SECONDS`
- `ENV_CACHE_DIR`, `ENV_CACHE_MAX_AGE_HOURS`, `ENV_CACHE_MAX_BYTES`, `ENV_STALE_LOCK_AGE_HOURS`, `ENV_MIN_FREE_BYTES`, `ENV_CACHE_CLEANUP_INTERVAL_SECONDS`
- `CACHE_KEY_VERSION`, `CACHE_CLEANUP_LOCK_NAME`, `CACHE_CLEANUP_LAST_NAME`, `CACHE_INTERNAL_FILES`
- Functions: `_safe_cache_stem`, `_build_cache_paths`, `_is_protected_cache_root`, `_cleanup_cache_dir`, `_touch_cache_cleanup_marker`, `_run_cache_cleanup`, `_maybe_run_cache_cleanup`, `_ensure_cache_storage`, `_check_cache_dir`, `_cache_lock`
- The entire `lifespan` async context manager (FastAPI is instantiated without `lifespan=`)
- CLI flags: `--cache-dir`, `--cache-max-age-hours`, `--cache-max-bytes`, `--stale-lock-age-hours`, `--min-free-bytes`, `--cache-cleanup-interval-seconds`
- All references in `serve_image` to the cache (`_build_cache_paths`, `_cache_lock`, `_maybe_run_cache_cleanup`, `_ensure_cache_storage`, the "cache hit" branch, the "filled by another worker" branch).

**Rename:**
- `IMGSBACKUP_DIRS` → `IMGSBACKUP_READ_DIRS` (still a list for read fallback). Keep `/mnt/imgsbackup/imgs3` as the only element by default.
- `CACHE_CONVERSION_SLOTS_DIR_NAME` stays (used by `_conversion_slot`), but the parent dir is now `state_dir`, not `cache_dir`.

**Add:**
- `DEFAULT_STATE_DIR = "./state"` and `ENV_STATE_DIR = "IMGSERVE_STATE_DIR"`.
- `DEFAULT_IMGSBACKUP_PRIMARY = "/mnt/imgsbackup/imgs3"` and `ENV_IMGSBACKUP_PRIMARY = "IMGSERVE_IMGSBACKUP_DIR"` — single writable backup root.
- `PASSTHROUGH_EXTS = {".mp4", ".mov", ".m4v", ".html", ".pdf"}` lives in `converter.py` alongside the other format sets.
- Helper `is_passthrough(ext)` in `converter.py`: returns `True` iff `ext in PASSTHROUGH_EXTS` **or** `ext not in WEB_FORMATS and ext not in CONVERT_FORMATS and ext not in RAW_EXTS` — i.e. "anything we explicitly mark passthrough, or anything we don't know how to convert."
- Content-Type for the new types is handled by extending `converter.get_content_type`. No separate map in `server.py`.
- `promote_to_imgsbackup(local_path, folder, filename) -> bool` — see pseudocode above. Returns `True` on success, `False` on any `OSError`. Logs success at INFO and failure at WARNING with the OS error message. Cleans up the `.tmp` file on failure with a suppressed `OSError`.
- `_check_imgsbackup_write(imgsbackup_dir, min_free_bytes)` replaces `_check_cache_dir`. Tries `tempfile.NamedTemporaryFile(dir=imgsbackup_dir, prefix=".health-", delete=True)` and `shutil.disk_usage(imgsbackup_dir)`. Same response shape as the old function.

**`serve_image` becomes** (logical replacement, full implementation in plan):
1. Validate `folder`, `filename`.
2. If `format` is `None` or `"webp"`, call `find_optimized_webp` and serve on hit.
3. `src = join(IMGS_DIR, folder, filename)`. If not `isfile(src)`, raise 404.
4. `ext = splitext(filename)[1].lower()`. If `is_passthrough(ext)`, stream the source with `Cache-Control: DIRECT_SOURCE_CACHE_CONTROL` and the right `Content-Type`. Return.
5. Determine `out_fmt`: `(format or "webp").lower()`. Validate against `FORMAT_TO_EXT`, else 400.
6. Acquire `conversion_slot`. Inside: create `tmp_out = tempfile.mkstemp(prefix="imgserve-", suffix=FORMAT_TO_EXT[out_fmt], dir="/tmp")`. Call `convert_image(src, tmp_out, out_fmt)`. On failure, `os.unlink(tmp_out)` and raise 500.
7. If `out_fmt == "webp"`, call `promote_to_imgsbackup(tmp_out, folder, filename)` (ignore return value beyond logging).
8. Return a `FileResponse` for `tmp_out` with `Cache-Control: IMMUTABLE_CACHE_CONTROL`, and attach a `BackgroundTask(os.unlink, tmp_out)` so FastAPI deletes the temp after the response is sent.

**`/health`:** the `cache` check becomes `imgsbackup` and probes `IMGSBACKUP_PRIMARY` for writability + free space against `--health-min-free-bytes`. The `optimized_webp` read check stays. The `fallback_source` check stays.

**`main`:** removes deleted flags; adds `--state-dir` and `--imgsbackup-dir`; keeps `--imgs-dir`, `--conversion-slots`, `--conversion-slot-timeout-seconds`, `--health-min-free-bytes`, `--workers`, `--host`, `--port`. Startup log lines reflect the new structure.

### `converter.py`

**Add:**
- `RAW_EXTS = {".nef", ".arw", ".dng", ".cr2", ".cr3", ".raf", ".rw2", ".orf", ".pef", ".srw"}`.
- `convert_with_rawpy(src, dst, fmt) -> bool` — short-circuits `False` for non-RAW; otherwise opens with `rawpy.imread`, runs `raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB, output_bps=8)`, wraps the resulting numpy array in `Image.fromarray(rgb)`, saves as the requested format (WebP `quality=92, method=6`; JPEG `quality=95`; PNG default). Soft dependency: `ImportError` returns `False` with one warning log.
- Extend `get_content_type` with `.mp4 / .mov / .m4v / .html / .pdf`.
- `PASSTHROUGH_EXTS` and `is_passthrough(ext)` (defined above) live here and are imported by `server.py`.

**Change:**
- `convert_image`: insert `("rawpy", convert_with_rawpy)` as the first entry of `backends`. Behavior unchanged for non-RAW because `convert_with_rawpy` returns `False` immediately.
- `convert_with_pillow`: after `img.load()` and before any mode conversion, if `img.info.get("icc_profile")` is present and `img.mode in ("RGB", "RGBA", "L", "LA")`, apply `PIL.ImageCms.profileToProfile(img, srcProfile=ImageCms.ImageCmsProfile(io.BytesIO(img.info["icc_profile"])), destProfile=ImageCms.createProfile("sRGB"), outputMode="RGBA" if has_transparency(img) else "RGB")`. Best-effort: wrap in `try/except` and log at debug on failure (fall through to today's behavior).
- `choose_output_format`: change the no-transparency default from `"png"` to `"webp"`.

**Dependencies:**
- Add `rawpy` to `requirements.txt` (creating the file if missing). Mark it optional in a comment so dev installs without it still run.
- Confirm `Pillow` is already present (it is — imported at module top).

---

## Data flow examples

### Example 1 — uncached PSD
```
GET /imgs/fillop/48e04f71...d956b4.psd
  → find_optimized_webp → MISS
  → src exists at /mnt/storagebox/imgs/fillop/48e04f71...d956b4.psd
  → ext .psd not passthrough; out_fmt=webp
  → conversion_slot acquired
  → /tmp/imgserve-abc123.webp produced via Pillow
  → promote: copy /tmp/... → /mnt/imgsbackup/imgs3/fillop/.48e04f71...d956b4.<pid>.<rand>.tmp
                rename → /mnt/imgsbackup/imgs3/fillop/48e04f71...d956b4.webp
  → FileResponse(/tmp/imgserve-abc123.webp), BackgroundTask deletes /tmp file after send.
```

### Example 2 — second hit on the same source
```
GET /imgs/fillop/48e04f71...d956b4.psd
  → find_optimized_webp → HIT (/mnt/imgsbackup/imgs3/fillop/48e04f71...d956b4.webp)
  → FileResponse, Cache-Control: immutable. No conversion.
```

### Example 3 — RAW file
```
GET /imgs/foo/IMG_0123.nef
  → MISS in imgsbackup
  → ext .nef not passthrough; out_fmt=webp
  → convert_image tries rawpy first → libraw demosaic → sRGB array → WebP
  → promote, serve. Correct colors.
```

### Example 4 — video passthrough
```
GET /imgs/foo/clip.mp4
  → MISS in imgsbackup (find_optimized_webp checks for clip.webp; absent)
  → src exists; ext .mp4 is passthrough
  → FileResponse(src, Content-Type: video/mp4, Cache-Control: public, max-age=3600)
  → Nothing written to imgsbackup.
```

### Example 5 — explicit `?format=png`
```
GET /imgs/foo/photo.psd?format=png
  → format != webp, so skip find_optimized_webp shortcut
  → convert in /tmp to PNG
  → promote step skipped (out_fmt != webp)
  → serve, delete tmp.
```

### Example 6 — imgsbackup mount down
```
GET /imgs/foo/photo.psd
  → MISS
  → convert in /tmp → OK
  → promote: copy raises OSError(EIO)
  → log WARNING, suppress tmp_dest cleanup error
  → still serve /tmp file. Next request will retry the conversion.
```

---

## Error handling

| Condition | Behavior |
|---|---|
| Source not found | 404 |
| Invalid folder/filename | 400 |
| Unsupported `?format=` value | 400 |
| Conversion slot timeout | 503 (unchanged) |
| All conversion backends fail | 500 (unchanged) |
| imgsbackup write fails after successful convert | 200 + WebP bytes + WARNING log |
| `rawpy` not installed for RAW source | One-time WARNING log, fall through to existing backends (likely produces the channel-separation artifact, but matches today's degraded behavior) |
| `/health` imgsbackup write probe fails | 503 with details in JSON body |

---

## Testing

New `tests/` directory with `pytest` + FastAPI `TestClient` (or `httpx.AsyncClient`). All tests use `tmp_path`-based source and imgsbackup dirs — no real network mounts.

| File | What it covers |
|---|---|
| `tests/conftest.py` | Fixture: build app with overridden `app.state.imgs_dir`, `app.state.imgsbackup_dir`, `app.state.state_dir` pointing at `tmp_path` subdirs. |
| `tests/test_passthrough.py` | Request `.mp4`, `.mov`, `.m4v`, `.pdf`, `.html` fixtures → assert response bytes == source bytes, assert correct `Content-Type`, assert no file appears in imgsbackup. |
| `tests/test_writeback.py` | Tiny `.psd` fixture (or `.tif` if simpler) → first request returns WebP and creates `imgsbackup/.../X.webp`; second request returns same bytes and does NOT call `convert_image` (assert via `unittest.mock.patch` on `server.convert_image` for the second request). |
| `tests/test_writeback_failure.py` | `chmod 0500` on imgsbackup dir → request still returns 200 with valid WebP, log captures the warning. |
| `tests/test_concurrent_convert.py` | `ThreadPoolExecutor` fires 4 simultaneous requests for the same fresh source → all 4 get byte-equivalent WebP, exactly one `X.webp` exists in imgsbackup, no `.tmp` files remain. |
| `tests/test_raw_demosaic.py` | Tiny DNG fixture (or skip if rawpy unavailable). Convert, assert output is WebP, assert RGB histogram has non-trivial spread on all 3 channels (sanity check against channel separation). |
| `tests/test_format_variant.py` | Request `?format=png` → Content-Type `image/png`, no file in imgsbackup, second identical request also has no file in imgsbackup. |
| `tests/test_health.py` | `/health` returns 200 with all-OK fixtures, 503 with read-only imgsbackup. |

---

## Migration / rollout

- No database. No external state to migrate.
- The existing local `cache/` directory becomes unused. Operator deletes it manually after deploy (or leaves it; it just sits there).
- Existing files in `/mnt/imgsbackup/imgs3/` are forward-compatible — the read lookup is unchanged.
- Existing systemd unit / launch script needs updating only if it passes the now-removed `--cache-*` flags. They become unrecognized and `argparse` will exit; the deploy must drop them.

## Open questions

None. All Q1–Q6 and F1–F3 answered.
