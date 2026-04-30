#!/usr/bin/env python3
"""
imgServe — Internal image server with on-the-fly format conversion.

Serves images from /mnt/storagebox/imgs/ with automatic conversion of
non-web-friendly formats (PSD, TIFF, DNG, NEF, ARW, JXR, etc.) to
PNG/JPG/WebP. Converted images are cached on the local SSD.

Binds to 127.0.0.1 only — not accessible from the internet.

Usage:
  python3 server.py
  python3 server.py --port 8100
  python3 server.py --cache-dir /path/to/cache

Request examples:
  GET /fillop/48e04f71...d956b4.psd         → auto-converts to PNG
  GET /fillop/48e04f71...d956b4.psd?format=webp  → converts to WebP
  GET /bilka/abc123.jpg                       → served directly
  GET /health                                 → health check
"""

import os
import logging
import argparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from converter import needs_conversion, convert_image, get_content_type, FORMAT_TO_EXT

# Configuration
IMGS_BASE = "/mnt/storagebox/imgs"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("imgserve")

app = FastAPI(title="imgServe", docs_url=None, redoc_url=None)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/imgs/{folder}/{filename}")
async def serve_image(
    folder: str,
    filename: str,
    format: str | None = Query(None, description="Output format: png, jpg, webp"),
):
    # Sanitize path components — prevent directory traversal
    if ".." in folder or ".." in filename or "/" in folder or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid path")

    src_path = os.path.join(IMGS_BASE, folder, filename)

    if not os.path.isfile(src_path):
        raise HTTPException(status_code=404, detail="Image not found")

    ext = os.path.splitext(filename)[1].lower()
    basename = os.path.splitext(filename)[0]

    # Validate requested format
    out_fmt = None
    if format:
        format = format.lower()
        if format not in FORMAT_TO_EXT:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        out_fmt = format

    # If no conversion needed and no format override, serve directly
    if not needs_conversion(ext) and not out_fmt:
        content_type = get_content_type(ext)
        return FileResponse(src_path, media_type=content_type)

    # If format override on a web format (e.g. PNG→WebP), still convert
    # Default output format for conversions
    if not out_fmt:
        out_fmt = "png"

    out_ext = FORMAT_TO_EXT[out_fmt]
    cache_filename = f"{basename}{out_ext}"
    cache_path = os.path.join(CACHE_DIR, folder, cache_filename)

    # Serve from cache if available
    if os.path.isfile(cache_path):
        content_type = get_content_type(out_ext)
        return FileResponse(cache_path, media_type=content_type)

    # Convert
    logger.info(f"Converting {folder}/{filename} → {out_fmt}")
    success = convert_image(src_path, cache_path, out_fmt)

    if not success:
        raise HTTPException(status_code=500, detail="Conversion failed")

    content_type = get_content_type(out_ext)
    return FileResponse(cache_path, media_type=content_type)


def main():
    global IMGS_BASE, CACHE_DIR

    parser = argparse.ArgumentParser(description="imgServe — Internal image server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    parser.add_argument("--imgs-dir", default=IMGS_BASE, help=f"Images directory (default: {IMGS_BASE})")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help=f"Cache directory (default: {CACHE_DIR})")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers (default: 16)")
    args = parser.parse_args()

    IMGS_BASE = args.imgs_dir
    CACHE_DIR = args.cache_dir
    os.makedirs(CACHE_DIR, exist_ok=True)

    logger.info(f"Serving images from: {IMGS_BASE}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"Listening on: {args.host}:{args.port}")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
