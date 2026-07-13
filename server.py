#!/usr/bin/env python3
"""Read-only image server for the temporary groceryImgsOnly dataset.

This branch intentionally serves only exact image files below
``/mnt/groceryImgsOnly/thumbnails/``. It has no fallback source tree, format
conversion, or write-back path, so a missing file always produces a 404.
"""

import argparse
import logging
import mimetypes
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

DEFAULT_IMAGES_DIR = "/mnt/groceryImgsOnly/thumbnails"
DEFAULT_PORT = 8101
DEFAULT_WORKERS = 2
ALLOWED_FOLDERS = frozenset({"coop", "salling", "dagrofa", "rema1000"})
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("imgserve")


def _validate_path_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or ".." in value or "\x00" in value:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")

    separators = {os.sep}
    if os.path.altsep:
        separators.add(os.path.altsep)

    if any(separator in value for separator in separators):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")

    return value


def _check_readable_dir(path: str) -> dict[str, object]:
    status: dict[str, object] = {"path": path}
    try:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a directory")
        if not os.access(path, os.R_OK | os.X_OK):
            raise PermissionError(f"{path} is not readable")
        with os.scandir(path):
            pass
    except OSError as exc:
        status["ok"] = False
        status["error"] = str(exc)
        return status

    status["ok"] = True
    return status


def _health_payload(images_dir: str) -> tuple[dict[str, object], int]:
    root = _check_readable_dir(images_dir)
    folders = {
        folder: _check_readable_dir(os.path.join(images_dir, folder))
        for folder in sorted(ALLOWED_FOLDERS)
    }
    healthy = bool(root.get("ok")) and all(
        bool(folder_status.get("ok")) for folder_status in folders.values()
    )
    return {
        "status": "ok" if healthy else "error",
        "mode": "grocery-images-only",
        "read_only": True,
        "checks": {"root": root, "folders": folders},
    }, (200 if healthy else 503)


def _image_media_type(filename: str) -> str | None:
    media_type, _ = mimetypes.guess_type(filename)
    if media_type is None or not media_type.startswith("image/"):
        return None
    return media_type


def _is_file_within(path: str, directory: str) -> bool:
    resolved_path = os.path.realpath(path)
    resolved_directory = os.path.realpath(directory)
    try:
        return (
            os.path.commonpath((resolved_path, resolved_directory)) == resolved_directory
            and os.path.isfile(resolved_path)
        )
    except (OSError, ValueError):
        return False


def create_app(images_dir: str | None = None, log_config: bool = True) -> FastAPI:
    app = FastAPI(title="imgServe groceryImgsOnly", docs_url=None, redoc_url=None)
    app.state.images_dir = os.path.abspath(images_dir or DEFAULT_IMAGES_DIR)

    if log_config:
        logger.info(
            "App configured: mode=grocery-images-only read_only=true images_dir=%s "
            "allowed_folders=%s port=%s",
            app.state.images_dir,
            ",".join(sorted(ALLOWED_FOLDERS)),
            DEFAULT_PORT,
        )

    @app.get("/health")
    def health(request: Request):
        payload, status_code = _health_payload(request.app.state.images_dir)
        return JSONResponse(payload, status_code=status_code)

    @app.get("/imgs/{folder}/{filename}")
    def serve_image(request: Request, folder: str, filename: str):
        folder = _validate_path_component(folder, "folder")
        filename = _validate_path_component(filename, "filename")

        if folder not in ALLOWED_FOLDERS:
            logger.info(
                "Image request rejected: pid=%s reason=folder-not-allowed image=%s/%s",
                os.getpid(),
                folder,
                filename,
            )
            raise HTTPException(status_code=404, detail="Image not found")

        media_type = _image_media_type(filename)
        folder_path = os.path.join(request.app.state.images_dir, folder)
        image_path = os.path.join(folder_path, filename)
        if media_type is None or not _is_file_within(image_path, folder_path):
            logger.info(
                "Image request miss: pid=%s image=%s/%s path=%s",
                os.getpid(),
                folder,
                filename,
                image_path,
            )
            raise HTTPException(status_code=404, detail="Image not found")

        logger.info(
            "Serving grocery image: pid=%s image=%s/%s path=%s",
            os.getpid(),
            folder,
            filename,
            image_path,
        )
        return FileResponse(
            image_path,
            media_type=media_type,
            headers={"Cache-Control": IMMUTABLE_CACHE_CONTROL},
        )

    return app


app = create_app(log_config=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="imgServe — read-only groceryImgsOnly image server"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of workers (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.workers <= 0:
        parser.error("--workers must be > 0")

    logger.info("Mode: grocery-images-only (read-only, exact-file lookup)")
    logger.info("Image root: %s", DEFAULT_IMAGES_DIR)
    logger.info("Allowed folders: %s", ", ".join(sorted(ALLOWED_FOLDERS)))
    logger.info("Listening on: %s:%s", args.host, args.port)

    uvicorn.run(
        "server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
