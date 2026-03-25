#!/usr/bin/env python3
"""
Batch-convert referenced product images into training-friendly outputs.

Default behavior:
  - reads image paths from /root/productImages.txt
  - processes every path listed in the source file
  - loads files from /mnt/storagebox/imgs
  - writes resized images to /root/newImages
  - preserves aspect ratio and downsizes the longest edge to 1024 px
  - converts embedded ICC profiles to sRGB where present
  - saves transparent images as PNG and opaque images as JPEG
  - never deletes or modifies the original files

Example:
  python3 prepare_training_images.py

If you want to process only selected top-level folders from the list, use:
  python3 prepare_training_images.py --include-root rema1000 --include-root netto
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from PIL import Image, ImageCms, ImageOps

try:
    import rawpy  # type: ignore
except ImportError:
    rawpy = None


Image.MAX_IMAGE_PIXELS = None

SRGB_PROFILE = ImageCms.createProfile("sRGB")
SRGB_CMS = ImageCms.ImageCmsProfile(SRGB_PROFILE)

LOGGER = logging.getLogger("prepare_training_images")
RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}
TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class Job:
    rel_path: PurePosixPath
    src_path: Path


@dataclass(frozen=True)
class Result:
    status: str
    rel_path: str
    source_path: str
    output_path: str = ""
    loader: str = ""
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", type=Path, default=Path("/root/productImages.txt"))
    parser.add_argument("--src-root", type=Path, default=Path("/mnt/storagebox/imgs"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/newImages"))
    parser.add_argument("--max-edge", type=int, default=1024, help="Longest edge after resize")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker count")
    parser.add_argument(
        "--opaque-format",
        choices=("jpeg", "png"),
        default="jpeg",
        help="Output format for images without transparency",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for opaque outputs")
    parser.add_argument(
        "--include-root",
        action="append",
        dest="include_roots",
        help="Repeat to keep specific top-level folders from the path list",
    )
    parser.add_argument(
        "--export-alpha-mask",
        action="store_true",
        help="Save alpha masks as additional *_mask.png files when transparency is present",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-generated output files",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N matching paths")
    parser.add_argument("--progress-every", type=int, default=250, help="Log progress every N processed files")
    return parser.parse_args()


def normalized_include_roots(args: argparse.Namespace) -> set[str] | None:
    if args.include_roots:
        return {root.strip().strip("/").lower() for root in args.include_roots if root.strip()}
    return None


def parse_relative_path(raw_line: str) -> PurePosixPath | None:
    text = raw_line.strip()
    if not text:
        return None

    raw_path = PurePosixPath(text)
    parts = list(raw_path.parts)
    if parts and parts[0] == "/":
        parts = parts[1:]
    if not parts:
        return None
    if any(part in ("", ".", "..") for part in parts):
        return None

    return PurePosixPath(*parts)


def collect_jobs(
    source_list: Path,
    src_root: Path,
    include_roots: set[str] | None,
    limit: int | None,
) -> tuple[list[Job], Counter]:
    jobs: list[Job] = []
    stats: Counter = Counter()

    with source_list.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            rel_path = parse_relative_path(raw_line)
            if rel_path is None:
                stats["invalid_or_empty"] += 1
                continue

            top_level = rel_path.parts[0].lower()
            if include_roots is not None and top_level not in include_roots:
                stats["filtered_out"] += 1
                stats[f"filtered_out:{top_level}"] += 1
                continue

            jobs.append(Job(rel_path=rel_path, src_path=src_root / Path(*rel_path.parts)))
            stats["selected"] += 1
            stats[f"selected:{top_level}"] += 1

            if limit is not None and len(jobs) >= limit:
                break

    return jobs, stats


def has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        alpha = img.getchannel("A")
        extrema = alpha.getextrema()
        return extrema[0] < 255
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False


def load_with_pillow(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        try:
            opened.seek(0)
        except (AttributeError, EOFError):
            pass
        image = ImageOps.exif_transpose(opened)
        image.load()
        return image.copy()


def load_with_rawpy(path: Path) -> Image.Image:
    if rawpy is None:
        raise RuntimeError("rawpy is not installed")

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(output_bps=8, use_camera_wb=True)
    return Image.fromarray(rgb, mode="RGB")


def load_via_ffmpeg(path: Path) -> Image.Image:
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_file.close()
    tmp_path = Path(tmp_file.name)
    try:
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(tmp_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"ffmpeg exited with {result.returncode}")
        return load_with_pillow(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_via_convert(path: Path) -> Image.Image:
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_file.close()
    tmp_path = Path(tmp_file.name)
    try:
        command = ["convert", f"{path}[0]", str(tmp_path)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"convert exited with {result.returncode}")
        return load_with_pillow(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_via_tiffcp(path: Path) -> Image.Image:
    if path.suffix.lower() not in TIFF_EXTENSIONS:
        raise RuntimeError("tiffcp only applies to TIFF inputs")

    tmp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_file.close()
    tmp_path = Path(tmp_file.name)
    try:
        result = subprocess.run(
            ["tiffcp", str(path), str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"tiffcp exited with {result.returncode}")
        return load_with_pillow(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_image(path: Path) -> tuple[Image.Image, str]:
    ext = path.suffix.lower()
    loaders: list[tuple[str, Callable[[Path], Image.Image]]] = []

    if ext in RAW_EXTENSIONS and rawpy is not None:
        loaders.append(("rawpy", load_with_rawpy))

    loaders.append(("pillow", load_with_pillow))

    if ext == ".pdf":
        loaders.extend([("convert", load_via_convert), ("ffmpeg", load_via_ffmpeg)])
    elif ext in TIFF_EXTENSIONS:
        loaders.extend([("convert", load_via_convert), ("tiffcp", load_via_tiffcp), ("ffmpeg", load_via_ffmpeg)])
    elif ext in RAW_EXTENSIONS:
        loaders.extend([("ffmpeg", load_via_ffmpeg), ("convert", load_via_convert)])
    else:
        loaders.extend([("convert", load_via_convert), ("ffmpeg", load_via_ffmpeg)])

    errors: list[str] = []
    for loader_name, loader in loaders:
        try:
            image = loader(path)
            return image, loader_name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader_name}: {exc}")

    raise RuntimeError("; ".join(errors))


def resize_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    if max(image.size) <= max_edge:
        return image

    resized = image.copy()
    resized.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return resized


def convert_to_srgb(image: Image.Image) -> Image.Image:
    """Convert image to sRGB color space if it has an embedded ICC profile."""
    icc_data = image.info.get("icc_profile")
    if not icc_data:
        return image
    try:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_data))
        target_mode = "RGBA" if image.mode == "RGBA" else "RGB"
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert(target_mode)
        return ImageCms.profileToProfile(image, src_profile, SRGB_CMS, outputMode=target_mode)
    except (ImageCms.PyCMSError, OSError):
        return image


def normalize_for_output(image: Image.Image, keep_alpha: bool) -> Image.Image:
    image = convert_to_srgb(image)

    if keep_alpha:
        if image.mode != "RGBA":
            return image.convert("RGBA")
        return image

    if image.mode == "RGB":
        return image

    if image.mode in ("RGBA", "LA", "PA", "P") and has_transparency(image):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background

    return image.convert("RGB")


def output_paths(
    output_root: Path,
    rel_path: PurePosixPath,
    transparent: bool,
    opaque_format: str,
    export_alpha_mask: bool,
) -> tuple[Path, Path | None]:
    base_output = output_root / Path(*rel_path.parts)
    if transparent:
        image_output = base_output.with_suffix(".png")
        mask_output = image_output.with_name(f"{image_output.stem}_mask.png") if export_alpha_mask else None
        return image_output, mask_output

    output_suffix = ".jpg" if opaque_format == "jpeg" else ".png"
    return base_output.with_suffix(output_suffix), None


def save_output(image: Image.Image, destination: Path, opaque_format: str, jpeg_quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.suffix.lower() == ".png":
        image.save(destination, format="PNG", optimize=True, compress_level=6)
        return

    if opaque_format != "jpeg":
        raise ValueError(f"Unsupported opaque format: {opaque_format}")

    image.save(
        destination,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
        progressive=True,
        subsampling=0,
    )


def save_alpha_mask(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    alpha = image.getchannel("A")
    alpha.save(destination, format="PNG", optimize=True, compress_level=6)


def process_job(job: Job, args: argparse.Namespace) -> Result:
    if not job.src_path.exists():
        return Result(
            status="missing",
            rel_path=str(job.rel_path),
            source_path=str(job.src_path),
            detail="source file not found",
        )

    try:
        image, loader_name = load_image(job.src_path)
        transparent = has_transparency(image)
        image_output, mask_output = output_paths(
            output_root=args.output_root,
            rel_path=job.rel_path,
            transparent=transparent,
            opaque_format=args.opaque_format,
            export_alpha_mask=args.export_alpha_mask,
        )

        if not args.overwrite and image_output.exists() and (mask_output is None or mask_output.exists()):
            return Result(
                status="skipped_existing",
                rel_path=str(job.rel_path),
                source_path=str(job.src_path),
                output_path=str(image_output),
                loader=loader_name,
                detail="output already exists",
            )

        resized = resize_to_max_edge(image, args.max_edge)
        prepared = normalize_for_output(resized, keep_alpha=transparent)
        save_output(prepared, image_output, args.opaque_format, args.jpeg_quality)
        if mask_output is not None:
            save_alpha_mask(prepared, mask_output)

        return Result(
            status="converted",
            rel_path=str(job.rel_path),
            source_path=str(job.src_path),
            output_path=str(image_output),
            loader=loader_name,
            detail=f"{image.size[0]}x{image.size[1]} -> {prepared.size[0]}x{prepared.size[1]}",
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            status="failed",
            rel_path=str(job.rel_path),
            source_path=str(job.src_path),
            detail=str(exc),
        )


def write_result_logs(
    result: Result,
    manifest_handle,
    missing_handle,
    failed_handle,
) -> None:
    manifest_handle.write(
        "\t".join(
            [
                result.status,
                result.rel_path,
                result.source_path,
                result.output_path,
                result.loader,
                result.detail.replace("\n", " "),
            ]
        )
        + "\n"
    )
    if result.status == "missing":
        missing_handle.write(f"{result.rel_path}\t{result.source_path}\n")
    elif result.status == "failed":
        failed_handle.write(f"{result.rel_path}\t{result.source_path}\t{result.detail}\n")


def main() -> int:
    args = parse_args()

    if args.max_edge <= 0:
        raise SystemExit("--max-edge must be greater than 0")
    if args.workers <= 0:
        raise SystemExit("--workers must be greater than 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")
    if not args.source_list.is_file():
        raise SystemExit(f"Source list not found: {args.source_list}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    include_roots = normalized_include_roots(args)
    jobs, selection_stats = collect_jobs(
        source_list=args.source_list,
        src_root=args.src_root,
        include_roots=include_roots,
        limit=args.limit,
    )

    if not jobs:
        LOGGER.error("No matching images found in %s", args.source_list)
        return 1

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    manifest_path = log_dir / f"manifest-{timestamp}.tsv"
    missing_path = log_dir / f"missing-{timestamp}.txt"
    failed_path = log_dir / f"failed-{timestamp}.txt"
    summary_path = log_dir / f"summary-{timestamp}.json"

    LOGGER.info("Selected %s image paths from %s", len(jobs), args.source_list)
    LOGGER.info("Source root: %s", args.src_root)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Resize longest edge to %s px", args.max_edge)
    LOGGER.info("Workers: %s", args.workers)
    if include_roots is None:
        LOGGER.info("Top-level root filter: disabled")
    else:
        LOGGER.info("Top-level root filter: %s", ", ".join(sorted(include_roots)))
    if rawpy is None:
        LOGGER.info("rawpy not installed; RAW files will use Pillow/ffmpeg fallbacks")

    counters: Counter = Counter()
    loaders: Counter = Counter()
    start_time = time.time()

    with (
        manifest_path.open("w", encoding="utf-8") as manifest_handle,
        missing_path.open("w", encoding="utf-8") as missing_handle,
        failed_path.open("w", encoding="utf-8") as failed_handle,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        manifest_handle.write("status\trel_path\tsource_path\toutput_path\tloader\tdetail\n")

        future_to_job = {executor.submit(process_job, job, args): job for job in jobs}
        total = len(future_to_job)

        for index, future in enumerate(concurrent.futures.as_completed(future_to_job), start=1):
            result = future.result()
            counters[result.status] += 1
            if result.loader:
                loaders[result.loader] += 1

            write_result_logs(
                result=result,
                manifest_handle=manifest_handle,
                missing_handle=missing_handle,
                failed_handle=failed_handle,
            )

            if index % args.progress_every == 0 or index == total:
                elapsed = time.time() - start_time
                rate = index / elapsed if elapsed else 0.0
                LOGGER.info(
                    "Processed %s/%s (%.1f%%) at %.2f files/sec | converted=%s skipped=%s missing=%s failed=%s",
                    index,
                    total,
                    (index / total) * 100.0,
                    rate,
                    counters["converted"],
                    counters["skipped_existing"],
                    counters["missing"],
                    counters["failed"],
                )

    summary = {
        "timestamp": timestamp,
        "source_list": str(args.source_list),
        "src_root": str(args.src_root),
        "output_root": str(output_root),
        "max_edge": args.max_edge,
        "workers": args.workers,
        "opaque_format": args.opaque_format,
        "jpeg_quality": args.jpeg_quality,
        "include_roots": None if include_roots is None else sorted(include_roots),
        "export_alpha_mask": args.export_alpha_mask,
        "overwrite": args.overwrite,
        "limit": args.limit,
        "selection_stats": dict(selection_stats),
        "result_counts": dict(counters),
        "loader_counts": dict(loaders),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "manifest_path": str(manifest_path),
        "missing_path": str(missing_path),
        "failed_path": str(failed_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    LOGGER.info("Summary written to %s", summary_path)
    if counters["failed"]:
        LOGGER.warning("Some files failed. See %s", failed_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
