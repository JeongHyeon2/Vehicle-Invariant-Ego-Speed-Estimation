#!/usr/bin/env python3
"""Extract aligned 10 or 30 FPS PNG frames from the public MP4 release."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2


SOURCE_FPS = 30.0
OUTPUT_FIELDS = [
    "frame_index",
    "source_frame_index",
    "time_sec",
    "speed_kmh",
    "speed_mps",
    "png_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract aligned PNGs and labels. The default 10 FPS selects source "
            "frames 0, 3, 6, ... from each 30 FPS recording."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Public original-video dataset containing recordings/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-fps", type=int, choices=(10, 30), default=10)
    parser.add_argument(
        "--recording",
        action="append",
        default=[],
        help="Recording to extract; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=3,
        metavar="0-9",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-output-frames", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_labels(path: Path, fps: float) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["frame_index", "time_sec", "speed_kmh", "speed_mps"]
        if reader.fieldnames != expected:
            raise RuntimeError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)

    for index, row in enumerate(rows):
        if int(row["frame_index"]) != index:
            raise RuntimeError(f"Non-contiguous frame_index in {path} at row {index}")
        if abs(float(row["time_sec"]) - index / fps) > 5.1e-7:
            raise RuntimeError(f"Timestamp mismatch in {path} at row {index}")
    return rows


def recording_names(dataset_root: Path, requested: list[str]) -> list[str]:
    recordings_root = dataset_root / "recordings"
    if not recordings_root.is_dir():
        raise RuntimeError(f"Missing recordings directory: {recordings_root}")
    available = sorted(path.name for path in recordings_root.iterdir() if path.is_dir())
    if not requested:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise RuntimeError(f"Unknown recording(s): {', '.join(missing)}")
    return list(dict.fromkeys(requested))


def extract_recording(
    dataset_root: Path,
    output_root: Path,
    name: str,
    output_fps: int,
    compression: int,
    overwrite: bool,
    limit: int | None,
) -> dict[str, int | float | str]:
    source_dir = dataset_root / "recordings" / name
    video_path = source_dir / f"{name}.mp4"
    csv_path = source_dir / f"{name}.csv"
    destination = output_root / name
    frames_dir = destination / "frames"

    if not video_path.is_file() or not csv_path.is_file():
        raise RuntimeError(f"Expected {name}.mp4 and {name}.csv in {source_dir}")
    if destination.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists; use --overwrite: {destination}")
        shutil.rmtree(destination)
    frames_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(source_fps - SOURCE_FPS) > 1e-3:
        capture.release()
        raise RuntimeError(f"Expected 30 FPS, got {source_fps} for {video_path}")

    labels = read_labels(csv_path, source_fps)
    if video_frames != len(labels):
        capture.release()
        raise RuntimeError(
            f"Frame/label mismatch for {name}: video={video_frames}, CSV={len(labels)}"
        )

    stride = int(round(source_fps / output_fps))
    selected: list[dict[str, int | str]] = []
    source_index = 0
    output_index = 0
    while source_index < len(labels):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Video ended early at frame {source_index}: {video_path}")

        if source_index % stride == 0:
            filename = f"frame_{output_index:06d}.png"
            relative_path = f"frames/{filename}"
            if not cv2.imwrite(
                str(frames_dir / filename),
                frame,
                [cv2.IMWRITE_PNG_COMPRESSION, compression],
            ):
                capture.release()
                raise RuntimeError(f"Could not write {frames_dir / filename}")

            label = labels[source_index]
            selected.append(
                {
                    "frame_index": output_index,
                    "source_frame_index": source_index,
                    "time_sec": label["time_sec"],
                    "speed_kmh": label["speed_kmh"],
                    "speed_mps": label["speed_mps"],
                    "png_path": relative_path,
                }
            )
            output_index += 1
            if limit is not None and output_index >= limit:
                break
        source_index += 1

    capture.release()
    with (destination / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(selected)

    return {
        "recording": name,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "stride": stride,
        "source_labeled_frames": len(labels),
        "extracted_frames": len(selected),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    recordings_root = dataset_root / "recordings"
    if output_root == recordings_root or recordings_root in output_root.parents:
        raise RuntimeError("Output directory must not be inside dataset/recordings")
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in recording_names(dataset_root, args.recording):
        summary = extract_recording(
            dataset_root,
            output_root,
            name,
            args.output_fps,
            args.png_compression,
            args.overwrite,
            args.max_output_frames,
        )
        summaries.append(summary)
        print(
            f"{name}: extracted {summary['extracted_frames']} aligned frames "
            f"at {args.output_fps} FPS",
            flush=True,
        )

    with (output_root / "extraction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
