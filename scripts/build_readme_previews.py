#!/usr/bin/env python3

"""Build lightweight autoplay GIF previews for the README video gallery."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def build_preview(
    source: Path,
    destination: Path,
    *,
    seconds: float,
    output_fps: float,
    width: int,
    colors: int,
) -> None:
    capture = cv2.VideoCapture(str(source))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Unable to read frame rate: {source}")

    sample_step = max(1, round(source_fps / output_fps))
    frame_limit = round(seconds * source_fps)
    frames: list[Image.Image] = []
    frame_index = 0

    while frame_index < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_step == 0:
            height = round(frame.shape[0] * width / frame.shape[1])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).resize(
                (width, height), Image.Resampling.LANCZOS
            )
            frames.append(image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT))
        frame_index += 1

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 / output_fps)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=Path("assets/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/previews"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--colors", type=int, default=96)
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No MP4 files found under {args.video_dir}")

    for video in videos:
        output = args.output_dir / f"{video.stem}.gif"
        build_preview(
            video,
            output,
            seconds=args.seconds,
            output_fps=args.fps,
            width=args.width,
            colors=args.colors,
        )
        print(f"{video.name} -> {output} ({output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
