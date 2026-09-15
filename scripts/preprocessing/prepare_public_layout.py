#!/usr/bin/env python3
"""Adapt public 10 FPS PNG folders to the preprocessing layout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path


# Public model-ready names do not follow the order of the original release.
# This mapping is fixed from the released pack provenance and frame counts.
PUBLIC_SEQUENCE_TO_RECORDING = {
    "avante_01": "avante_4",
    "avante_02": "avante_5",
    "avante_03": "avante_1",
    "avante_04": "avante_2",
    "avante_05": "avante_3",
    "carnival_01": "carnival_1",
    "carnival_02": "carnival_2",
    "carnival_03": "carnival_3",
    "malibu_01": "malibu_4",
    "malibu_02": "malibu_1",
    "malibu_03": "malibu_3",
    "malibu_04": "malibu_2",
    "sonata_01": "sonata_1",
    "xm3_01": "xm3_1",
}

# Kept so the script can also adapt archived internal split definitions.
LEGACY_DRIVE_TO_RECORDING = {
    "AVANTE_251022_indong_middle": "avante_1",
    "AVANTE_251022_indong_school": "avante_2",
    "AVANTE_251022_school_indong": "avante_3",
    "AVANTE_251001_school_indong": "avante_4",
    "AVANTE_251002_indong_school": "avante_5",
    "SUV_251123_1": "carnival_1",
    "SUV_251123_2": "carnival_2",
    "SUV_251123_3": "carnival_3",
    "MALIBU_251010_1": "malibu_1",
    "MALIBU_251010_3": "malibu_2",
    "MALIBU_251010_2": "malibu_3",
    "MALIBU_KIDI_4": "malibu_4",
    "SONATA_251103": "sonata_1",
    "XM3_251103_minsoo": "xm3_1",
}

RAW_RECORDING_NAMES = set(PUBLIC_SEQUENCE_TO_RECORDING.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-root",
        type=Path,
        required=True,
        help="Output of extract_frames.py --output-fps 10",
    )
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-frames",
        action="store_true",
        help="Copy PNG folders instead of linking them (requires substantial space)",
    )
    return parser.parse_args()


def sequence_to_drive(sequence: str) -> str:
    without_date = re.sub(r"^\d{4}_\d{2}_\d{2}_", "", sequence)
    return re.sub(r"_\d{4}_sync$", "", without_date)


def recording_for_sequence(sequence: str) -> str:
    if sequence in PUBLIC_SEQUENCE_TO_RECORDING:
        return PUBLIC_SEQUENCE_TO_RECORDING[sequence]
    if sequence.lower() in RAW_RECORDING_NAMES:
        return sequence.lower()
    drive = sequence_to_drive(sequence)
    if drive in LEGACY_DRIVE_TO_RECORDING:
        return LEGACY_DRIVE_TO_RECORDING[drive]
    raise RuntimeError(f"No public recording mapping for sequence: {sequence}")


def image_group_for_sequence(sequence: str) -> str:
    if sequence in PUBLIC_SEQUENCE_TO_RECORDING or sequence.lower() in RAW_RECORDING_NAMES:
        return sequence
    return sequence_to_drive(sequence)


def all_sequences(splits: dict) -> list[str]:
    return sorted(
        {
            sequence
            for fold in splits.values()
            for partition in ("train", "valid", "test")
            for sequence in fold.get(partition, [])
        }
    )


def read_10fps_labels(path: Path, frames_dir: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty label CSV: {path}")

    required = {
        "frame_index",
        "source_frame_index",
        "time_sec",
        "speed_kmh",
        "speed_mps",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

    for index, row in enumerate(rows):
        if int(row["frame_index"]) != index:
            raise RuntimeError(f"Non-contiguous frame_index in {path} at row {index}")
        if int(row["source_frame_index"]) != index * 3:
            raise RuntimeError(
                f"{path} is not a 10 FPS stride-3 extraction at row {index}; "
                "run extract_frames.py --output-fps 10"
            )
        if abs(float(row["time_sec"]) - index / 10.0) > 5.1e-7:
            raise RuntimeError(f"Unexpected 10 FPS timestamp in {path} at row {index}")

    pngs = list(frames_dir.glob("frame_*.png"))
    if len(pngs) != len(rows):
        raise RuntimeError(
            f"PNG/CSV count mismatch in {frames_dir}: {len(pngs)} != {len(rows)}"
        )
    return rows


def materialize_frames(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
    copy_frames: bool,
) -> None:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            if destination.is_symlink() and destination.resolve() == source.resolve():
                return
            raise RuntimeError(f"Destination exists; use --overwrite: {destination}")
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_frames:
        shutil.copytree(source, destination)
        return

    relative_source = os.path.relpath(source.resolve(), destination.parent)
    try:
        destination.symlink_to(relative_source, target_is_directory=True)
    except OSError as error:
        raise RuntimeError(
            "Could not create the frame-directory symlink. On Windows, enable "
            "Developer Mode or rerun with --copy-frames."
        ) from error


def main() -> None:
    args = parse_args()
    frames_root = args.frames_root.resolve()
    output_root = args.output_root.resolve()
    image_root = output_root / "image_root"
    label_root = output_root / "label_root"

    with args.splits_json.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    sequences = all_sequences(splits)
    if not sequences:
        raise RuntimeError(f"No sequences found in {args.splits_json}")

    labels_by_recording: dict[str, list[dict[str, str]]] = {}
    for sequence in sequences:
        recording = recording_for_sequence(sequence)
        source_dir = frames_root / recording
        frames_dir = source_dir / "frames"
        csv_path = source_dir / f"{recording}.csv"
        if recording not in labels_by_recording:
            labels_by_recording[recording] = read_10fps_labels(csv_path, frames_dir)
        materialize_frames(
            frames_dir,
            image_root / image_group_for_sequence(sequence) / "frames_10fps_stride3",
            overwrite=args.overwrite,
            copy_frames=args.copy_frames,
        )

    for sequence in sequences:
        recording = recording_for_sequence(sequence)
        destination = label_root / sequence / "label.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.overwrite:
            raise RuntimeError(f"Destination exists; use --overwrite: {destination}")
        with destination.open("w", encoding="utf-8") as handle:
            for row in labels_by_recording[recording]:
                handle.write(
                    f"{int(row['frame_index'])} {float(row['speed_kmh']):.9f}\n"
                )

    label_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.splits_json, label_root / "holdout_splits.json")
    summary = {
        "sequences": len(sequences),
        "physical_recordings": len(labels_by_recording),
        "image_root": str(image_root),
        "label_root": str(label_root),
        "source_sampling": "30 FPS source frames 0,3,6,... re-indexed to 10 FPS",
        "frame_materialization": "copy" if args.copy_frames else "symlink",
    }
    with (output_root / "layout_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
