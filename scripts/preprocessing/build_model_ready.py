#!/usr/bin/env python3
"""Run or resume the complete original-video to model-ready pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL_NAMES = (
    "EgoSpeed_original_mp4_per_frame_csv_20260914",
    "EgoSpeed_original_mp4",
    "EgoSpeedOriginal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 48 x 86 packed tensors and SmartROI masks. Standard "
            "sibling paths are auto-detected when path options are omitted."
        )
    )
    parser.add_argument("--original-dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=REPOSITORY_ROOT / "splits" / "holdout_splits.json",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--depth-batch", type=int, default=4)
    parser.add_argument("--flow-batch", type=int, default=4)
    parser.add_argument("--smartroi-depth-batch", type=int, default=1)
    parser.add_argument("--smartroi-flow-batch", type=int, default=1)
    parser.add_argument("--smartroi-chunk-centers", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--copy-frames", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing packs, masks, extraction folders, and layout",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing original and model-ready releases only",
    )
    return parser.parse_args()


def has_original_layout(path: Path) -> bool:
    return (path / "recordings").is_dir() and (path / "manifest.csv").is_file()


def discover_original_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not has_original_layout(root):
            raise RuntimeError(f"Invalid original dataset root: {root}")
        return root

    environment = os.environ.get("EGOSPEED_ORIGINAL_ROOT")
    candidates = []
    if environment:
        candidates.append(Path(environment).expanduser())
    for name in DEFAULT_ORIGINAL_NAMES:
        candidates.extend(
            [REPOSITORY_ROOT.parent / name / "dataset", REPOSITORY_ROOT.parent / name]
        )
    for candidate in candidates:
        candidate = candidate.resolve()
        if has_original_layout(candidate):
            return candidate
    raise RuntimeError(
        "Could not find the original dataset. Pass --original-dataset-root or "
        "set EGOSPEED_ORIGINAL_ROOT."
    )


def output_root_from_args(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    environment = os.environ.get("EGOSPEED_DATASET_ROOT")
    if environment:
        return Path(environment).expanduser().resolve()
    return (REPOSITORY_ROOT.parent / "EgoSpeedDataset").resolve()


def run(command: list[str]) -> None:
    print(f"\n$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def require_preprocessing_environment() -> None:
    modules = {
        "cv2": "opencv-python",
        "PIL": "pillow",
        "timm": "timm",
        "torchvision": "torchvision",
    }
    missing = [
        package
        for module, package in modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing preprocessing dependencies: "
            + ", ".join(missing)
            + ". Install a CUDA PyTorch/torchvision build, then run "
            + 'pip install -e ".[preprocessing]".'
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. Full-resolution "
            "MiDaS/RAFT preprocessing requires a CUDA-enabled PyTorch build."
        )


def read_splits(path: Path) -> tuple[dict, list[str]]:
    with path.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    sequences = sorted(
        {
            sequence
            for fold in splits.values()
            for partition in ("train", "valid", "test")
            for sequence in fold.get(partition, [])
        }
    )
    if not sequences:
        raise RuntimeError(f"No sequences in split file: {path}")
    return splits, sequences


def csv_row_count(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate_original(root: Path) -> list[str]:
    failures: list[str] = []
    manifest_path = root / "manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 14:
        failures.append(f"manifest contains {len(manifest)} recordings, expected 14")

    total_video_frames = 0
    total_csv_rows = 0
    for row in manifest:
        name = row["recording"]
        directory = root / "recordings" / name
        video_path = directory / f"{name}.mp4"
        csv_path = directory / f"{name}.csv"
        if not video_path.is_file():
            failures.append(f"missing video: {video_path}")
            continue
        if not csv_path.is_file():
            failures.append(f"missing CSV: {csv_path}")
            continue
        rows = csv_row_count(csv_path)
        video_frames = int(row["video_frames"])
        manifest_rows = int(row["csv_rows"])
        if not (video_frames == manifest_rows == rows):
            failures.append(
                f"{name}: manifest video={video_frames}, manifest CSV={manifest_rows}, "
                f"actual CSV={rows}"
            )
        if int(row["unlabeled_tail_frames"]) != 0:
            failures.append(f"{name}: manifest reports an unlabeled video tail")
        total_video_frames += video_frames
        total_csv_rows += rows

    validation_path = root / "validation.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "PASS":
            failures.append("validation.json status is not PASS")
        if validation.get("total_video_frames") != total_video_frames:
            failures.append("validation.json total_video_frames does not match manifest")
        if validation.get("total_csv_rows") != total_csv_rows:
            failures.append("validation.json total_csv_rows does not match CSV files")
    else:
        failures.append(f"missing validation report: {validation_path}")
    return failures


def extracted_recording_complete(
    original_root: Path, frames_root: Path, recording: str
) -> bool:
    source_csv = original_root / "recordings" / recording / f"{recording}.csv"
    output_directory = frames_root / recording
    output_csv = output_directory / f"{recording}.csv"
    frames_directory = output_directory / "frames"
    if not output_csv.is_file() or not frames_directory.is_dir():
        return False
    expected = (csv_row_count(source_csv) + 2) // 3
    if csv_row_count(output_csv) != expected:
        return False
    return sum(1 for _ in frames_directory.glob("frame_*.png")) == expected


def preprocessing_layout_complete(layout_root: Path, sequences: list[str]) -> bool:
    split_copy = layout_root / "label_root" / "holdout_splits.json"
    if not split_copy.is_file() or not (layout_root / "layout_summary.json").is_file():
        return False
    return all(
        (layout_root / "label_root" / sequence / "label.txt").is_file()
        and (
            layout_root
            / "image_root"
            / sequence
            / "frames_10fps_stride3"
        ).is_dir()
        for sequence in sequences
    )


def model_dataset_complete(root: Path, sequences: list[str]) -> bool:
    return packs_complete(root, sequences) and masks_complete(root, sequences)


def packs_complete(root: Path, sequences: list[str]) -> bool:
    return all(
        (root / "packed" / sequence / "packed_depthnorm64.pt").is_file()
        for sequence in sequences
    )


def masks_complete(root: Path, sequences: list[str]) -> bool:
    return all(
        (root / "smartroi_masks" / f"{sequence}__smartroi_mask_u8.pt").is_file()
        for sequence in sequences
    )


def validate_model_dataset(root: Path, splits_path: Path) -> None:
    run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "inspect_dataset.py"),
            "--dataset-root",
            str(root),
            "--splits",
            str(splits_path),
        ]
    )


def write_dataset_metadata(root: Path, splits_path: Path, sequences: list[str]) -> None:
    metadata_root = root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(splits_path, metadata_root / "holdout_splits.json")

    rows = []
    pack_bytes = 0
    mask_bytes = 0
    for sequence in sequences:
        pack_path = root / "packed" / sequence / "packed_depthnorm64.pt"
        mask_path = root / "smartroi_masks" / f"{sequence}__smartroi_mask_u8.pt"
        try:
            pack = torch.load(
                pack_path, map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:
            pack = torch.load(pack_path, map_location="cpu")
        rows.append(
            {
                "sequence": sequence,
                "vehicle": sequence.rsplit("_", 1)[0].upper(),
                "frames": len(pack["frames"]),
            }
        )
        pack_bytes += pack_path.stat().st_size
        mask_bytes += mask_path.stat().st_size

    with (metadata_root / "sequence_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "vehicle", "frames"])
        writer.writeheader()
        writer.writerows(rows)

    release_metadata = {
        "release": "EgoSpeed model-ready 48x86 dataset",
        "clip_length": 13,
        "input_channels": ["grayscale", "raw_flow_u", "raw_flow_v"],
        "smartroi_source": "MiDaS relative depth and RAFT flow at 1920x1080",
        "depthnorm_input": False,
        "layout_version": 2,
        "sequence_naming": "<vehicle_model>_<recording_number>",
        "logical_sequences": len(sequences),
        "physical_packs": len(sequences),
        "physical_masks": len(sequences),
        "pack_bytes": pack_bytes,
        "mask_bytes": mask_bytes,
        "pack_keys_required": [
            "frames",
            "flow_rate64",
            "speeds_mps",
            "frame_indices",
        ],
    }
    (metadata_root / "release_metadata.json").write_text(
        json.dumps(release_metadata, indent=2) + "\n", encoding="utf-8"
    )

    readme = """# EgoSpeed Model-Ready Dataset

This directory was generated by `EgoSpeed-SmartROI/scripts/preprocessing/build_model_ready.py`.
It contains 48 x 86 grayscale/flow tensor packs, synchronized speed labels,
and precomputed SmartROI masks. Place this directory next to the
`EgoSpeed-SmartROI` repository and run `python scripts/inspect_dataset.py` from
the repository root before training or evaluation.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    original_root = discover_original_root(args.original_dataset_root)
    output_root = output_root_from_args(args.output_root)
    work_root = (
        args.work_root.expanduser().resolve()
        if args.work_root is not None
        else (REPOSITORY_ROOT.parent / "EgoSpeedPreprocessWork").resolve()
    )
    splits_path = args.splits_json.expanduser().resolve()
    _, sequences = read_splits(splits_path)

    print(f"Repository:       {REPOSITORY_ROOT}")
    print(f"Original dataset: {original_root}")
    print(f"Model dataset:    {output_root}")
    print(f"Work directory:   {work_root}")
    print(f"Sequences:        {len(sequences)}")

    original_failures = validate_original(original_root)
    if original_failures:
        raise RuntimeError("Original dataset validation failed:\n- " + "\n- ".join(original_failures))
    print("Original dataset validation: PASS")

    if model_dataset_complete(output_root, sequences) and not args.overwrite:
        validate_model_dataset(output_root, splits_path)
        print("Model-ready dataset already complete; no preprocessing was needed.")
        return
    if args.validate_only:
        raise RuntimeError(f"Model-ready dataset is incomplete: {output_root}")

    require_preprocessing_environment()
    free_bytes = shutil.disk_usage(work_root.parent).free
    print(f"Available work-disk space: {free_bytes / (1024**3):.1f} GiB")
    if free_bytes < 250 * 1024**3:
        print(
            "WARNING: full-resolution PNG extraction can require hundreds of GiB; "
            "consider moving --work-root to a larger disk.",
            flush=True,
        )

    frames_root = work_root / "frames_10fps"
    layout_root = work_root / "layout"
    frames_root.mkdir(parents=True, exist_ok=True)
    recording_names = sorted(
        path.name for path in (original_root / "recordings").iterdir() if path.is_dir()
    )
    incomplete = [
        name
        for name in recording_names
        if args.overwrite
        or not extracted_recording_complete(original_root, frames_root, name)
    ]
    if incomplete:
        command = [
            sys.executable,
            str(Path(__file__).with_name("extract_frames.py")),
            "--dataset-root",
            str(original_root),
            "--output-dir",
            str(frames_root),
            "--output-fps",
            "10",
        ]
        for recording in incomplete:
            command.extend(["--recording", recording])
        if args.overwrite or any((frames_root / name).exists() for name in incomplete):
            command.append("--overwrite")
        run(command)
    else:
        print("10 FPS extraction: already complete")

    if not all(
        extracted_recording_complete(original_root, frames_root, name)
        for name in recording_names
    ):
        raise RuntimeError("10 FPS extraction validation failed")

    if args.overwrite or not preprocessing_layout_complete(layout_root, sequences):
        command = [
            sys.executable,
            str(Path(__file__).with_name("prepare_public_layout.py")),
            "--frames-root",
            str(frames_root),
            "--splits-json",
            str(splits_path),
            "--output-root",
            str(layout_root),
        ]
        if args.overwrite or layout_root.exists():
            command.append("--overwrite")
        if args.copy_frames:
            command.append("--copy-frames")
        run(command)
    else:
        print("Preprocessing layout: already complete")

    tensor_command = [
        sys.executable,
        str(Path(__file__).with_name("preprocess_48x86_tensors.py")),
        "--label_root",
        str(layout_root / "label_root"),
        "--image_root",
        str(layout_root / "image_root"),
        "--out_root",
        str(output_root / "packed"),
        "--splits_json",
        str(layout_root / "label_root" / "holdout_splits.json"),
        "--target_hw",
        "360,640",
        "--out_hw",
        "48,86",
        "--fps",
        "10",
        "--depth_batch",
        str(args.depth_batch),
        "--flow_batch",
        str(args.flow_batch),
        "--gpu",
        str(args.gpu),
    ]
    if args.overwrite:
        tensor_command.append("--overwrite")
    if args.overwrite or not packs_complete(output_root, sequences):
        run(tensor_command)
    else:
        print("48 x 86 tensor packs: already complete")

    mask_command = [
        sys.executable,
        str(Path(__file__).with_name("precompute_smartroi.py")),
        "--label_root",
        str(layout_root / "label_root"),
        "--image_root",
        str(layout_root / "image_root"),
        "--out_mask_root",
        str(output_root / "smartroi_masks"),
        "--splits_json",
        str(layout_root / "label_root" / "holdout_splits.json"),
        "--source_hw",
        "1080,1920",
        "--out_hw",
        "48,86",
        "--window_t",
        "13",
        "--chunk_centers",
        str(args.smartroi_chunk_centers),
        "--mask_batch",
        "1",
        "--depth_batch",
        str(args.smartroi_depth_batch),
        "--flow_batch",
        str(args.smartroi_flow_batch),
        "--fps",
        "10",
        "--depth_low_pct",
        "0.15",
        "--depth_high_pct",
        "0.85",
        "--motion_thresh",
        "0.02",
        "--consistency_thresh",
        "0.3",
        "--gpu",
        str(args.gpu),
    ]
    if args.amp:
        mask_command.append("--amp")
    if args.overwrite:
        mask_command.append("--overwrite")
    if args.overwrite or not masks_complete(output_root, sequences):
        run(mask_command)
    else:
        print("SmartROI masks: already complete")

    write_dataset_metadata(output_root, splits_path, sequences)
    validate_model_dataset(output_root, splits_path)
    print("\nModel-ready dataset build: PASS")


if __name__ == "__main__":
    main()
