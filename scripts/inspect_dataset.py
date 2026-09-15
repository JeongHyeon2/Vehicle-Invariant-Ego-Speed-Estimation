#!/usr/bin/env python3
"""Validate a downloaded model-ready dataset before training."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from egospeed.data import resolve_dataset_roots


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        help="Directory containing packed/ and smartroi_masks/ (auto-detected if omitted)",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--mask-root")
    parser.add_argument("--splits", default="splits/holdout_splits.json")
    args = parser.parse_args()
    try:
        data_root, mask_root = resolve_dataset_roots(
            args.dataset_root,
            args.data_root,
            args.mask_root,
        )
    except ValueError as error:
        parser.error(str(error))

    with Path(args.splits).open(encoding="utf-8") as handle:
        splits = json.load(handle)
    sequences = sorted(
        {
            sequence
            for fold in splits.values()
            for subset in ("train", "valid", "test")
            for sequence in fold.get(subset, [])
        }
    )

    failures = []
    physical_packs = set()
    total_frames = 0
    dataset_root = data_root.parent if data_root.name == "packed" else None
    manifest_rows = {}
    if dataset_root is not None:
        manifest_path = dataset_root / "metadata" / "sequence_manifest.csv"
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
                manifest_rows = {
                    row["sequence"]: int(row["frames"])
                    for row in csv.DictReader(handle)
                }
        metadata_splits = dataset_root / "metadata" / "holdout_splits.json"
        if metadata_splits.is_file():
            with metadata_splits.open(encoding="utf-8") as handle:
                dataset_splits = json.load(handle)
            if dataset_splits != splits:
                failures.append(
                    f"dataset split metadata differs from {Path(args.splits).resolve()}"
                )
    for sequence in sequences:
        pack_path = data_root / sequence / "packed_depthnorm64.pt"
        mask_path = mask_root / f"{sequence}__smartroi_mask_u8.pt"
        if not pack_path.exists():
            failures.append(f"missing pack: {pack_path}")
            continue
        if not mask_path.exists():
            failures.append(f"missing mask: {mask_path}")
            continue

        pack = torch_load(pack_path)
        mask_payload = torch_load(mask_path)
        missing = {"frames", "flow_rate64", "speeds_mps", "frame_indices"}.difference(pack)
        if missing:
            failures.append(f"{sequence}: missing keys {sorted(missing)}")
            continue
        mask = mask_payload.get("masks_u8", mask_payload.get("masks"))
        if mask is None:
            failures.append(f"{sequence}: mask file has no masks_u8/masks tensor")
            continue

        frames = pack["frames"]
        flow = pack["flow_rate64"]
        speeds = pack["speeds_mps"]
        frame_indices = pack["frame_indices"]
        if tuple(frames.shape[1:]) != (1, 48, 86):
            failures.append(f"{sequence}: frames shape is {tuple(frames.shape)}")
        if tuple(flow.shape[1:]) != (2, 48, 86):
            failures.append(f"{sequence}: flow shape is {tuple(flow.shape)}")
        if tuple(mask.shape[1:]) != (48, 86):
            failures.append(f"{sequence}: mask shape is {tuple(mask.shape)}")
        if not (
            len(frames)
            == len(flow)
            == len(speeds)
            == len(frame_indices)
            == len(mask)
        ):
            failures.append(f"{sequence}: inconsistent frame counts")
        if frames.dtype != torch.float16:
            failures.append(f"{sequence}: frames dtype is {frames.dtype}, expected float16")
        if flow.dtype != torch.float16:
            failures.append(f"{sequence}: flow dtype is {flow.dtype}, expected float16")
        if speeds.dtype != torch.float32:
            failures.append(f"{sequence}: speeds dtype is {speeds.dtype}, expected float32")
        if mask.dtype != torch.uint8:
            failures.append(f"{sequence}: mask dtype is {mask.dtype}, expected uint8")
        expected_indices = torch.arange(len(frame_indices), dtype=frame_indices.dtype)
        if not torch.equal(frame_indices.cpu(), expected_indices):
            failures.append(f"{sequence}: frame_indices are not contiguous from zero")
        mask_indices = mask_payload.get("frame_indices")
        if mask_indices is not None and not torch.equal(
            frame_indices.cpu().to(mask_indices.dtype), mask_indices.cpu()
        ):
            failures.append(f"{sequence}: pack/mask frame_indices differ")
        if sequence in manifest_rows and manifest_rows[sequence] != len(frames):
            failures.append(
                f"{sequence}: manifest frames={manifest_rows[sequence]}, pack={len(frames)}"
            )
        total_frames += len(frames)
        physical_packs.add(os.path.realpath(pack_path))

    report = {
        "logical_sequences": len(sequences),
        "physical_packs": len(physical_packs),
        "logical_frames_across_split_entries": total_frames,
        "metadata_sequences": len(manifest_rows),
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
