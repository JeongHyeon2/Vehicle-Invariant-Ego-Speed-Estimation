#!/usr/bin/env python3
"""Validate a downloaded model-ready dataset before training."""

from __future__ import annotations

import argparse
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
        help="Directory containing packed/ and smartroi_masks/",
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
        missing = {"frames", "flow_rate64", "speeds_mps"}.difference(pack)
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
        if tuple(frames.shape[1:]) != (1, 48, 86):
            failures.append(f"{sequence}: frames shape is {tuple(frames.shape)}")
        if tuple(flow.shape[1:]) != (2, 48, 86):
            failures.append(f"{sequence}: flow shape is {tuple(flow.shape)}")
        if tuple(mask.shape[1:]) != (48, 86):
            failures.append(f"{sequence}: mask shape is {tuple(mask.shape)}")
        if not (len(frames) == len(flow) == len(speeds) == len(mask)):
            failures.append(f"{sequence}: inconsistent frame counts")
        total_frames += len(frames)
        physical_packs.add(os.path.realpath(pack_path))

    report = {
        "logical_sequences": len(sequences),
        "physical_packs": len(physical_packs),
        "logical_frames_across_split_entries": total_frames,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
