#!/usr/bin/env python3
"""Build 48 x 86 grayscale, RAFT-flow, speed, and depth tensor packs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def sequence_to_image_group(sequence: str) -> str:
    without_date = re.sub(r"^\d{4}_\d{2}_\d{2}_", "", sequence)
    return re.sub(r"_\d{4}_sync$", "", without_date)


def parse_hw(spec: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(spec).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected H,W, e.g. 48,86")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("H,W must be positive")
    return height, width


def read_labels(path: Path) -> tuple[list[int], torch.Tensor]:
    frame_ids: list[int] = []
    speeds_kmh: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            frame_ids.append(int(parts[0].replace(".png", "")))
            speeds_kmh.append(float(parts[1]))
    if frame_ids != list(range(len(frame_ids))):
        raise RuntimeError(f"Frame IDs must be contiguous from zero: {path}")
    return frame_ids, torch.tensor(speeds_kmh, dtype=torch.float32)


def load_sequences(label_root: Path, splits_json: Path | None, sequences: str) -> list[str]:
    if sequences:
        return sorted({item.strip() for item in sequences.split(",") if item.strip()})
    if splits_json and splits_json.exists():
        with splits_json.open(encoding="utf-8") as handle:
            splits = json.load(handle)
        return sorted(
            {
                sequence
                for fold in splits.values()
                for partition in ("train", "valid", "test")
                for sequence in fold.get(partition, [])
            }
        )
    return sorted(path.name for path in label_root.iterdir() if path.is_dir())


def image_path(image_root: Path, image_group: str, frame_id: int) -> Path:
    return (
        image_root
        / image_group
        / "frames_10fps_stride3"
        / f"frame_{frame_id:06d}.png"
    )


def load_rgb_resized(path: Path, target_hw: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image)
    return TF.resize(tensor, list(target_hw), antialias=True)


def build_grayscale_frames(
    frame_ids: list[int],
    image_group: str,
    image_root: Path,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    frames = torch.empty(
        len(frame_ids), 1, out_hw[0], out_hw[1], dtype=torch.float16
    )
    for index, frame_id in enumerate(frame_ids):
        image = Image.open(image_path(image_root, image_group, frame_id)).convert("RGB")
        rgb = TF.to_tensor(image)
        gray = TF.resize(TF.rgb_to_grayscale(rgb), list(out_hw), antialias=True)
        frames[index] = (gray * 2.0 - 1.0).half()
    return frames


@dataclass
class Models:
    depth_net: torch.nn.Module
    depth_transform: object
    raft: torch.nn.Module
    raft_transform: object


def load_models(device: torch.device) -> Models:
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    print("Loading MiDaS DPT_Large...", flush=True)
    depth_net = torch.hub.load(
        "intel-isl/MiDaS", "DPT_Large", trust_repo=True
    ).to(device).eval()
    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", trust_repo=True
    )

    print("Loading torchvision RAFT-Large...", flush=True)
    weights = Raft_Large_Weights.DEFAULT
    raft = raft_large(weights=weights, progress=True).to(device).eval()
    return Models(
        depth_net=depth_net,
        depth_transform=midas_transforms.dpt_transform,
        raft=raft,
        raft_transform=weights.transforms(),
    )


def robust_relative_inverse_depth(
    depth: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    batch = depth.shape[0]
    flat = depth.reshape(batch, -1).float()
    q01 = torch.quantile(flat, 0.01, dim=1).view(batch, 1, 1, 1)
    shifted = (depth.float() - q01 + eps).clamp_min(eps)
    median = shifted.reshape(batch, -1).median(dim=1).values.view(batch, 1, 1, 1)
    return (shifted / median.clamp_min(eps)).clamp(0.25, 4.0).half()


def compute_depth(
    frame_ids: list[int],
    image_group: str,
    image_root: Path,
    target_hw: tuple[int, int],
    out_hw: tuple[int, int],
    device: torch.device,
    depth_net: torch.nn.Module,
    depth_transform: object,
    batch_size: int,
) -> torch.Tensor:
    count = len(frame_ids)
    height, width = target_hw
    output = torch.empty(
        count, 1, out_hw[0], out_hw[1], dtype=torch.float16
    )
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        inputs = []
        for frame_id in frame_ids[start:end]:
            image = Image.open(image_path(image_root, image_group, frame_id)).convert("RGB")
            image = image.resize((width, height), Image.BILINEAR)
            inputs.append(depth_transform(np.array(image)))
        with torch.inference_mode():
            prediction = depth_net(torch.cat(inputs, dim=0).to(device))
            prediction = F.interpolate(
                prediction.unsqueeze(1).float(),
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )
            relative_depth = robust_relative_inverse_depth(prediction)
        output[start:end] = relative_depth.cpu()
        print(f"    depth {end}/{count}", flush=True)
    return output


def compute_flow_rates(
    frame_ids: list[int],
    image_group: str,
    image_root: Path,
    target_hw: tuple[int, int],
    out_hw: tuple[int, int],
    fps: float,
    device: torch.device,
    raft: torch.nn.Module,
    raft_transform: object,
    batch_size: int,
) -> torch.Tensor:
    count = len(frame_ids)
    height, width = target_hw
    output = torch.zeros(count, 2, out_hw[0], out_hw[1], dtype=torch.float16)
    if count <= 1:
        return output

    target_indices = list(range(1, count))
    for start in range(0, len(target_indices), batch_size):
        batch_indices = target_indices[start : start + batch_size]
        previous, current = [], []
        for index in batch_indices:
            previous.append(
                load_rgb_resized(
                    image_path(image_root, image_group, frame_ids[index - 1]), target_hw
                )
            )
            current.append(
                load_rgb_resized(
                    image_path(image_root, image_group, frame_ids[index]), target_hw
                )
            )

        previous_batch, current_batch = raft_transform(
            torch.stack(previous), torch.stack(current)
        )
        with torch.inference_mode():
            flow = raft(
                previous_batch.to(device), current_batch.to(device)
            )[-1].float()
            flow[:, 0].mul_(float(fps) / float(width))
            flow[:, 1].mul_(float(fps) / float(height))
            small_flow = F.interpolate(
                flow, size=out_hw, mode="bilinear", align_corners=False
            )
        output[batch_indices] = small_flow.cpu().half()
        print(f"    flow {batch_indices[-1] + 1}/{count}", flush=True)

    output[0] = output[1]
    return output


def same_label_signature(first: str, second: str, label_root: Path) -> bool:
    ids_a, speeds_a = read_labels(label_root / first / "label.txt")
    ids_b, speeds_b = read_labels(label_root / second / "label.txt")
    return ids_a == ids_b and bool(
        torch.allclose(speeds_a, speeds_b, atol=1e-4, rtol=0.0)
    )


def link_pack(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def process_sequence(
    sequence: str,
    args: argparse.Namespace,
    device: torch.device,
    models: Models,
) -> Path:
    output_path = Path(args.out_root) / sequence / args.out_name
    if output_path.exists() and not args.overwrite:
        print(f"  [SKIP] exists: {output_path}", flush=True)
        return output_path

    label_path = Path(args.label_root) / sequence / "label.txt"
    frame_ids, speeds_kmh = read_labels(label_path)
    if not frame_ids:
        raise RuntimeError(f"{sequence}: empty label file")
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
        speeds_kmh = speeds_kmh[: args.max_frames]

    image_group = sequence_to_image_group(sequence)
    first_frame = image_path(Path(args.image_root), image_group, frame_ids[0])
    if not first_frame.exists():
        raise FileNotFoundError(f"{sequence}: missing frame: {first_frame}")

    print(
        f"  sequence={sequence} N={len(frame_ids)} "
        f"flow@{args.target_hw[1]}x{args.target_hw[0]} "
        f"store={args.out_hw[1]}x{args.out_hw[0]}",
        flush=True,
    )
    started = time.time()
    frames = build_grayscale_frames(
        frame_ids, image_group, Path(args.image_root), args.out_hw
    )
    depth = compute_depth(
        frame_ids,
        image_group,
        Path(args.image_root),
        args.target_hw,
        args.out_hw,
        device,
        models.depth_net,
        models.depth_transform,
        args.depth_batch,
    )
    flow = compute_flow_rates(
        frame_ids,
        image_group,
        Path(args.image_root),
        args.target_hw,
        args.out_hw,
        args.fps,
        device,
        models.raft,
        models.raft_transform,
        args.flow_batch,
    )

    payload = {
        "frames": frames,
        "speeds_mps": (speeds_kmh / 3.6).float(),
        "frame_indices": torch.tensor(frame_ids, dtype=torch.int32),
        "flow_rate64": flow,
        "depth_rel64": depth,
        "meta": {
            "method": "depthnorm64_v2",
            "source_frames": "frames_10fps_stride3",
            "image_group": image_group,
            "source_hw": (1080, 1920),
            "flow_compute_hw": tuple(args.target_hw),
            "stored_hw": tuple(args.out_hw),
            "fps": float(args.fps),
            "flow_units": "image_fraction_per_second",
            "flow_formula": "u_rate=u_px/W*fps, v_rate=v_px/H*fps",
            "depth_units": "robust_relative_inverse_depth_per_frame",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"  [OK] {output_path} ({output_path.stat().st_size / 1e6:.1f} MB, "
        f"{(time.time() - started) / 60:.1f} min)",
        flush=True,
    )
    return output_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--splits_json", default="")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--out_name", default="packed_depthnorm64.pt")
    parser.add_argument("--target_hw", type=parse_hw, default=(360, 640))
    parser.add_argument("--out_hw", type=parse_hw, default=(48, 86))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--depth_batch", type=int, default=4)
    parser.add_argument("--flow_batch", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0, help="Debug only")
    args = parser.parse_args(argv)

    label_root = Path(args.label_root)
    splits_path = Path(args.splits_json) if args.splits_json else None
    sequences = load_sequences(label_root, splits_path, args.sequences)
    if not sequences:
        raise RuntimeError("No sequences to process")

    groups: dict[str, list[str]] = {}
    for sequence in sequences:
        groups.setdefault(sequence_to_image_group(sequence), []).append(sequence)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Sequences: {len(sequences)} | image groups: {len(groups)}")
    models = load_models(device)

    for group_index, (image_group, group_sequences) in enumerate(sorted(groups.items()), 1):
        group_sequences = sorted(group_sequences)
        primary = group_sequences[0]
        print(
            f"\n[{group_index}/{len(groups)}] {image_group}: "
            f"{len(group_sequences)} sequence(s)",
            flush=True,
        )
        primary_path = process_sequence(primary, args, device, models)
        for sequence in group_sequences[1:]:
            destination = Path(args.out_root) / sequence / args.out_name
            if destination.exists() and not args.overwrite:
                continue
            if same_label_signature(primary, sequence, label_root):
                link_pack(primary_path, destination)
                print(f"  link {sequence} -> {primary}", flush=True)
            else:
                process_sequence(sequence, args, device, models)

    print("\nDone.")


if __name__ == "__main__":
    main()
