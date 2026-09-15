#!/usr/bin/env python3
"""Compute full-resolution SmartROI and store only resized 48 x 86 masks."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from contextlib import nullcontext
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
        raise argparse.ArgumentTypeError("expected H,W or HxW")
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


def load_rgb_resized(path: Path, hw: tuple[int, int]) -> torch.Tensor:
    height, width = hw
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    return TF.to_tensor(image)


def same_label_signature(first: str, second: str, label_root: Path) -> bool:
    ids_a, speeds_a = read_labels(label_root / first / "label.txt")
    ids_b, speeds_b = read_labels(label_root / second / "label.txt")
    return ids_a == ids_b and bool(
        torch.allclose(speeds_a, speeds_b, atol=1e-4, rtol=0.0)
    )


def link_mask(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


class SmartROI:
    def __init__(
        self,
        depth_low_pct: float = 0.15,
        depth_high_pct: float = 0.85,
        motion_thresh: float = 0.02,
        consistency_thresh: float = 0.3,
    ) -> None:
        self.depth_low_pct = float(depth_low_pct)
        self.depth_high_pct = float(depth_high_pct)
        self.motion_thresh = float(motion_thresh)
        self.consistency_thresh = float(consistency_thresh)

    def __call__(self, depth: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Return a soft full-resolution mask from depth and temporal flow.

        Args:
            depth: ``(B,T,H,W)`` robust relative inverse depth.
            flow: ``(B,T,2,H,W)`` image-fraction/second flow.
        """
        depth_mean = depth.mean(dim=1)
        batch, height, width = depth_mean.shape
        flat = depth_mean.reshape(batch, -1)
        low = flat.quantile(self.depth_low_pct, dim=1, keepdim=True).view(batch, 1, 1)
        high = flat.quantile(self.depth_high_pct, dim=1, keepdim=True).view(batch, 1, 1)
        depth_mask = ((depth_mean > low) & (depth_mean < high)).float()

        flow_magnitude = torch.sqrt(
            flow[:, :, 0].square() + flow[:, :, 1].square() + 1e-6
        )
        motion_mask = (flow_magnitude.mean(dim=1) > self.motion_thresh).float()

        if flow.shape[1] > 1:
            consistency = F.cosine_similarity(
                flow[:, :-1], flow[:, 1:], dim=2
            ).mean(dim=1)
            consistency_mask = (consistency > self.consistency_thresh).float()
        else:
            consistency_mask = torch.ones(
                batch, height, width, device=depth.device
            )

        bonus = (motion_mask * 0.3 + consistency_mask * 0.2).clamp(max=0.5)
        mask = depth_mask * (0.5 + bonus)
        if height > 10 and width > 10:
            mask = F.avg_pool2d(
                mask.unsqueeze(1), kernel_size=5, stride=1, padding=2
            ).squeeze(1)
        return mask.clamp(0.0, 1.0)


def robust_relative_inverse_depth(
    depth: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    batch = depth.shape[0]
    flat = depth.reshape(batch, -1).float()
    q01 = torch.quantile(flat, 0.01, dim=1).view(batch, 1, 1, 1)
    shifted = (depth.float() - q01 + eps).clamp_min(eps)
    median = shifted.reshape(batch, -1).median(dim=1).values.view(batch, 1, 1, 1)
    return (shifted / median.clamp_min(eps)).clamp(0.25, 4.0).half()


def load_models(device: torch.device):
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
    return depth_net, midas_transforms.dpt_transform, raft, weights.transforms()


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def compute_depth_context(
    frame_ids: list[int],
    image_group: str,
    image_root: Path,
    source_hw: tuple[int, int],
    device: torch.device,
    depth_net: torch.nn.Module,
    depth_transform: object,
    batch_size: int,
    use_amp: bool,
) -> torch.Tensor:
    height, width = source_hw
    output = torch.empty(
        len(frame_ids), 1, height, width, dtype=torch.float16
    )
    for start in range(0, len(frame_ids), batch_size):
        end = min(start + batch_size, len(frame_ids))
        inputs = []
        for frame_id in frame_ids[start:end]:
            image = Image.open(image_path(image_root, image_group, frame_id)).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.BILINEAR)
            inputs.append(depth_transform(np.array(image)))
        with torch.inference_mode(), amp_context(device, use_amp):
            prediction = depth_net(
                torch.cat(inputs, dim=0).to(device, non_blocking=True)
            )
            prediction = F.interpolate(
                prediction.unsqueeze(1).float(),
                size=source_hw,
                mode="bilinear",
                align_corners=False,
            )
            relative_depth = robust_relative_inverse_depth(prediction)
        output[start:end] = relative_depth.cpu()
    return output


def compute_flow_context(
    frame_ids: list[int],
    image_group: str,
    image_root: Path,
    source_hw: tuple[int, int],
    fps: float,
    device: torch.device,
    raft: torch.nn.Module,
    raft_transform: object,
    batch_size: int,
    use_amp: bool,
) -> torch.Tensor:
    height, width = source_hw
    output = torch.zeros(
        len(frame_ids), 2, height, width, dtype=torch.float16
    )
    if len(frame_ids) <= 1:
        return output

    target_indices = list(range(1, len(frame_ids)))
    for start in range(0, len(target_indices), batch_size):
        batch_indices = target_indices[start : start + batch_size]
        previous, current = [], []
        for index in batch_indices:
            previous.append(
                load_rgb_resized(
                    image_path(image_root, image_group, frame_ids[index - 1]), source_hw
                )
            )
            current.append(
                load_rgb_resized(
                    image_path(image_root, image_group, frame_ids[index]), source_hw
                )
            )
        previous_batch, current_batch = raft_transform(
            torch.stack(previous), torch.stack(current)
        )
        with torch.inference_mode(), amp_context(device, use_amp):
            flow = raft(
                previous_batch.to(device, non_blocking=True),
                current_batch.to(device, non_blocking=True),
            )[-1].float()
            flow[:, 0].mul_(float(fps) / float(width))
            flow[:, 1].mul_(float(fps) / float(height))
        output[batch_indices] = flow.cpu().half()
    output[0] = output[1]
    return output


def process_sequence(
    sequence: str,
    args: argparse.Namespace,
    device: torch.device,
    models: tuple,
    smartroi: SmartROI,
) -> bool:
    output_path = Path(args.out_mask_root) / f"{sequence}__smartroi_mask_u8.pt"
    if output_path.exists() and not args.overwrite:
        print(f"  [SKIP] exists: {output_path}", flush=True)
        return True

    label_path = Path(args.label_root) / sequence / "label.txt"
    frame_ids, speeds_kmh = read_labels(label_path)
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
        speeds_kmh = speeds_kmh[: args.max_frames]
    count = len(frame_ids)
    if count == 0:
        print(f"  [SKIP] empty: {sequence}", flush=True)
        return False

    image_group = sequence_to_image_group(sequence)
    first_frame = image_path(Path(args.image_root), image_group, frame_ids[0])
    if not first_frame.exists():
        raise FileNotFoundError(f"{sequence}: missing frame: {first_frame}")

    output_masks = torch.empty(
        count, args.out_hw[0], args.out_hw[1], dtype=torch.uint8
    )
    half_window = int(args.window_t) // 2
    depth_net, depth_transform, raft, raft_transform = models
    started = time.time()
    print(
        f"  sequence={sequence} N={count} "
        f"SmartROI@{args.source_hw[1]}x{args.source_hw[0]} "
        f"-> {args.out_hw[1]}x{args.out_hw[0]}",
        flush=True,
    )

    for center_start in range(0, count, args.chunk_centers):
        center_end = min(center_start + args.chunk_centers, count)
        context_start = max(0, center_start - half_window - 1)
        context_end = min(count, center_end + half_window)
        context_ids = frame_ids[context_start:context_end]

        depth_context = compute_depth_context(
            context_ids,
            image_group,
            Path(args.image_root),
            args.source_hw,
            device,
            depth_net,
            depth_transform,
            args.depth_batch,
            args.amp,
        )
        flow_context = compute_flow_context(
            context_ids,
            image_group,
            Path(args.image_root),
            args.source_hw,
            args.fps,
            device,
            raft,
            raft_transform,
            args.flow_batch,
            args.amp,
        )

        relative_positions = torch.arange(args.window_t, dtype=torch.long) - half_window
        for batch_start in range(center_start, center_end, args.mask_batch):
            batch_end = min(batch_start + args.mask_batch, center_end)
            centers = torch.arange(batch_start, batch_end, dtype=torch.long).unsqueeze(1)
            global_indices = (centers + relative_positions.unsqueeze(0)).clamp(
                0, count - 1
            )
            local_indices = (global_indices - context_start).clamp(
                0, len(context_ids) - 1
            )
            depth_window = (
                depth_context[local_indices]
                .squeeze(2)
                .to(device, non_blocking=True)
                .float()
            )
            flow_window = flow_context[local_indices].to(
                device, non_blocking=True
            ).float()
            with torch.inference_mode():
                full_mask = smartroi(depth_window, flow_window)
                small_mask = F.interpolate(
                    full_mask.unsqueeze(1),
                    size=args.out_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            output_masks[batch_start:batch_end] = (
                (small_mask.clamp(0, 1) * 255.0)
                .round()
                .to(torch.uint8)
                .cpu()
            )

        del depth_context, flow_context
        print(
            f"    masks {center_end}/{count} "
            f"elapsed={(time.time() - started) / 60:.1f} min",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "masks_u8": output_masks,
            "method": "SmartROI_stream_fullres_to_u8",
            "source_hw": tuple(args.source_hw),
            "mask_hw": tuple(args.out_hw),
            "source_frames": "frames_10fps_stride3",
            "image_group": image_group,
            "fps": float(args.fps),
            "window_t": int(args.window_t),
            "depth_low_pct": float(args.depth_low_pct),
            "depth_high_pct": float(args.depth_high_pct),
            "motion_thresh": float(args.motion_thresh),
            "consistency_thresh": float(args.consistency_thresh),
            "speeds_mps": (speeds_kmh / 3.6).half(),
            "frame_indices": torch.tensor(frame_ids, dtype=torch.int32),
        },
        output_path,
    )
    print(
        f"  [OK] {sequence}: {output_path} "
        f"mean={float(output_masks.float().mean() / 255.0):.3f}",
        flush=True,
    )
    return True


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--out_mask_root", required=True)
    parser.add_argument("--splits_json", default="")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--source_hw", type=parse_hw, default=(1080, 1920))
    parser.add_argument("--out_hw", type=parse_hw, default=(48, 86))
    parser.add_argument("--window_t", type=int, default=13)
    parser.add_argument("--chunk_centers", type=int, default=32)
    parser.add_argument("--mask_batch", type=int, default=1)
    parser.add_argument("--depth_batch", type=int, default=1)
    parser.add_argument("--flow_batch", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0, help="Debug only")
    parser.add_argument("--depth_low_pct", type=float, default=0.15)
    parser.add_argument("--depth_high_pct", type=float, default=0.85)
    parser.add_argument("--motion_thresh", type=float, default=0.02)
    parser.add_argument("--consistency_thresh", type=float, default=0.3)
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
    smartroi = SmartROI(
        depth_low_pct=args.depth_low_pct,
        depth_high_pct=args.depth_high_pct,
        motion_thresh=args.motion_thresh,
        consistency_thresh=args.consistency_thresh,
    )

    successful = 0
    for group_index, (image_group, group_sequences) in enumerate(sorted(groups.items()), 1):
        group_sequences = sorted(group_sequences)
        primary = group_sequences[0]
        print(
            f"\n[{group_index}/{len(groups)}] {image_group}: "
            f"{len(group_sequences)} sequence(s)",
            flush=True,
        )
        primary_ok = process_sequence(primary, args, device, models, smartroi)
        successful += int(primary_ok)
        primary_path = Path(args.out_mask_root) / f"{primary}__smartroi_mask_u8.pt"

        for sequence in group_sequences[1:]:
            destination = Path(args.out_mask_root) / f"{sequence}__smartroi_mask_u8.pt"
            if destination.exists() and not args.overwrite:
                successful += 1
                continue
            if primary_ok and same_label_signature(primary, sequence, label_root):
                link_mask(primary_path, destination)
                print(f"  link {sequence} -> {primary}", flush=True)
                successful += 1
            else:
                successful += int(
                    process_sequence(sequence, args, device, models, smartroi)
                )

    print(f"\nDone. successful={successful}/{len(sequences)}")


if __name__ == "__main__":
    main()
