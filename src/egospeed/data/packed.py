"""Dataset loader for the released model-ready EgoSpeed tensors."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


VEHICLE_ALIASES = {
    "AVANTE": "AVANTE",
    "MALIBU": "MALIBU",
    "SONATA": "SONATA",
    "CARNIVAL": "CARNIVAL",
    "SUV": "CARNIVAL",  # Legacy checkpoint and experiment token.
    "XM3": "XM3",
}


def _torch_load(path: Path, *, mmap: bool = False):
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, mmap=mmap, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_splits(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def extract_vehicle(sequence_name: str) -> str:
    for token in sequence_name.upper().split("_"):
        if token in VEHICLE_ALIASES:
            return VEHICLE_ALIASES[token]
    raise ValueError(f"Cannot infer vehicle from sequence name: {sequence_name}")


def build_vehicle_map(sequences: Iterable[str]) -> dict[str, int]:
    vehicles = sorted({extract_vehicle(name) for name in sequences})
    return {vehicle: index for index, vehicle in enumerate(vehicles)}


class PackedEgoSpeedDataset(Dataset):
    """Return C=3 clips, clip-level speed targets, vehicle IDs, and SmartROI masks.

    A packed sequence must provide `frames`, `flow_rate64`, and `speeds_mps`.
    `depth_rel64` may be present for provenance but is not consumed by the final
    model. Precomputed SmartROI masks are loaded from a separate directory.
    """

    def __init__(
        self,
        sequences: Iterable[str],
        vehicle_map: dict[str, int],
        data_root: str | Path,
        mask_root: str | Path,
        *,
        pack_name: str = "packed_depthnorm64.pt",
        clip_length: int = 13,
        stride: int = 10,
        temporal_split: str | None = None,
        split_ratio: float = 0.8,
        min_speed_mps: float | None = 0.5,
        max_speed_mps: float | None = None,
        flow_rate_clip: float = 2.0,
        target_weight_start: float = 1.0,
        target_weight_end: float = 2.0,
        cache: dict | None = None,
    ) -> None:
        if temporal_split not in {None, "train", "val"}:
            raise ValueError("temporal_split must be None, 'train', or 'val'")
        if not 0.0 < split_ratio < 1.0:
            raise ValueError("split_ratio must lie between 0 and 1")

        self.data_root = Path(data_root)
        self.mask_root = Path(mask_root)
        # Published sequences use CARNIVAL, while the released seed-42
        # checkpoints may still store the historical SUV class key.
        self.vehicle_map = {
            VEHICLE_ALIASES.get(vehicle.upper(), vehicle.upper()): index
            for vehicle, index in vehicle_map.items()
        }
        self.pack_name = pack_name
        self.clip_length = int(clip_length)
        self.flow_rate_clip = float(flow_rate_clip)
        self._weights = torch.linspace(
            float(target_weight_start),
            float(target_weight_end),
            steps=self.clip_length,
        )
        self._packs = cache if cache is not None else {}
        self._masks: dict[str, tuple[torch.Tensor, float]] = {}
        self.samples: list[tuple[str, str, int, float, int]] = []

        for sequence_name in sequences:
            pack_path = self.data_root / sequence_name / self.pack_name
            mask_path = self.mask_root / f"{sequence_name}__smartroi_mask_u8.pt"
            if not pack_path.exists():
                raise FileNotFoundError(f"Missing packed sequence: {pack_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing SmartROI mask: {mask_path}")

            real_path = os.path.realpath(pack_path)
            if real_path not in self._packs:
                packed = _torch_load(Path(real_path), mmap=True)
                required = {"frames", "flow_rate64", "speeds_mps"}
                missing = required.difference(packed)
                if missing:
                    raise KeyError(f"{pack_path} is missing keys: {sorted(missing)}")
                self._packs[real_path] = {
                    "frames": packed["frames"],
                    "flow": packed["flow_rate64"],
                    "speeds": packed["speeds_mps"],
                }

            mask_payload = _torch_load(mask_path, mmap=True)
            if "masks_u8" in mask_payload:
                mask = mask_payload["masks_u8"]
                mask_scale = 255.0
            elif "masks" in mask_payload:
                mask = mask_payload["masks"]
                mask_scale = 1.0
            else:
                raise KeyError(f"{mask_path} must contain 'masks_u8' or 'masks'")

            packed = self._packs[real_path]
            if tuple(mask.shape[-2:]) != tuple(packed["frames"].shape[-2:]):
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=tuple(packed["frames"].shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                mask_scale = 1.0
            self._masks[sequence_name] = (mask, mask_scale)

            n_frames = min(
                len(packed["frames"]),
                len(packed["flow"]),
                len(packed["speeds"]),
                len(mask),
            )
            if n_frames < self.clip_length:
                continue

            boundary = int(n_frames * split_ratio)
            if temporal_split == "train":
                frame_start = 0
                # Preserve the one-clip guard gap used by the final experiments.
                frame_end = boundary - self.clip_length
            elif temporal_split == "val":
                frame_start = boundary
                frame_end = n_frames
            else:
                frame_start = 0
                frame_end = n_frames

            last_start = min(frame_end, n_frames) - self.clip_length
            if last_start < frame_start:
                continue

            vehicle_id = self.vehicle_map.get(extract_vehicle(sequence_name), -1)
            for start in range(max(0, frame_start), last_start + 1, int(stride)):
                frame_speeds = packed["speeds"][start : start + self.clip_length].float()
                target = float((frame_speeds * self._weights).sum() / self._weights.sum())
                if min_speed_mps is not None and target < min_speed_mps:
                    continue
                if max_speed_mps is not None and target >= max_speed_mps:
                    continue
                self.samples.append((real_path, sequence_name, start, target, vehicle_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        real_path, sequence_name, start, target, vehicle_id = self.samples[index]
        packed = self._packs[real_path]
        stop = start + self.clip_length
        gray = packed["frames"][start:stop].float()
        flow = packed["flow"][start:stop].float()
        if self.flow_rate_clip > 0:
            flow = flow.clamp(-self.flow_rate_clip, self.flow_rate_clip)
            flow = flow / self.flow_rate_clip
        clip = torch.cat([gray, flow], dim=1)

        masks, scale = self._masks[sequence_name]
        mask = (masks[start:stop].float() / scale).clamp(0.0, 1.0)
        return (
            clip,
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(vehicle_id, dtype=torch.long),
            mask,
        )
