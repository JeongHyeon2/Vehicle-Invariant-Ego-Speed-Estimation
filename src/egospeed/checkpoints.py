"""Checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from egospeed.models import EgoSpeedSmartROI


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = payload.get("model", payload)
    vehicle_map = payload.get("vehicle_map", {}) if isinstance(payload, dict) else {}
    num_vehicles = len(vehicle_map) or 4
    num_blocks = int(payload.get("num_blocks", 1)) if isinstance(payload, dict) else 1
    model = EgoSpeedSmartROI(num_vehicles=num_vehicles, num_blocks=num_blocks).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload
