"""Shared training and evaluation routines."""

from __future__ import annotations

import numpy as np
import torch


def predict(model, loader, device: torch.device):
    model.eval()
    ground_truth = []
    prediction = []
    with torch.no_grad():
        for clip, speed, _vehicle_id, mask in loader:
            clip = clip.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            output = model.predict_speed(clip, mask)
            ground_truth.append(speed.numpy().reshape(-1))
            prediction.append(output.cpu().numpy().reshape(-1))
    return np.concatenate(ground_truth), np.concatenate(prediction)


def mean_absolute_error(ground_truth, prediction) -> float:
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    return float(np.abs(prediction - ground_truth).mean())
