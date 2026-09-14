"""Regression metrics used in the paper."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    ground_truth,
    prediction,
    *,
    min_speed_mps: float = 0.5,
    max_speed_mps: float = 20.0,
) -> dict[str, float | int]:
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if ground_truth.shape != prediction.shape:
        raise ValueError("ground_truth and prediction must have the same shape")
    selected = (ground_truth >= min_speed_mps) & (ground_truth < max_speed_mps)
    if not np.any(selected):
        raise ValueError("No predictions fall inside the requested evaluation range")
    error = prediction[selected] - ground_truth[selected]
    return {
        "n": int(selected.sum()),
        "mae_mps": float(np.abs(error).mean()),
        "rmse_mps": float(np.sqrt(np.square(error).mean())),
        "bias_mps": float(error.mean()),
    }
