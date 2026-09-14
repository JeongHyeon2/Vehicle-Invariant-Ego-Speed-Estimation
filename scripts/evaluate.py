#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from egospeed.checkpoints import load_checkpoint
from egospeed.data import (
    PackedEgoSpeedDataset,
    build_vehicle_map,
    load_splits,
    resolve_dataset_roots,
)
from egospeed.engine import predict
from egospeed.metrics import regression_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/final.json")
    parser.add_argument(
        "--dataset-root",
        help="Directory containing packed/ and smartroi_masks/",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--mask-root")
    parser.add_argument("--splits", default="splits/holdout_splits.json")
    parser.add_argument("--holdout", default="")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint_path,
    config_path,
    data_root,
    mask_root,
    splits_path,
    output_dir,
    *,
    holdout="",
    device_name="auto",
):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    splits = load_splits(splits_path)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    holdout = holdout or checkpoint.get("holdout", "")
    if holdout not in splits:
        raise KeyError("Pass --holdout or use a checkpoint containing a valid holdout name")

    fold = splits[holdout]
    source_sequences = sorted(set(fold["train"] + fold.get("valid", [])))
    vehicle_map = checkpoint.get("vehicle_map") or build_vehicle_map(source_sequences)
    dataset = PackedEgoSpeedDataset(
        fold["test"],
        vehicle_map,
        data_root,
        mask_root,
        clip_length=config["clip_length"],
        stride=config["test_stride"],
        min_speed_mps=config["min_train_speed_mps"],
        max_speed_mps=None,
        flow_rate_clip=config["flow_rate_clip"],
        target_weight_start=config["target_weight_start"],
        target_weight_end=config["target_weight_end"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    ground_truth, prediction = predict(model, loader, device)
    metrics = regression_metrics(
        ground_truth,
        prediction,
        min_speed_mps=config["evaluation_min_speed_mps"],
        max_speed_mps=config["evaluation_max_speed_mps"],
    )
    metrics.update({"holdout": holdout, "checkpoint": str(Path(checkpoint_path).resolve())})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_dir / f"{holdout}_predictions.csv",
        np.column_stack([ground_truth, prediction]),
        delimiter=",",
        header="gt_mps,pred_mps",
        comments="",
    )
    with (output_dir / f"{holdout}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main():
    args = parse_args()
    try:
        data_root, mask_root = resolve_dataset_roots(
            args.dataset_root,
            args.data_root,
            args.mask_root,
        )
    except ValueError as error:
        raise SystemExit(error) from error
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.config,
        data_root,
        mask_root,
        args.splits,
        args.output_dir,
        holdout=args.holdout,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
