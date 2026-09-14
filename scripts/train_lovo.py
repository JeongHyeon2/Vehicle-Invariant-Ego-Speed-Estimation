#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from egospeed.data import (
    PackedEgoSpeedDataset,
    build_vehicle_map,
    load_splits,
    resolve_dataset_roots,
)
from egospeed.engine import mean_absolute_error, predict
from egospeed.metrics import regression_metrics
from egospeed.models import EgoSpeedSmartROI, dann_weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.json")
    parser.add_argument(
        "--dataset-root",
        help="Directory containing packed/ and smartroi_masks/",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--mask-root")
    parser.add_argument("--splits", default="splits/holdout_splits.json")
    parser.add_argument("--output-dir", default="runs/final")
    parser.add_argument("--holdout", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_dataset(
    sequences,
    vehicle_map,
    args,
    config,
    *,
    split,
    stride,
    cache=None,
):
    return PackedEgoSpeedDataset(
        sequences,
        vehicle_map,
        args.data_root,
        args.mask_root,
        clip_length=config["clip_length"],
        stride=stride,
        temporal_split=split,
        split_ratio=config["train_split_ratio"],
        min_speed_mps=config["min_train_speed_mps"],
        max_speed_mps=config["max_train_speed_mps"],
        flow_rate_clip=config["flow_rate_clip"],
        target_weight_start=config["target_weight_start"],
        target_weight_end=config["target_weight_end"],
        cache=cache,
    )


def checkpoint_payload(model, epoch, val_mae, holdout, vehicle_map, config, seed):
    return {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "val_mae": float(val_mae),
        "holdout": holdout,
        "vehicle_map": vehicle_map,
        "model_version": "rawflow48x86_smartroi_disent",
        "input_mode": "rawflow",
        "input_channels": 3,
        "smartroi_enabled": True,
        "flow_rate_clip": config["flow_rate_clip"],
        "disent_type": config["disentanglement"],
        "num_blocks": config["disentanglement_blocks"],
        "test_window_stride": config["test_stride"],
        "seed": int(seed),
        "config": config,
    }


def train_fold(holdout, fold, args, config, device):
    fold_dir = Path(args.output_dir) / f"seed{args.seed}" / holdout
    fold_dir.mkdir(parents=True, exist_ok=True)

    source_sequences = sorted(set(fold["train"] + fold.get("valid", [])))
    vehicle_map = build_vehicle_map(source_sequences)
    cache = {}
    train_dataset = make_dataset(
        source_sequences,
        vehicle_map,
        args,
        config,
        split="train",
        stride=config["train_stride"],
        cache=cache,
    )
    val_dataset = make_dataset(
        source_sequences,
        vehicle_map,
        args,
        config,
        split="val",
        stride=config["train_stride"],
        cache=cache,
    )
    test_dataset = make_dataset(
        fold["test"],
        vehicle_map,
        args,
        config,
        split=None,
        stride=config["test_stride"],
    )
    print(
        f"[{holdout}] train={len(train_dataset)} val={len(val_dataset)} "
        f"test={len(test_dataset)} vehicles={vehicle_map}"
    )

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = EgoSpeedSmartROI(
        num_vehicles=len(vehicle_map),
        latent_dim=config["latent_dim"],
        grl_lambda=config["grl_lambda"],
        num_blocks=config["disentanglement_blocks"],
    ).to(device)

    if args.dry_run:
        clip, speed, vehicle_id, mask = next(iter(train_loader))
        with torch.no_grad():
            output = model(clip.to(device), mask.to(device))
        print(
            f"[{holdout}] dry run OK: clip={tuple(clip.shape)} "
            f"speed={tuple(speed.shape)} vehicle={tuple(vehicle_id.shape)} "
            f"prediction={tuple(output['speed_pred'].shape)}"
        )
        return {"holdout": holdout, "dry_run": True}

    speed_loss = nn.SmoothL1Loss()
    vehicle_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    total_iterations = max(1, config["epochs"] * len(train_loader))
    warmup_iterations = max(1, int(total_iterations * config["lambda_adv_warmup"]))
    global_iteration = 0
    best_val_mae = float("inf")
    best_path = fold_dir / "best.pt"
    validations_without_improvement = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running = {"speed": 0.0, "vehicle": 0.0, "adversarial": 0.0}
        for clip, speed, vehicle_id, mask in train_loader:
            clip = clip.to(device, non_blocking=True)
            speed = speed.to(device, non_blocking=True)
            vehicle_id = vehicle_id.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            adversarial_weight = dann_weight(
                global_iteration,
                warmup_iterations,
                config["lambda_adv_max"],
            )

            optimizer.zero_grad(set_to_none=True)
            output = model(clip, mask)
            loss_speed = speed_loss(output["speed_pred"], speed)
            loss_vehicle = vehicle_loss(output["vehicle_pred_private"], vehicle_id)
            loss_adversarial = vehicle_loss(output["vehicle_pred_adv"], vehicle_id)
            loss = (
                loss_speed
                + config["lambda_vehicle"] * loss_vehicle
                + adversarial_weight * loss_adversarial
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()

            running["speed"] += float(loss_speed.item())
            running["vehicle"] += float(loss_vehicle.item())
            running["adversarial"] += float(loss_adversarial.item())
            global_iteration += 1

        should_validate = epoch == 1 or epoch % config["validation_every"] == 0
        if not should_validate:
            continue

        val_true, val_pred = predict(model, val_loader, device)
        val_mae = mean_absolute_error(val_true, val_pred)
        scheduler.step(val_mae)
        row = {
            "epoch": epoch,
            "train_speed_loss": running["speed"] / max(1, len(train_loader)),
            "train_vehicle_loss": running["vehicle"] / max(1, len(train_loader)),
            "train_adversarial_loss": running["adversarial"] / max(1, len(train_loader)),
            "val_mae_mps": val_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps({"holdout": holdout, **row}))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            validations_without_improvement = 0
            torch.save(
                checkpoint_payload(model, epoch, val_mae, holdout, vehicle_map, config, args.seed),
                best_path,
            )
        else:
            validations_without_improvement += 1
            if validations_without_improvement >= config["early_stopping_patience"]:
                break

    with (fold_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_true, test_pred = predict(model, test_loader, device)
    metrics = regression_metrics(
        test_true,
        test_pred,
        min_speed_mps=config["evaluation_min_speed_mps"],
        max_speed_mps=config["evaluation_max_speed_mps"],
    )
    np.savetxt(
        fold_dir / "predictions.csv",
        np.column_stack([test_true, test_pred]),
        delimiter=",",
        header="gt_mps,pred_mps",
        comments="",
    )
    result = {
        "holdout": holdout,
        "seed": args.seed,
        "best_epoch": checkpoint["epoch"],
        "best_val_mae_mps": checkpoint["val_mae"],
        **metrics,
    }
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


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
    args.data_root = data_root
    args.mask_root = mask_root
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    splits = load_splits(args.splits)
    holdouts = sorted(splits) if args.holdout == "all" else [args.holdout]
    unknown = set(holdouts).difference(splits)
    if unknown:
        raise KeyError(f"Unknown holdout(s): {sorted(unknown)}")

    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"device={device} seed={args.seed} holdouts={holdouts}")
    results = [train_fold(name, splits[name], args, config, device) for name in holdouts]
    output = Path(args.output_dir) / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
