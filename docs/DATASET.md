# Dataset Format

## Public Releases

Two separate downloads are recommended.

1. **Model-ready release**: resized grayscale frames, raw optical-flow rates,
   synchronized speed labels, split metadata, and precomputed SmartROI masks.
   This is sufficient to train and evaluate the code in this repository.
2. **Original synchronized release**: 1920 x 1080 dashcam MP4 files and OBD CSV
   logs. This is useful for independent preprocessing and future research, but
   it is not read directly by the training script.

## Model-Ready Layout

```text
EgoSpeedDataset/
  packed/
    2025_..._AVANTE_..._sync/
      packed_depthnorm64.pt
    ...
  smartroi_masks/
    2025_..._AVANTE_..._sync__smartroi_mask_u8.pt
    ...
```

The split file is versioned with the source code at
`splits/holdout_splits.json`.

After extraction, the dataset is passed as one directory:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/holdout_avante/best.pt \
  --dataset-root /path/to/EgoSpeedDataset
```

## Required Packed Tensors

Each `packed_depthnorm64.pt` is a PyTorch dictionary. The final model requires:

| Key | Shape | Typical dtype | Description |
|---|---|---|---|
| `frames` | `(N, 1, 48, 86)` | `float16` | Grayscale, normalized to approximately `[-1, 1]` |
| `flow_rate64` | `(N, 2, 48, 86)` | `float16` | RAFT horizontal/vertical flow rates |
| `speeds_mps` | `(N,)` | `float32` | Synchronized speed in m/s |
| `frame_indices` | `(N,)` | `int32` | Optional source frame index |
| `meta` | dictionary | - | Optional preprocessing metadata |

`depth_rel64` may appear in the internal packs. It is not read by the final C=3
model and can be omitted from the compact public release because the SmartROI
masks are already precomputed.

Each mask file contains `masks_u8` with shape `(N, 48, 86)` and dtype `uint8`.
The loader divides it by 255 to obtain the SmartROI weights in `[0, 1]`.

## Clip Target

A sample contains 13 consecutive frames. Its speed target is a linearly
weighted average of the 13 synchronized frame speeds, with weights increasing
from 1.0 to 2.0 toward the most recent frame. Training and validation clips use
stride 10, and test clips use non-overlapping stride 13.

## Measured Uncompressed Sizes

Sizes below use unique physical payloads rather than counting linked aliases
more than once.

| Release content | Size |
|---|---:|
| Original unique MP4 + OBD CSV files for publication | 23.098 GiB |
| Current raw working directory, including duplicate/bad backup files | 26.141 GiB |
| Existing 48 x 86 packs, including relative depth | 2.997 GiB |
| Precomputed SmartROI masks | 0.375 GiB |
| Existing model-ready bundle | 3.372 GiB |
| Compact model-ready bundle without relative depth | about 2.623 GiB |
| Grayscale + speed/index tensors only | about 0.750 GiB |

The grayscale-only size is not sufficient to run the proposed model because
the model also requires raw optical flow and SmartROI masks.

The publication figure for the original release removes one exact duplicate
Avante MP4/CSV pair and an obsolete backup copy. The retained and removed
copies were checked by SHA-256 rather than filename or size alone.

## Vehicle Name

The legacy internal token `SUV` identifies the Kia Carnival recordings. Use
`Carnival` in public descriptions, tables, and figures.
