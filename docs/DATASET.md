# Dataset Format

## Public Releases

Two separate `.tar.zst` downloads are provided.

1. **Model-ready release**: resized grayscale frames, raw optical-flow rates,
   synchronized speed labels, split metadata, and precomputed SmartROI masks.
   This is sufficient to train and evaluate the code in this repository.
2. **Original synchronized release**: 1920 x 1080 dashcam MP4 files and OBD CSV
   logs. This is useful for independent preprocessing and future research, but
   it is not read directly by the training script.

The complete original-to-model-ready pipeline is documented in
[`PREPROCESSING.md`](PREPROCESSING.md). With the standard sibling directory
layout it can be run or resumed with
`python scripts/preprocessing/build_model_ready.py --amp`.

| Archive | Extracted root | Compressed size | SHA-256 |
|---|---|---:|---|
| `EgoSpeed_model_ready_48x86_20260914.tar.zst` | `EgoSpeedDataset/` | 2.041 GiB | `15DF1C3A09AFE28BB7CD1EA6F22F4C598B37ABADFD4FF2DEB6521F522B506CAE` |
| `EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst` | `EgoSpeed_original_mp4_per_frame_csv_20260914/dataset/` | 22.966 GiB | `96AB406560C34516672E3F2C432DA5ED381BDD85C5AAF6CA158F7668874F91CC` |

Extract both archives from the directory that contains the cloned repository:

```bash
tar --zstd -xf EgoSpeed_model_ready_48x86_20260914.tar.zst
tar --zstd -xf EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst
```

## Model-Ready Layout

```text
EgoSpeedDataset/
  packed/
    avante_01/
      packed_depthnorm64.pt
    ...
  smartroi_masks/
    avante_01__smartroi_mask_u8.pt
    ...
  metadata/
    holdout_splits.json
    sequence_manifest.csv
    release_metadata.json
```

The 14 physical recordings use anonymized public names of the form
`<vehicle_model>_<recording_number>`:

- `avante_01` through `avante_05`
- `malibu_01` through `malibu_04`
- `sonata_01`
- `carnival_01` through `carnival_03`
- `xm3_01`

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

## Release Sizes

Sizes below were measured from the finalized release directories and archives.

| Release content | Size |
|---|---:|
| Original synchronized directory | 22.989 GiB |
| Original synchronized `.tar.zst` | 22.966 GiB |
| Model-ready directory | 2.623 GiB |
| Model-ready `.tar.zst` | 2.041 GiB |

The original archive compresses only slightly because H.264 MP4 is already a
compressed format. The model-ready archive compresses more effectively because
it contains tensor and mask payloads.

The original release contains 292,248 aligned 30 FPS frame/label pairs. The
model-ready release contains 97,421 frames sampled at 10 FPS. Every released
MP4 has exactly as many video frames as its corresponding per-frame CSV has
rows; unlabeled video tails are not included.

## Vehicle Names

All dataset paths and metadata use `carnival`. The code accepts the historical
`SUV` class token only to remain compatible with the included pretrained
checkpoint and its `holdout_suv` fold identifier.
