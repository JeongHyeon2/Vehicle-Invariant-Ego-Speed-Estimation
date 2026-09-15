---
license: cc-by-nc-4.0
pretty_name: EgoSpeed Multi-Vehicle
language:
- en
tags:
- ego-speed-estimation
- dashcam
- computer-vision
- domain-generalization
- optical-flow
- smartroi
size_categories:
- 100K<n<1M
---

# EgoSpeed Multi-Vehicle Dataset

EgoSpeed Multi-Vehicle is a synchronized in-vehicle dashcam and OBD-speed
dataset for vision-based ego-vehicle speed estimation and cross-vehicle domain
generalization. It accompanies the paper **Vehicle-Invariant Ego-Speed
Estimation from In-Vehicle Dashcam Videos** and the official
[EgoSpeed-SmartROI implementation](https://github.com/JeongHyeon2/Vehicle-Invariant-Ego-Speed-Estimation).

The release contains recordings from five vehicle models: Avante, Malibu,
Sonata, Carnival, and XM3. It is distributed in two forms so users can either
run the released model immediately or reproduce the complete preprocessing
pipeline from the synchronized source videos.

## Downloads

| File | Size | Use |
|---|---:|---|
| `EgoSpeed_model_ready_48x86_20260914.tar.zst` | 2.041 GiB | Direct training and evaluation |
| `EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst` | 22.966 GiB | Reproduce preprocessing from synchronized 1080p MP4 and per-frame speed CSV files |

Download only the model-ready archive:

```bash
hf download Jeonghyeon3575/EgoSpeed-MultiVehicle \
  --repo-type dataset \
  --include "EgoSpeed_model_ready_48x86_20260914.tar.zst" \
  --local-dir .
```

Download the complete release:

```bash
hf download Jeonghyeon3575/EgoSpeed-MultiVehicle \
  --repo-type dataset \
  --local-dir .
```

## Dataset Summary

| Item | Value |
|---|---:|
| Vehicle models | 5 |
| Physical recordings | 14 |
| Original video resolution | 1920 x 1080 |
| Original frame rate | 30 FPS |
| Original aligned frame/label pairs | 292,248 |
| Model-ready sampling rate | 10 FPS |
| Model-ready frames | 97,421 |
| Model input resolution | 48 x 86 |
| Temporal clip length | 13 frames |

The original release contains five Avante recordings, four Malibu recordings,
three Carnival recordings, one Sonata recording, and one XM3 recording.

## Original Synchronized Release

Extract the archive:

```bash
tar --zstd -xf EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst
```

It produces:

```text
EgoSpeed_original_mp4_per_frame_csv_20260914/
  dataset/
    README.md
    manifest.csv
    validation.json
    extract_frames.py
    recordings/
      avante_1/
        avante_1.mp4
        avante_1.csv
      ...
```

Each recording CSV contains:

| Column | Meaning |
|---|---|
| `frame_index` | Zero-based index of the corresponding video frame |
| `time_sec` | Frame timestamp, equal to `frame_index / 30` |
| `speed_kmh` | Synchronized speed in km/h |
| `speed_mps` | Synchronized speed in m/s |

The finalized OBD-speed signal is sampled at each video-frame timestamp. Any
video tail without a valid speed label was removed without re-encoding the
video. Consequently, every released MP4 has exactly the same number of frames
as its paired CSV has data rows.

The public files do not contain a shared hardware-trigger record for the camera
and OBD logger. Therefore, a constant inter-device offset cannot be
independently estimated again from the released file metadata; users should use
the provided synchronized per-frame labels.

## Model-Ready Release

Extract the archive:

```bash
tar --zstd -xf EgoSpeed_model_ready_48x86_20260914.tar.zst
```

It produces:

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

Each `packed_depthnorm64.pt` is a PyTorch dictionary containing:

| Key | Shape | Typical dtype | Description |
|---|---|---|---|
| `frames` | `(N, 1, 48, 86)` | `float16` | Grayscale frames normalized with `gray * 2 - 1` |
| `flow_rate64` | `(N, 2, 48, 86)` | `float16` | RAFT-Large horizontal and vertical flow rates |
| `speeds_mps` | `(N,)` | `float32` | Synchronized speed labels in m/s |
| `frame_indices` | `(N,)` | integer | Indices in the 10 FPS model stream |
| `depth_rel64` | `(N, 1, 48, 86)` | `float16` | Relative inverse depth retained for historical compatibility |

The final model reads `frames`, `flow_rate64`, and `speeds_mps` from the pack.
Relative depth is not a model input; it is used only in the offline SmartROI
construction. Each mask file contains `masks_u8` with shape `(N, 48, 86)` and
dtype `uint8`.

Public model-ready sequence names use the form
`<vehicle_model>_<recording_number>`, such as `avante_01` and `carnival_03`.
Their numbering is a fixed public alias mapping and should not be assumed to
match the raw recording numbers. The official preprocessing adapter performs
this mapping automatically.

## Quick Start

Place the extracted model-ready dataset next to the cloned code repository:

```text
parent_directory/
  EgoSpeed-SmartROI/
  EgoSpeedDataset/
```

Then run:

```bash
git clone https://github.com/JeongHyeon2/Vehicle-Invariant-Ego-Speed-Estimation.git EgoSpeed-SmartROI
cd EgoSpeed-SmartROI
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python scripts/inspect_dataset.py
python scripts/evaluate.py --dry-run
python scripts/train_lovo.py --dry-run --holdout holdout_avante
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

`inspect_dataset.py`, `evaluate.py`, and `train_lovo.py` automatically discover
a sibling directory named `EgoSpeedDataset`. Use `--dataset-root` or the
`EGOSPEED_DATASET_ROOT` environment variable for another location.

Run a complete training fold after the dry run succeeds:

```bash
python scripts/train_lovo.py --seed 42 --holdout holdout_avante
```

## Reproduce the Model-Ready Data

Keep all three extracted components under one parent directory:

```text
parent_directory/
  EgoSpeed-SmartROI/
  EgoSpeedDataset/
  EgoSpeed_original_mp4_per_frame_csv_20260914/
    dataset/
      recordings/
```

Install a CUDA-enabled PyTorch/torchvision build and the preprocessing extras,
then run the resumable builder from the code repository:

```bash
pip install -e ".[preprocessing]"
python scripts/preprocessing/build_model_ready.py --amp
```

The pipeline extracts the 10 FPS RGB stream, builds normalized 48 x 86
grayscale and RAFT flow tensors, computes full-resolution MiDaS DPT-Large and
RAFT-Large SmartROI masks, writes `EgoSpeedDataset`, and validates the result.
The first run requires internet access for pretrained weights and a CUDA GPU.
See the detailed
[preprocessing documentation](https://github.com/JeongHyeon2/Vehicle-Invariant-Ego-Speed-Estimation/blob/main/docs/PREPROCESSING.md).

## Evaluation Protocol

The official code provides five leave-one-vehicle-out folds. In each fold, all
recordings from one vehicle model are held out for testing. The released code
contains five seed-42 checkpoints and supports the three paper seeds 42, 45,
and 46.

## Checksums

```text
15DF1C3A09AFE28BB7CD1EA6F22F4C598B37ABADFD4FF2DEB6521F522B506CAE  EgoSpeed_model_ready_48x86_20260914.tar.zst
96AB406560C34516672E3F2C432DA5ED381BDD85C5AAF6CA158F7668874F91CC  EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst
```

## Intended Use and Limitations

This dataset is intended for non-commercial research on ego-speed estimation,
video regression, motion representation learning, and vehicle-domain
generalization. It is not intended for identity recognition, surveillance, or
attempts to identify road users or vehicles.

The data were collected with a limited set of five vehicle models and camera
configurations. Performance measured on this release should not be interpreted
as validation for all vehicles, cameras, roads, countries, weather conditions,
or safety-critical deployment. The speed labels and predictions must not be
used as the sole input to real-world vehicle control.

The original release contains real-world road video. Users must comply with the
dataset license and applicable privacy and data-protection requirements.

## License

The dataset is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Attribution is
required and commercial use is not permitted under this license. Source code
in the companion GitHub repository is separately licensed under GPL-3.0.

## Citation

```bibtex
@article{kim2026egospeed,
  title={Vehicle-Invariant Ego-Speed Estimation from In-Vehicle Dashcam Videos},
  author={Kim, Jeonghyeon and Kim, Youngwon and Lee, Jun Seong},
  journal={IEEE Access},
  year={2026}
}
```

The citation entry will be updated with volume, issue, pages, and DOI after
publication.

## Questions and Issues

For questions about the data, preprocessing, or released checkpoints, open an
issue in the
[official GitHub repository](https://github.com/JeongHyeon2/Vehicle-Invariant-Ego-Speed-Estimation/issues).
