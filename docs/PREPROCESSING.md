# Reproduce the 48 x 86 Model-Ready Dataset

This pipeline starts from the public synchronized MP4/CSV release and creates
the packed tensors and SmartROI masks consumed by the training code.

## Requirements

Use Python 3.10 or later and a CUDA-enabled PyTorch installation. Install a
PyTorch and torchvision build appropriate for the machine's CUDA version, then
install the project with its preprocessing extras:

```bash
pip install -e ".[preprocessing]"
```

Preprocessing additionally uses Pillow, NumPy, OpenCV, and timm. MiDaS
DPT-Large is loaded through `torch.hub`, and RAFT-Large uses the official
torchvision weights. The first run requires internet access to download those
pretrained weights.

The commands below assume these locations:

```text
/data/EgoSpeedOriginal/dataset/  # synchronized recordings/ folders
/data/EgoSpeedFrames10/          # extracted full-resolution PNGs
/data/EgoSpeedPreprocess/        # linked preprocessing layout
/data/EgoSpeedDataset/           # final packed tensors and masks
```

The public archives already contain compatible top-level directories. Extract
them next to the cloned repository with:

```bash
tar --zstd -xf EgoSpeed_model_ready_48x86_20260914.tar.zst
tar --zstd -xf EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst
```

After extraction, `build_model_ready.py` automatically discovers
`../EgoSpeed_original_mp4_per_frame_csv_20260914/dataset` and writes
`../EgoSpeedDataset` unless explicit paths are supplied.

## Recommended: one-command build

When the repository, original release, and output dataset use their standard
sibling names, run this from `EgoSpeed-SmartROI`:

```bash
python scripts/preprocessing/build_model_ready.py --amp
```

The command automatically finds
`../EgoSpeed_original_mp4_per_frame_csv_20260914/dataset`, writes or resumes
intermediate files under `../EgoSpeedPreprocessWork`, creates
`../EgoSpeedDataset`, and runs the final validator. Completed extraction,
tensor, and mask stages are reused when the command is restarted.

If the folders are elsewhere, specify them explicitly:

```bash
python scripts/preprocessing/build_model_ready.py \
  --original-dataset-root /data/EgoSpeedOriginal/dataset \
  --work-root /data/EgoSpeedPreprocessWork \
  --output-root /data/EgoSpeedDataset \
  --gpu 0 \
  --amp
```

Run only the release checks, without starting preprocessing, using:

```bash
python scripts/preprocessing/build_model_ready.py --validate-only
```

The following sections show the individual stages invoked by the one-command
builder.

## 1. Extract the 10 FPS PNG stream

Run from the repository root:

```bash
python scripts/preprocessing/extract_frames.py \
  --dataset-root /data/EgoSpeedOriginal/dataset \
  --output-fps 10 \
  --output-dir /data/EgoSpeedFrames10
```

This extracts full-resolution RGB source frames `0, 3, 6, ...`. The resulting
model stream is 10 FPS, not 3 FPS. Every generated PNG has one row in its
recording CSV, including both its new 10 FPS index and original 30 FPS index.

The complete release produces 97,421 full-resolution PNGs. PNG storage can be
substantially larger than the MP4 release.

## 2. Build the preprocessing layout

The public split contains 14 sequence names backed by the 14 physical
recordings. The adapter applies the released sequence-to-recording mapping,
creates `label.txt` files, and links each public sequence to its PNG folder.
It also accepts raw recording names such as `avante_1` if a custom split uses
those names. Public names such as `avante_01` must use the fixed provenance
mapping because their numbering is not the same as the raw release numbering.

```bash
python scripts/preprocessing/prepare_public_layout.py \
  --frames-root /data/EgoSpeedFrames10 \
  --splits-json splits/holdout_splits.json \
  --output-root /data/EgoSpeedPreprocess
```

On Windows, directory symlinks require Developer Mode or administrator
permission. If links are unavailable, add `--copy-frames`; this duplicates all
PNGs and therefore requires substantial additional space.

## 3. Generate the 48 x 86 tensor packs

```bash
python scripts/preprocessing/preprocess_48x86_tensors.py \
  --label_root /data/EgoSpeedPreprocess/label_root \
  --image_root /data/EgoSpeedPreprocess/image_root \
  --out_root /data/EgoSpeedDataset/packed \
  --splits_json /data/EgoSpeedPreprocess/label_root/holdout_splits.json \
  --target_hw 360,640 \
  --out_hw 48,86 \
  --fps 10 \
  --depth_batch 4 \
  --flow_batch 4 \
  --gpu 0
```

Each `packed_depthnorm64.pt` contains:

| Key | Shape | Description |
|---|---|---|
| `frames` | `(N,1,48,86)` | grayscale normalized as `gray * 2 - 1` |
| `flow_rate64` | `(N,2,48,86)` | RAFT-Large raw flow rates |
| `speeds_mps` | `(N,)` | synchronized speeds in m/s |
| `frame_indices` | `(N,)` | 10 FPS model-frame indices |
| `depth_rel64` | `(N,1,48,86)` | robust relative inverse depth |

The final model reads `frames`, `flow_rate64`, and `speeds_mps`; it does not use
`depth_rel64` as an input. The flow-rate conversion after RAFT inference at
`360 x 640` is:

```text
u_rate = u_px / width  * fps
v_rate = v_px / height * fps
```

The historical output filename retains `64` for checkpoint compatibility even
though the stored spatial resolution is `48 x 86`.

## 4. Generate SmartROI masks

```bash
python scripts/preprocessing/precompute_smartroi.py \
  --label_root /data/EgoSpeedPreprocess/label_root \
  --image_root /data/EgoSpeedPreprocess/image_root \
  --out_mask_root /data/EgoSpeedDataset/smartroi_masks \
  --splits_json /data/EgoSpeedPreprocess/label_root/holdout_splits.json \
  --source_hw 1080,1920 \
  --out_hw 48,86 \
  --window_t 13 \
  --chunk_centers 32 \
  --mask_batch 1 \
  --depth_batch 1 \
  --flow_batch 1 \
  --fps 10 \
  --depth_low_pct 0.15 \
  --depth_high_pct 0.85 \
  --motion_thresh 0.02 \
  --consistency_thresh 0.3 \
  --gpu 0 \
  --amp
```

This reproduces the full-resolution SmartROI path used by the final experiment:

1. MiDaS DPT-Large estimates relative inverse depth.
2. RAFT-Large estimates consecutive optical flow at `1920 x 1080`.
3. Every center frame uses a clamped 13-frame context window.
4. Relative-depth values between the 15th and 85th percentiles form the depth region.
5. Mean flow magnitude and temporal flow consistency modulate the region.
6. The soft mask is resized to `48 x 86` and stored as `uint8`.

Each `<sequence>__smartroi_mask_u8.pt` contains `masks_u8` with shape
`(N,48,86)`. Full-resolution MiDaS and RAFT processing is computationally
expensive; use a GPU compute node rather than a login node.

## 5. Validate

```bash
python scripts/inspect_dataset.py --dataset-root /data/EgoSpeedDataset
```

The validator checks that every public sequence has a pack and mask, all stored
spatial shapes are `48 x 86`, and the frame, flow, speed, and mask counts agree.

## Optional smoke test

Before launching the full job, use a separate temporary output location and a
small frame limit:

```bash
python scripts/preprocessing/preprocess_48x86_tensors.py \
  --label_root /data/EgoSpeedPreprocess/label_root \
  --image_root /data/EgoSpeedPreprocess/image_root \
  --out_root /data/EgoSpeedDataset_smoke/packed \
  --sequences avante_01 \
  --max_frames 20 \
  --gpu 0
```

The corresponding mask smoke test uses the same `--sequences avante_01` and
`--max_frames 20` options with a temporary `--out_mask_root`.
