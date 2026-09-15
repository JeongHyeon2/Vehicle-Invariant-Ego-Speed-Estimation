# Vehicle-Invariant Ego-Speed Estimation from In-Vehicle Dashcam Videos

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

**Jeonghyeon Kim, Youngwon Kim, and Jun Seong Lee**

Official PyTorch implementation and qualitative results for our IEEE Access
paper.

The method estimates ego-vehicle speed from 13-frame dashcam clips and is
designed to generalize to vehicles that are not observed during training. It
combines a FlexiNet backbone, SmartROI feature masking, and vehicle-adversarial
disentanglement.

The repository includes the final model and training code plus five pretrained
LOVO checkpoints for immediate evaluation. The model-ready dataset is
distributed separately.

## Results

Results use five-fold leave-one-vehicle-out evaluation, three random seeds, and
the ground-truth range `0.5 <= speed < 20 m/s`.

| Method | MAE (m/s) | RMSE (m/s) |
|---|---:|---:|
| 3DCMA | 2.617 +/- 0.132 | 3.250 +/- 0.153 |
| FlexiNet | 1.781 +/- 0.038 | 2.365 +/- 0.018 |
| **EgoSpeed-SmartROI** | **1.300 +/- 0.024** | **1.819 +/- 0.041** |

The proposed model reduces MAE by 27.0% and RMSE by 23.1% relative to FlexiNet.

## Qualitative Videos

The following autoplay previews compare the ground-truth speed, FlexiNet
prediction, and our prediction on unseen holdout vehicles. Click any preview to
play the complete 15-second MP4. Error colors are green (`<= 1 m/s`), yellow
(`1--2 m/s`), and red (`> 2 m/s`).

These clips were selected to illustrate qualitative behavior and should not be
interpreted as aggregate test results. The complete quantitative results are
reported above and in the paper.

### Avante

<p align="center"><a href="assets/videos/avante_01.mp4"><img src="assets/previews/avante_01.gif" width="100%" alt="Avante example 1"></a><br><b>Example 1</b></p>
<p align="center"><a href="assets/videos/avante_02.mp4"><img src="assets/previews/avante_02.gif" width="100%" alt="Avante example 2"></a><br><b>Example 2</b></p>
<p align="center"><a href="assets/videos/avante_03.mp4"><img src="assets/previews/avante_03.gif" width="100%" alt="Avante example 3"></a><br><b>Example 3</b></p>

### Malibu

<p align="center"><a href="assets/videos/malibu_01.mp4"><img src="assets/previews/malibu_01.gif" width="100%" alt="Malibu example 1"></a><br><b>Example 1</b></p>
<p align="center"><a href="assets/videos/malibu_02.mp4"><img src="assets/previews/malibu_02.gif" width="100%" alt="Malibu example 2"></a><br><b>Example 2</b></p>
<p align="center"><a href="assets/videos/malibu_03.mp4"><img src="assets/previews/malibu_03.gif" width="100%" alt="Malibu example 3"></a><br><b>Example 3</b></p>

### Sonata

<p align="center"><a href="assets/videos/sonata_01.mp4"><img src="assets/previews/sonata_01.gif" width="100%" alt="Sonata example 1"></a><br><b>Example 1</b></p>
<p align="center"><a href="assets/videos/sonata_02.mp4"><img src="assets/previews/sonata_02.gif" width="100%" alt="Sonata example 2"></a><br><b>Example 2</b></p>
<p align="center"><a href="assets/videos/sonata_03.mp4"><img src="assets/previews/sonata_03.gif" width="100%" alt="Sonata example 3"></a><br><b>Example 3</b></p>

### Carnival

<p align="center"><a href="assets/videos/carnival_01.mp4"><img src="assets/previews/carnival_01.gif" width="100%" alt="Carnival example 1"></a><br><b>Example 1</b></p>
<p align="center"><a href="assets/videos/carnival_02.mp4"><img src="assets/previews/carnival_02.gif" width="100%" alt="Carnival example 2"></a><br><b>Example 2</b></p>
<p align="center"><a href="assets/videos/carnival_03.mp4"><img src="assets/previews/carnival_03.gif" width="100%" alt="Carnival example 3"></a><br><b>Example 3</b></p>

### XM3

<p align="center"><a href="assets/videos/xm3_01.mp4"><img src="assets/previews/xm3_01.gif" width="100%" alt="XM3 example 1"></a><br><b>Example 1</b></p>
<p align="center"><a href="assets/videos/xm3_02.mp4"><img src="assets/previews/xm3_02.gif" width="100%" alt="XM3 example 2"></a><br><b>Example 2</b></p>
<p align="center"><a href="assets/videos/xm3_03.mp4"><img src="assets/previews/xm3_03.gif" width="100%" alt="XM3 example 3"></a><br><b>Example 3</b></p>

## Final Model

- Clip: `13 x 3 x 48 x 86`
- Channels: grayscale + raw optical-flow rates `(u, v)`
- SmartROI: MiDaS relative depth + RAFT optical flow, generated offline
- Backbone: CMA -> AFT -> SFE/MFE -> DIG
- Speed branch: SmartROI-masked feature, Res3D block, global average pooling
- Vehicle branch: unmasked feature, Res3D block, global average pooling
- Training-only heads: private vehicle classifier and GRL vehicle classifier
- Regression loss: SmoothL1
- Orthogonality and HSIC losses: not used

Depth-normalized flow is **not** an input to the final model. Relative depth is
used only when constructing the precomputed SmartROI masks.

![Qualitative SmartROI examples](assets/smartroi_examples.png)

## Installation

Python 3.10 or later and a CUDA-capable PyTorch installation are recommended.

Linux/macOS:

```bash
cd EgoSpeed-SmartROI
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Windows PowerShell:

```powershell
cd EgoSpeed-SmartROI
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

For a CUDA build, install the appropriate PyTorch wheel for the local CUDA
version before running `pip install -e .`.

To reproduce the model-ready dataset from the original MP4/CSV release, install
the additional preprocessing dependencies instead:

```bash
pip install -e ".[preprocessing]"
```

## Repository Layout

```text
EgoSpeed-SmartROI/
  checkpoints/             Five seed-42 LOVO pretrained models
  assets/videos/           Fifteen FlexiNet-vs-Ours comparison videos
  assets/previews/         Autoplay GIF previews used in this README
  configs/final.json       Final paper hyperparameters
  scripts/train_lovo.py    Training entry point
  scripts/evaluate.py      Evaluation entry point
  scripts/preprocessing/   Original MP4 to model-ready dataset pipeline
  splits/                  Five vehicle holdout definitions
  src/egospeed/            Model, data loader, losses, and metrics
  tests/                    Release smoke tests
```

## Dataset

The data are distributed separately from this Git repository as two Zstandard
archives:

| Archive | Compressed size | Purpose |
|---|---:|---|
| `EgoSpeed_model_ready_48x86_20260914.tar.zst` | 2.041 GiB | Direct training and evaluation |
| `EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst` | 22.966 GiB | Reproduce preprocessing from synchronized 1080p MP4/CSV pairs |

The five seed-42 pretrained checkpoints are already included under
`checkpoints/`. The dataset archives are not committed to Git because of their
size. The public Zenodo, Hugging Face, or other data-record URLs will be added
to this section when the dataset upload is published.

Verify downloaded archives before extraction:

```text
15DF1C3A09AFE28BB7CD1EA6F22F4C598B37ABADFD4FF2DEB6521F522B506CAE  EgoSpeed_model_ready_48x86_20260914.tar.zst
96AB406560C34516672E3F2C432DA5ED381BDD85C5AAF6CA158F7668874F91CC  EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst
```

Clone the repository and extract both archives under the same parent
directory:

```bash
git clone https://github.com/JeongHyeon2/Vehicle-Invariant-Ego-Speed-Estimation.git EgoSpeed-SmartROI
tar --zstd -xf EgoSpeed_model_ready_48x86_20260914.tar.zst
tar --zstd -xf EgoSpeed_original_mp4_per_frame_csv_20260914.tar.zst
```

The archives are packaged so those commands produce this exact layout:

```text
parent_directory/
  EgoSpeed-SmartROI/
  EgoSpeedDataset/
  EgoSpeed_original_mp4_per_frame_csv_20260914/
    dataset/
      recordings/
```

Only `EgoSpeed-SmartROI` and `EgoSpeedDataset` are needed for immediate
training and evaluation. The original MP4/CSV release is needed only to
reproduce the model-ready tensors and masks from scratch.

The model-ready archive should be extracted as follows:

```text
EgoSpeedDataset/
  packed/
    avante_01/
      packed_depthnorm64.pt
    ...
    carnival_03/
      packed_depthnorm64.pt
  smartroi_masks/
    avante_01__smartroi_mask_u8.pt
    ...
    carnival_03__smartroi_mask_u8.pt
  metadata/
    holdout_splits.json
    sequence_manifest.csv
    release_metadata.json
```

The historical pack filename contains `64`, but the tensors used by the final
model have shape `48 x 86`. Public sequence names follow
`<vehicle_model>_<recording_number>`, such as `malibu_02`. The public split
definition is included at
[`splits/holdout_splits.json`](splits/holdout_splits.json). See
[`docs/DATASET.md`](docs/DATASET.md) for the tensor schema and measured sizes.
To regenerate these tensors and masks from the synchronized MP4/CSV release,
follow [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md).

For the shortest setup, place the extracted dataset next to this repository:

```text
parent_directory/
  EgoSpeed-SmartROI/
  EgoSpeedDataset/
```

Validate a downloaded dataset before training:

```bash
python scripts/inspect_dataset.py
python scripts/evaluate.py --dry-run
python scripts/train_lovo.py --dry-run --holdout holdout_avante
```

When `EgoSpeedDataset` is next to this repository it is detected automatically.
For another location, pass `--dataset-root /path/to/EgoSpeedDataset` or set the
`EGOSPEED_DATASET_ROOT` environment variable.

To regenerate `EgoSpeedDataset` from the original archive, install the
preprocessing extras and run the resumable one-command pipeline:

```bash
pip install -e ".[preprocessing]"
python scripts/preprocessing/build_model_ready.py --amp
```

The first preprocessing run downloads the pretrained MiDaS DPT-Large and
RAFT-Large weights and requires a CUDA-enabled PyTorch environment. See
[`docs/PREPROCESSING.md`](docs/PREPROCESSING.md) for resource requirements and
the individual pipeline stages.

## Quick Evaluation

The released code evaluates model-ready packed data. Direct inference from an
arbitrary raw MP4 first requires the offline preprocessing pipeline.

Evaluate one included pretrained fold:

```bash
python scripts/evaluate.py
```

This defaults to the included Avante holdout checkpoint. Select another fold
with `--checkpoint checkpoints/holdout_malibu/best.pt`, for example.
To verify checkpoint inference on one batch before the full evaluation, run
`python scripts/evaluate.py --dry-run`.

Five seed-42 checkpoints are included for Avante, Malibu, Sonata, Carnival, and
XM3. Each checkpoint is evaluated on the vehicle named by its directory.
Predictions and metrics are written under `outputs/evaluation/`.

Evaluate all five included holdouts on Linux/macOS:

```bash
for vehicle in avante malibu sonata suv xm3; do
  python scripts/evaluate.py \
    --checkpoint "checkpoints/holdout_${vehicle}/best.pt" \
    --dataset-root ../EgoSpeedDataset
done
```

Windows PowerShell:

```powershell
foreach ($vehicle in "avante", "malibu", "sonata", "suv", "xm3") {
  python scripts/evaluate.py `
    --checkpoint "checkpoints/holdout_$vehicle/best.pt" `
    --dataset-root ../EgoSpeedDataset
}
```

## Training

Train all five holdout folds for one seed:

```bash
python scripts/train_lovo.py --config configs/final.json --dataset-root ../EgoSpeedDataset --seed 42 --holdout all
```

With the standard sibling layout, the shorter equivalent is:

```bash
python scripts/train_lovo.py --seed 42 --holdout all
```

Before a long training run, verify one complete data/model batch:

```bash
python scripts/train_lovo.py --dry-run --holdout holdout_avante
```

Repeat with seeds `45` and `46` for the paper protocol. To verify the data and
model connection without training, add `--dry-run --holdout holdout_avante`.

The best checkpoint is selected using validation MAE. The test set is evaluated
only after training, and test predictions are not used for checkpoint selection.

## Pretrained Checkpoints

Five actual validation-best checkpoints from seed 42 of the final experiment
are included under `checkpoints/` (approximately 135 MiB total). They cover all
five LOVO holdouts, and their model tensors are identical to the archived
experiment checkpoints. See [`checkpoints/README.md`](checkpoints/README.md)
for provenance and SHA-256 checksums.

## Reproducibility

The exact hyperparameters are stored in [`configs/final.json`](configs/final.json).
The final setup uses AdamW, 100 epochs, batch size 16, learning rate `1e-3`,
weight decay `1e-4`, and train/validation/test strides of 10/10/13. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for loss weights and details.

## Naming

Dataset sequences use the public vehicle-model names `avante`, `malibu`,
`sonata`, `carnival`, and `xm3`. The legacy fold identifier `holdout_suv` is
retained only so the included Carnival checkpoint remains backward compatible.

## Citation

```bibtex
@article{kim2026egospeed,
  title={Vehicle-Invariant Ego-Speed Estimation from In-Vehicle Dashcam Videos},
  author={Kim, Jeonghyeon and Kim, Youngwon and Lee, Jun Seong},
  journal={IEEE Access},
  year={2026}
}
```

## Acknowledgment and License

The backbone is adapted from
[FlexiNet](https://github.com/Geekgineer/FlexiNet) by Ibrahim et al. This
repository is distributed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE) and [`THIRD_PARTY.md`](THIRD_PARTY.md).
