# Pretrained Checkpoints

The repository includes five actual validation-best checkpoints from seed 42
of the final paper experiment:

- Architecture: final Ours, raw flow + SmartROI + disentanglement, no DepthNorm
- Total size: 141,551,775 bytes (approximately 135 MiB)

The model tensors are bit-identical to the archived experiment checkpoints.
Only the machine-local `mask_root` metadata was removed for publication; the
seed and SmartROI-enabled fields were added explicitly.

Each model was trained on four source vehicles and selected without using the
named holdout vehicle's test set. These are LOVO models, not separately trained
all-vehicle deployment models.

Layout:

```text
checkpoints/
  holdout_avante/best.pt
  holdout_malibu/best.pt
  holdout_sonata/best.pt
  holdout_suv/best.pt
  holdout_xm3/best.pt
```

| Public vehicle | Checkpoint directory | Best epoch |
|---|---|---:|
| Avante | `holdout_avante` | 100 |
| Malibu | `holdout_malibu` | 95 |
| Sonata | `holdout_sonata` | 90 |
| Carnival | `holdout_suv` | 100 |
| XM3 | `holdout_xm3` | 80 |

The loader accepts the original experiment `best.pt` format and verifies the
model state dictionary strictly.

Verify the files after cloning:

```bash
sha256sum --check checkpoints/SHA256SUMS
```

Run a checkpoint against the separately distributed model-ready dataset:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/holdout_avante/best.pt \
  --dataset-root /path/to/EgoSpeedDataset
```

The complete paper result uses 15 independently trained models: three seeds
times five LOVO folds. Seeds 45 and 46 can be regenerated with
`scripts/train_lovo.py`.
