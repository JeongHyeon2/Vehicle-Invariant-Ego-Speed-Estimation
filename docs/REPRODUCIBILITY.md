# Reproducibility Notes

## Final Input

- Shape: `(B, 13, 3, 48, 86)`
- Channels: grayscale, horizontal raw flow rate, vertical raw flow rate
- Raw flow clipping: `[-2, 2]`, followed by division by 2
- SmartROI is applied to the speed branch only
- The vehicle branch receives the unmasked DIG feature

## Objective

```text
L_total = L_speed + lambda_vehicle * L_vehicle + lambda_adv(t) * L_adversarial
```

- `L_speed`: SmoothL1
- `lambda_vehicle`: 1.0
- `lambda_adv_max`: 0.1
- `lambda_adv_warmup`: first 30% of all training iterations
- GRL coefficient: 1.0
- Orthogonality loss: not used
- HSIC loss: not used

`lambda_adv(t)` follows the logistic DANN schedule and approaches 0.1 during
the warmup period. It is held near its maximum after the first 30% of training.

## Optimization

- AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Batch size: 16
- Maximum epochs: 100
- Gradient norm clipping: 1.0
- Validation every five epochs, plus epoch one
- Best checkpoint selected by validation MAE
- Seeds: 42, 45, 46

## Data Split

Five leave-one-vehicle-out folds are defined in `splits/holdout_splits.json`.
For each fold, the listed source-vehicle train and validation sequences are
combined and then split in temporal order. The first 80% is used for training,
the final 20% for validation, with a one-clip guard interval between them.

The archived final runs excluded targets below 0.5 m/s while constructing
training clips and did not apply an upper speed cutoff during optimization.
Paper metrics are computed only for `0.5 <= ground truth < 20 m/s`. This exact
behavior is represented by `max_train_speed_mps: null` and
`evaluation_max_speed_mps: 20.0` in `configs/final.json`.
