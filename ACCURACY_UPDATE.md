# Default VideoMAE V2 experiment settings (v7)

This file records the settings implemented by the v7 notebook, shell wrapper, and command-line defaults. The main `README.md` is the authoritative usage guide.

## ViT-S distilled default

- Architecture: `videomaev2_vit_s_distilled`
- Input: 16 frames at 224 x 224
- Temporal stride: 4 when adjacent extracted frames represent approximately 30 fps
- Mini-batch: 2
- Gradient accumulation: 4
- Effective batch: 8
- Validation/test clips: 10 / 10
- Evaluation clip-forward chunk: 1
- Head learning rate: `3e-4`
- Top backbone learning rate: `3e-5`
- Layer-wise learning-rate decay: `0.90`
- Weight decay: `0.05`
- Label smoothing: `0.10`
- Dropout: `0.35`
- Stochastic depth: `0.10`
- Warmup: 5 epochs
- Maximum epochs: 45
- Minimum epochs: 15
- Early-stopping patience: 12
- Gradient checkpointing: enabled
- CUDA AMP: enabled

## ViT-B distilled switch

Use:

```text
architecture                  videomaev2_vit_b_distilled
mini-batch                    1
gradient accumulation         8
stochastic depth              0.15
```

The effective batch remains 8. Keep the split, temporal sampling, learning rates, augmentation, and validation protocol fixed for the first comparison.

## Experiment policy

- Checkpoint selection and early stopping use validation only.
- Test evaluation is a separate explicit command.
- Reuse one `splits.json` for architecture and seed comparisons.
- Use `drop_train` for leakage-safe handling of confirmed exact official train/test duplicates.
- Treat the resulting protocol as official-derived and decontaminated.
