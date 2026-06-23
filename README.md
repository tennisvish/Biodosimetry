# ConvPatch-ViT: Radiation Biodosimetry from 53BP1 Fluorescence Microscopy

A Vision Transformer that predicts radiation dose (Gy) from single-nucleus 53BP1 fluorescence microscopy images, without manual foci counting.

**Test performance:** R² = 0.635, MAE = 0.145 Gy (vs. R² = 0.553, MAE = 0.203 Gy for a polynomial foci-counting baseline)

This is the reference implementation for the ConvPatch-ViT described in *[paper title / venue TBD]*.

## Overview

Standard dosimetry from 53BP1 foci relies on manually counting discrete nuclear foci, which is labor-intensive and discards biological signal not captured by foci counts alone (e.g. dense, sub-resolution DSB clusters, texture, cell-cycle effects). ConvPatch-ViT instead predicts dose directly from raw fluorescence images.

It modifies the standard ViT patch embedding by replacing direct patch extraction with a 3-stage convolutional pipeline, giving each patch token a locally-contextualized feature representation before the transformer sees it — important for resolving the small (~3–8 px) 53BP1 foci that would otherwise be split across patch boundaries.

## Architecture

```
Input (224×224×1, single-channel 53BP1)
  → ConvPatchEncoder
      conv 3×3 (s=1) → BN → GELU
      conv 3×3 (s=1) → BN → GELU
      conv 14×14 (s=14) → BN → GELU   [patch splitting, d=256]
      → 256 patch tokens + CLS token → learned positional embedding
  → 6 × TransformerBlock
      pre-norm multi-head self-attention (8 heads, d_k=32)
      MLP: Dense(2d) → GELU → DepthwiseConv2D(3×3) on patch grid → Dense(d)
  → LayerNorm → CLS token
  → Dense(512) → Dense(256) → Dense(128) → Dense(1)   [GELU, dropout]
  → predicted dose (Gy)
```

The depthwise convolution inside each transformer block's MLP reshapes patch tokens back to their spatial grid, preserving locality between adjacent patches throughout the network depth — a local inductive bias layered on top of global self-attention.

| Hyperparameter | Value |
|---|---|
| Patch size | 14 |
| Embedding dim | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Dropout | 0.15 |
| Loss | MSE (on z-normalized dose) |
| Optimizer | AdamW (weight decay 0.02, grad clip norm 1.0) |
| LR schedule | Linear warmup (5 epochs) → cosine decay with warm restarts |
| Batch size | 32 |
| Max epochs | 100 (early stopping, patience 90) |

## Dataset

[NASA BPS Microscopy Benchmark Training Dataset](https://registry.opendata.aws/nasa-bps-training-data/) — mouse fibroblast nuclei imaged by 53BP1 fluorescence.

This model uses only the **X-ray (low-LET)** subset at **4 hours post-exposure**, across 3 doses (0, 0.1, 1 Gy): **13,847 images** total. Iron (high-LET) data is excluded — particle traversal is stochastic at the single-track level, so dose can't be learned per-image the way it can for uniform X-ray exposure.

Split: 70% train / 20% val / 10% test, stratified by dose (`random_state=42`).

## Preprocessing

```
16-bit TIFF → clip intensities to [400, 4000] (sensor range)
            → CLAHE (clipLimit=2.0, tileGridSize=8×8)
            → resize to 224×224
            → normalize to [0,1] via (I − 400) / 3600
```

Training-only augmentation: horizontal/vertical flips, 90°-multiple rotations, brightness scaling (×0.8–1.2), and mixup (λ ~ Uniform(0.3, 0.7)).

## Results

| Model | R² | MAE (Gy) |
|---|---|---|
| Basic ViT, MAE loss | 0.389 | 0.164 |
| Basic ViT, MSE loss | 0.545 | 0.188 |
| **ConvPatch-ViT, MSE loss (this model)** | **0.635** | **0.146** |
| Polynomial foci-count baseline | 0.553 | 0.203 |

Evaluated on the held-out test set (1,385 images). Predictions are also evaluated with 8-way test-time augmentation (TTA); see script output for the TTA vs. no-TTA comparison.

## Usage

### Requirements
```
tensorflow
numpy
pandas
scikit-learn
opencv-python
matplotlib
```

### Data layout

Update the paths at the top of the script to point to your local copies:
```python
CSV_PATH   = '<path to metadata CSV with columns: filename, hr_post_exposure, particle_type, dose_Gy>'
IMAGES_DIR = '<path to .tif image directory>'
```

### Train and evaluate
```bash
python convpatch_vit_final.py
```

This runs the full pipeline: data loading/splitting, training with early stopping, test-set evaluation (standard + TTA), per-dose MAE breakdown, and saves:

- `convpatch_vit_final.keras` — trained model
- `convpatch_vit_final_preds.csv` — per-sample true/predicted doses
- `convpatch_vit_final_results.png` — predicted-vs-true scatter, prediction distributions by dose, training curves

### Inference on a new image
```python
from tensorflow import keras
model = keras.models.load_model('convpatch_vit_final.keras')

img = load_image('path/to/image.tif')          # see preprocessing above
pred = model.predict(img[None, ...])
dose_gy = pred[0][0] * stats['std'] + stats['mean']   # de-normalize using training stats
```

`predict_with_tta()` (8 geometric augmentations averaged) instead of a single forward pass but not included in paper as it is not part of training architecture.

## Limitations

- Trained on only 3 discrete dose levels (0, 0.1, 1 Gy); interpolation between 0.1 and 1 Gy is unvalidated.
- X-ray (low-LET) only — does not generalize to other radiation types without retraining, but Iron Poisson analysis can be used for population level patterns.
- Single cell line, single imaging source/protocol; cross-lab generalization requires further validation.
- Fixed to the 4-hour post-exposure timepoint.

See the accompanying manuscript for full discussion.

## Citation

If you use this code, please cite:

TBD

## License

[Add license]
