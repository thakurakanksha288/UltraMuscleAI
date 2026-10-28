# UMUD Muscle Architecture Pipeline

Automated estimation of **pennation angle (PA)**, **fascicle length (FL)** and **muscle thickness (MT)** from B-mode ultrasound for the UMUD challenge. Two segmentation ensembles (aponeurosis, fascicle) feed a rater-faithful geometry stage, followed by calibration and temporal smoothing.

```
image ──► apo U-Net++ ensemble ──► superficial/deep curves ──► MT (3-point perpendicular)
      └─► fasc U-Net++ ensemble ──► PCA fascicle lines ──┬──► PA (vs. deep tangent)
                                                          └──► FL (extrapolated intersections)
                                        px→mm calibration ─► temporal smoothing ─► submission.csv
```

## Why this design scores well

| Component                                                             | Accuracy lever                                                                                              |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Duplicate-aware 5-fold CV (`data.make_folds`)                       | the organisers flagged duplicate images; grouping them avoids inflated validation and leaky model selection |
| U-Net++ with scSE + ConvNeXt/EfficientNet encoders (timm via `smp`) | strong pretrained features; two architectures = heterogeneous ensemble                                      |
| `bce_dice_boundary` for aponeurosis                                 | sharper mask borders → lower MT error (MT tolerance is tight: 1 mm)                                        |
| `focal_dice` for fascicles                                          | thin sparse structures are heavily imbalanced                                                               |
| EMA weights, cosine LR + warmup, AMP, grad-clip                       | stable convergence, better generalisation to unseen devices (Lumify in test only)                           |
| Device-style augmentation (gamma, CLAHE, noise, blur, affine ±8°)   | test devices/muscles differ from training; no vertical flip to keep superficial/deep ordering               |
| Fold ensemble + h-flip TTA + multi-scale                              | smoother probability maps → more stable component extraction                                               |
| Robust quadratic aponeurosis fit with linear extrapolation            | mirrors manual extrapolation when fascicles exit the frame                                                  |
| Length-weighted medians over fascicles, orientation consensus         | resists spurious short components                                                                           |
| `FL ≈ MT / sin(PA)` closure + structure-tensor fallback            | no image is left without a prediction                                                                       |
| Sequence detection + median blending                                  | exploits the 5-frame video snippets in the test set for temporal stability                                  |
| Clipping to published physiological ranges                            | bounded worst-case error on the normalised MAE                                                              |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU with ≥12 GB VRAM recommended (batch 8 @ 512²). On CPU use `--encoder resnet34 --img-size 384 --batch-size 4`.

## Data layout

```
data/
  apo_imgs/  apo_masks/      # 1048 pairs (matched by file stem)
  fasc_imgs/ fasc_masks/     # 2761 pairs
  test_images/               # 309 images
  sample_submission.csv
```

## Train

```bash
python -m umud.train --task apo  --data-root data --loss bce_dice_boundary
python -m umud.train --task fasc --data-root data --loss focal_dice
```

Key options: `--arch` (`Unet`, `UnetPlusPlus`, `FPN`, `DeepLabV3Plus`, `Segformer`), `--encoder` (`resnet34`, `tu-convnext_small`, `tu-efficientnet_b4`, …), `--img-size`, `--epochs`, `--folds 0 1`. Checkpoints land in `runs/<task>/<arch>_<encoder>/fold{k}.pt` and store their config so inference is self-describing.

## Validate (out-of-fold)

```bash
python -m umud.validate --task apo  --runs runs
python -m umud.validate --task fasc --runs runs
```

Reports Dice at several thresholds and, more importantly, the **downstream geometry agreement** (MT px, PA deg, FL px) between architecture derived from predicted masks vs reference masks — the closest offline proxy for the UMUD score. Use it to pick `--apo-thr/--fasc-thr`.

## Predict

```bash
python -m umud.predict --data-root data --runs runs --out submission.csv \
    --scales 0.875 1.0 1.125 --prior-scale --overlays overlays
```

`submission_debug.csv` records the calibration source, detection status and fascicle count per image; `overlays/` draws the fitted curves and fascicles for visual QA.

# Create virtual environment

python -m venv .venv

# Activate it (Windows PowerShell)

.venv\Scripts\Activate.ps1

# Install requirements

pip install -r requirements.txt

### Pixel → mm calibration (read this)

Masks give pixels; the leaderboard wants millimetres. `umud/scale.py` resolves px/mm in this order:

1. `--scale-csv scales.csv` (`image_id,px_per_mm`) – best; fill it if you know acquisition depth/device
2. `--device-json device_scales.json` (`{"HxW": px_per_mm}`) – per-device defaults keyed by image size
3. automatic detection of the depth-ruler ticks on the image margin
4. `--prior-scale`: pick among candidate depths (3–8 cm) the one making MT/FL most physiologically plausible, else `--default-depth-cm`

Inspect `scale_src` in the debug CSV. If many test images fall to `default_depth`/`prior`, create `device_scales.json` from a few images with visible rulers — this is the single largest remaining error source.

## Ensemble reproducibility rule

The challenge requires the final method to re-run on a new dataset with no leaderboard. This pipeline satisfies that: everything is derived from the training masks and fixed heuristics; no leaderboard feedback is used in any step.

## Project layout

```
umud/config.py       ranges, priors, CLI config
umud/data.py         pairing, fingerprint dedup folds, augmentations, inference pre/post sizing
umud/models.py       smp model factory, Dice/Focal/Boundary combo losses
umud/train.py        k-fold trainer (EMA, AMP, cosine LR)
umud/infer.py        fold ensemble + TTA + multi-scale
umud/postprocess.py  curve fitting, fascicle lines, PA/FL/MT geometry, overlays
umud/scale.py        px/mm calibration strategies
umud/predict.py      submission writer with temporal smoothing
umud/validate.py     OOF Dice + geometry-agreement proxy metric
run_all.sh           full reproducible run
```

## License

MIT – see `LICENSE`.
