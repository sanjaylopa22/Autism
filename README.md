# ASD Facial Image Classification — Transfer Learning Pipeline

Deep learning pipeline for automated Autism Spectrum Disorder (ASD) classification from facial
images, using pretrained CNN architectures (VGG-16, ResNet-50, InceptionV3, EfficientNet-B0,
EfficientNet-B7) fine-tuned via transfer learning on ImageNet weights, with soft- and
hard-voting ensembles, 5-fold stratified cross-validation, and offline data augmentation.

## Contents

| Script | Description |
|---|---|
| `Autism_Classification_LocalDrive_new.py` | Full pipeline — VGG-16 (with and without dropout), InceptionV3, EfficientNet-B0, EfficientNet-B7, plus soft-vote and hard-vote ensembles |
| `ResNet50_Classification_LocalDrive.py` | Standalone ResNet-50-only pipeline (same stages, single architecture, with CV checkpoint/resume support) |

Both scripts run the identical data pipeline — deduplication, subject-independence
verification, offline augmentation, final-model training, test-set evaluation, and 5-fold
cross-validation — over the models each one covers.

# Experimented Datasets

https://data.mendeley.com/datasets/f9dycfvwbt/3

https://www.kaggle.com/datasets/prayashdas/autistic-children-facial-image-dataset

https://data.mendeley.com/datasets/b33pf78h62/1

## Dataset layout

```
dataset_autism/
├── train/
│   ├── autistic/
│   └── non_autistic/
├── valid/
│   ├── autistic/
│   └── non_autistic/
└── test/
    ├── autistic/
    └── non_autistic/
```

- Folder names are matched case-insensitively (`train`/`Train`/`TRAIN`, etc.).
- Set `DATASET_ROOT` near the top of each script to your dataset's location.
- Subject IDs are derived from the leading token of each filename (split on `_`). If your
  filenames don't encode a real subject/patient ID (e.g. plain sequential `001.jpg`,
  `002.jpg` per folder), the subject-independence check cannot verify anything meaningful —
  see [Known limitations](#known-limitations).

## Setup

```bash
python -m venv autism_env
source autism_env/bin/activate   # or autism_env\Scripts\activate on Windows
python Autism_Classification_LocalDrive_final_all_models.py   # installs its own dependencies on first run
```

Both scripts `pip install` their dependencies automatically on startup:
`tensorflow`, `opencv-python`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`,
`albumentations`.

**GPU note:** both scripts include compatibility workarounds for RTX 5090 / Blackwell GPUs
(forcing eager execution, disabling XLA). If you're on older hardware these are harmless
no-ops; if you hit CUDA/XLA errors on other GPUs, those env vars at the top of each script
are the first place to look.

## Pipeline stages

1. **Deduplication** — perceptual hashing removes exact/near-duplicate images across all
   three splits before anything else runs.
2. **Subject-independence check** — verifies no subject appears in more than one split
   (train/valid/test), to catch identity leakage before it silently inflates results.
3. **Offline augmentation (Albumentations)** — each training image is copied once, plus
   10 augmented variants (horizontal flip, ±10° rotation, hue/saturation/value shift, gamma
   contrast, Gaussian blur/noise, random resized crop to 224×224). Written to
   `augmented_train/` (or `augmented_train_resnet50/`). The build step verifies per-class
   counts against the current source folder and automatically rebuilds if they don't match —
   so a stale augmented set left over from a previous dataset version can't be silently reused.
4. **Final model training** — one pass per architecture, frozen ImageNet backbone, trained
   head only. Models and training curves are saved under `saved_models/` (or
   `saved_models_resnet50/`); a saved model is loaded and reused on subsequent runs instead of
   retraining.
5. **Test-set evaluation** — classification report + confusion matrix per model, plus
   hard-vote and soft-vote ensemble reports (full pipeline only).
6. **5-fold stratified cross-validation** — retrains each architecture from scratch per fold
   on pooled train+valid+test data, reporting per-fold and mean±std accuracy/precision/
   recall/F1.

## Known issues fixed in this version

Several bugs were identified and corrected during development — documented here since they're
easy to reintroduce if this pipeline is extended to new architectures:

- **EfficientNet-B0/B7 double-rescaling.** `tf.keras.applications.EfficientNet*` includes a
  built-in `Rescaling` layer expecting raw `[0, 255]` pixel input. Feeding it the same `[0, 1]`
  images used for the other models divided by 255 twice, collapsing every input near zero and
  causing training to get stuck at `loss ≈ ln(2)`, accuracy pinned at 50%, predicting a single
  class. Fixed with an explicit `Rescaling(255.0)` layer before the EfficientNet base.
- **ResNet-50 / InceptionV3 frozen-BatchNorm preprocessing mismatch.** Both architectures have
  BatchNorm layers throughout, frozen with running statistics tuned for their specific
  preprocessing (`resnet50.preprocess_input`: BGR + ImageNet-mean-centered; `inception_v3.
  preprocess_input`: scaled to `[-1, 1]`) — not the generic `[0, 1]` rescale used elsewhere.
  This only surfaces once the backbone is frozen (a full fine-tune can adapt around it), and
  manifests identically to the EfficientNet bug: collapse to a single predicted class. Fixed by
  baking `Rescaling(255.0)` + the architecture-specific `preprocess_input` into the model graph.
  VGG-16 has no BatchNorm and is unaffected by this class of bug.
- **Subject-independence check ignoring valid↔test overlap.** The original pass/fail condition
  only checked train↔valid and train↔test, silently ignoring a valid↔test overlap of 100% in
  one run. Fixed to check all three pairwise overlaps.
- **Stale augmented-training-set cache.** The original cache check only verified the augmented
  folder was non-empty, so a dataset reorganization (different per-class counts) would be
  silently ignored and training would run on a stale, potentially imbalanced augmented set.
  Fixed to verify exact expected-vs-actual per-class counts before skipping regeneration.
- **`Lambda`-layer preprocessing functions don't survive `save`/`load_model` by default.**
  Keras serializes a `Lambda` layer's wrapped function by name only; reloading raises
  `Could not locate function 'preprocess_input'` unless the function is passed back in via
  `custom_objects` at load time.

## Known limitations

- **Subject ID extraction assumes a naming convention.** `get_subject_ids()` takes the
  filename's leading token (split on `_`) as a proxy subject ID. If your dataset uses generic
  sequential filenames per folder (e.g. `001.jpg`) with no real subject encoding, this check
  cannot detect genuine identity leakage — it will report a coincidental "overlap" or "no
  overlap" that isn't meaningful. Verify your filename convention before trusting this check.
- **ResNet-50's frozen backbone increases the effective head size substantially** (ResNet-50's
  un-pooled output is `7×7×2048` vs. VGG-16's `7×7×512`), so its `Dense(256)` head alone has
  ~25.7M parameters — worth knowing if you're comparing parameter counts or overfitting risk
  across architectures.
- **GPU memory contention.** Long training runs interrupted by hard crashes (e.g. driver-level
  OOM aborts) have been observed to leave orphaned CUDA processes holding VRAM after the
  Python process exits. If a run OOMs immediately on a GPU that should have ample free memory,
  check `nvidia-smi` for stray processes before assuming a code-level bug.
- **CV fold checkpointing (ResNet-50 script) relies on deterministic fold generation.**
  `StratifiedKFold(..., random_state=42)` reproduces identical folds across runs *provided*
  the underlying image list (`os.listdir()` order) hasn't changed — e.g. don't add/remove
  files from the augmented training folder between a crash and a resume.

## Output artifacts

Each script writes to its own model directory (`saved_models/` or `saved_models_resnet50/`):

- `<model>_model.keras` / `.h5` — trained model weights
- `<model>_history.csv` — per-epoch training history
- `<model>_curves.png` — accuracy/loss curves
- `dataset_distribution.png` — class counts per split
- `cv_results_<model>.csv` — per-fold + mean±std cross-validation metrics
- `cv_accuracy_<model>.png` / `cv_accuracy_all_models.png` — CV accuracy bar charts
- `cv_checkpoint_resnet50.json` (ResNet-50 script only) — fold-level resume checkpoint

## Disclaimer

This is a research/experimental pipeline, not a validated clinical or diagnostic tool. Results
should not be used to make or support any individual diagnostic decision.
