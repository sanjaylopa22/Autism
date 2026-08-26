# ─── RTX 5090 / Blackwell compatibility fixes ────────────────────────────────
import os
os.environ["TF_XLA_FLAGS"]              = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"]                 = "--xla_gpu_enable_triton_gemm=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "2"
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"
os.environ["TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"] = "0"

# ─── Cell 1: Install Dependencies ──────────────────────────────────────────────
import subprocess, sys

packages = [
    'tensorflow',
    'opencv-python',
    'scikit-learn',
    'matplotlib',
    'numpy',
    'shap',
]
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

print('All packages installed successfully.')

# ─── CONFIDENTIALITY NOTE ───────────────────────────────────────────────────────
# This script never displays, plots, or saves any actual facial pixel data.
# Explanations are computed by dividing each image into an abstract GRID_SIZE
# x GRID_SIZE grid of anonymous cells (e.g. "row 2, col 5") and reporting the
# SHAP value (contribution to the prediction) PER GRID POSITION. All saved
# outputs are numeric heatmaps/bar charts over grid positions — none of them
# render, embed, or reconstruct the underlying image content.

# ─── Cell 2: Dataset + Model Paths ──────────────────────────────────────────────
from pathlib import Path

# ▶▶▶ SET THIS to the same dataset root used for training ◀◀◀
DATASET_ROOT = Path.home() / 'Documents/Autism/dataset_autism'

if not DATASET_ROOT.exists():
    raise FileNotFoundError(f'Dataset root not found: {DATASET_ROOT}')


def find_model_dir(search_roots, markers=('vgg16_model.keras', 'vgg16_model.h5')):
    """Auto-detect the saved-models folder by searching for a directory that
    actually contains a trained model file."""
    for root in search_roots:
        if not root.exists():
            continue
        for marker in markers:
            for hit in root.rglob(marker):
                return hit.parent
    return None


_search_roots = [Path.cwd(), Path.cwd().parent, Path.home()]
MODEL_DIR = find_model_dir(_search_roots)

if MODEL_DIR is None:
    raise FileNotFoundError(
        'Could not auto-locate a saved_models folder containing trained '
        'models (looked for vgg16_model.keras / .h5) under:\n'
        + '\n'.join(f'  - {r}' for r in _search_roots) +
        '\n\nEither run the training script first, or set MODEL_DIR '
        'explicitly to the correct path.'
    )
print('Auto-detected MODEL_DIR:', MODEL_DIR)


def find_folder(base, names):
    for name in names:
        for path in base.rglob(name):
            if path.is_dir():
                return str(path) + '/'
    return None


test_folders = find_folder(DATASET_ROOT, ['test', 'Test', 'TEST'])
assert test_folders, 'Test folder not found — check DATASET_ROOT.'
print('Test folder:', test_folders)

CLASSES    = ['non_autistic', 'autistic']
IMG_SIZE   = (224, 224)
OUTPUT_DIR = Path('explainability_shap')
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Cell 3: GPU Setup ──────────────────────────────────────────────────────────
import tensorflow as tf
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
print('GPUs available:', len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ─── Cell 4: Imports ────────────────────────────────────────────────────────────
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

import shap

print('Imports complete.')

# ─── Cell 5: Load Trained Models ────────────────────────────────────────────────
def _resolve_path(name):
    keras_path = MODEL_DIR / f'{name}.keras'
    h5_path    = MODEL_DIR / f'{name}.h5'
    path = keras_path if keras_path.exists() else h5_path
    if not path.exists():
        raise FileNotFoundError(
            f'Could not find {keras_path} or {h5_path}. '
            'Make sure training has completed for this model.'
        )
    return str(path)


print('Loading models...')
models = {
    'VGG-16':          load_model(_resolve_path('vgg16_model')),
    'ResNet-50':       load_model(
        _resolve_path('resnet50_model'),
        custom_objects={'preprocess_input': resnet50_preprocess_input}
    ),
    'InceptionV3':     load_model(_resolve_path('inceptionv3_model')),
    'EfficientNet-B0': load_model(_resolve_path('efficientnetb0_model')),
    'EfficientNet-B7': load_model(_resolve_path('efficientnetb7_model')),
}
MODEL_NAMES = list(models.keys())
print(f'Loaded {len(models)} models: {MODEL_NAMES}')

# ─── Cell 6: Prediction Functions ──────────────────────────────────────────────
PREDICT_BATCH_SIZE = 16


def make_predict_fn(model):
    def predict_fn(images_uint8):
        images_float = images_uint8.astype(np.float32) / 255.0
        return model.predict(images_float, batch_size=PREDICT_BATCH_SIZE, verbose=0)
    return predict_fn


individual_predict_fns = {name: make_predict_fn(m) for name, m in models.items()}


def ensemble_predict_fn(images_uint8):
    """Soft-vote ensemble: average the 5 base models' probability outputs."""
    images_float = images_uint8.astype(np.float32) / 255.0
    all_probs = [
        m.predict(images_float, batch_size=PREDICT_BATCH_SIZE, verbose=0)
        for m in models.values()
    ]
    return np.mean(all_probs, axis=0)


print('Prediction functions ready.')

# ─── Cell 7: Select Sample Test Images ─────────────────────────────────────────
NUM_IMAGES_PER_CLASS = 2
NUM_MISCLASSIFIED    = 2

candidates = []
for cls in CLASSES:
    cls_dir = os.path.join(test_folders, cls)
    for fname in os.listdir(cls_dir):
        candidates.append((os.path.join(cls_dir, fname), CLASSES.index(cls), cls))

print(f'Scoring {len(candidates)} test images with the ensemble to pick examples...')

scored = []
for path, true_idx, true_name in candidates:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    probs = ensemble_predict_fn(np.array([img]))[0]
    pred_idx = int(np.argmax(probs))
    scored.append({
        # NOTE: 'image' is kept only in memory for this run, to feed the
        # model — it is never written to disk or plotted anywhere below.
        'path': path, 'image': img, 'true_idx': true_idx, 'true_name': true_name,
        'pred_idx': pred_idx, 'pred_name': CLASSES[pred_idx],
        'confidence': float(probs[pred_idx]), 'correct': pred_idx == true_idx,
    })

correct_examples = [s for s in scored if s['correct']]
wrong_examples    = [s for s in scored if not s['correct']]

selected = []
for cls_idx, cls_name in enumerate(CLASSES):
    cls_correct = [s for s in correct_examples if s['true_idx'] == cls_idx]
    selected.extend(cls_correct[:NUM_IMAGES_PER_CLASS])
selected.extend(wrong_examples[:NUM_MISCLASSIFIED])

# Anonymous labels used everywhere downstream instead of filenames/paths —
# keeps saved outputs free of anything that could identify a specific
# subject's image file.
for i, s in enumerate(selected):
    s['anon_id'] = f'sample_{i+1:02d}'

print(f'Selected {len(selected)} images for explanation.')
if not wrong_examples:
    print('Note: zero misclassifications found among scored test images — '
          'all selected examples are correct predictions.')

# ─── Cell 8: Grid-Segment Kernel SHAP ──────────────────────────────────────────
# Each image is divided into a GRID_SIZE x GRID_SIZE grid of anonymous cells.
# Kernel SHAP treats "cell is present / cell is masked out" as the feature
# space (not raw pixels), so every SHAP value corresponds to a GRID POSITION
# ("row 3, col 5"), never to image content. Masked-out cells are replaced
# with a blurred version of the same image (standard image-SHAP baseline) —
# this blurred image exists only transiently in memory for prediction and is
# never saved or plotted.
GRID_SIZE   = 7                    # 7x7 = 49 grid cells per image
N_FEATURES  = GRID_SIZE * GRID_SIZE
NUM_SAMPLES = 500                  # Kernel SHAP perturbation samples per explanation


def make_grid_masked_image(image, active_mask, baseline):
    h, w, _ = image.shape
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
    out = image.copy()
    for idx in range(N_FEATURES):
        if active_mask[idx] == 0:
            r, c = divmod(idx, GRID_SIZE)
            y0, y1 = r * cell_h, (r + 1) * cell_h if r < GRID_SIZE - 1 else h
            x0, x1 = c * cell_w, (c + 1) * cell_w if c < GRID_SIZE - 1 else w
            out[y0:y1, x0:x1] = baseline[y0:y1, x0:x1]
    return out


def make_shap_predict_fn(image, baseline, predict_fn, class_idx):
    """Wraps a model's predict_fn so SHAP sees a function of GRID MASKS
    (shape (n_samples, N_FEATURES)) -> probability of class_idx, never
    pixels directly."""
    def f(mask_matrix):
        imgs = np.stack([
            make_grid_masked_image(image, row, baseline) for row in mask_matrix
        ])
        return predict_fn(imgs)[:, class_idx]
    return f


def compute_grid_shap(image, predict_fn, class_idx):
    baseline = cv2.GaussianBlur(image, (31, 31), 0)
    f = make_shap_predict_fn(image, baseline, predict_fn, class_idx)
    background = np.zeros((1, N_FEATURES))          # fully-masked (baseline) reference
    instance   = np.ones((1, N_FEATURES))            # the real image = all cells present
    explainer  = shap.KernelExplainer(f, background)
    values     = explainer.shap_values(instance, nsamples=NUM_SAMPLES, silent=True)
    return np.array(values).flatten().reshape(GRID_SIZE, GRID_SIZE)


def plot_grid_heatmap(shap_grid, ax, title, vmax):
    """Plots ONLY the abstract GRID_SIZE x GRID_SIZE SHAP-value grid — a
    plain numeric heatmap with no image content whatsoever. vmax is shared
    across all panels in a figure so color intensity is comparable between
    models and the figure's single colorbar is accurate for every panel."""
    im = ax.imshow(shap_grid, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(GRID_SIZE)); ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel('Grid Column'); ax.set_ylabel('Grid Row')
    ax.set_title(title, fontsize=9)
    return im


all_grid_shap = {name: [] for name in MODEL_NAMES}
ensemble_grids = []   # tracked separately so the aggregate summary can include it

for example in selected:
    print(f"\nExplaining {example['anon_id']}: true={example['true_name']}, "
          f"ensemble_pred={example['pred_name']} "
          f"({'correct' if example['correct'] else 'WRONG'}, "
          f"confidence={example['confidence']:.3f})")
    print('  NOTE: each panel below explains support for the class the ENSEMBLE '
          "predicted — a model that itself disagreed with the ensemble will show "
          "low/negative values here, not an explanation of its own top class.")

    class_idx = example['pred_idx']   # explain the class the ensemble predicted

    # Compute every model's grid first, so we can share one color scale
    # across all panels — otherwise each panel silently normalizes to its
    # own max and a single shared colorbar becomes misleading.
    grids = {}
    for model_name in MODEL_NAMES:
        print(f'  Computing SHAP for {model_name}...')
        grids[model_name] = compute_grid_shap(
            example['image'], individual_predict_fns[model_name], class_idx
        )
        all_grid_shap[model_name].append(grids[model_name])

    print('  Computing SHAP for Ensemble (soft-vote)...')
    ens_grid = compute_grid_shap(example['image'], ensemble_predict_fn, class_idx)
    ensemble_grids.append(ens_grid)
    grids['Ensemble (Soft-Vote)'] = ens_grid

    shared_vmax = max(np.abs(g).max() for g in grids.values()) or 1e-8

    n_panels = len(grids)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 4))
    for ax, (name, grid) in zip(axes, grids.items()):
        im = plot_grid_heatmap(grid, ax, name, shared_vmax)

    fig.suptitle(
        f"{example['anon_id']} — True: {example['true_name']} | "
        f"Predicted: {example['pred_name']} ({example['confidence']:.2f}) | "
        f"SHAP values for predicted class, by grid position",
        fontsize=10
    )
    fig.colorbar(im, ax=axes, shrink=0.6, label='SHAP value (higher = supports predicted class)')

    status = 'correct' if example['correct'] else 'MISCLASSIFIED'
    out_path = OUTPUT_DIR / f"shap_grid_{example['anon_id']}_{example['true_name']}_{status}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}  (grid-position heatmap only — no image content)')

    # Top-contributing grid cells as a bar chart (position labels only)
    fig, ax = plt.subplots(figsize=(7, 4))
    flat = ens_grid.flatten()
    top_idx = np.argsort(np.abs(flat))[::-1][:10]
    labels = [f'({i // GRID_SIZE},{i % GRID_SIZE})' for i in top_idx]
    ax.barh(labels[::-1], flat[top_idx][::-1],
            color=['crimson' if v > 0 else 'steelblue' for v in flat[top_idx][::-1]])
    ax.set_xlabel('SHAP value'); ax.set_ylabel('Grid cell (row, col)')
    ax.set_title(f"{example['anon_id']} — Top 10 grid cells, Ensemble", fontsize=10)
    plt.tight_layout()
    top_path = OUTPUT_DIR / f"shap_top_cells_{example['anon_id']}.png"
    plt.savefig(top_path, dpi=150)
    plt.close()
    print(f'  Saved: {top_path}')

print(f'\nAll SHAP grid-position plots saved to: {OUTPUT_DIR}/ (no facial images included)')

# ─── Cell 9: Aggregate SHAP Summary Across Selected Images ─────────────────────
# Averages |SHAP value| per grid cell across all selected images, per model
# (including the Ensemble) — shows which grid POSITIONS tend to matter most
# in general. Still purely positional/numeric; no image content.
agg_names = MODEL_NAMES + ['Ensemble (Soft-Vote)']
mean_abs_grids = {
    name: np.mean([np.abs(g) for g in all_grid_shap[name]], axis=0)
    for name in MODEL_NAMES
}
mean_abs_grids['Ensemble (Soft-Vote)'] = np.mean([np.abs(g) for g in ensemble_grids], axis=0)

shared_vmax_agg = max(g.max() for g in mean_abs_grids.values()) or 1e-8

fig, axes = plt.subplots(1, len(agg_names), figsize=(3.6 * len(agg_names), 4))
if len(agg_names) == 1:
    axes = [axes]

for ax, name in zip(axes, agg_names):
    im = ax.imshow(mean_abs_grids[name], cmap='viridis', vmin=0, vmax=shared_vmax_agg)
    ax.set_xticks(range(GRID_SIZE)); ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel('Grid Column'); ax.set_ylabel('Grid Row')
    ax.set_title(name, fontsize=9)

fig.suptitle(f'Mean |SHAP value| by grid position, across {len(selected)} sample images', fontsize=11)
fig.colorbar(im, ax=axes, shrink=0.6, label='Mean |SHAP value|')
plt.savefig(OUTPUT_DIR / 'shap_aggregate_grid_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Aggregate grid-importance plot saved: {OUTPUT_DIR / 'shap_aggregate_grid_importance.png'}")

# ─── Cell 10: Model Agreement Summary ──────────────────────────────────────────
import pandas as pd

print('\nScoring model agreement with ensemble across all test images...')
agreement_rows = []
for s in scored:
    img_batch = np.array([s['image']])
    row = {'true': s['true_name'], 'ensemble_pred': s['pred_name']}
    for name, fn in individual_predict_fns.items():
        probs = fn(img_batch)[0]
        row[name] = CLASSES[int(np.argmax(probs))]
    agreement_rows.append(row)

agreement_df = pd.DataFrame(agreement_rows)   # numeric/text only — no images
for name in MODEL_NAMES:
    agreement_rate = (agreement_df[name] == agreement_df['ensemble_pred']).mean()
    print(f'  {name:16s} agrees with ensemble on {agreement_rate:.1%} of test images')

agreement_df.to_csv(OUTPUT_DIR / 'model_agreement_summary.csv', index=False)
print(f'\nAgreement summary saved to: {OUTPUT_DIR / "model_agreement_summary.csv"}')
