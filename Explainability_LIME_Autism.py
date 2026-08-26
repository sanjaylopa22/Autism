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
    'scikit-image',
    'matplotlib',
    'numpy',
    'lime',
]
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

print('All packages installed successfully.')

# ─── Cell 2: Dataset + Model Paths ──────────────────────────────────────────────
from pathlib import Path

# ▶▶▶ SET THIS to the same dataset root used for training ◀◀◀
DATASET_ROOT = Path.home() / 'Documents/Autism/dataset_autism'
# ▶▶▶ SET THIS to the same saved_models folder produced by training ◀◀◀
MODEL_DIR    = Path('saved_models')

if not DATASET_ROOT.exists():
    raise FileNotFoundError(f'Dataset root not found: {DATASET_ROOT}')
if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f'Model directory not found: {MODEL_DIR}\n'
        'Run the training script first so trained models exist to explain.'
    )


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
OUTPUT_DIR = Path('explainability_lime')
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

from lime import lime_image
from skimage.segmentation import mark_boundaries

print('Imports complete.')

# ─── Cell 5: Load Trained Models ────────────────────────────────────────────────
# Only the 5 base architectures that make up the soft-vote ENSEMBLE are loaded
# here (matches the training script's ensemble definition). VGG-16-Dropout is
# a standalone ablation, not part of the ensemble, so it's excluded — add it
# below the same way if you also want it explained individually.
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

# ─── Cell 6: Prediction Functions for LIME ─────────────────────────────────────
# LIME's explain_instance() calls this function with a batch of perturbed
# images as uint8 RGB arrays in [0, 255] (LIME's default image format) and
# expects back an (N, num_classes) array of class probabilities.
#
# Every model here was trained on [0, 1]-scaled images (each architecture's
# specific preprocessing — Caffe-style mean-centering for ResNet-50, etc. —
# is baked INSIDE the saved model graph itself), so the only conversion
# needed here is uint8[0,255] -> float[0,1]. Batched internally at a safe
# size to avoid the large-batch GPU OOM issues seen during CV training.
PREDICT_BATCH_SIZE = 16


def make_predict_fn(model):
    def predict_fn(images_uint8):
        images_float = images_uint8.astype(np.float32) / 255.0
        return model.predict(images_float, batch_size=PREDICT_BATCH_SIZE, verbose=0)
    return predict_fn


individual_predict_fns = {name: make_predict_fn(m) for name, m in models.items()}


def ensemble_predict_fn(images_uint8):
    """Soft-vote ensemble: average the 5 base models' probability outputs —
    matches the ensemble definition used in training/evaluation."""
    images_float = images_uint8.astype(np.float32) / 255.0
    all_probs = [
        m.predict(images_float, batch_size=PREDICT_BATCH_SIZE, verbose=0)
        for m in models.values()
    ]
    return np.mean(all_probs, axis=0)


print('Prediction functions ready.')

# ─── Cell 7: Select Sample Test Images ─────────────────────────────────────────
# Picks a mix of images the ensemble got right AND wrong (if any misclassified
# examples exist) — misclassifications are usually the most informative case
# for explainability, since they show what the model was looking at when it
# got fooled.
NUM_IMAGES_PER_CLASS = 2   # correct examples per class
NUM_MISCLASSIFIED    = 2   # misclassified examples, if available (any class)

candidates = []   # (path, true_label_idx, true_label_name)
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

print(f'Selected {len(selected)} images for explanation '
      f'({len(selected) - min(len(wrong_examples), NUM_MISCLASSIFIED)} correct, '
      f'{min(len(wrong_examples), NUM_MISCLASSIFIED)} misclassified).')
if not wrong_examples:
    print('Note: the ensemble had zero misclassifications among test images scored above — '
          'all selected examples are correct predictions.')

# ─── Cell 8: Run LIME + Save Explanation Panels ────────────────────────────────
# LIME segments the image into superpixels, randomly perturbs which
# superpixels are shown/hidden, observes how the prediction changes, and
# fits a local linear model to identify which superpixels drove the
# prediction. NUM_SAMPLES controls how many perturbations are evaluated —
# higher is more stable but proportionally slower (each sample triggers a
# forward pass through every model being explained for that panel).
NUM_SAMPLES  = 500   # lower (e.g. 200) for a quick look; raise for stability
NUM_FEATURES = 8     # number of top superpixels highlighted per explanation

explainer = lime_image.LimeImageExplainer()


def explain_and_plot(image_uint8, predict_fn, ax, title):
    explanation = explainer.explain_instance(
        image_uint8, predict_fn,
        top_labels=2, hide_color=0, num_samples=NUM_SAMPLES
    )
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=NUM_FEATURES, hide_rest=False
    )
    ax.imshow(mark_boundaries(temp / 255.0, mask))
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    return CLASSES[top_label]


for i, example in enumerate(selected):
    print(f'\nExplaining image {i+1}/{len(selected)}: '
          f'true={example["true_name"]}, ensemble_pred={example["pred_name"]} '
          f'({"correct" if example["correct"] else "WRONG"}, '
          f'confidence={example["confidence"]:.3f})')

    n_panels = len(MODEL_NAMES) + 2   # + original + ensemble
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))

    axes[0].imshow(example['image'])
    axes[0].set_title(
        f"Original\nTrue: {example['true_name']}\n"
        f"Ensemble: {example['pred_name']} ({example['confidence']:.2f})",
        fontsize=9
    )
    axes[0].axis('off')

    for j, model_name in enumerate(MODEL_NAMES):
        print(f'  Running LIME on {model_name}...')
        explain_and_plot(
            example['image'], individual_predict_fns[model_name],
            axes[j + 1], model_name
        )

    print('  Running LIME on Ensemble (soft-vote)...')
    explain_and_plot(
        example['image'], ensemble_predict_fn, axes[-1], 'Ensemble (Soft-Vote)'
    )

    plt.tight_layout()
    status = 'correct' if example['correct'] else 'MISCLASSIFIED'
    out_path = OUTPUT_DIR / f'lime_{i+1:02d}_{example["true_name"]}_{status}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')

print(f'\nAll explanation panels saved to: {OUTPUT_DIR}/')

# ─── Cell 9: Model Agreement Summary ────────────────────────────────────────────
# Beyond visual explanations, a quick numeric summary of how often each
# individual model agrees with the ensemble's final decision — useful context
# for interpreting *why* certain LIME panels disagree with each other.
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

agreement_df = pd.DataFrame(agreement_rows)
for name in MODEL_NAMES:
    agreement_rate = (agreement_df[name] == agreement_df['ensemble_pred']).mean()
    print(f'  {name:16s} agrees with ensemble on {agreement_rate:.1%} of test images')

agreement_df.to_csv(OUTPUT_DIR / 'model_agreement_summary.csv', index=False)
print(f'\nAgreement summary saved to: {OUTPUT_DIR / "model_agreement_summary.csv"}')
