# ─── RTX 5090 / Blackwell compatibility fixes ────────────────────────────────
# RTX 5090 (Compute Capability 12.0a) is not yet supported by TF XLA/Triton.
# Force eager execution to bypass all XLA graph compilation entirely.
import os
os.environ["TF_XLA_FLAGS"]              = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"]                 = "--xla_gpu_enable_triton_gemm=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "2"
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"
# Must be set before tensorflow import
os.environ["TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"] = "0"

# ─── Cell 1: Install Dependencies ──────────────────────────────────────────────
import subprocess, sys

packages = [
    'tensorflow',
    'opencv-python',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'pandas',
    'numpy',
    'albumentations',
]

for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

print('All packages installed successfully.')

# ─── Cell 2: Local Dataset Path Setup ──────────────────────────────────────────
from pathlib import Path

# ▶▶▶  SET THIS TO YOUR DATASET ROOT FOLDER  ◀◀◀
DATASET_ROOT = Path.home() / 'Documents/Autism/dataset_autism'

# ── Validate the path exists ──────────────────────────────────────────────────
if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f'Dataset root not found: {DATASET_ROOT}\n'
        'Please update DATASET_ROOT to the correct path.'
    )

print('Dataset root:', DATASET_ROOT)
print('\nContents:')
for item in sorted(DATASET_ROOT.rglob('*')):
    if item.is_dir():
        print(' ', item)

# ── Auto-detect train / valid / test folders ──────────────────────────────────
def find_folder(base, names):
    """Search recursively for a folder matching any of the given names."""
    for name in names:
        for path in base.rglob(name):
            if path.is_dir():
                return str(path) + '/'
    return None

train_folders = find_folder(DATASET_ROOT, ['train', 'Train', 'TRAIN'])
valid_folders = find_folder(DATASET_ROOT, ['valid', 'val', 'Valid', 'Val', 'validation'])
test_folders  = find_folder(DATASET_ROOT, ['test',  'Test',  'TEST'])

print('\nDetected splits:')
print('  Train folder:', train_folders)
print('  Valid folder:', valid_folders)
print('  Test  folder:', test_folders)

if valid_folders is None:
    print('\nWARNING: No validation folder found. '
          'Create a valid/ folder or rename your existing validation folder.')

assert train_folders, 'Train folder not found — check your folder names.'
assert test_folders,  'Test folder not found — check your folder names.'

# ── Output directory for saved models ────────────────────────────────────────
MODEL_DIR = Path('saved_models_resnet50')
MODEL_DIR.mkdir(exist_ok=True)

print('\nSetup complete. MODEL_DIR:', MODEL_DIR)

# ─── Cell 3: GPU Check ─────────────────────────────────────────────────────────
import tensorflow as tf

# Disable XLA JIT at runtime level (belt-and-suspenders with env vars above)
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
print('TensorFlow version:', tf.__version__)
print('GPUs available:', len(gpus))

if gpus:
    for gpu in gpus:
        print('  GPU detected:', gpu.name)
    # Allow memory growth to prevent TF from allocating all VRAM at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Memory growth enabled.')
else:
    print('WARNING: No GPU detected. Training will run on CPU and will be very slow.')
    print('Make sure CUDA and cuDNN are installed and your GPU drivers are up to date.')

print('\nCUDA built with TF:', tf.test.is_built_with_cuda())
print('GPU available for TF:', bool(gpus))

# ─── Cell 4: Imports ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import random
import cv2
import hashlib
import itertools
from collections import Counter

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves plots to files
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import albumentations as A

CLASSES = ['non_autistic', 'autistic']

print('All imports successful.')

# ─── Cell 5: Perceptual Hash Deduplication ─────────────────────────────────────
def perceptual_hash(image_path, hash_size=8):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return hashlib.sha256(img_resized.tobytes()).hexdigest()


def collect_all_images(folder_roots):
    records = []
    for folder_root in folder_roots:
        for cls in CLASSES:
            cls_path = os.path.join(folder_root, cls)
            if not os.path.exists(cls_path):
                continue
            for fname in os.listdir(cls_path):
                full_path = os.path.join(cls_path, fname)
                records.append((full_path, cls))
    return records


def deduplicate_images(records):
    seen_hashes    = set()
    unique_records = []
    duplicates_removed = 0
    for path, cls in records:
        h = perceptual_hash(path)
        if h is None:
            continue
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_records.append((path, cls))
        else:
            duplicates_removed += 1
    print(f'Total images before deduplication : {len(records)}')
    print(f'Duplicates removed                : {duplicates_removed}')
    print(f'Unique images after deduplication : {len(unique_records)}')
    return unique_records


all_records    = collect_all_images([train_folders, valid_folders, test_folders])
unique_records = deduplicate_images(all_records)

class_counts = Counter([cls for _, cls in unique_records])
print('\nClass distribution after deduplication:')
for cls, cnt in class_counts.items():
    print(f'  {cls}: {cnt}')

# ─── Cell 6: Dataset Distribution ───────────────────────────────────────────────
def count_images(folder_root):
    counts = {}
    for cls in CLASSES:
        cls_path = os.path.join(folder_root, cls)
        counts[cls] = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
    return counts


quantity_tr = count_images(train_folders)
quantity_va = count_images(valid_folders) if valid_folders else {}
quantity_te = count_images(test_folders)

quantity_train = pd.DataFrame(list(quantity_tr.items()), columns=['class', 'count'])
quantity_valid = pd.DataFrame(list(quantity_va.items()), columns=['class', 'count'])
quantity_test  = pd.DataFrame(list(quantity_te.items()), columns=['class', 'count'])

figure, ax = plt.subplots(1, 3, figsize=(20, 5))
sns.barplot(x='class', y='count', data=quantity_train, ax=ax[0]).set_title('Training Set')
sns.barplot(x='class', y='count', data=quantity_valid, ax=ax[1]).set_title('Validation Set')
sns.barplot(x='class', y='count', data=quantity_test,  ax=ax[2]).set_title('Test Set')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'dataset_distribution.png', dpi=150)
plt.close()

print('Final image counts:')
print(f'  Train      : {sum(quantity_tr.values())}')
print(f'  Validation : {sum(quantity_va.values()) if quantity_va else 0}')
print(f'  Test       : {sum(quantity_te.values())}')

# ─── Cell 7: Subject-Independence Check ────────────────────────────────────────
def get_subject_ids(folder_root, id_delimiter='_', id_position=0):
    subject_ids = set()
    for cls in CLASSES:
        cls_path = os.path.join(folder_root, cls)
        if not os.path.exists(cls_path):
            continue
        for fname in os.listdir(cls_path):
            subject_id = fname.split(id_delimiter)[id_position]
            subject_ids.add(subject_id)
    return subject_ids


train_subjects = get_subject_ids(train_folders)
valid_subjects = get_subject_ids(valid_folders) if valid_folders else set()
test_subjects  = get_subject_ids(test_folders)

train_valid_overlap = train_subjects & valid_subjects
train_test_overlap  = train_subjects & test_subjects
valid_test_overlap  = valid_subjects & test_subjects

print(f'Unique subjects in train : {len(train_subjects)}')
print(f'Unique subjects in valid : {len(valid_subjects)}')
print(f'Unique subjects in test  : {len(test_subjects)}')
print(f'Train-Valid overlap : {len(train_valid_overlap)} (should be 0)')
print(f'Train-Test  overlap : {len(train_test_overlap)}  (should be 0)')
print(f'Valid-Test  overlap : {len(valid_test_overlap)}  (should be 0)')

if (len(train_test_overlap) == 0 and len(train_valid_overlap) == 0
        and len(valid_test_overlap) == 0):
    print('\n✓ Split is subject-independent. No identity leakage detected.')
else:
    print('\n⚠ Subject overlap detected. Review the split before proceeding.')

# ─── Cell 8: Offline Data Augmentation (Albumentations) ────────────────────────
# Each original training image is augmented AUGS_PER_IMAGE times to increase
# diversity and reduce overfitting. Augmentation is applied to the TRAINING
# SET ONLY — validation and test stay untouched so evaluation reflects real,
# unaltered images.
#
# Key techniques:
#   Geometric       : horizontal flip, random rotation (±10°)
#   Color/Brightness: hue/saturation/value shifts, gamma contrast adjustment
#   Noise/Quality   : Gaussian blur, Gaussian noise
#   Resize/Crop     : random resized crop to 224×224
AUGS_PER_IMAGE = 10
AUG_TRAIN_ROOT = Path('augmented_train_resnet50')

augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                          val_shift_limit=15, p=0.6),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(std_range=(0.05, 0.15), p=1.0),
    ], p=0.5),
    A.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
])


def build_augmented_train_set(source_root, dest_root, augs_per_image=10, force=False):
    """Copy original training images into dest_root, then generate
    `augs_per_image` Albumentations-augmented variants of each. Verifies
    per-class counts against the current source before deciding to skip, so
    a stale augmented set from a previous, differently-sized train/ folder
    is automatically detected and rebuilt rather than silently reused."""
    dest_root = Path(dest_root)
    if dest_root.exists() and not force:
        expected_counts = {}
        for cls in CLASSES:
            src_cls_dir = os.path.join(source_root, cls)
            n_source = len(os.listdir(src_cls_dir)) if os.path.exists(src_cls_dir) else 0
            expected_counts[cls] = n_source * (1 + augs_per_image)

        actual_counts = {}
        stale = False
        for cls in CLASSES:
            cls_dir = dest_root / cls
            actual_counts[cls] = len(list(cls_dir.glob('*'))) if cls_dir.exists() else 0
            if actual_counts[cls] != expected_counts[cls]:
                stale = True

        if not stale and any(actual_counts.values()):
            print(f'Augmented training set at {dest_root} matches current source '
                  f'counts {expected_counts} — skipping regeneration.')
            return str(dest_root) + '/'
        else:
            print(f'Augmented training set at {dest_root} is stale or missing '
                  f'(found {actual_counts}, expected {expected_counts}) — rebuilding.')
            import shutil
            shutil.rmtree(dest_root)

    if dest_root.exists() and force:
        import shutil
        shutil.rmtree(dest_root)

    for cls in CLASSES:
        (dest_root / cls).mkdir(parents=True, exist_ok=True)

    total_written = 0
    for cls in CLASSES:
        src_cls_dir = os.path.join(source_root, cls)
        if not os.path.exists(src_cls_dir):
            continue
        for fname in os.listdir(src_cls_dir):
            src_path = os.path.join(src_cls_dir, fname)
            img = cv2.imread(src_path)
            if img is None:
                continue

            stem, ext = os.path.splitext(fname)

            # 1) Copy the original, untouched, into the augmented set.
            orig_dest = dest_root / cls / fname
            cv2.imwrite(str(orig_dest), img)
            total_written += 1

            # 2) Generate augmented variants of it.
            for i in range(augs_per_image):
                augmented = augmentation_pipeline(image=img)['image']
                aug_dest = dest_root / cls / f'{stem}_aug{i+1}{ext}'
                cv2.imwrite(str(aug_dest), augmented)
                total_written += 1

    print(f'Augmented training set built at {dest_root} — {total_written} images '
          f'(original + {augs_per_image}x augmented per image).')
    return str(dest_root) + '/'


augmented_train_folders = build_augmented_train_set(
    train_folders, AUG_TRAIN_ROOT, augs_per_image=AUGS_PER_IMAGE
)

# ─── Cell 9: Data Generators ────────────────────────────────────────────────────
# Augmentation was already applied offline above, so these generators only
# rescale — applying additional shear/zoom on top would over-augment.
train_datagen      = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
test_datagen       = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    augmented_train_folders, batch_size=32, shuffle=True,
    class_mode='binary', target_size=(224, 224), classes=CLASSES
)
validation_generator = validation_datagen.flow_from_directory(
    valid_folders, shuffle=False, batch_size=32,
    class_mode='binary', target_size=(224, 224), classes=CLASSES
)
test_generator = test_datagen.flow_from_directory(
    test_folders, shuffle=False, batch_size=32,
    class_mode='binary', target_size=(224, 224), classes=CLASSES
)

class_indices   = validation_generator.class_indices
class_names     = list(class_indices.keys())
inv_map_classes = {v: k for k, v in class_indices.items()}
print('Class indices:', class_indices)
print('Inverse map  :', inv_map_classes)

# ─── Cell 10: Helper Functions ──────────────────────────────────────────────────
def save_history(history, model_name):
    hist_df = pd.DataFrame(history.history)
    with open(MODEL_DIR / f'{model_name}_history.csv', mode='w') as f:
        hist_df.to_csv(f)


def plot_history(history, model_name, epoches):
    x = list(range(1, epoches + 1))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(x, history.history['accuracy'],     label='Training',   linestyle='-')
    axes[0].plot(x, history.history['val_accuracy'], label='Validation', linestyle='--')
    axes[0].set_xlabel('Epochs'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{model_name} — Accuracy'); axes[0].legend()

    axes[1].plot(x, history.history['loss'],     label='Training',   linestyle='-')
    axes[1].plot(x, history.history['val_loss'], label='Validation', linestyle='--')
    axes[1].set_xlabel('Epochs'); axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{model_name} — Loss'); axes[1].legend()

    plt.tight_layout()
    plt.savefig(MODEL_DIR / f'{model_name}_curves.png', dpi=150)
    plt.close()


def clf_report(true_value, model_pred, class_names):
    TP_count = [true_value[i] == model_pred[i] for i in range(len(true_value))]
    print(f'Accuracy: {np.sum(TP_count)/len(TP_count):.4f}')
    plt.figure(figsize=(7, 7))
    cm = confusion_matrix(true_value, model_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
    plt.title('Confusion Matrix'); plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() * 0.8
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='black' if cm[i, j] > thresh else 'white')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'test_confusion_matrix_resnet50.png', dpi=150)
    plt.close()
    print(classification_report(true_value, model_pred, target_names=class_names))


def show_few_images(number_of_examples=6, model=None):
    figure1, ax1 = plt.subplots(
        number_of_examples, len(os.listdir(test_folders)),
        figsize=(12, 4 * number_of_examples)
    )
    ax1 = ax1.reshape(-1)
    for ax in ax1:
        ax.axis('off')
    axs = 0
    for folder in os.listdir(test_folders):
        image_ids = os.listdir(os.path.join(test_folders, folder))
        for j in [random.randrange(0, len(image_ids)) for _ in range(number_of_examples)]:
            display = plt.imread(os.path.join(test_folders, folder, image_ids[j]))
            figure1.tight_layout()
            ax1[axs].imshow(display)
            title = 'True: ' + folder
            if model is not None:
                img_input = np.array([cv2.resize(display, (224, 224))])
                pred = model.predict(img_input, verbose=0).argmax()
                title += f'\nResNet-50: {"Autistic" if pred==1 else "Non-Autistic"}'
            ax1[axs].set_title(title, fontsize=8)
            axs += 1
    plt.savefig(MODEL_DIR / 'sample_predictions_resnet50.png', dpi=150)
    plt.close()


print('Helper functions defined.')

# ─── Cell 11: Sample Images and Config ─────────────────────────────────────────
show_few_images(6)

tf.keras.backend.clear_session()
epoches = 100
print(f'Training for {epoches} epochs.')

# ─── Cell 12: ResNet-50 (Final Model) ──────────────────────────────────────────
# Check both .keras (new) and .h5 (legacy) saved models
_resnet50_keras = str(MODEL_DIR / 'resnet50_model.keras')
_resnet50_h5    = str(MODEL_DIR / 'resnet50_model.h5')
_resnet50_path  = _resnet50_keras if os.path.exists(_resnet50_keras) else _resnet50_h5

if os.path.exists(_resnet50_path):
    print('ResNet-50 already trained — loading from', _resnet50_path)
    resnet50_final_model = load_model(_resnet50_path)
    resnet50_final_model.compile(loss='sparse_categorical_crossentropy',
        optimizer=SGD(learning_rate=0.01, momentum=0.7), metrics=['accuracy'], run_eagerly=True)
    resnet50_history = None
else:
    # NOTE on preprocessing: ResNet-50 has BatchNorm after nearly every conv
    # layer, frozen with running statistics tuned for Caffe-style
    # preprocessing (RGB->BGR, zero-centered by ImageNet channel means — NOT
    # a [0,1] rescale). Feeding [0,1] pixels into these frozen BN layers
    # pushes activations into a degenerate range and can collapse
    # predictions to a single class. Rescaling(255.0) undoes the outer
    # 1/255 from the generator, then resnet50.preprocess_input applies the
    # exact preprocessing the frozen backbone expects.
    resnet50_input = tf.keras.Input(shape=(224, 224, 3))
    resnet50_rescaled = tf.keras.layers.Rescaling(255.0)(resnet50_input)
    resnet50_preprocessed = tf.keras.layers.Lambda(
        resnet50_preprocess_input, output_shape=(224, 224, 3)
    )(resnet50_rescaled)
    resnet50_base = ResNet50(
        weights='imagenet', include_top=False, input_tensor=resnet50_preprocessed
    )
    # Backbone frozen — only the new Dense head is trained.
    for layer in resnet50_base.layers:
        layer.trainable = False

    resnet50_x = Flatten()(resnet50_base.output)
    resnet50_x = Dense(256, activation='relu')(resnet50_x)
    resnet50_x = Dense(2, activation='sigmoid')(resnet50_x)
    resnet50_final_model = Model(inputs=resnet50_input, outputs=resnet50_x)
    resnet50_final_model.summary()
    resnet50_final_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=SGD(learning_rate=0.01, momentum=0.7),
        metrics=['accuracy'],
        run_eagerly=True
    )
    resnet50_history = resnet50_final_model.fit(
        train_generator, epochs=epoches, validation_data=validation_generator
    )
    resnet50_final_model.save(_resnet50_path)
    save_history(resnet50_history, 'resnet50')
    plot_history(resnet50_history, 'ResNet-50', epoches)
    print('ResNet-50 saved to', _resnet50_path)

# ─── Cell 13: Test Set Evaluation ──────────────────────────────────────────────
true_value  = []
resnet_pred = []

for folder in os.listdir(test_folders):
    test_image_ids = os.listdir(os.path.join(test_folders, folder))
    for image_id in test_image_ids:
        path = os.path.join(test_folders, folder, image_id)
        true_value.append(class_indices[folder])

        img = cv2.resize(cv2.imread(path), (224, 224))
        img_input = np.array([img / 255.0])

        r = resnet50_final_model.predict(img_input, verbose=0).argmax()
        resnet_pred.append(r)

print(f'Test samples evaluated: {len(true_value)}')
print('=' * 60)
print('ResNet-50 — Test Set Classification Report')
print('=' * 60)
clf_report(true_value, resnet_pred, class_names)

# ─── Cell 14: Sample Predictions on Test Images ────────────────────────────────
show_few_images(6, resnet50_final_model)

# ─── Cell 15: Prepare Full Dataset for Cross-Validation ────────────────────────
# Uses the augmented training set (original + 10x Albumentations variants)
# for the train portion, and the raw, unaugmented valid/test folders.
all_image_paths, all_labels = [], []

for folder_root in [augmented_train_folders, valid_folders, test_folders]:
    if folder_root is None:
        continue
    for cls in CLASSES:
        cls_path = os.path.join(folder_root, cls)
        if not os.path.exists(cls_path):
            continue
        for fname in os.listdir(cls_path):
            all_image_paths.append(os.path.join(cls_path, fname))
            all_labels.append(class_indices[cls])

all_image_paths = np.array(all_image_paths)
all_labels      = np.array(all_labels)

print(f'Total images for cross-validation : {len(all_image_paths)}')
print(f'Class distribution                : {Counter(all_labels)}')

# ─── Cell 16: Load Images Helper ───────────────────────────────────────────────
def load_images(paths, target_size=(224, 224)):
    images = []
    for path in paths:
        img = cv2.resize(cv2.imread(str(path)), target_size)
        images.append(img / 255.0)
    return np.array(images)


def build_fold_resnet50():
    # Same frozen-BatchNorm/preprocessing correction as the final model above.
    inputs   = tf.keras.Input(shape=(224, 224, 3))
    rescaled = tf.keras.layers.Rescaling(255.0)(inputs)
    preprocessed = tf.keras.layers.Lambda(
        resnet50_preprocess_input, output_shape=(224, 224, 3)
    )(rescaled)
    base = ResNet50(weights='imagenet', include_top=False, input_tensor=preprocessed)
    for layer in base.layers:
        layer.trainable = False
    x = Flatten()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(2,   activation='sigmoid')(x)
    m = Model(inputs, x)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer=SGD(learning_rate=0.01, momentum=0.7),
              metrics=['accuracy'], run_eagerly=True)
    return m


print('Cross-validation helpers defined.')

# ─── Cell 17: Stratified 5-Fold Cross-Validation (ResNet-50 only) ──────────────
N_SPLITS  = 5
CV_EPOCHS = 30

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cv_results = {'fold': [], 'accuracy': [],
              'precision_autistic': [], 'recall_autistic': [], 'f1_autistic': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(all_image_paths, all_labels)):
    print('\n' + '=' * 50)
    print(f' Fold {fold+1} / {N_SPLITS}')
    print('=' * 50)

    X_train_fold = load_images(all_image_paths[train_idx])
    y_train_fold = all_labels[train_idx]
    X_val_fold   = load_images(all_image_paths[val_idx])
    y_val_fold   = all_labels[val_idx]

    print(f'Train: {len(X_train_fold)} | Val: {len(X_val_fold)}')

    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(False)

    fold_model = build_fold_resnet50()
    fold_model.fit(
        X_train_fold, y_train_fold,
        epochs=CV_EPOCHS, batch_size=32,
        validation_data=(X_val_fold, y_val_fold),
        verbose=1
    )

    probs    = fold_model.predict(X_val_fold, verbose=0)
    y_pred   = probs.argmax(axis=1)
    report   = classification_report(
        y_val_fold, y_pred, target_names=class_names, output_dict=True
    )
    acc = np.mean(y_pred == y_val_fold)

    cv_results['fold'].append(fold + 1)
    cv_results['accuracy'].append(acc)
    cv_results['precision_autistic'].append(report['autistic']['precision'])
    cv_results['recall_autistic'].append(report['autistic']['recall'])
    cv_results['f1_autistic'].append(report['autistic']['f1-score'])

    print(f'ResNet-50 Fold {fold+1} Accuracy: {acc:.4f}')
    print(classification_report(y_val_fold, y_pred, target_names=class_names))

# ─── Cell 18: Print + Save CV Summary ──────────────────────────────────────────
def make_summary_df(results):
    df = pd.DataFrame(results)
    summary = pd.DataFrame([{
        'fold':               'Mean ± Std',
        'accuracy':           f"{df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}",
        'precision_autistic': f"{df['precision_autistic'].mean():.4f} ± {df['precision_autistic'].std():.4f}",
        'recall_autistic':    f"{df['recall_autistic'].mean():.4f} ± {df['recall_autistic'].std():.4f}",
        'f1_autistic':        f"{df['f1_autistic'].mean():.4f} ± {df['f1_autistic'].std():.4f}",
    }])
    return pd.concat([df, summary], ignore_index=True)


cv_summary_df = make_summary_df(cv_results)
print('===== 5-Fold Cross-Validation Summary (ResNet-50) =====')
print(cv_summary_df.to_string(index=False))
cv_summary_df.to_csv(MODEL_DIR / 'cv_results_resnet50.csv', index=False)

# ─── Cell 19: Cross-Validation Bar Chart ───────────────────────────────────────
mean_acc = np.mean(cv_results['accuracy'])
plt.figure(figsize=(8, 5))
plt.bar([f'Fold {i}' for i in cv_results['fold']], cv_results['accuracy'],
        color='darkorange', edgecolor='black', alpha=0.85)
plt.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean = {mean_acc:.4f}')
plt.xlabel('Fold'); plt.ylabel('Accuracy')
plt.title('ResNet-50 — 5-Fold Cross-Validation Accuracy')
plt.ylim(0, 1); plt.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / 'cv_accuracy_resnet50.png', dpi=150)
plt.close()
print('CV chart saved: cv_accuracy_resnet50.png')

# ─── Cell 20: Reload Saved Model (run this if session restarts) ───────────────
# Uncomment and run this cell if you need to reload the model after a crash

# from tensorflow.keras.models import load_model
# resnet50_final_model = load_model(str(MODEL_DIR / 'resnet50_model.keras'))
# print('Model reloaded successfully.')
