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
    'albumentations'
]

for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

print('All packages installed successfully.')

# ─── Cell 2: Local Dataset Path Setup ──────────────────────────────────────────
import os
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
MODEL_DIR = Path('saved_models')
MODEL_DIR.mkdir(exist_ok=True)

print('\nSetup complete. MODEL_DIR:', MODEL_DIR)

# ─── Cell 5: GPU Check ─────────────────────────────────────────────────────────
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

# Optional: verify CUDA
print('\nCUDA built with TF:', tf.test.is_built_with_cuda())
print('GPU available for TF:', tf.test.is_gpu_available())

# ─── Cell 6: Imports ───────────────────────────────────────────────────────────
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

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from keras import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dropout, Dense, Flatten

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7

import albumentations as A

CLASSES = ['non_autistic', 'autistic']

# Output directory for saved models
MODEL_DIR = Path('saved_models')
MODEL_DIR.mkdir(exist_ok=True)

print('All imports successful.')

# ─── Cell 7: Perceptual Hash Deduplication ─────────────────────────────────────
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
    seen_hashes   = set()
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

# ─── Cell 8: Dataset Distribution ──────────────────────────────────────────────
def count_images(folder_root):
    counts = {}
    for cls in CLASSES:
        cls_path = os.path.join(folder_root, cls)
        if os.path.exists(cls_path):
            counts[cls] = len(os.listdir(cls_path))
        else:
            counts[cls] = 0
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

# ─── Cell 9: Subject-Independence Check ────────────────────────────────────────
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

# ─── Cell 9.5: Offline Data Augmentation (Albumentations) ──────────────────────
# Each original training image is augmented AUGS_PER_IMAGE times to increase
# diversity and reduce overfitting. Augmentation is applied to the TRAINING
# SET ONLY — validation and test stay untouched so evaluation reflects
# real, unaltered images. All 5 models train on this same augmented set.
#
# Key techniques:
#   Geometric      : horizontal flip, random rotation (±10°)
#   Color/Brightness: hue/saturation/value shifts, gamma contrast adjustment
#   Noise/Quality  : Gaussian blur, Gaussian noise
#   Resize/Crop    : random resized crop to 224×224
AUGS_PER_IMAGE   = 10
AUG_TRAIN_ROOT   = Path('augmented_train')

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
    `augs_per_image` Albumentations-augmented variants of each. Skips work
    if dest_root already exists and is non-empty (set force=True to redo)."""
    dest_root = Path(dest_root)
    if dest_root.exists() and not force:
        # FIX: previously this only checked "does the folder exist and have
        # *any* files in it" — so a stale augmented_train/ left over from a
        # PREVIOUS, differently-sized train/ folder (e.g. before a dataset
        # reorganization) would be silently reused forever, producing class
        # imbalance or wrong totals with no warning. Now we verify each
        # class's expected count (source images * (1 + augs_per_image))
        # actually matches what's on disk before deciding to skip.
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

# ─── Cell 10: Data Generators ──────────────────────────────────────────────────
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

# ─── Cell 11: Helper Functions ──────────────────────────────────────────────────
def save_history(history, model_name):
    hist_df = pd.DataFrame(history.history)
    with open(MODEL_DIR / f'{model_name}_history.csv', mode='w') as f:
        hist_df.to_csv(f)


def mode(my_list):
    ct = Counter(my_list)
    max_value = max(ct.values())
    return [key for key, value in ct.items() if value == max_value]


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
    plt.tight_layout(); plt.close()
    print(classification_report(true_value, model_pred, target_names=class_names))


def show_few_images(number_of_examples=6, models=None):
    figure1, ax1 = plt.subplots(
        number_of_examples, len(os.listdir(test_folders)),
        figsize=(20, 4 * number_of_examples)
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
            if models:
                img_input = np.array([cv2.resize(display, (224, 224))])
                preds = [m.predict(img_input, verbose=0).argmax() for m in models]
                names = ['VGG-16', 'ResNet-50', 'InceptionV3', 'EfficientNet-B0', 'EfficientNet-B7']
                for name, pred in zip(names, preds):
                    title += f'\n{name}: {"Autistic" if pred==1 else "Non-Autistic"}'
                ensemble = mode(preds)
                title += f'\nEnsemble: {"Autistic" if ensemble==[1] else "Non-Autistic"}'
            ax1[axs].set_title(title, fontsize=7)
            axs += 1
    plt.close()


print('Helper functions defined.')

# ─── Cell 12: Sample Images and Config ─────────────────────────────────────────
show_few_images(6)

tf.keras.backend.clear_session()
epoches = 100
print(f'Training for {epoches} epochs per model.')

# ─── Cell 13: VGG-16 ───────────────────────────────────────────────────────────
from tensorflow.keras.models import load_model

# Check both .keras (new) and .h5 (legacy) saved models
_vgg16_keras = str(MODEL_DIR / 'vgg16_model.keras')
_vgg16_h5    = str(MODEL_DIR / 'vgg16_model.h5')
_vgg16_path  = _vgg16_keras if os.path.exists(_vgg16_keras) else _vgg16_h5
if os.path.exists(_vgg16_path):
    print('VGG-16 already trained — loading from', _vgg16_path)
    vgg16_final_model = load_model(_vgg16_path)
    vgg16_final_model.compile(loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001), metrics=['accuracy'], run_eagerly=True)
    vgg16_history     = None
else:
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg16_base.layers:
        layer.trainable = False
    vgg_x = Flatten()(vgg16_base.output)
    vgg_x = Dense(256, activation='relu')(vgg_x)
    vgg_x = Dense(2, activation='sigmoid')(vgg_x)
    vgg16_final_model = Model(vgg16_base.input, vgg_x)
    vgg16_final_model.summary()
    vgg16_final_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy'],
        run_eagerly=True
    )
    vgg16_history = vgg16_final_model.fit(
        train_generator, epochs=epoches, validation_data=validation_generator
    )
    vgg16_final_model.save(_vgg16_path)
    save_history(vgg16_history, 'vgg16')
    plot_history(vgg16_history, 'VGG-16', epoches)
    print('VGG-16 saved to', _vgg16_path)

# ─── Cell 13b: VGG-16 (Dropout variant) ────────────────────────────────────────
# Second VGG-16 head with Dropout(0.5) added before the output layer, trained
# for direct comparison against the no-dropout VGG-16 above.
_vgg16drop_keras = str(MODEL_DIR / 'vgg16_dropout_model.keras')
_vgg16drop_h5    = str(MODEL_DIR / 'vgg16_dropout_model.h5')
_vgg16drop_path  = _vgg16drop_keras if os.path.exists(_vgg16drop_keras) else _vgg16drop_h5
if os.path.exists(_vgg16drop_path):
    print('VGG-16 (Dropout) already trained — loading from', _vgg16drop_path)
    vgg16_dropout_final_model = load_model(_vgg16drop_path)
    vgg16_dropout_final_model.compile(loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001), metrics=['accuracy'], run_eagerly=True)
    vgg16_dropout_history     = None
else:
    vgg16drop_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg16drop_base.layers:
        layer.trainable = False
    vgg16drop_x = Flatten()(vgg16drop_base.output)
    vgg16drop_x = Dense(256, activation='relu')(vgg16drop_x)
    vgg16drop_x = Dropout(0.5)(vgg16drop_x)
    vgg16drop_x = Dense(2, activation='sigmoid')(vgg16drop_x)
    vgg16_dropout_final_model = Model(vgg16drop_base.input, vgg16drop_x)
    vgg16_dropout_final_model.summary()
    vgg16_dropout_final_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy'],
        run_eagerly=True
    )
    vgg16_dropout_history = vgg16_dropout_final_model.fit(
        train_generator, epochs=epoches, validation_data=validation_generator
    )
    vgg16_dropout_final_model.save(_vgg16drop_path)
    save_history(vgg16_dropout_history, 'vgg16_dropout')
    plot_history(vgg16_dropout_history, 'VGG-16-Dropout', epoches)
    print('VGG-16 (Dropout) saved to', _vgg16drop_path)

# ─── Cell 14: ResNet-50 ────────────────────────────────────────────────────────
# Check both .keras (new) and .h5 (legacy) saved models
_resnet50_keras = str(MODEL_DIR / 'resnet50_model.keras')
_resnet50_h5    = str(MODEL_DIR / 'resnet50_model.h5')
_resnet50_path  = _resnet50_keras if os.path.exists(_resnet50_keras) else _resnet50_h5

if os.path.exists(_resnet50_path):
    print('ResNet-50 already trained — loading from', _resnet50_path)
    # FIX: the model contains a Lambda layer wrapping resnet50_preprocess_input.
    # Lambda layers save the wrapped function by NAME only ("preprocess_input"),
    # so on load, Keras needs to be told explicitly what that name maps to —
    # otherwise it raises "Could not locate function 'preprocess_input'".
    resnet50_final_model = load_model(
        _resnet50_path,
        custom_objects={'preprocess_input': resnet50_preprocess_input}
    )
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

# ─── Cell 15: InceptionV3 ──────────────────────────────────────────────────────
# Check both .keras (new) and .h5 (legacy) saved models
_inceptionv3_keras = str(MODEL_DIR / 'inceptionv3_model.keras')
_inceptionv3_h5    = str(MODEL_DIR / 'inceptionv3_model.h5')
_inceptionv3_path  = _inceptionv3_keras if os.path.exists(_inceptionv3_keras) else _inceptionv3_h5
if os.path.exists(_inceptionv3_path):
    print('InceptionV3 already trained — loading from', _inceptionv3_path)
    inceptionv3_final_model = load_model(_inceptionv3_path)
    inceptionv3_final_model.compile(loss='sparse_categorical_crossentropy',
        optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'], run_eagerly=True)
    inceptionv3_history     = None
else:
    inceptionv3_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in inceptionv3_base.layers:
        layer.trainable = False
    inceptionv3_x = Flatten()(inceptionv3_base.output)
    inceptionv3_x = Dense(512, activation='relu')(inceptionv3_x)
    inceptionv3_x = Dropout(0.2)(inceptionv3_x)
    inceptionv3_x = Dense(2, activation='sigmoid')(inceptionv3_x)
    inceptionv3_final_model = Model(inputs=inceptionv3_base.input, outputs=inceptionv3_x)
    inceptionv3_final_model.summary()
    inceptionv3_final_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=RMSprop(learning_rate=0.0001),
        metrics=['accuracy'],
        run_eagerly=True
    )
    inceptionv3_history = inceptionv3_final_model.fit(
        train_generator, epochs=epoches, validation_data=validation_generator
    )
    inceptionv3_final_model.save(_inceptionv3_path)
    save_history(inceptionv3_history, 'inceptionv3')
    plot_history(inceptionv3_history, 'InceptionV3', epoches)
    print('InceptionV3 saved to', _inceptionv3_path)

# ─── Cell 16: EfficientNet-B0 ──────────────────────────────────────────────────
# Check both .keras (new) and .h5 (legacy) saved models
_efficientnetb0_keras = str(MODEL_DIR / 'efficientnetb0_model.keras')
_efficientnetb0_h5    = str(MODEL_DIR / 'efficientnetb0_model.h5')
_efficientnetb0_path  = _efficientnetb0_keras if os.path.exists(_efficientnetb0_keras) else _efficientnetb0_h5
if os.path.exists(_efficientnetb0_path):
    print('EfficientNet-B0 already trained — loading from', _efficientnetb0_path)
    efficientnetb0_final_model = load_model(_efficientnetb0_path)
    efficientnetb0_final_model.compile(loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001), metrics=['accuracy'], run_eagerly=True)
    efficientnetb0_history     = None
else:
    # FIX: EfficientNet's built-in preprocessing expects raw pixel values in
    # [0, 255]. Our generators feed [0, 1] (rescale=1/255), same as the other
    # models, which made EfficientNet divide by 255 a second time internally
    # — crushing inputs to near-zero and killing gradients (loss stuck at
    # ln(2)=0.693, accuracy pinned at 50%). An explicit Rescaling(255.0)
    # layer undoes the outer scaling before the base model's own
    # normalization is applied.
    efficientnetb0_input    = tf.keras.Input(shape=(224, 224, 3))
    efficientnetb0_rescaled = tf.keras.layers.Rescaling(255.0)(efficientnetb0_input)
    efficientnetb0_base = EfficientNetB0(
        weights='imagenet', include_top=False, input_tensor=efficientnetb0_rescaled
    )
    for layer in efficientnetb0_base.layers:
        layer.trainable = False
    efficientnetb0_x = Flatten()(efficientnetb0_base.output)
    efficientnetb0_x = Dense(512, activation='relu')(efficientnetb0_x)
    efficientnetb0_x = Dense(256, activation='relu')(efficientnetb0_x)
    efficientnetb0_x = Dropout(0.5)(efficientnetb0_x)
    efficientnetb0_x = Dense(2, activation='sigmoid')(efficientnetb0_x)
    efficientnetb0_final_model = Model(inputs=efficientnetb0_input, outputs=efficientnetb0_x)
    efficientnetb0_final_model.summary()
    efficientnetb0_final_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy'],
        run_eagerly=True
    )
    efficientnetb0_history = efficientnetb0_final_model.fit(
        train_generator, epochs=epoches, validation_data=validation_generator
    )
    efficientnetb0_final_model.save(_efficientnetb0_path)
    save_history(efficientnetb0_history, 'efficientnetb0')
    plot_history(efficientnetb0_history, 'EfficientNet-B0', epoches)
    print('EfficientNet-B0 saved to', _efficientnetb0_path)

# ─── Cell 17: EfficientNet-B7 ──────────────────────────────────────────────────
# EfficientNet-B7 has very large intermediate activations — reduce batch size
# to 8 to avoid OOM on the 32GB RTX 5090 with 490MB model + activation maps.
from tensorflow.keras.preprocessing.image import ImageDataGenerator as _IDG
# Augmentation already applied offline (see Cell 9.5) — rescale only here.
_b7_train_gen = _IDG(rescale=1/255).flow_from_directory(
    augmented_train_folders, batch_size=8, shuffle=True,
    class_mode='binary', target_size=(224, 224), classes=CLASSES
)
_b7_valid_gen = _IDG(rescale=1/255).flow_from_directory(
    valid_folders, shuffle=False, batch_size=8,
    class_mode='binary', target_size=(224, 224), classes=CLASSES
)

# Check both .keras (new) and .h5 (legacy) saved models
_efficientnetb7_keras = str(MODEL_DIR / 'efficientnetb7_model.keras')
_efficientnetb7_h5    = str(MODEL_DIR / 'efficientnetb7_model.h5')
_efficientnetb7_path  = _efficientnetb7_keras if os.path.exists(_efficientnetb7_keras) else _efficientnetb7_h5
if os.path.exists(_efficientnetb7_path):
    print('EfficientNet-B7 already trained — loading from', _efficientnetb7_path)
    efficientnetb7_final_model = load_model(_efficientnetb7_path)
    efficientnetb7_final_model.compile(loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001), metrics=['accuracy'], run_eagerly=True)
    efficientnetb7_history     = None
else:
    # FIX: same double-rescale bug as EfficientNet-B0 — undo the outer
    # 1/255 scaling before EfficientNet's own internal normalization.
    efficientnetb7_input    = tf.keras.Input(shape=(224, 224, 3))
    efficientnetb7_rescaled = tf.keras.layers.Rescaling(255.0)(efficientnetb7_input)
    efficientnetb7_base = EfficientNetB7(
        weights='imagenet', include_top=False, input_tensor=efficientnetb7_rescaled
    )
    for layer in efficientnetb7_base.layers:
        layer.trainable = False
    efficientnetb7_x = Flatten()(efficientnetb7_base.output)
    efficientnetb7_x = Dense(512, activation='relu')(efficientnetb7_x)
    efficientnetb7_x = Dense(256, activation='relu')(efficientnetb7_x)
    efficientnetb7_x = Dropout(0.5)(efficientnetb7_x)
    efficientnetb7_x = Dense(2, activation='sigmoid')(efficientnetb7_x)
    efficientnetb7_final_model = Model(inputs=efficientnetb7_input, outputs=efficientnetb7_x)
    efficientnetb7_final_model.summary()
    efficientnetb7_final_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy'],
        run_eagerly=True
    )
    efficientnetb7_history = efficientnetb7_final_model.fit(
        _b7_train_gen, epochs=epoches, validation_data=_b7_valid_gen
    )
    efficientnetb7_final_model.save(_efficientnetb7_path)
    save_history(efficientnetb7_history, 'efficientnetb7')
    # FIX: labels were incorrectly showing EfficientNet-B0 for B7 in original code
    plot_history(efficientnetb7_history, 'EfficientNet-B7', epoches)
    print('EfficientNet-B7 saved to', _efficientnetb7_path)

# ─── Cell 18: Combined Training Curves ─────────────────────────────────────────
# Only plot curves for models that were trained this run (history is None for loaded models)
x = list(range(1, epoches + 1))
model_names = ['VGG-16', 'VGG-16-Dropout', 'ResNet-50', 'InceptionV3', 'EfficientNet-B0', 'EfficientNet-B7']
histories   = [vgg16_history, vgg16_dropout_history, resnet50_history, inceptionv3_history,
               efficientnetb0_history, efficientnetb7_history]

trained_pairs = [(n, h) for n, h in zip(model_names, histories) if h is not None]

if trained_pairs:
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    for name, hist in trained_pairs:
        ep = list(range(1, len(hist.history['accuracy']) + 1))
        axes[0, 0].plot(ep, hist.history['accuracy'],     label=name)
        axes[0, 1].plot(ep, hist.history['val_accuracy'], label=name)
        axes[1, 0].plot(ep, hist.history['loss'],         label=name)
        axes[1, 1].plot(ep, hist.history['val_loss'],     label=name)

    titles  = ['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']
    ylabels = ['Accuracy', 'Accuracy', 'Loss', 'Loss']
    for ax, title, ylabel in zip(axes.flatten(), titles, ylabels):
        ax.set_title(title); ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel); ax.legend()

    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'all_models_curves.png', dpi=150)
    plt.close()
    print('Combined curves saved.')
else:
    print('All models loaded from disk — skipping combined training curves plot.')

# ─── Cell 19: Test Set Evaluation ──────────────────────────────────────────────
true_value              = []
vgg_pred                = []
vgg_dropout_pred        = []
resnet_pred             = []
inceptionv3_pred        = []
efficientnetb0_pred     = []
efficientnetb7_pred     = []
ensembled_pred_hard     = []   # majority vote over predicted classes
ensemble_probs_all      = []   # per-image list of softmax-style probability vectors, for soft-voting

# The 5-model ensemble — VGG-16 (no-dropout), ResNet-50, InceptionV3,
# EfficientNet-B0, EfficientNet-B7. VGG-16-Dropout is reported separately as
# an ablation and is NOT part of the ensemble.

# FIX: iterate over test_folders, not valid_folders
for folder in os.listdir(test_folders):
    test_image_ids = os.listdir(os.path.join(test_folders, folder))
    for image_id in test_image_ids:
        path = os.path.join(test_folders, folder, image_id)
        true_value.append(class_indices[folder])

        img = cv2.resize(cv2.imread(path), (224, 224))
        img_input = np.array([img / 255.0])

        vgg_probs = vgg16_final_model.predict(img_input, verbose=0)
        vgg_drop_probs = vgg16_dropout_final_model.predict(img_input, verbose=0)
        resnet_probs = resnet50_final_model.predict(img_input, verbose=0)
        inc_probs = inceptionv3_final_model.predict(img_input, verbose=0)
        b0_probs = efficientnetb0_final_model.predict(img_input, verbose=0)
        b7_probs = efficientnetb7_final_model.predict(img_input, verbose=0)

        v   = vgg_probs.argmax()
        vd  = vgg_drop_probs.argmax()
        r   = resnet_probs.argmax()
        inc = inc_probs.argmax()
        b0  = b0_probs.argmax()
        b7  = b7_probs.argmax()

        vgg_pred.append(v)
        vgg_dropout_pred.append(vd)
        resnet_pred.append(r)
        inceptionv3_pred.append(inc)
        efficientnetb0_pred.append(b0)
        efficientnetb7_pred.append(b7)

        # Hard voting: majority vote over each model's predicted class
        ensembled_pred_hard.append(mode([v, r, inc, b0, b7]))

        # Soft voting: average the 5 models' probability vectors, then argmax
        ensemble_probs_all.append(
            np.mean([vgg_probs[0], resnet_probs[0], inc_probs[0], b0_probs[0], b7_probs[0]], axis=0)
        )

ensembled_pred_hard_flat = [c[0] for c in ensembled_pred_hard]
ensembled_pred_soft_flat = list(np.argmax(np.array(ensemble_probs_all), axis=1))
print(f'Test samples evaluated: {len(true_value)}')

# ─── Cell 20: Classification Reports ───────────────────────────────────────────
report_data = [
    ('VGG-16',                 vgg_pred),
    ('VGG-16-Dropout',         vgg_dropout_pred),
    ('ResNet-50',              resnet_pred),
    ('InceptionV3',            inceptionv3_pred),
    ('EfficientNet-B0',        efficientnetb0_pred),
    ('EfficientNet-B7',        efficientnetb7_pred),
    ('Ensemble (Hard-Vote)',   ensembled_pred_hard_flat),
    ('Ensemble (Soft-Vote)',   ensembled_pred_soft_flat),
]

for model_name, preds in report_data:
    print('=' * 60)
    print(f'{model_name} — Test Set Classification Report')
    print('=' * 60)
    clf_report(true_value, preds, class_names)

# ─── Cell 21: Prepare Full Dataset for Cross-Validation ────────────────────────
# Uses the augmented training set (original + 10x Albumentations variants)
# for the train portion, and the raw, unaugmented valid/test folders — same
# split composition the final models trained on above.
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

# ─── Cell 22: Load Images Helper ───────────────────────────────────────────────
def load_images(paths, target_size=(224, 224)):
    images = []
    for path in paths:
        img = cv2.resize(cv2.imread(str(path)), target_size)
        images.append(img / 255.0)
    return np.array(images)


# ── Per-fold model builders (all with run_eagerly for RTX 5090) ───────────────
def build_fold_vgg16():
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers: layer.trainable = False
    x = Flatten()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(2,   activation='sigmoid')(x)
    m = Model(base.input, x)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'], run_eagerly=True)
    return m

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

def build_fold_inceptionv3():
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers: layer.trainable = False
    x = Flatten()(base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2,   activation='sigmoid')(x)
    m = Model(base.input, x)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.0001),
              metrics=['accuracy'], run_eagerly=True)
    return m

def build_fold_efficientnetb0():
    # FIX: load_images() (used for all CV folds) divides by 255, same as the
    # other models' generators. EfficientNet already rescales internally, so
    # feeding it [0,1] input divides by 255 twice, crushing the signal and
    # producing the stuck-at-50%-accuracy / loss=ln(2) failure seen in CV.
    # Undo the outer scaling with Rescaling(255.0) before the base model.
    inputs   = tf.keras.Input(shape=(224, 224, 3))
    rescaled = tf.keras.layers.Rescaling(255.0)(inputs)
    base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=rescaled)
    for layer in base.layers: layer.trainable = False
    x = Flatten()(base.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2,   activation='sigmoid')(x)
    m = Model(inputs, x)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'], run_eagerly=True)
    return m

def build_fold_efficientnetb7():
    # FIX: same double-rescale bug as build_fold_efficientnetb0().
    inputs   = tf.keras.Input(shape=(224, 224, 3))
    rescaled = tf.keras.layers.Rescaling(255.0)(inputs)
    base = EfficientNetB7(weights='imagenet', include_top=False, input_tensor=rescaled)
    for layer in base.layers: layer.trainable = False
    x = Flatten()(base.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2,   activation='sigmoid')(x)
    m = Model(inputs, x)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'], run_eagerly=True)
    return m

def build_fold_vgg16_dropout():
    # Dropout ablation of build_fold_vgg16(), for direct comparison. NOT part
    # of the 5-model ensemble — reported alongside it as a standalone row.
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers: layer.trainable = False
    x = Flatten()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2,   activation='sigmoid')(x)
    m = Model(base.input, x)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'], run_eagerly=True)
    return m

FOLD_BUILDERS = {
    'VGG-16':           build_fold_vgg16,
    'ResNet-50':        build_fold_resnet50,
    'InceptionV3':      build_fold_inceptionv3,
    'EfficientNet-B0':  build_fold_efficientnetb0,
    'EfficientNet-B7':  build_fold_efficientnetb7,
}

# Trained and reported separately from FOLD_BUILDERS — not included in the
# ensemble average, since the ensemble is defined over the 5 base
# architectures (one entry per architecture), matching the abstract.
EXTRA_FOLD_BUILDERS = {
    'VGG-16-Dropout':   build_fold_vgg16_dropout,
}

print('Cross-validation helpers defined.')

# ─── Cell 23: Stratified 5-Fold Cross-Validation (All Models + Ensemble) ───────
import json

N_SPLITS  = 5
CV_EPOCHS = 30

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Checkpoint file so a crash/OOM mid-run doesn't lose already-completed folds.
# random_state=42 makes skf.split() deterministic across runs on the same
# data, so re-running this script reproduces the exact same 5 folds and can
# safely skip any fold already recorded here. Since every model + both
# ensembles are trained together within a single fold iteration below, the
# checkpoint granularity is per-fold (covering all models at once), not
# per-model.
CV_CHECKPOINT_PATH = MODEL_DIR / 'cv_checkpoint_all_models.json'

if CV_CHECKPOINT_PATH.exists():
    with open(CV_CHECKPOINT_PATH) as f:
        all_cv_results = json.load(f)
    completed_folds = set(next(iter(all_cv_results.values()))['fold'])
    print(f'Resuming from checkpoint — already completed folds: {sorted(completed_folds)}')
else:
    # Results store: one dict per model, per extra ablation model, and per ensemble variant
    all_cv_results = {
        name: {'fold': [], 'accuracy': [],
               'precision_autistic': [], 'recall_autistic': [], 'f1_autistic': []}
        for name in (list(FOLD_BUILDERS.keys()) + list(EXTRA_FOLD_BUILDERS.keys())
                     + ['Ensemble (Hard-Vote)', 'Ensemble (Soft-Vote)'])
    }
    completed_folds = set()

for fold, (train_idx, val_idx) in enumerate(skf.split(all_image_paths, all_labels)):
    fold_num = fold + 1
    if fold_num in completed_folds:
        print(f'\nSkipping Fold {fold_num} / {N_SPLITS} — already completed (checkpoint).')
        continue

    print('\n' + '='*50)
    print('='*50)
    print(f' Fold {fold_num} / {N_SPLITS}')
    print('='*50)

    X_train_fold = load_images(all_image_paths[train_idx])
    y_train_fold = all_labels[train_idx]
    X_val_fold   = load_images(all_image_paths[val_idx])
    y_val_fold   = all_labels[val_idx]

    print(f'Train: {len(X_train_fold)} | Val: {len(X_val_fold)}')

    fold_probs = []   # collect softmax outputs for ensemble averaging

    for model_name, builder in FOLD_BUILDERS.items():
        print(f'\n  -- Training {model_name} --')
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)

        # Use smaller batch for B7 to avoid OOM
        bs = 8 if 'B7' in model_name else 32

        fm = builder()
        fm.fit(
            X_train_fold, y_train_fold,
            epochs=CV_EPOCHS, batch_size=bs,
            validation_data=(X_val_fold, y_val_fold),
            verbose=1
        )

        # Per-model softmax probs
        probs = fm.predict(X_val_fold, verbose=0)   # shape (N, 2)
        fold_probs.append(probs)

        y_pred_m = probs.argmax(axis=1)
        report_m = classification_report(
            y_val_fold, y_pred_m, target_names=class_names, output_dict=True
        )
        acc_m = np.mean(y_pred_m == y_val_fold)

        all_cv_results[model_name]['fold'].append(fold + 1)
        all_cv_results[model_name]['accuracy'].append(acc_m)
        all_cv_results[model_name]['precision_autistic'].append(
            report_m['autistic']['precision'])
        all_cv_results[model_name]['recall_autistic'].append(
            report_m['autistic']['recall'])
        all_cv_results[model_name]['f1_autistic'].append(
            report_m['autistic']['f1-score'])

        print(f'  {model_name} Fold {fold+1} Accuracy: {acc_m:.4f}')
        print(classification_report(y_val_fold, y_pred_m, target_names=class_names))

    # ── Extra ablation models (trained + reported, NOT part of the ensemble) ──
    for model_name, builder in EXTRA_FOLD_BUILDERS.items():
        print(f'\n  -- Training {model_name} (ablation, not in ensemble) --')
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(False)

        fm = builder()
        fm.fit(
            X_train_fold, y_train_fold,
            epochs=CV_EPOCHS, batch_size=32,
            validation_data=(X_val_fold, y_val_fold),
            verbose=1
        )

        probs    = fm.predict(X_val_fold, verbose=0)
        y_pred_m = probs.argmax(axis=1)
        report_m = classification_report(
            y_val_fold, y_pred_m, target_names=class_names, output_dict=True
        )
        acc_m = np.mean(y_pred_m == y_val_fold)

        all_cv_results[model_name]['fold'].append(fold + 1)
        all_cv_results[model_name]['accuracy'].append(acc_m)
        all_cv_results[model_name]['precision_autistic'].append(
            report_m['autistic']['precision'])
        all_cv_results[model_name]['recall_autistic'].append(
            report_m['autistic']['recall'])
        all_cv_results[model_name]['f1_autistic'].append(
            report_m['autistic']['f1-score'])

        print(f'  {model_name} Fold {fold+1} Accuracy: {acc_m:.4f}')
        print(classification_report(y_val_fold, y_pred_m, target_names=class_names))

    # ── Ensemble: soft-voting (average softmax probabilities) ────────────────
    ens_probs       = np.mean(fold_probs, axis=0)   # shape (N, 2)
    y_pred_ens_soft = ens_probs.argmax(axis=1)
    report_ens_soft = classification_report(
        y_val_fold, y_pred_ens_soft, target_names=class_names, output_dict=True
    )
    acc_ens_soft = np.mean(y_pred_ens_soft == y_val_fold)

    all_cv_results['Ensemble (Soft-Vote)']['fold'].append(fold + 1)
    all_cv_results['Ensemble (Soft-Vote)']['accuracy'].append(acc_ens_soft)
    all_cv_results['Ensemble (Soft-Vote)']['precision_autistic'].append(
        report_ens_soft['autistic']['precision'])
    all_cv_results['Ensemble (Soft-Vote)']['recall_autistic'].append(
        report_ens_soft['autistic']['recall'])
    all_cv_results['Ensemble (Soft-Vote)']['f1_autistic'].append(
        report_ens_soft['autistic']['f1-score'])

    print(f'\n  *** Ensemble (Soft-Vote) Fold {fold+1} Accuracy: {acc_ens_soft:.4f} ***')

    # ── Ensemble: hard-voting (majority vote over each model's predicted class) ──
    per_model_preds = [p.argmax(axis=1) for p in fold_probs]   # list of (N,) arrays, one per model
    y_pred_ens_hard = np.array([
        mode([per_model_preds[m][i] for m in range(len(per_model_preds))])[0]
        for i in range(len(y_val_fold))
    ])
    report_ens_hard = classification_report(
        y_val_fold, y_pred_ens_hard, target_names=class_names, output_dict=True
    )
    acc_ens_hard = np.mean(y_pred_ens_hard == y_val_fold)

    all_cv_results['Ensemble (Hard-Vote)']['fold'].append(fold + 1)
    all_cv_results['Ensemble (Hard-Vote)']['accuracy'].append(acc_ens_hard)
    all_cv_results['Ensemble (Hard-Vote)']['precision_autistic'].append(
        report_ens_hard['autistic']['precision'])
    all_cv_results['Ensemble (Hard-Vote)']['recall_autistic'].append(
        report_ens_hard['autistic']['recall'])
    all_cv_results['Ensemble (Hard-Vote)']['f1_autistic'].append(
        report_ens_hard['autistic']['f1-score'])

    print(f'  *** Ensemble (Hard-Vote) Fold {fold+1} Accuracy: {acc_ens_hard:.4f} ***')

    # Save checkpoint immediately after each fold — if the next fold OOMs or
    # crashes, this run's progress across all models and both ensembles for
    # every completed fold up to now is not lost.
    with open(CV_CHECKPOINT_PATH, 'w') as f:
        json.dump(all_cv_results, f)
    print(f'Checkpoint saved: {CV_CHECKPOINT_PATH}')


# ─── Cell 24: Print + Save All CV Summaries ────────────────────────────────────
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

for model_name, results in all_cv_results.items():
    df = make_summary_df(results)
    print(f'===== 5-Fold Cross-Validation Summary ({model_name}) =====')
    print(df.to_string(index=False))
    safe_name = model_name.lower().replace('-', '').replace(' ', '_')
    df.to_csv(MODEL_DIR / f'cv_results_{safe_name}.csv', index=False)

# ─── Cell 25: Cross-Validation Bar Chart — All Models + Ensembles ─────────────
model_order  = (list(FOLD_BUILDERS.keys()) + list(EXTRA_FOLD_BUILDERS.keys())
                + ['Ensemble (Hard-Vote)', 'Ensemble (Soft-Vote)'])
model_colors = {
    'VGG-16':               'steelblue',
    'VGG-16-Dropout':       'cornflowerblue',
    'ResNet-50':            'darkorange',
    'InceptionV3':          'green',
    'EfficientNet-B0':      'red',
    'EfficientNet-B7':      'purple',
    'Ensemble (Hard-Vote)': 'dimgray',
    'Ensemble (Soft-Vote)': 'black',
}

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

for ax, model_name in zip(axes, model_order):
    results = all_cv_results[model_name]
    accs    = results['accuracy']
    mean_a  = np.mean(accs)
    color   = model_colors[model_name]

    ax.bar([f'Fold {i}' for i in results['fold']], accs,
           color=color, edgecolor='black', alpha=0.8)
    ax.axhline(y=mean_a, color='red', linestyle='--',
               label=f'Mean = {mean_a:.4f}')
    ax.set_title(f'{model_name} — 5-Fold CV Accuracy')
    ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1); ax.legend(fontsize=8)

# 8 panels for 8 model_order entries — grid is fully used, nothing to hide.
for ax in axes[len(model_order):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'cv_accuracy_all_models.png', dpi=150)
plt.close()
print('\nCV bar charts saved: cv_accuracy_all_models.png')

# ── Ensemble-specific larger standalone chart (both voting strategies) ────────
fig, ens_axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ens_name, color in zip(ens_axes,
                                ['Ensemble (Hard-Vote)', 'Ensemble (Soft-Vote)'],
                                ['dimgray', 'black']):
    ens_results = all_cv_results[ens_name]
    ens_accs    = ens_results['accuracy']
    ax.bar([f'Fold {i}' for i in ens_results['fold']], ens_accs,
           color=color, edgecolor='white', alpha=0.85)
    ax.axhline(y=np.mean(ens_accs), color='red', linestyle='--',
               label=f"Mean = {np.mean(ens_accs):.4f}")
    ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{ens_name} — 5-Fold CV Accuracy')
    ax.set_ylim(0, 1); ax.legend()
plt.tight_layout()
plt.savefig(MODEL_DIR / 'cv_accuracy_ensemble.png', dpi=150)
plt.close()
print('Ensemble CV chart saved: cv_accuracy_ensemble.png')

# ─── Cell 25: Sample Predictions on Test Images ────────────────────────────────
models_list = [
    vgg16_final_model, resnet50_final_model, inceptionv3_final_model,
    efficientnetb0_final_model, efficientnetb7_final_model
]
show_few_images(6, models_list)

# ─── Cell 26: Reload Saved Models (run this if session restarts) ───────────────
# Uncomment and run this cell if you need to reload models after a crash

# from tensorflow.keras.models import load_model
# vgg16_final_model          = load_model(str(MODEL_DIR / 'vgg16_model.h5'))
# resnet50_final_model       = load_model(
#     str(MODEL_DIR / 'resnet50_model.h5'),
#     custom_objects={'preprocess_input': resnet50_preprocess_input}
# )
# inceptionv3_final_model    = load_model(str(MODEL_DIR / 'inceptionv3_model.h5'))
# efficientnetb0_final_model = load_model(str(MODEL_DIR / 'efficientnetb0_model.h5'))
# efficientnetb7_final_model = load_model(str(MODEL_DIR / 'efficientnetb7_model.h5'))
# print('Models reloaded successfully.')
