"""Shared image loading only. Model definitions and training stay in their own files."""
import csv
import os
from pathlib import Path
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# TensorFlow 2.21 can miss cuSOLVER's pip location on Linux; preload that library.
# This is an environment fix, not part of the model or preprocessing.
import ctypes
import sysconfig
cuda_solver = Path(sysconfig.get_path('purelib')) / 'nvidia/cusolver/lib/libcusolver.so.11'
if cuda_solver.exists():
    ctypes.CDLL(str(cuda_solver))
import numpy as np
import tensorflow as tf
import keras

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = ['Normal', 'Tuberculosis']


def configure_training():
    keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.mixed_precision.set_global_policy('mixed_float16' if gpus else 'float32')
    print('Device:', gpus or 'CPU (training will be slow)', flush=True)


def read_image(path):
    array = np.load(path.decode(), allow_pickle=False)
    if array.shape != (3, 224, 224) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f'Invalid prepared image: {path}')
    # Files stay unchanged. Convert CHW to HWC and undo ImageNet standardization.
    # Models accept CLAHE-processed pixels in [0,255] and normalize internally.
    pixels = (array.transpose(1, 2, 0) * STD + MEAN) * 255.0
    return np.clip(pixels, 0, 255).astype(np.float32)


def load_data(batch_size=16, accumulation_steps=2):
    with (ROOT / 'FINAL_DATASET/manifest.csv').open() as file:
        rows = list(csv.DictReader(file))
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty manifest or duplicate IDs')
    for row in rows:
        if row['split'] not in ['train', 'val', 'test'] or row['label'] not in ['0', '1']:
            raise ValueError('Invalid label or split')
        if (row['source'] == 'shenzhen') != (row['split'] == 'test'):
            raise ValueError('Shenzhen must be reserved exclusively for testing')
    for field in ['group', 'pixel_sha256']:
        seen = {}
        for row in rows:
            if not row[field]:
                raise ValueError(f'Missing {field}')
            if row[field] in seen and seen[row[field]] != row['split']:
                raise ValueError(f'{field} overlaps between splits')
            seen[row[field]] = row['split']

    datasets, labels_by_split = {}, {}
    for split in ['train', 'val', 'test']:
        selected = [r for r in rows if r['split'] == split]
        paths = [str(ROOT / r['file_path']) for r in selected]
        labels = np.array([int(r['label']) for r in selected], dtype=np.int32)
        if set(labels) != {0, 1}:
            raise ValueError(f'{split} must contain both classes')
        labels_by_split[split] = labels
        data = tf.data.Dataset.from_tensor_slices((paths, labels))
        if split == 'train':
            data = data.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
            # Complete accumulation windows only. Shuffle changes the unused tail each epoch.
            used = len(paths) // (batch_size * accumulation_steps) * batch_size * accumulation_steps
            if used == 0:
                raise ValueError('Not enough training images for one accumulation window')
            data = data.take(used)
            print(f'Train: {len(paths)} images; using {used} per epoch ({len(paths)-used} shuffled tail images omitted)')
        else:
            print(f'{split}: {len(paths)} images, all evaluated')
        def load(path, label):
            image = tf.numpy_function(read_image, [path], tf.float32)
            image.set_shape((224, 224, 3))
            return image, label
        data = data.map(load, num_parallel_calls=4, deterministic=True)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 4
        datasets[split] = data.batch(batch_size).with_options(options).prefetch(1)
    return datasets['train'], datasets['val'], datasets['test'], labels_by_split


def augment_images(images):
    # Keras disables these random layers automatically during validation/prediction.
    augmentation = keras.Sequential([
        keras.layers.RandomRotation(10 / 360, fill_mode='constant', dtype='float32'),
        keras.layers.RandomFlip('horizontal', dtype='float32'),
        keras.layers.RandomZoom((-0.1, 0.1), fill_mode='constant', dtype='float32'),
        keras.layers.RandomBrightness(0.1, value_range=(0, 255), dtype='float32'),
        keras.layers.RandomContrast(0.1, value_range=(0, 255), dtype='float32'),
    ], name='augmentation')
    return augmentation(images)
