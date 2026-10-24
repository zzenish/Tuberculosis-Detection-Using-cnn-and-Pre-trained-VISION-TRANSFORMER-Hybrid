"""Shared file handling only. Each numbered script contains its own image operation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def stage_path(name):
    return ROOT / name if name == "FINAL_DATASET" else ROOT / "Preprocessing" / name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = {0: 'Normal', 1: 'Tuberculosis'}


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(path):
    # Paths stay valid when FINAL_PROJECT is moved as one folder.
    return Path(path).absolute().relative_to(ROOT).as_posix()


def setup_logging(path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(path)], force=True)


def read_image(path, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def run_stage(input_folder, output_folder, operation, transform, extension, parameters):
    parser = argparse.ArgumentParser(description=operation)
    parser.add_argument('--workers', type=int, default=4, help='Concurrent image workers; no GPU required')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    incoming = stage_path(input_folder) / 'manifest.csv'
    if not incoming.is_file():
        raise FileNotFoundError(f'Run the preceding step first: missing {incoming}')
    rows = read_csv(incoming)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate-ID input manifest')
    if not (stage_path(input_folder) / 'stage_info.json').is_file():
        raise ValueError('Previous stage is incomplete: its completion record is missing')
    previous = json.loads((stage_path(input_folder) / 'stage_info.json').read_text())
    if previous.get('manifest_sha256') != digest(incoming):
        raise ValueError('Previous manifest changed after its stage completed')
    output = stage_path(output_folder)
    # Refuse stale mixtures; never overwrite source or previous-stage images.
    output.mkdir(exist_ok=False)
    setup_logging(output / 'process.log')
    cv2.setNumThreads(1)
    for split in ['train', 'val', 'test']:
        for label in CLASSES.values():
            (output / split / label).mkdir(parents=True)

    def process(row):
        result = transform(ROOT / row['file_path'])
        if not np.isfinite(result).all():
            raise ValueError(f'Nonfinite output for {row["id"]}')
        destination = output / row['split'] / CLASSES[int(row['label'])] / (row['id'] + extension)
        temporary = destination.with_name(destination.stem + '.tmp' + extension)
        if extension == '.npy':
            with temporary.open('wb') as f:
                np.save(f, result, allow_pickle=False)
        elif not cv2.imwrite(str(temporary), result):
            raise OSError(f'Failed to save {destination}')
        temporary.replace(destination)
        return dict(row, file_path=relative(destination))

    logging.info('Starting %s on %d images', operation, len(rows))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = []
        for i, row in enumerate(pool.map(process, rows), 1):
            completed.append(row)
            if i % 500 == 0 or i == len(rows):
                logging.info('%s: %d/%d', output_folder, i, len(rows))
    write_csv(output / 'manifest.csv', completed)
    write_json(output / 'stage_info.json', dict(
        status='complete', operation=operation, parameters=parameters, count=len(completed),
        script=Path(sys.argv[0]).name, script_sha256=digest(sys.argv[0]),
        source_manifest=relative(incoming), source_manifest_sha256=digest(incoming),
        manifest_sha256=digest(output / 'manifest.csv'),
        counts=dict(Counter(f'{r["split"]}:{CLASSES[int(r["label"])]}' for r in completed)),
        finished_utc=datetime.now(timezone.utc).isoformat()))
    logging.info('Complete: %s. Original and preceding-stage files remain unchanged.', output_folder)
