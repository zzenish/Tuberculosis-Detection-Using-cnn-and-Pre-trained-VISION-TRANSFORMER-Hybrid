"""Audit source labels/provenance, remove duplicates, and create grouped splits."""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from openpyxl import load_workbook
from sklearn.model_selection import StratifiedGroupKFold

from pipeline_utils import ROOT, CLASSES, read_csv, write_csv, write_json, setup_logging, relative, digest
import os
from datetime import datetime, timezone


def tbx_label(annotation):
    tags = {t['name'] for t in annotation.get('tags', [])}
    if tags == {'healthy'}:
        return 0, 'healthy'
    if tags and tags <= {'active_tb', 'latent_tb', 'active&latent_tb'}:
        return 1, '+'.join(sorted(tags))
    if tags == {'sick_but_non-tb'}:
        return None, 'sick_but_non_tb'
    if not tags:
        return None, 'unlabeled'
    raise ValueError(f'Unexpected/conflicting TBX tags: {tags}')


def source_is_nlm(url):
    # This exact article is the supplied metadata's Montgomery/Shenzhen provenance.
    # Do not exclude all NIH/NIAID sources (e.g. Belarus).
    return 'pmc4256233' in url.lower() or 'tuberculosis-chest-x-ray-datasets' in url.lower()


def metadata_rows(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(wb.active.values)
        headers = [str(x).strip().upper() for x in next(rows)]
        if not {'FILE NAME', 'URL'} <= set(headers):
            raise ValueError(f'Missing metadata columns: {path}')
        return [dict(zip(headers, r)) for r in rows if any(x is not None for x in r)]
    finally:
        wb.close()


def patient_groups(rows, source):
    """Union all explicit same-patient references, including transitive references."""
    ids = {Path(r['study_id']).stem for r in rows}
    parent = {x: x for x in ids}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in rows:
        own = Path(row['study_id']).stem
        for ref in re.findall(r'(?:MCUCXR|CHNCXR)_\d+_[01]', row['findings']):
            if ref not in ids:
                raise ValueError(f'Unresolved patient reference {ref}')
            a, b = root(own), root(ref)
            parent[max(a, b)] = min(a, b)
    return {x: f'{source}:{root(x)}' for x in ids}


def inventory(raw):
    included, excluded = [], []
    def add(path, source, label, subtype, group=None, provenance=''):
        item = dict(id=f'{source}__{path.stem}', source=source, label=label,
                    subtype=subtype, group=group or f'{source}:{path.stem}',
                    original_path=str(path.resolve()), provenance=provenance)
        included.append(item)
    for ann in sorted((raw / 'tbx11k-DatasetNinja').glob('*/ann/*.json')):
        path = ann.parent.parent / 'img' / ann.name.removesuffix('.json')
        if not path.is_file():
            raise FileNotFoundError(path)
        label, subtype = tbx_label(json.loads(ann.read_text()))
        if label is None:
            excluded.append(dict(path=str(path.resolve()), reason=subtype, duplicate_of=''))
        else:
            add(path, 'tbx11k', label, subtype)
    kaggle = raw / 'Tawsifur Rahman' / 'TB_Chest_Radiography_Database'
    for label, name in enumerate(['Normal', 'Tuberculosis']):
        seen = set()
        for row in metadata_rows(kaggle / f'{name}.metadata.xlsx'):
            filename = str(row['FILE NAME']).strip()
            if not filename.lower().endswith('.png'):
                filename += '.png'
            if Path(filename).name != filename or filename in seen:
                raise ValueError(f'Invalid/duplicate metadata filename: {filename}')
            seen.add(filename)
            path = kaggle / name / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            url = str(row['URL'] or '')
            if not url:
                raise ValueError(f'Missing provenance: {filename}')
            if source_is_nlm(url):
                excluded.append(dict(path=str(path.resolve()), reason='nlm_provenance_overlap', duplicate_of=url))
            else:
                add(path, 'kaggle', label, name, provenance=url)
        actual = {p.name for p in (kaggle / name).glob('*.png')}
        if seen != actual:
            raise ValueError(f'Metadata/image mismatch in {name}: {seen ^ actual}')
    for source in ['montgomery', 'shenzhen']:
        rows = read_csv(raw / source / f'{source}_metadata.csv')
        groups = patient_groups(rows, source)
        seen = set()
        for row in rows:
            path = raw / source / 'images' / 'images' / row['study_id']
            if not path.is_file() or path.name in seen:
                raise ValueError(f'Missing or duplicate study: {path}')
            seen.add(path.name)
            match = re.fullmatch(r'(MCUCXR|CHNCXR)_\d+_([01])', path.stem)
            if not match:
                raise ValueError(f'Unknown label convention: {path}')
            add(path, source, int(match[2]), 'source_binary_label', groups[path.stem], row['findings'])
        if seen != {p.name for p in path.parent.glob('*.png')}:
            raise ValueError(f'Metadata/image mismatch: {source}')
    if not included or len({r['id'] for r in included}) != len(included):
        raise ValueError('Empty inventory or duplicate identifiers')
    return included, excluded


def pixel_hash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Unreadable image: {path}')
    return hashlib.sha256(str(img.shape).encode() + img.tobytes()).hexdigest()


def deduplicate(rows, exclusions):
    # Keep test originals first; keep Montgomery originals before derivative sources.
    priority = {'shenzhen': 0, 'montgomery': 1, 'tbx11k': 2, 'kaggle': 3}
    hashes, clean = {}, []
    for i, row in enumerate(sorted(rows, key=lambda r: (priority[r['source']], r['id']))):
        h = pixel_hash(row['original_path'])
        row['pixel_sha256'] = h
        if h in hashes:
            previous = hashes[h]
            if previous['label'] != row['label']:
                raise ValueError(f'Conflicting labels for identical pixels: {row["id"]}, {previous["id"]}')
            if row['source'] in {'montgomery', 'shenzhen'}:
                raise ValueError('Duplicate original in protected cohort; review instead of silently dropping it')
            exclusions.append(dict(path=row['original_path'], reason='identical_pixels', duplicate_of=previous['original_path']))
        else:
            hashes[h] = row
            clean.append(row)
        if i % 500 == 0:
            logging.info('Pixel audit %d/%d', i, len(rows))
    return clean


def split_rows(rows, seed):
    pool = [r for r in rows if r['source'] != 'shenzhen']
    strata = [f'{r["source"]}:{r["label"]}' for r in pool]
    groups = [r['group'] for r in pool]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    # A single seeded fold; never optimize splits using model performance.
    train, val = next(splitter.split(np.zeros(len(pool)), strata, groups))
    for idx in train:
        pool[idx]['split'] = 'train'
    for idx in val:
        pool[idx]['split'] = 'val'
    for r in rows:
        if r['source'] == 'shenzhen':
            r['split'] = 'test'
    for key in ['group', 'pixel_sha256']:
        sets = [{r[key] for r in rows if r['split'] == s} for s in ['train', 'val', 'test']]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    return sorted(rows, key=lambda r: (r['split'], r['id']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=ROOT / 'Datasets')
    p.add_argument('--output', type=Path, default=ROOT / 'Preprocessing' / 'CLEANED_DATASET')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--copy', action='store_true', help='Copy images instead of symlinking read-only inputs')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    setup_logging(args.output / 'cleaning.log')
    rows, excluded = inventory(args.raw)
    before = Counter(f'{r["source"]}:{r["label"]}' for r in rows)
    rows = split_rows(deduplicate(rows, excluded), args.seed)
    for row in rows:
        destination = args.output / row['split'] / CLASSES[int(row['label'])] / (row['id'] + '.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.copy:
            shutil.copy2(row['original_path'], destination)
        else:
            destination.symlink_to(os.path.relpath(row['original_path'], destination.parent))
        row['original_path'] = relative(row['original_path'])
        row['file_path'] = relative(destination)
    write_csv(args.output / 'manifest.csv', rows)
    for item in excluded:
        item['path'] = relative(item['path'])
        if item['reason'] == 'identical_pixels':
            item['duplicate_of'] = relative(item['duplicate_of'])
    write_csv(args.output / 'excluded.csv', excluded, ['path', 'reason', 'duplicate_of'])
    summary = dict(target='Normal vs dataset-labeled TB (not confirmed active disease)', seed=args.seed,
                   before_pixel_dedup=dict(before), counts=dict(Counter(f'{r["split"]}:{r["source"]}:{r["label"]}' for r in rows)),
                   exclusions=dict(Counter(r['reason'] for r in excluded)),
                   limitations=['Unknown TBX test labels excluded.', 'Exact pixel hashing does not detect all resized/cropped duplicates.',
                                'Only explicit patient references can be grouped; other patient identities unavailable.',
                                '80/20 is approximate to preserve patient groups.'])
    write_json(args.output / 'audit.json', summary)
    write_json(args.output / 'stage_info.json', dict(status='complete', operation='Metadata cleaning, duplicate checks and grouped split', count=len(rows), manifest_sha256=digest(args.output / 'manifest.csv'), finished_utc=datetime.now(timezone.utc).isoformat()))
    logging.info('Complete: %s', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
