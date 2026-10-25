#!/bin/bash
# Run all stages in order using the project's own environment.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="../.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Create .venv in the project root and install requirements.txt first."
    exit 1
fi
if [ ! -d ../Datasets ]; then
    echo "Missing Datasets in the project root. Copy the original images AND metadata first."
    exit 1
fi

scripts=(01_clean_and_split.py 02_grayscale.py 03_resize.py 04_contrast_normalization.py 05_channel_replication.py 06_intensity_scaling.py 07_zscore_standardization.py 08_prepare_final_dataset.py)
outputs=(CLEANED_DATASET GRAYSCALE RESIZING CONTRAST_NORMALIZATION CHANNEL_REPLICATION INTENSITY_SCALING Z_SCORE_STANDARDIZATION ../FINAL_DATASET)
for i in "${!scripts[@]}"; do
    script="${scripts[$i]}"
    output="${outputs[$i]}"
    if [ -e "$output" ]; then
        # Only skip a completed stage with an intact manifest and existing files.
        "$PYTHON" - "$output" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
folder = Path(sys.argv[1])
try:
    manifest = folder / 'manifest.csv'
    info = json.loads((folder / 'stage_info.json').read_text())
    assert info['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest(), 'manifest changed'
    with manifest.open() as file:
        rows = list(csv.DictReader(file))
    assert rows and len(rows) == info['count'], 'incorrect image count'
    assert all((Path('..') / row['file_path']).is_file() for row in rows), 'missing image or broken link'
except (OSError, ValueError, KeyError, AssertionError) as exc:
    raise SystemExit(f'{folder}: incomplete or invalid stage ({exc}). Preserve/move this output and any later stages before rerunning. Nothing has been deleted.')
print(f'Keeping completed stage: {folder}')
PY
    else
        echo "Starting $script"
        "$PYTHON" -u "$script"
        echo "Finished $script"
    fi
done

echo "Verifying every saved stage and generating the preview…"
"$PYTHON" -u 09_verify_preprocessing.py
echo "Preprocessing complete. Model inputs are in FINAL_DATASET."
