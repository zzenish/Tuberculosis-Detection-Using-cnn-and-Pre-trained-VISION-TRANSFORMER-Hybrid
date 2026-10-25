#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="../.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Create .venv with Python 3.11 and install the root requirements.txt first."
    exit 1
fi

for model in CNN ViT Hybrid; do
    output="$model"
    if [ -f "$output/results.json" ] && [ -f "$output/best_model.keras" ]; then
        echo "$model already completed in $output; keeping those results."
        continue
    fi
    if [ -e "$output" ]; then
        echo "$output contains an incomplete run. Check its log and move it before restarting."
        exit 1
    fi
    echo "Starting $model training using the original unweighted recipe"
    mkdir -p "$output"
    "$PYTHON" -u "$model.py" 2>&1 | tee "$output/console.log"
done

echo "Finished. Models, logs and results are in CNN/, ViT/ and Hybrid/."
