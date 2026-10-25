#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Create .venv and install requirements.txt first."
    exit 1
fi
if [ ! -f Training/Hybrid/best_model.keras ]; then
    echo "Missing Training/Hybrid/best_model.keras. Complete Hybrid training first."
    exit 1
fi
# app.py contains the complete UI and prediction code.
echo "Opening the project at http://localhost:8501 — press Ctrl+C to stop."
exec .venv/bin/python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.maxUploadSize=20 --browser.gatherUsageStats=false "$@"
