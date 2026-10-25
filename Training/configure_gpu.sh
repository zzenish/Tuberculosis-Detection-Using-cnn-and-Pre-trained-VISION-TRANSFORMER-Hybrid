#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# TensorFlow official Linux setup: expose pip-installed CUDA libraries.
.venv/bin/python - <<'PY'
from pathlib import Path
import importlib.util,os
folder=Path(importlib.util.find_spec('tensorflow').origin).parent
count=0
for library in (folder.parent/'nvidia').glob('*/lib/*.so*'):
 link=folder/library.name
 if not link.exists():
  link.symlink_to(os.path.relpath(library,folder));count+=1
print('TensorFlow GPU library links created:',count)
PY
.venv/bin/python -c "from Training.dataset import tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
