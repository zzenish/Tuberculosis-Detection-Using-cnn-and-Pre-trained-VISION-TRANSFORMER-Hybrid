# Run the final project

Run these commands from FINAL_PROJECT. The scripts also work when called from another directory.

## Open the demonstration

```bash
./run_detection.sh
```

Open the local URL printed in the terminal, normally http://localhost:8501. If that port is occupied, use `./run_detection.sh --server.port 8502`. Stop the server with Ctrl+C.

The app includes image assessment with model selection, CNN Grad-CAM, downloadable assessment records, validation and Shenzhen results, training curves and methodology. Uploads remain in memory; there is no patient database. Select an 8-bit PNG/JPEG chest X-ray. It does not detect whether an uploaded image is anatomically appropriate.

The launcher exports the selected trained checkpoints to UI/models once, then reuses matching exports. These deployment bundles contain inference weights without optimizer history. This is the only intentional extra model copy; it keeps inference independent of datasets and recovery checkpoints. No old parent-project trained model is used. Inference runs on CPU and requires no weight download.

## Preprocessing

```bash
./preprocess.sh
```

Runs 01 through 08 sequentially for missing stages, then runs 09 to verify all saved stages. Existing complete stages with matching manifest fingerprints are skipped. Incomplete/mismatched output directories stop the pipeline; nothing is deleted or overwritten. On the current completed project this primarily performs verification. To recreate from scratch, use a separate project copy with scripts and Datasets but without generated outputs. Do not remove source data.

## Training

```bash
./train.sh
./train.sh --model cnn
./train.sh --model vit
./train.sh --model hybrid
```

The all-model default runs CNN, ViT and hybrid sequentially. Existing completed runs are skipped. It will not retrain your completed hybrid. An unfinished run stops with instructions; after its process has stopped, use `./train.sh --model cnn --resume` to resume from last.pt. Code, dataset and settings must match the original run. No forced overwrite option is provided. Explicit outputs are selected per model, avoiding the raw Python command's hybrid default output.

## Final evaluation

```bash
./evaluate.sh
```

Evaluates the best checkpoint of each model on validation and Shenzhen. Saves per-image predictions, per-source metrics and the generalization gap under Training/runs/<model>/final_evaluation. Repeated calls skip matching completed evaluations. It does not change weights or tune against test results.

## Environment and moving the folder

Scripts first use FINAL_PROJECT/.venv, then the existing parent .venv, or an explicit PROJECT_PYTHON executable. To make this folder independent after moving, create a new environment here (do not move an old virtual environment):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

requirements.txt is the single installation file for preprocessing, training, evaluation and the UI. Use Python 3.11. Its package versions match the tested environment, omitting machine-specific CUDA suffixes. Pip installs their transitive dependencies automatically. The environment freeze in logs/requirements-tested.txt is a historical record, not another installation file.

The UI runs on CPU. GPU training additionally requires a compatible NVIDIA driver and CUDA-enabled PyTorch build; requirements.txt cannot install system drivers. See https://pytorch.org/get-started/locally/ if CUDA is unavailable. A clean installation on a different operating system has not been tested.

On Windows, install with the same requirements.txt, then launch the UI from FINAL_PROJECT using `python -m streamlit run app.py`. Keep UI/models in the copied project. The shell launchers and their fcntl-based pipeline locking target Linux/WSL; use WSL for those scripts.

Datasets, relative manifests, and stage links stay valid when the whole FINAL_PROJECT folder moves together. Launchers set the Hugging Face cache explicitly: existing parent cache when present, otherwise FINAL_PROJECT/.cache/huggingface. New pretrained training on a fresh computer may need a download. Saved UI models do not.

Keep Training source files unchanged if you need to resume old experiments, because checkpoints record their fingerprints. All new UI and launcher code is outside Training. Logs and submission guides are editable.

## Folder roles

- Datasets: original source collections.
- Preprocessing: individual scripts and saved intermediate operations.
- FINAL_DATASET: canonical prepared arrays and split manifest.
- Training: CNN.py, VIT.py, merge_and_train.py and model-specific experiment outputs.
- UI: inference helpers and exported deployment models.
- app.py: Streamlit interface.
- scripts: shell environment selection and safe pipeline orchestration.
- logs: environment records, verification and launcher lock.

A .lock file is a coordination file, not a model. __pycache__ is Python's regenerated import cache. Neither belongs in a results presentation.
