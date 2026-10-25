# Run FINAL_PROJECT

This guide describes the current TensorFlow/Keras implementation. Commands below start in the project root. The project folder may have a different name after cloning. Python 3.11 on Ubuntu/Linux is the tested setup; installation on another computer still needs to be checked there.

## 1. Install the environment

Install Python 3.11 with venv support, then run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Create a new environment on each computer; do not copy .venv. requirements.txt is the only dependency list. Shell launchers use this environment directly, so activation is optional. An NVIDIA driver is a system dependency, not something pip installs. UI inference uses CPU. GPU training requires a compatible NVIDIA GPU and driver. If TensorFlow misses the pip CUDA libraries on Linux, run:

```bash
bash Training/configure_gpu.sh
```

This prints detected GPUs. CPU training is possible but much slower. Initial ViT/Hybrid training needs internet access to download the pretrained KerasHub preset. Saved-model inference does not need that download. Available memory, storage and package availability must be checked on the destination computer.

## 2. Choose your workflow

### Use the UI with existing models

No raw datasets or FINAL_DATASET are needed. Copy the following files from each completed run, keeping these paths:

```text
Training/CNN/best_model.keras
Training/CNN/results.json
Training/CNN/training_history.csv
Training/ViT/best_model.keras
Training/ViT/results.json
Training/ViT/training_history.csv
Training/Hybrid/best_model.keras
Training/Hybrid/results.json
Training/Hybrid/training_history.csv
```

Copying each complete model output folder is simplest. The JSON/CSV files are required by the interface, not optional decorations. These folders are intentionally excluded from Git. Then run:

```bash
bash startUI.sh
```

Open http://localhost:8501. Stop that terminal's server with Ctrl+C. If another app owns that port, stop that server or use `bash startUI.sh --server.port=8502` and open the URL Streamlit prints. Only one server can use a port.

app.py contains the entire UI, image validation, preprocessing, prediction and Grad-CAM. For memory isolation it invokes its own internal prediction mode in a short-lived CPU process; there are no separate UI Python helpers. .streamlit/config.toml contains the original theme and must be committed. The app shows one selected model's prediction, CNN/Hybrid Grad-CAM, a three-model evaluation table, training curves and downloadable assessment records. Images are processed in memory, not saved. Scores are not calibrated disease probabilities.

### Train using already prepared data

Copy FINAL_DATASET, including manifest.csv and every referenced .npy file, into the root. No earlier preprocessing stages are needed for training. Run:

```bash
bash Training/train.sh
```

This trains CNN, ViT and Hybrid in order, using batch 16, accumulation 2, mixed precision on GPU and up to 30 epochs with early stopping. Each script validates after each epoch, selects its best checkpoint by validation loss, and then evaluates the Shenzhen test split. The recipe is unweighted and the pretrained ViT backbone is frozen. Hybrid shares architecture functions with CNN.py and ViT.py; it does not load their trained checkpoints.

Each Training/CNN, Training/ViT or Training/Hybrid folder contains best_model.keras, console.log, training_history.csv, settings.json, evaluation.txt and results.json. Completed runs are skipped by train.sh. Incomplete output folders stop the launcher; move them somewhere safe before restarting. Automatic resume is not implemented. To retrain a completed model, first move its output folder. Never remove source scripts or data.

### Recreate preprocessing from raw datasets

Copy the full Datasets folder, including metadata, keeping the source layout expected by 01_clean_and_split.py:

```text
Datasets/
  tbx11k-DatasetNinja/<source subset>/ann/*.json
  tbx11k-DatasetNinja/<source subset>/img/<images>
  Tawsifur Rahman/TB_Chest_Radiography_Database/
    Normal.metadata.xlsx
    Tuberculosis.metadata.xlsx
    Normal/*.png
    Tuberculosis/*.png
  montgomery/montgomery_metadata.csv
  montgomery/images/images/*.png
  shenzhen/shenzhen_metadata.csv
  shenzhen/images/images/*.png
```

Source links are in datasetLinks.md. Different downloads may use different folder layouts; images alone are insufficient because cleaning depends on metadata and provenance. Do not rename or omit source metadata to force the script to proceed.

```bash
bash Preprocessing/run.sh
```

The launcher changes into its own folder, uses root .venv and runs steps 01–09 in order. Existing stages are skipped only when their completion record, manifest hash, recorded count and referenced files pass checks. Step 09 then verifies the saved image operations and generates verification reports and the preview. Existing stages are not silently regenerated when source data or code changes: for a different dataset or recipe, use a fresh project copy without generated outputs.

If a stage was interrupted, preserve/move that output folder and any downstream generated stages before restarting. Never delete Datasets. The launcher does not delete or overwrite stage folders. Individual numbered scripts remain runnable with `.venv/bin/python Preprocessing/01_clean_and_split.py` (and the corresponding later filename). Scripts 01–08 refuse existing output directories. To recheck existing preprocessing alone:

```bash
.venv/bin/python Preprocessing/09_verify_preprocessing.py
```

All intermediate outputs are saved inside Preprocessing; FINAL_DATASET stays at the root. Steps are cleaning/splitting, grayscale, AREA resize to 224×224, CLAHE (2.0, 8×8), three-channel replication, divide by 255, ImageNet standardization, CHW conversion and verification. Allow substantial disk space: the three float32 stages together use roughly 16 GB, in addition to raw images and PNG stages.

Cleaning uses source metadata, overlap provenance and exact decoded-pixel hashes. It is not guaranteed to identify all resized/cropped duplicates. The current source files yield 6,508 training, 1,628 validation and 662 Shenzhen test images. Development sources use an approximately 80/20 grouped split; Shenzhen is entirely test-only. Class labels are Normal and dataset-labelled TB, not uniformly confirmed active disease. The metadata does not identify every patient's repeated images.

## 3. Portability and limitations

Keep relative folder structure intact when moving the project. Cleaning creates relative symlinks into Datasets by default; moving the complete project on Linux preserves them. For a fresh workflow needing physical copies, run step 01 with `--copy` before the launcher. Do not move only CLEANED_DATASET without its source images.

The shell launchers target Linux/WSL. Native Windows uses `.venv\\Scripts\\python.exe -m streamlit run app.py` after creating its own environment and installing dependencies; the Bash launchers and Linux GPU repair script do not run in ordinary PowerShell. Native Windows and macOS installations have not been verified. Do not assume native Windows TensorFlow GPU training is supported by this setup.

Current RUN_PROJECT.md and Training/README.md are the operational references. Older DOCX study guides and dated WORK_LOG entries may describe earlier implementations; consult current code before citing architectural details. Previous experimental backups are local history, not runtime dependencies.

Shenzhen has been reviewed across experiments, so report it as a repeatedly evaluated external dataset, not a newly untouched benchmark. The UI is an academic demonstration, not a clinical diagnostic system.
