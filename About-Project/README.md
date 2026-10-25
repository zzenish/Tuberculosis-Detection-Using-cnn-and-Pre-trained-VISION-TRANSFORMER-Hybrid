> Latest status: UI and launchers are implemented; all three final evaluations are complete. See RUN_PROJECT.md at the project root and Training/FINAL_RESULTS.md for current results. Older pending-evaluation statements below are historical.

# FINAL_PROJECT — step-by-step preprocessing

**Preprocessing is verified. Modular hybrid training was authorized and launched on 24 October 2026; see Training/README.md and its live run logs.**

Each numbered script performs one stage and saves its complete output. Later scripts read the previous saved stage, rather than silently repeating all operations from the raw images. This makes the work inspectable and reproducible for your college demonstration.

**Completed and reverified, 24 October 2026:** all8,798images passed every saved-stage check. FINAL_DATASET contains6,508training,1,628validation and662test arrays, each float32(3,224,224). The existing outputs are ready; the commands below explain reproduction, not unfinished work.

[Open the before/after walkthrough](Preprocessing/PREPROCESSING_PREVIEW.png) · [Verification record](VERIFICATION_REPORT.json) · [Actual pixel values](PIXEL_WALKTHROUGH.json)

## Current layout and study documents

- `Datasets/`: original images and source metadata.
- `Preprocessing/`: numbered scripts and full intermediate outputs. These are evidence of each operation, not unnecessary files.
- `FINAL_DATASET/`: canonical model input, containing the existing train/val/test split.
- `Training/`: CNN.py, VIT.py, merge_and_train.py, supporting modules, study guide and active run logs.
- `about_dataset.docx`: detailed dataset and viva guide.

An extra copy at `Preprocessing/FINAL_DATASET` was preserved. It is not used as model input or by the verifier. Do not scan both copies as one dataset. Intermediate manifests and all 8,798 clean-image links were repaired after the move. The root final manifest is unchanged and every stage passed verification again.

The table below lists script basenames and output basenames. Scripts and all intermediate outputs live inside `Preprocessing/`; only the canonical `FINAL_DATASET` stays at the project root. Verification JSON files stay at the root and the visual preview is under `Preprocessing/`.

## The sequence

| Run this script | Saved folder | Actual saved values |
|---|---|---|
| `01_clean_and_split.py` | `CLEANED_DATASET` | Links to selected originals; labels, exclusions and splits |
| `02_grayscale.py` | `GRAYSCALE` | Single-channel uint8 PNGs, original dimensions |
| `03_resize.py` | `RESIZING` | Single-channel uint8 PNGs,224×224 |
| `04_contrast_normalization.py` | `CONTRAST_NORMALIZATION` | CLAHE-enhanced single-channel PNGs |
| `05_channel_replication.py` | `CHANNEL_REPLICATION` | Three-channel uint8 PNGs,224×224×3 |
| `06_intensity_scaling.py` | `INTENSITY_SCALING` | Float32.npy arrays,224×224×3,range[0,1] |
| `07_zscore_standardization.py` | `Z_SCORE_STANDARDIZATION` | ImageNet-standardized float32.npy arrays,HWC |
| `08_prepare_final_dataset.py` | `FINAL_DATASET` | Contiguous float32.npy arrays,3×224×224,CHW |
| `09_verify_preprocessing.py` | Root verification/preview files | Checks every image at every stage; creates a visual walkthrough |

Folder names use underscores so terminal commands are easier to type. `pipeline_utils.py` shares only file handling, manifests and logging; the actual operation is visible in its corresponding numbered script. Cleaning rules are implemented directly in script01. This folder does not import the older project's code.

## Folder arrangement

Every processing stage has the same arrangement:

```text
RESIZING/                       # same layout for the other stages
├── train/
│   ├── Normal/
│   └── Tuberculosis/
├── val/
│   ├── Normal/
│   └── Tuberculosis/
├── test/                       # Shenzhen only
│   ├── Normal/
│   └── Tuberculosis/
├── manifest.csv                # exactly which images, labels and sources
├── stage_info.json             # completion record, settings and manifest hashes
└── process.log                 # progress records
```

`CLEANED_DATASET` has `cleaning.log`, `audit.json` and `excluded.csv` instead of `process.log`. Its images are **relative symbolic links**, not additional copies of the original images. The next stages contain real independently saved files. Keep the current structure with `Datasets` at the project root and cleaned images under `Preprocessing/CLEANED_DATASET` so its links remain valid. Moving all of FINAL_PROJECT together preserves these relative links and manifest paths.

## How to run manually

On this laptop the already-installed environment can run preprocessing. From a terminal inside FINAL_PROJECT:

```bash
../.venv/bin/python Preprocessing/01_clean_and_split.py
../.venv/bin/python Preprocessing/02_grayscale.py
../.venv/bin/python Preprocessing/03_resize.py
../.venv/bin/python Preprocessing/04_contrast_normalization.py
../.venv/bin/python Preprocessing/05_channel_replication.py
../.venv/bin/python Preprocessing/06_intensity_scaling.py
../.venv/bin/python Preprocessing/07_zscore_standardization.py
../.venv/bin/python Preprocessing/08_prepare_final_dataset.py
../.venv/bin/python Preprocessing/09_verify_preprocessing.py
```

Run in order. Once a stage has completed, its results are already saved: **do not blindly rerun these commands into existing folders**. Scripts01–08 refuse an existing output directory to prevent accidental mixing/overwriting. To recreate from scratch, use a separate project copy with source data and scripts but no generated stage folders. If a stage was interrupted, preserve/rename that incomplete output folder before restarting that stage; do not delete your source data. Script09 is read-only with respect to the datasets and can regenerate its verification/preview reports.

For a new computer, create an environment inside FINAL_PROJECT and install all project dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Then use `.venv/bin/python` instead of `../.venv/bin/python`. The single requirements.txt includes preprocessing, PyTorch/timm training and Streamlit UI dependencies. Preprocessing itself does not need a GPU. Image operations use four CPU worker threads by default; scripts02–08 accept `--workers 2` if fewer are desired.

All full intermediate stages are saved, not just a few demonstration samples. This intentionally uses additional disk space. Each float-array stage alone occupies about5.3GB; there are three such stages. PNG stages use additional space, especially the original-resolution grayscale stage.

## What to explain in your presentation

### 1. Cleaning comes before image enhancement

Class0 is Normal; class1 is **dataset-labeled Tuberculosis**. The available labels do not consistently establish confirmed active disease.

- TBX: retain3,800 labeled healthy and799 TB images. Exclude3,800 sick/non-TB and3,303 unlabeled images. Do not invent labels to meet the earlier requested5,000/1,200 counts.
- Kaggle: remove800 Montgomery/Shenzhen-derived overlaps by filename joins to source metadata; then remove one additional exact decoded-pixel duplicate. Final count3,399.
- Montgomery: retain138 originals. Group known repeated patients before splitting.
- Shenzhen: retain662 originals separately for testing.
- Pool cleaned TBX,Kaggle,Montgomery and make the same seeded grouped80/20 split. Known patient groups and exact pixel hashes do not cross splits.

Expected counts with the supplied unchanged datasets:

| Split | Normal | TB | Total |
|---|---:|---:|---:|
| train |5,579|929|6,508|
| val |1,395|233|1,628|
| test |326|336|662|
| total |7,300|1,498|8,798|

Exclusions are recorded rather than deleting source images. Provenance exclusion, exact-image deduplication and patient grouping are different protections. Near-duplicate detection is not exhaustive; undocumented patient identities remain a limitation.

Shenzhen was already evaluated in the earlier project. This reorganization does not turn it into a newly unseen test set. No model is trained or evaluated by these preprocessing scripts.

### 2. Grayscale and resizing are separate

Step02 changes channels only, preserving the original resolution. Step03 changes the size to224×224 using `cv2.INTER_AREA`. The square resize may change aspect ratio; there is no automatic lung crop or segmentation.

### 3. Contrast normalization means CLAHE here

Step04 applies `cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))`. The8×8 setting describes a grid of64regions, not eight-pixel tiles. CLAHE adjusts local contrast; it does not divide values by255. Correct implementation alone does not prove improved model performance.

### 4. Three channels do not create new color information

Step05 copies the same grayscale values into three channels for pretrained-model compatibility. All three channels are identical at this point. The image looks the same as the single-channel image; the difference is its stored shape.

### 5. Scaling and standardization are different operations

Step06 computes `float32(pixel)/255.0`, so values lie in[0,1]. Step07 computes `(scaled - mean)/std` using mean[0.485,0.456,0.406] and std[0.229,0.224,0.225]. These are fixed ImageNet statistics, not each image's own measured mean/std. Standardized values can be negative or larger than1; the channels differ numerically because their statistics differ.

These float values are saved as.npy, not PNG. Converting standardized data directly to ordinary PNG would clip or quantize the actual values. For presentation, PREPROCESSING_PREVIEW.png reverses normalization for viewing; it does not replace or modify the stored arrays.

### 6. FINAL_DATASET changes layout only

Step08 transposes `(height,width,channels)` to `(channels,height,width)` and ensures contiguous storage. It does not apply CLAHE or normalization again. Each file is float32(3,224,224). Later training can convert an array to a tensor and group16images into shape(16,3,224,224), but no training is included now.

### 7. Verification is part of the work

Step09 checks every saved stage against the exact operation applied to the preceding stage. It also independently recomputes the full transform from the raw image and compares it with FINAL_DATASET. It checks labels/IDs stay aligned, original decoded pixels match their cleaning hashes, grouped splits remain disjoint, and Shenzhen remains test-only.

After it passes, inspect:

- `VERIFICATION_REPORT.json`: machine-readable pass/counts/checks.
- `Preprocessing/PREPROCESSING_PREVIEW.png`: one training X-ray shown through all operations.
- `PIXEL_WALKTHROUGH.json`: actual numeric values for the same224×224pixel through the float stages.
- Each stage's `manifest.csv` and `stage_info.json`: complete saved-data lineage.
- `WORK_LOG.md`: actual execution history.

## Inspect a final array without PyTorch

From FINAL_PROJECT:

```python
import csv
from pathlib import Path
import numpy as np

root = Path.cwd()
with (root / 'FINAL_DATASET/manifest.csv').open() as file:
    row = next(csv.DictReader(file))
array = np.load(root / row['file_path'], allow_pickle=False)
print(row['id'], row['label'], row['split'])
print(array.shape, array.dtype)  # (3,224,224), float32
```

Study and run the individual scripts so you can explain the actual operations. The saved stages and logs are evidence of reproducible processing, not a substitute for understanding the code. The user has now greenlit model training; see Training/README.md.
