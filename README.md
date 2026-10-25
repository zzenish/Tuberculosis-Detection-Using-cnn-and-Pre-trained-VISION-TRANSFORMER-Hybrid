# Tuberculosis Detection from Chest X-rays

### A hybrid approach using a CNN and a pretrained Vision Transformer

This is our final-year Computer Engineering project. We built it to compare three approaches to chest X-ray classification: a custom CNN, a pretrained Vision Transformer (ViT), and a hybrid model that combines features from both.

The question behind the project is simple: does combining the two models help when the X-rays come from a different dataset than the one used for training?

In our current experiment, the hybrid gave the best overall performance on the separate Shenzhen test set. There is still a noticeable gap between validation and test performance, so we treat this as a research project rather than a finished diagnostic tool.

> This project is for academic demonstration. Its predictions are not medical diagnoses. The classes are **Normal** and **dataset-labelled Tuberculosis**; the source labels do not consistently establish confirmed active TB.

## What is included

- A preprocessing pipeline with each operation saved separately, so the intermediate images and arrays can be inspected.
- Three TensorFlow/Keras training scripts: `CNN.py`, `ViT.py` and `Hybrid.py`.
- Validation during training and a separate cross-dataset test evaluation.
- A Streamlit interface for selecting a model, uploading an X-ray and reviewing its prediction.
- CNN Grad-CAM visualisations for the CNN and hybrid, plus evaluation tables and training curves.

## How the models work

The CNN has four convolutional blocks with 32, 64, 128 and 256 filters. Global average pooling produces a 256-value feature vector.

The ViT uses the KerasHub `vit_base_patch16_224_imagenet21k` preset. Its classification-token embedding is projected to 256 values. We keep the pretrained backbone frozen to make training manageable on a laptop GPU with 4 GB of VRAM.

The hybrid sends the same image through both branches, concatenates their features into a 512-value vector, and passes it through a classification head. It combines **features**, rather than averaging the two models' final predictions. The hybrid does not load the independently trained CNN and ViT checkpoints.

## Dataset and preprocessing

We use TBX11K, the Kaggle TB Chest Radiography Database, Montgomery and Shenzhen. The cleaning script uses the available annotations and metadata to select the two classes, exclude sick-but-non-TB images, remove known source overlaps and check for identical decoded images.

TBX11K, cleaned Kaggle and Montgomery form the development pool. That pool uses an approximately **80/20 training–validation split**, preserving the patient groups that can be identified from the metadata. Shenzhen is kept separate for testing.

| Split | Normal | Tuberculosis | Total |
|---|---:|---:|---:|
| Training | 5,579 | 929 | 6,508 |
| Validation | 1,395 | 233 | 1,628 |
| Shenzhen test | 326 | 336 | 662 |

This is **not an 80/10/10 split**. These counts describe our cleaned copy of the source datasets; they should not be assumed for a different download.

The preprocessing sequence is:

```text
Clean and split
    → Grayscale
    → Resize to 224 × 224 (OpenCV INTER_AREA)
    → CLAHE (clip limit 2.0, tile grid 8 × 8)
    → Replicate into three channels
    → Scale by 1/255
    → ImageNet standardization
    → Save float32 arrays in (3, 224, 224) format
    → Verify the saved stages
```

Each step has its own script and output folder. Random rotation, horizontal flipping, zoom and small brightness/contrast changes are applied during training only. Validation and test images are not augmented.

[Dataset download links](datasetLinks.md) · [Saved preprocessing preview](Preprocessing/PREPROCESSING_PREVIEW.png)

## Results

These are the results from the current completed runs. Each checkpoint was selected using validation loss.

| Model | Validation accuracy | Shenzhen accuracy | TB recall | Specificity | TB F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 86.55% | 67.37% | 90.77% | 43.25% | 73.85% | 0.8524 |
| ViT | 95.88% | 66.01% | 76.79% | 54.91% | 69.64% | 0.7355 |
| Hybrid | 93.37% | **78.55%** | 86.31% | 70.55% | **80.33%** | **0.8670** |

Recall, specificity, F1 and AUC in this table are measured on Shenzhen. Recall measures how many TB images were detected; specificity measures how many Normal images were correctly identified.

The CNN detected more TB cases, but also produced more false alarms. The hybrid had the strongest overall balance in this run. On the 662 test images, it correctly classified 290 TB images and 230 Normal images, missing 46 TB images and misclassifying 96 Normal images.

Shenzhen results have been reviewed across several experiments. Although Shenzhen is excluded from training, it should not be described as a newly untouched final benchmark. These results also do not establish that the hybrid will outperform the baselines on every dataset or training run.

## Repository layout

```text
.
├── Preprocessing/        # Numbered scripts 01–09 and run.sh
├── Training/            # CNN.py, ViT.py, Hybrid.py, dataset.py and train.sh
├── About-Project/       # Setup guide, project notes and work log
├── .streamlit/          # Interface theme
├── app.py               # Complete UI, inference and Grad-CAM implementation
├── startUI.sh           # Launches the interface
├── requirements.txt     # Dependencies for the current implementation
└── datasetLinks.md      # Source dataset links
```

Raw datasets, generated preprocessing folders, `.venv` and trained model outputs are **not included in Git**. A fresh clone contains the code, not a ready-to-run set of trained models.

## Getting started

Our tested environment is Ubuntu Linux with Python 3.11. Training was developed on an AMD Ryzen 7 6800H laptop with an NVIDIA RTX 3050 Ti, 4 GB VRAM. The interface runs inference on CPU, so a GPU is not required to use saved models.

From the repository root, create a new environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Do not copy an existing `.venv` from another computer. The scripts use the environment in the repository root, so activating it is optional.

### Open the interface

First obtain the completed model outputs or train the models yourself. For each of `CNN`, `ViT` and `Hybrid`, place these files in its corresponding folder:

```text
Training/<model>/best_model.keras
Training/<model>/results.json
Training/<model>/training_history.csv
```

The UI needs the result files as well as the checkpoint. Copying each complete model output folder is the simplest option. No raw datasets are needed for this workflow.

```bash
bash startUI.sh
```

Open **http://localhost:8501**, select a model and upload an 8-bit PNG or JPEG chest X-ray. Stop the server with `Ctrl+C`. If the port is occupied, use:

```bash
bash startUI.sh --server.port=8502
```

Uploaded images are processed in memory and are not saved by the app. Prediction scores are not calibrated probabilities of disease. Grad-CAM shows contributions from the CNN branch; it is not a lesion outline or an explanation of the transformer's attention.

### Recreate the dataset

Place the original images **and metadata** in `Datasets/`. The exact expected folder structure is documented in the [setup guide](About-Project/RUN_PROJECT.md).

```bash
bash Preprocessing/run.sh
```

This runs steps 01–09 in order. It checks existing stage records before skipping completed work and finishes by verifying the saved operations. It stops on incomplete stages instead of overwriting them. For a changed dataset or preprocessing recipe, use a fresh output copy.

Saving every intermediate operation uses substantial disk space. The three float32 array stages alone require roughly 16 GB, in addition to the original images and PNG stages.

### Train the models

Once `FINAL_DATASET/` contains the prepared arrays and `manifest.csv`, run:

```bash
bash Training/train.sh
```

This trains CNN, ViT and Hybrid in order. The current settings use a batch size of 16, gradient accumulation over two batches, mixed precision on GPU, AdamW, a learning-rate scheduler and early stopping. The maximum is 30 epochs. The current recipe does not use class weights.

Each model's checkpoint, logs, history and evaluation are saved under `Training/CNN/`, `Training/ViT/` or `Training/Hybrid/`. Completed runs are skipped. An incomplete run must be moved somewhere safe before restarting; automatic resume is not implemented.

Initial ViT and hybrid training needs internet access to download the pretrained weights. GPU training also needs a compatible NVIDIA driver. If TensorFlow cannot locate the installed CUDA libraries on Linux, the project includes:

```bash
bash Training/configure_gpu.sh
```

The shell launchers target Linux/WSL. Native Windows and macOS installations have not been verified. See the [full setup guide](About-Project/RUN_PROJECT.md) for transfer instructions and platform notes.

## Limitations and what we learned

The largest issue is the drop in performance on images from a different source. A high validation accuracy alone did not mean that a model would generalize well to Shenzhen. Comparing recall, false positives and AUC helped us understand that more clearly than accuracy alone.

The training data is imbalanced. Patient information is incomplete, and exact-pixel duplicate checks cannot catch every resized or cropped copy. The system also cannot check whether an uploaded image is actually a suitable chest X-ray.

These are part of the project findings. Better cross-source validation, more representative data and carefully controlled fine-tuning are areas for further work.

## Further documentation

- [Setup and running the project](About-Project/RUN_PROJECT.md)
- [Training files and outputs](Training/README.md)
- [Project documentation](About-Project/README.md)
- [Work log](About-Project/WORK_LOG.md)

The current code and setup guide describe the active TensorFlow implementation. Older study documents and historical log entries may refer to earlier versions.
