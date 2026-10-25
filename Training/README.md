# Training

Run all three models from this folder:

```bash
bash train.sh
```

The script uses the project's `.venv`. Install the root `requirements.txt` there first if needed. It trains CNN, then ViT, then Hybrid. Each model trains with validation after every epoch, restores its best validation checkpoint, and finally evaluates the separate Shenzhen test set. Training has not been started by this cleanup.

## Files you need

- `CNN.py`: CNN model, training, validation, testing and saving.
- `ViT.py`: pretrained ViT model, training, validation, testing and saving.
- `Hybrid.py`: joins CNN and ViT features, trains and evaluates the hybrid.
- `dataset.py`: loads FINAL_DATASET, checks split separation, sets up the GPU and applies training-only augmentation. All three scripts need this file.
- `train.sh`: runs the three scripts in order and saves their console logs.
- `configure_gpu.sh`: optional GPU library repair; run `bash configure_gpu.sh` if TensorFlow cannot find the GPU after installation.

Hybrid imports the architecture functions from CNN.py and ViT.py; this does not run the other models or load their trained checkpoints. Keep these files together.

## Outputs (created when you train)

Each model gets one folder: `CNN/`, `ViT/`, `Hybrid/`. Each contains:

- `best_model.keras`: the saved model selected by validation loss.
- `console.log`: terminal messages saved by train.sh.
- `training_history.csv`: loss and accuracy for each epoch.
- `evaluation.txt`: readable final validation/test results.
- `results.json`: the same evaluation metrics in a machine-readable format.
- `settings.json`: settings used for the run.

Completed models are skipped when running train.sh again. For a fresh run, move the old model output folder somewhere safe first. Incomplete runs also need to be moved before restarting; automatic resume is not implemented.

## Settings

Edit the settings near the start of main() in the model script: up to 30 epochs, batch size 16, gradient accumulation 2 (effective batch 32). Mixed precision is enabled on GPU. Early stopping may finish before 30 epochs. The ViT backbone is frozen to fit this laptop's GPU.

These scripts use the original unweighted recipe. Earlier experiments and checkpoints are archived outside Training, under About-Project/Code-Archive. Repeating training does not guarantee the same test accuracy.

The dataset split and preprocessing are unchanged. Random augmentation is enabled only during training. Validation and test are unaugmented. Python may recreate __pycache__; this is an automatic import cache, not training output.
