# FINAL_PROJECT — preprocessing-only work log

## 23 October 2026
1. User requested an independent, presentation-friendly preprocessing project with one operation per script and full saved intermediate outputs. Model training is explicitly deferred until the user's greenlight.
2. Inspected FINAL_PROJECT: initially empty; user confirmed completing the dataset copy. FINAL_PROJECT/Datasets now contains approximately8.6GiB of original source data.
3. Preserved the existing approved target (Normal vs dataset-labeled TB), provenance exclusions, decoded-pixel deduplication and grouped split rules. Local cleaning script is self-contained and imports no old project code. Records and clean-image links use project-relative paths.
4. Added separate grayscale, resizing, CLAHE, channel replication, intensity scaling, ImageNet standardization and CHW-format scripts. PNG stages preserve displayable pixel values; float stages use.npy to avoid clipping/quantization. pipeline_utils.py handles shared file IO/logging only. No training/UI/model files added.

5. Cleaning completed:6508train/1628validation/662Shenzhen test. All IDs,labels,sources,split assignments,patient groups and decoded-pixel hashes exactly match the prior audited split. Excluded3303unlabeled,3800sick/non-TB,800provenance overlaps and1exact pixel duplicate. Source images remain in Datasets; cleaned image links and manifests are relative to FINAL_PROJECT.
6. Ran grayscale,resizing,CLAHE,channel replication,intensity scaling,standardization and final CHW conversion in order. Every stage saved8798outputs with labels/splits/provenance preserved. PNG stages occupy approximately3.9GiB/248MiB/288MiB/757MiB respectively; each float32.npy stage approximately4.93GiB. Each stage has its own progress log and completed-stage metadata.
7. Chained the actual numbered transform functions on a synthetic color image; correct dimensions/dtypes at each step. Confirmed rerunning a completed stage fails before overwriting its manifest/output. Pinned preprocessing-only dependencies from the tested Python3.11 environment.
8. Full verification PASSED for all8798images at all stages. Every intermediate equals its prescribed operation, final arrays exactly equal a fresh calculation from raw originals, original decoded-pixel hashes remain unchanged, IDs/labels stay aligned, and groups/pixel hashes are disjoint across splits. Final arrays are finite contiguousfloat32(3,224,224). No model training or evaluation performed.
9. Created and visually inspected PREPROCESSING_PREVIEW.png. Training example tbx11k__tb0003: center pixel resized191→CLAHE204→scaled0.8→standardized[1.375546,1.535714,1.751111]; final CHW changes layout only. Float panels reverse normalization for display; no floating-point data was replaced by PNG. Actual values recorded in PIXEL_WALKTHROUGH.json.
10. Updated README with execution status and reproduction instructions. Stopped after preprocessing as requested. Training requires a later explicit user greenlight.

## 24 October 2026 — Relocation verification and viva guides
1. Inspected the user-reorganized layout. Canonical FINAL_DATASET remains at FINAL_PROJECT root; an extra Preprocessing/FINAL_DATASET copy was preserved and is not used.
2. Updated preprocessing root/stage resolution and reproduction commands for scripts under Preprocessing. Repaired 8,798 relative clean-image symlinks, intermediate manifest paths and stage lineage metadata. Raw images and saved array values were not reprocessed. Canonical final manifest stayed byte-for-byte unchanged; see RELOCATION_CHECK.json.
3. Full verification passed again for all 8,798 images and all stages on 24 October: exact stage mathematics, fresh-from-original equality, unchanged source pixels and disjoint known groups/hashes. See VERIFICATION_REPORT.json.
4. Created about_dataset.docx (7 pages) and Training/about_training.docx (10 pages), explaining actual sources/counts, exclusions, grouped 80/20 development split plus Shenzhen holdout, preprocessing, feature embeddings/fusion, frozen ViT, memory controls, validation/testing, earlier measured results and viva questions. Rendered both documents and visually checked every page; corrected diagram font alignment.
5. Created Training/README.md with clearly proposed module responsibilities. No new training code, optimizer updates or evaluations were run in FINAL_PROJECT. Earlier parent-project results remain explicitly distinguished from this planned phase.

## 24 October 2026 — Authorized modular hybrid training
- User explicitly authorized training and requested CNN.py, VIT.py and merge_and_train.py. Implemented these plus dataset/common/Grad-CAM helpers directly under FINAL_PROJECT/Training; no import dependency on old tb_project. Reused reviewed architecture/engine behavior and adapted relative final-manifest paths.
- Added configuration/source hashes, batch progress, running/completed status, atomic best/last checkpoints and resume support. Preserved existing split and frozen ViT default; no old CNN checkpoint transfer.
- GPU preflight initially skipped updates with default AMP scale; changed initial scale to 1024 and confirmed CNN/projection updates, frozen backbone, dimensions, and partial accumulation on 35 training images. Passed, peak allocation 741.27 MiB. No test images used.
- Full fresh hybrid training launched, PID recorded in Training/launch.json. Confirmed host GPU process active and batch progress advancing. Maximum 30 epochs, early stopping 7, batch16 accumulation2, validation after each epoch. Training does not automatically run final test.
- Updated Training README and about_training.docx to actual filenames/workflow; rendered and checked all 10 document pages. Earlier reference results remain explicitly separate. User authorization supersedes earlier do-not-train instruction.

## 24 October 2026 — Completed run and output organization
- Confirmed new hybrid training completed 19 epochs with early stopping; best.pt epoch 12 verified against minimum validation loss. Best validation accuracy96.8059%, precision92.4883%, recall84.5494%, F188.3408%, AUC0.981857. No new Shenzhen evaluation yet. Only hybrid trained in FINAL_PROJECT; all three older standalone experiments remain in parent runs.
- Added Training/START_HERE.md, hybrid/RESULTS.md and clear cnn/vit not-run READMEs. Grouped console/launch records under runs/hybrid/logs and preflight outputs under runs/hybrid/checks. Checkpoints/source/data unchanged; no unnecessary weight copies. Updated command guide. Left Python cache and open-document lock alone.

## Fresh standalone experiments authorized

User requested fresh CNN and ViT training for submission on 24 October 2026. Both run sequentially against the same FINAL_DATASET split; hybrid training is not repeated. No earlier TB classifier checkpoint is reused. ViT starts from generic pretrained weights and learns a fresh projection/head with its backbone frozen. Settings remain batch16, accumulation2, AMP, seed42, maximum30 epochs and validation early stopping7. Current queue state is in experiment_logs/standalone_status.json. Each completed run gets runs/<model>/RESULTS.md and logs/console.log. Shenzhen testing is separate.

## 24 October 2026 — Standalone models completed
- Fresh CNN completed 14 epochs, selected epoch7 by minimum validation loss: accuracy89.4963%, precision59.5092%, recall83.2618%, F169.4097%, AUC0.941702. Later epochs degraded; best retained honestly, no score-chasing reruns.
- Fresh frozen-backbone ViT completed13 epochs, selected epoch6: accuracy95.2088%, precision96.9697%, recall68.6695%, F180.4020%, AUC0.972306. No parent-project TB checkpoints reused.
- Both best checkpoints verified against history/model identity. All pre-existing hybrid run files verified unchanged by SHA256. New outputs, console logs and result explanations reside in respective runs/cnn and runs/vit folders. Added VALIDATION_RESULTS.md/JSON comparison; updated START_HERE. No new Shenzhen test evaluation performed.

## 24 October 2026 — Final UI and launch scripts
- Added root Streamlit app with assessment/model selection, CPU inference, CNN Grad-CAM, in-memory uploads, downloadable records, comparison metrics/history and methodology. Uses only fresh FINAL_PROJECT selected checkpoints exported under UI/models. Training code and existing weights unchanged.
- Added preprocess.sh, train.sh, evaluate.sh and run_detection.sh. Environment discovery supports local/parent venv or PROJECT_PYTHON; explicit Hugging Face cache setup removes launcher dependence on an old parent path. Pipeline lock prevents overlapping launcher jobs; complete stages/runs/evaluations skipped, mismatches fail safely. preprocess verifies all stages after skips.
- Completed final evaluation for CNN/ViT/hybrid: Shenzhen accuracy67.37/69.18/73.72%; preserved all per-image predictions and source metrics under each run/final_evaluation. Recomputed accuracy from saved predictions; matched. Recorded FINAL_RESULTS.md/JSON.
- Exported weights-only UI bundles; checked real inference for all three models, CNN/hybrid Grad-CAM, invalid upload rejection and exact raw-to-NPY preprocessing agreement. Streamlit AppTest passed all three pages. Browser visual inspection unavailable because local browser control is disabled; do not claim visual QA. Local server started at127.0.0.1:8502.
- Added RUN_PROJECT.md, full requirements specification and tested environment freeze. No clean-environment installation claimed. Full preprocess launcher verification is recorded in logs/preprocess_launcher.log.

- Final launcher checks passed: preprocessing verified all8798images; train skipped all3completed models without changing checkpoints; evaluate skipped3matching completed evaluations; UI launcher reused3matching exports. Existing hybrid weights and execution artifacts remain unchanged (RESULTS.md intentionally updated with test results).

## Single installation file
Consolidated FINAL_PROJECT requirements.txt into the complete pinned preprocessing/training/evaluation/UI dependency list, using tested torch/torchvision public versions without local CUDA suffixes. Removed requirements-full.txt and updated launcher/setup references. Historical logs/requirements-tested.txt retained as an environment record, not an installation file. Python3.11 recommended; CPU UI supported, GPU drivers and Linux/WSL launcher requirements documented. No claim of a clean install on another PC.
