#!/bin/bash

set -e

echo "Starting Pre-Processing pipeline:"

#python3 01_clean_and_split.py
echo "Finished 01_clean_and_split.py"

python3 02_grayscale.py
echo "Finished 02_grayscale.py"

python3 03_resize.py
echo "Finished 03_resize.py"

python3 04_contrast_normalization.py
echo "Finished 04_contrast_normalization.py"

python3 05_channel_replication.py
echo "Finished 05_channel_replication.py"

python3 06_intensity_scaling.py
echo "Finished 06_intensity_scaling.py"

python3 07_zscore_standardization.py
echo "Finished 07_zscore_standardization.py"

python3 08_prepare_final_dataset.py
echo "Finished 08_prepare_final_dataset.py"

python3 09_verify_preprocessing.py
echo "Preprocessing complete!"
