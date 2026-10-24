"""Step 2: convert selected original X-rays to one grayscale channel. Do not resize."""
import cv2
from pipeline_utils import read_image, run_stage


def grayscale(path):
    return read_image(path, cv2.IMREAD_GRAYSCALE)


if __name__ == '__main__':
    run_stage('CLEANED_DATASET', 'GRAYSCALE', 'Grayscale conversion only', grayscale, '.png',
              {'channels': 1, 'dtype': 'uint8', 'size': 'original dimensions'})
