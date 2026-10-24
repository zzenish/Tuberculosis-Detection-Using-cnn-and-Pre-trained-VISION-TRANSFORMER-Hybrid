"""Step 4: improve local contrast using CLAHE. This does not scale values to [0,1]."""
import cv2
from pipeline_utils import read_image, run_stage


def contrast_normalization(path):
    image = read_image(path)
    if image.shape != (224, 224):
        raise ValueError('CLAHE expects a resized grayscale image')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


if __name__ == '__main__':
    run_stage('RESIZING', 'CONTRAST_NORMALIZATION', 'CLAHE only', contrast_normalization, '.png',
              {'clip_limit': 2.0, 'tile_grid': [8, 8], 'dtype': 'uint8'})
