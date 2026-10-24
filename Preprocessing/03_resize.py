"""Step 3: resize grayscale images to exactly 224 x 224 using area interpolation."""
import cv2
from pipeline_utils import read_image, run_stage


def resize(path):
    image = read_image(path)
    if image.ndim != 2:
        raise ValueError('Resizing expects the preceding grayscale output')
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    run_stage('GRAYSCALE', 'RESIZING', 'Resize only', resize, '.png',
              {'height': 224, 'width': 224, 'interpolation': 'cv2.INTER_AREA'})
