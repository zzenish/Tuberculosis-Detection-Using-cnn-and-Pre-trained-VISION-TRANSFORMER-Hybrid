"""Step 6: convert uint8 [0,255] pixels to float32 [0,1]. Save HWC NumPy arrays."""
import numpy as np
from pipeline_utils import read_image, run_stage


def scale_intensity(path):
    image = read_image(path)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError('Expected the three-channel uint8 image')
    # OpenCV's BGR order has no effect here: all three input channels are identical.
    return image.astype(np.float32) / 255.0


if __name__ == '__main__':
    run_stage('CHANNEL_REPLICATION', 'INTENSITY_SCALING', 'Divide intensities by 255 only',
              scale_intensity, '.npy', {'shape': [224, 224, 3], 'dtype': 'float32', 'range': [0, 1]})
