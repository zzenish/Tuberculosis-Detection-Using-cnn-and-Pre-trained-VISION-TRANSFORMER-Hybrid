"""Step 7: apply fixed ImageNet mean/std to each scaled channel; keep HWC layout."""
import numpy as np
from pipeline_utils import MEAN, STD, run_stage


def standardize(path):
    image = np.load(path, allow_pickle=False)
    if image.shape != (224, 224, 3) or image.dtype != np.float32:
        raise ValueError('Expected scaled float32 HWC array')
    return (image - MEAN) / STD


if __name__ == '__main__':
    run_stage('INTENSITY_SCALING', 'Z_SCORE_STANDARDIZATION', 'ImageNet standardization only',
              standardize, '.npy', {'mean': MEAN.tolist(), 'std': STD.tolist(), 'layout': 'HWC'})
