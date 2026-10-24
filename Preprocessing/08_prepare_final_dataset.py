"""Step 8: change HWC to contiguous CHW. No further image enhancement or normalization."""
import numpy as np
from pipeline_utils import run_stage


def prepare_final(path):
    image = np.load(path, allow_pickle=False)
    if image.shape != (224, 224, 3) or image.dtype != np.float32:
        raise ValueError('Expected standardized float32 HWC array')
    return np.ascontiguousarray(image.transpose(2, 0, 1))


if __name__ == '__main__':
    run_stage('Z_SCORE_STANDARDIZATION', 'FINAL_DATASET', 'Convert HWC to CHW layout only',
              prepare_final, '.npy', {'shape': [3, 224, 224], 'dtype': 'float32', 'contiguous': True})
