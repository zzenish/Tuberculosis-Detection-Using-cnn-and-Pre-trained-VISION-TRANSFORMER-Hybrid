"""Step 5: replicate one grayscale channel into three identical channels."""
import numpy as np
from pipeline_utils import read_image, run_stage


def replicate_channels(path):
    image = read_image(path)
    if image.shape != (224, 224):
        raise ValueError('Expected the one-channel CLAHE image')
    return np.repeat(image[:, :, None], repeats=3, axis=2)


if __name__ == '__main__':
    run_stage('CONTRAST_NORMALIZATION', 'CHANNEL_REPLICATION', 'Channel replication only',
              replicate_channels, '.png', {'shape': [224, 224, 3], 'channels_identical': True})
