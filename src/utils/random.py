import os
import random

import numpy as np
import torch


def set_random_seed(seed_value=42, torch=True, tf=False):
    """
    Sets the seed for the various RNG to a predefined value.
    Can be used to increase reproducibility, e.g. for model comparison.
    """
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Python built-in pseudo-random generator
    import random
    random.seed(seed_value)
    # 3. Numpy pseudo-random generator
    import numpy as np
    np.random.seed(seed_value)
    if torch:
        # 4. PyTorch pseudo-random generator
        import torch
        torch.manual_seed(seed_value)
        # When running on the CuDNN backend, two further options must be set:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if tf:
        # 5. Tensorflow pseudo-random generator
        import tensorflow as tf
        tf.set_random_seed(seed_value)
