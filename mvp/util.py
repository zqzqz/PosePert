import os
import random
import numpy as np
import torch


def set_seed(seed=42, set_python=True, set_numpy=True, set_torch=True):
    if set_python:
        # Python & OS
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    if set_numpy:
        # NumPy
        np.random.seed(seed)
    
    if set_torch:
        # PyTorch (CPU & CUDA)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if multi-GPU

        # # Ensure deterministic behavior
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)  # new API (PyTorch 1.8+)