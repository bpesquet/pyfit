"""
Classification metrics
"""

import numpy as np
from pyfit import Tensor


def accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Compute accuracy: number of exact predictions / total number of predictions
    """
    return np.sum(y_pred == y_true) / y_true.size
