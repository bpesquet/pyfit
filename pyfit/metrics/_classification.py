"""
Classification metrics
"""

import numpy as np
from pyfit import Tensor


def accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Compute accuracy: number of exact predictions / total number of predictions
    """
    n_exact: int = np.sum(y_pred == y_true)
    n_total: int = y_true.size
    return n_exact / n_total
