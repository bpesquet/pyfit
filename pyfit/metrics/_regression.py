"""
Regression metrics
"""

import numpy as np
from pyfit import Tensor


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Compute Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)
