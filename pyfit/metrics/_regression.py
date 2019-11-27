"""
Regression metrics
"""

import numpy as np


def mean_squared_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Mean Squared Error
    """
    return np.mean((expected - predicted) ** 2)
