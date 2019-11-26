"""
Distance metrics
"""

import numpy as np
from pyfit.tensor import Tensor


def euclidean_distance(a: Tensor, b: Tensor) -> float:
    """
    Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
    """
    return np.sqrt(np.sum((a - b) ** 2))
