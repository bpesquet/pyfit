"""
Distance metrics
"""

from typing import Callable
import numpy as np
from pyfit import Tensor

# Distance function type
Distance = Callable[[Tensor, Tensor], float]


def euclidean_distance(a: Tensor, b: Tensor) -> float:
    """
    Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
    """
    return np.sqrt(np.sum((a - b) ** 2))
