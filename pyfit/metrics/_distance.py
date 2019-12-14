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
    squared_diff: Tensor = (a - b) ** 2
    sum_squared_diff: float = np.sum(squared_diff)
    eucl_dist: float = np.sqrt(sum_squared_diff)
    return eucl_dist
