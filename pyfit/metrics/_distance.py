"""
Distance metrics
"""

from typing import Callable
import numpy as np

# Distance function type
Distance = Callable[[np.ndarray, np.ndarray], float]


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
    """
    return np.sqrt(np.sum((a - b) ** 2))
