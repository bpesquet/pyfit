"""
Distance metrics
"""

import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
    """
    return np.sqrt(np.sum((a - b) ** 2))
