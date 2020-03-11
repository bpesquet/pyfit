"""
Data generation functions
"""

# Allow uppercase and non-snake_case names
# pylint: disable=invalid-name

from typing import Tuple
import numpy as np
from pyfit import Tensor


def make_multiclass(
    n_points: int = 500, n_dim: int = 2, n_classes: int = 3
) -> Tuple[Tensor, Tensor]:
    """
    Generate spiral-shaped data
    n: number of points per class
    d: dimensionality
    k: number of classes
    """

    np.random.seed(0)
    x = np.zeros((n_points * n_classes, n_dim))
    y = np.zeros(n_points * n_classes)
    for j in range(n_classes):
        ix = range(n_points * j, n_points * (j + 1))
        # radius
        r = np.linspace(0.0, 1, n_points)
        # theta
        t = np.linspace(j * 4, (j + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return x, y
