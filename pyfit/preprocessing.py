"""
Utilities for preparing data before training
"""

from typing import Tuple
from pyfit import Tensor


def train_test_split(x: Tensor, test_ratio: float = 0.25) -> Tuple[Tensor, Tensor]:
    """
    Split a tensor between training and test sets, according to a chosen test ratio
    """
    n_samples: int = x.shape[0]
    split_index: int = n_samples - round(n_samples * test_ratio)
    if x.ndim > 1:
        return x[:split_index, :], x[split_index:, :]
    return x[:split_index], x[split_index:]


def scale_min_max(x: Tensor) -> Tensor:
    """
    Scale a tensor into the [0..1] range
    """
    # Compute min and max feature-wise
    min_x: Tensor = x.min(axis=0)
    max_x: Tensor = x.max(axis=0)

    return (x - min_x) / (max_x - min_x)


def scale_standard(x: Tensor, *, mean: Tensor = None, std: Tensor = None) -> Tensor:
    """
    Scale a tensor to zero mean and unit variance
    """
    if mean is None:
        # Compute mean feature-wise
        mean = x.mean(axis=0)
    if std is None:
        # Compute standard deviation feature-wise
        std = x.std(axis=0)
    return (x - mean) / std
