"""
Utilities for preparing data before training
"""

from typing import Tuple
from pyfit.tensor import Tensor


def train_test_split(data: Tensor, test_ratio: float = 0.25) -> Tuple[Tensor, Tensor]:
    """
    Split data between training and test sets, according to a chosen test ratio
    """
    samples_count = data.shape[0]
    split_index = samples_count - round(samples_count * test_ratio)
    return data[:split_index, :], data[split_index:, :]


def scale_min_max(data: Tensor) -> Tensor:
    """
    Scale data into the [0..1] interval
    """
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))


def scale_standard(data: Tensor, mean: float = None, std: float = None) -> Tensor:
    """
    Scale data to zero mean and unit variance
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    return (data - mean) / std
