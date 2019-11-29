"""
Utilities for preparing data before training
"""

from typing import Tuple, Optional
import numpy as np


def train_test_split(data: np.ndarray, test_ratio: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data between training and test sets, according to a chosen test ratio
    """
    n_samples: int = data.shape[0]
    split_index: int = n_samples - round(n_samples * test_ratio)
    return data[:split_index, :], data[split_index:, :]


def scale_min_max(data: np.ndarray) -> np.ndarray:
    """
    Scale data into the [0..1] range
    """
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))


def scale_standard(data: np.ndarray, mean: Optional[np.ndarray] = None,
                   std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Scale data to zero mean and unit variance
    """
    if mean is None:
        # Compute mean feature-wise
        mean = data.mean(axis=0)
    if std is None:
        # Compute standard deviation feature-wise
        std = data.std(axis=0)
    return (data - mean) / std
