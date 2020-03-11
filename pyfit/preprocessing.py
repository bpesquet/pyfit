"""
Utilities for preparing data before training
"""

from typing import List
import numpy as np
from pyfit import Tensor


def train_test_split(*x: Tensor, test_ratio: float = 0.25) -> List[Tensor]:
    """
    Split and shuffle one or several same-length tensor(s) between training and test sets,
    according to a chosen test ratio
    """
    n_samples: int = x[0].shape[0]
    split_index: int = round(n_samples * (1 - test_ratio))
    # https://stackoverflow.com/a/4602224
    permutation: Tensor = np.random.permutation(n_samples)
    splitted = []
    for tensor in x:
        shuffled_tensor = tensor[permutation]
        if shuffled_tensor.ndim > 1:
            splitted.append(shuffled_tensor[:split_index, :])
            splitted.append(shuffled_tensor[split_index:, :])
        else:
            splitted.append(shuffled_tensor[:split_index])
            splitted.append(shuffled_tensor[split_index:])
    return splitted


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


def vectorize_sequences(sequences: Tensor, dimension: int = 10000) -> Tensor:
    """One-hot encode a vector of sequences into a binary matrix (number of sequences, dimension)"""

    # Example : [[3, 5]] -> [[0. 0. 0. 1. 0. 1. 0...]]

    results = np.zeros((len(sequences), dimension))
    # set specific indices of results[i] to 1s
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results
