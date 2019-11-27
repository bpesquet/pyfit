"""
A tensor is a multidimensional array of numbers
"""

from typing import Optional, Union
import numpy as np
# Use NumPy n-dimensional array class as our Tensor class
from numpy import ndarray as Tensor


# class Tensor:
#     def __init__(self, tensor: np.ndarray):
#         super().__init__()
#         self.tensor = tensor

#     def __getattr__(self, attr):
#         return getattr(self.tensor, attr)


def mean(x: Tensor, axis: Optional[int] = None) -> Union[float, Tensor]:
    """
    Compute the mean of a tensor
    """
    return np.mean(x, axis)


# def sum(x: Tensor, axis: Optional[int] = None) -> Union[float, Tensor]:
#     """
#     Compute the sum of a tensor
#     """
#     return np.sum(x, axis)


def sqrt(x: Tensor) -> Tensor:
    """
    Compute the square root of a tensor
    """
    return np.sqrt(x)
