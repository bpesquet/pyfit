"""
Unit tests for preprocessing functions
"""

# Docstrings are superfluous for test functions
# pylint: disable=missing-docstring

from math import isclose
import numpy as np
from pyfit.preprocessing import train_test_split, scale_min_max, scale_standard


def test_train_test_split() -> None:
    # Generate a random 1d tensor
    x = np.random.randint(1, 10, (17,))

    x_train, x_test = train_test_split(x, test_ratio=0.3)
    assert x_train.shape == (12,)
    assert x_test.shape == (5,)

    # Generate a random 2d tensor
    x = np.random.rand(30, 3)

    x_train, x_test = train_test_split(x, test_ratio=0.33)
    assert x_train.shape == (20, 3)
    assert x_test.shape == (10, 3)

    # Default ratio is 0.25
    x_train, x_test = train_test_split(x)
    assert x_train.shape == (22, 3)
    assert x_test.shape == (8, 3)


def test_train_test_split_2tensors() -> None:
    # Generate two tensors with same length (first dimension)
    x = np.random.rand(30, 3)
    y = np.random.rand(30)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    assert x_train.shape == (22, 3)
    assert x_test.shape == (8, 3)
    assert y_train.shape == (22,)
    assert y_test.shape == (8,)


def test_scale_min_max() -> None:
    # Generate a random (3,3) tensor with values between 1 and 10
    x = np.random.randint(1, 10, (3, 3))

    x_scaled = scale_min_max(x)
    assert x_scaled.min() == 0
    assert x_scaled.max() == 1


def test_scale_standard() -> None:
    # Generate a random tensor with int values between 1 and 10
    x = np.random.randint(1, 10, (3, 3))

    x_scaled = scale_standard(x)

    mean_scaled: float = x_scaled.mean()
    std_scaled: float = x_scaled.std()
    # https://stackoverflow.com/a/35325039
    assert isclose(mean_scaled, 0, abs_tol=1.0e-8)
    assert isclose(std_scaled, 1)


def test_scale_standard_mean_std() -> None:
    # Generate a random tensor with int values between 1 and 10
    x = np.random.randint(1, 10, (3, 4))

    # Scale with zero mean and unit variance (should do nothing)
    zero_mean = np.zeros(x.shape[1])
    unit_variance = np.ones(x.shape[1])
    x_scaled = scale_standard(x, mean=zero_mean, std=unit_variance)

    # Check that mean and standard deviation haven't been modified by scaling
    assert x_scaled.mean() == x.mean()
    assert x_scaled.std() == x.std()
