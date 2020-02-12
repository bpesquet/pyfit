"""
Unit tests for metrics
"""

# Docstrings are superfluous for test functions
# pylint: disable=missing-docstring

from math import isclose
import numpy as np
from pyfit.metrics import mean_squared_error, euclidean_distance, accuracy


def test_mse() -> None:
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    error = mean_squared_error(y_true, y_pred)
    assert isclose(error, (0.25 + 0.25 + 1) / 4)

    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])

    error = mean_squared_error(y_true, y_pred)
    assert isclose(error, (1.0 / 3 + 2.0 / 3 + 2.0 / 3) / 4)


def test_euclidean_distance() -> None:
    a = np.array([1, 1.5])
    b = np.array([3, 3])

    dist = euclidean_distance(a, b)
    assert isclose(dist, np.linalg.norm(a - b))

    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    dist = euclidean_distance(a, b)
    assert isclose(dist, np.linalg.norm(a - b))


def test_accuracy() -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 2, 1, 3])

    acc = accuracy(y_true, y_pred)
    assert acc == 0.5

    acc = accuracy(y_true, y_true)
    assert acc == 1.0
