"""
Unit tests for metrics
"""

# Docstring are superfluous for test functions
# pylint: disable=missing-docstring

import numpy as np
from sklearn.metrics import mean_squared_error as mse_sk, \
    euclidean_distances as eucl_dist_sk
from pyfit.metrics import mean_squared_error, euclidean_distance


def test_mse() -> None:
    expected_list = (np.array([3, -0.5, 2, 7]),
                     np.array([[0.5, 1], [-1, 1], [7, -6]]))
    predicted_list = (np.array([2.5, 0.0, 2, 8]),
                      np.array([[0, 2], [-1, 2], [8, -5]]))

    for expected, predicted in zip(expected_list, predicted_list):
        error = mean_squared_error(expected, predicted)
        error_sk = mse_sk(expected, predicted)
        assert error == error_sk


def test_euclidean_distance() -> None:
    # a = np.array([1, 1.5])
    # b = np.array([3, 3])

    # dist = euclidean_distance(a, b)
    # dist_sk = eucl_dist_sk(a, b)
    # assert dist == dist_sk

    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    dist = euclidean_distance(a, b)
    dist_sk = np.linalg.norm(a - b)
    assert dist == dist_sk
