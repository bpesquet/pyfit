"""
Unit tests for metrics
"""

# Docstring are superfluous for test functions
# pylint: disable=missing-docstring

import numpy as np
from sklearn.metrics import mean_squared_error as mse_sk
from pyfit.metrics import mean_squared_error


def test_mse() -> None:
    expected_list = (np.array([3, -0.5, 2, 7]),
                     np.array([[0.5, 1], [-1, 1], [7, -6]]))
    predicted_list = (np.array([2.5, 0.0, 2, 8]),
                      np.array([[0, 2], [-1, 2], [8, -5]]))

    for expected, predicted in zip(expected_list, predicted_list):
        error = mean_squared_error(expected, predicted)
        error_sk = mse_sk(expected, predicted)
        assert error == error_sk
