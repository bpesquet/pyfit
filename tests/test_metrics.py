"""
Unit tests for metrics
"""

import numpy as np
from sklearn.metrics import mean_squared_error as mse_sk
from mlcore.metrics import mean_squared_error


def test_mse():
    """
    Compare MSE results to scikit-learn's
    """

    expected = np.array([3, -0.5, 2, 7])
    predicted = np.array([2.5, 0.0, 2, 8])

    error = mean_squared_error(expected, predicted)
    error_sk = mse_sk(expected, predicted)
    assert error == error_sk

    expected = np.array([[0.5, 1], [-1, 1], [7, -6]])
    predicted = np.array([[0, 2], [-1, 2], [8, -5]])

    error = mean_squared_error(expected, predicted)
    error_sk = mse_sk(expected, predicted)
    assert error == error_sk
