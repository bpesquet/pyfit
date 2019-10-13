"""
Unit tests for loss
"""

from math import isclose
import numpy as np
from sklearn.metrics import mean_squared_error
from katalib.loss import MSE


def test_mse_loss():
    """
    Test MSE loss computation
    """

    actual = np.array([3, -0.5, 2, 7])
    predicted = np.array([2.5, 0.0, 2, 8])

    loss_sk = mean_squared_error(actual, predicted)
    loss = MSE().loss(actual, predicted)
    assert loss == loss_sk

    actual = np.array([[0.5, 1], [-1, 1], [7, -6]])
    predicted = np.array([[0, 2], [-1, 2], [8, -5]])

    loss_sk = mean_squared_error(actual, predicted)
    loss = MSE().loss(actual, predicted)
    assert loss == loss_sk


def test_mse_grad():
    """
    Test MSE gradient computation
    """

    grad = MSE().grad(3, 2.9)
    assert isclose(grad, -0.2)
