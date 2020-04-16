"""
Unit test for loss functions
"""

# pylint: disable=missing-docstring

from math import isclose
from pyfit.engine import Scalar, Vector
from pyfit.loss import MSELoss


def test_mse() -> None:
    y_true: Vector = [Scalar(3), Scalar(-0.5), Scalar(2), Scalar(7)]
    y_pred: Vector = [Scalar(2.5), Scalar(0.0), Scalar(2), Scalar(8)]
    loss = MSELoss()
    error: Scalar = loss(y_pred, y_true)
    assert isclose(error.data, 0.375)
