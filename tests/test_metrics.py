"""
Unit tests for metrics
"""

# pylint: disable=missing-docstring

from math import isclose
from pyfit.engine import Scalar, Vector
from pyfit.metrics import mean_squared_error, binary_accuracy


def test_mse() -> None:
    y_true: Vector = [Scalar(3), Scalar(-0.5), Scalar(2), Scalar(7)]
    y_pred: Vector = [Scalar(2.5), Scalar(0.0), Scalar(2), Scalar(8)]
    error: Scalar = mean_squared_error(y_true, y_pred)
    assert isclose(error.data, 0.375)


def test_binary_accuracy() -> None:
    y_true: Vector = [Scalar(0), Scalar(1), Scalar(2), Scalar(3)]
    y_pred: Vector = [Scalar(0), Scalar(2), Scalar(1), Scalar(3)]

    acc = binary_accuracy(y_true, y_pred)
    assert acc == 0.5

    acc = binary_accuracy(y_true, y_true)
    assert acc == 1.0
