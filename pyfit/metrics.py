"""
Metrics functions
"""

from pyfit.engine import Scalar, Vector


def mean_squared_error(y_true: Vector, y_pred: Vector) -> Scalar:
    """MSE loss"""

    total_squared_error: Scalar = sum(
        (
            (y_true_i - y_pred_i) * (y_true_i - y_pred_i)
            for (y_true_i, y_pred_i) in zip(y_true, y_pred)
        ),
        Scalar(0),
    )
    n_total: int = max(len(y_true), 1)
    return total_squared_error / n_total


def binary_accuracy(y_true: Vector, y_pred: Vector) -> float:
    """Binary accuracy"""

    n_exact: int = sum(
        (
            y_true_i.data == round(y_pred_i.data)
            for (y_true_i, y_pred_i) in zip(y_true, y_pred)
        )
    )
    n_total: int = max(len(y_true), 1)
    return n_exact / n_total
