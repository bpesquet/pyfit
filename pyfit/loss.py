"""
Loss functions
"""

# pylint: disable=too-few-public-methods

from pyfit.engine import Scalar, Vector


class MSELoss:
    """Mean Squared Error loss"""

    def __call__(self, y_pred: Vector, y_true: Vector) -> Scalar:
        """Compute MSE loss"""

        total_squared_error: Scalar = sum(
            (
                (y_true_i - y_pred_i) * (y_true_i - y_pred_i)
                for (y_true_i, y_pred_i) in zip(y_true, y_pred)
            ),
            Scalar(0),
        )
        return total_squared_error / max(len(y_true), 1)
