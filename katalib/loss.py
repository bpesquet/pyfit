"""
Loss functions quantify the gap between actual (expected) results and predictions.
"""

import numpy as np


class Loss:
    """
    Base class for all losses
    """

    def loss(self, actual, predicted):
        """Compute the loss between actual and predicted values"""
        raise NotImplementedError

    def grad(self, actual, predicted):
        """Compute the gradient (partial derivatives) of the loss"""
        raise NotImplementedError


class MSE(Loss):
    """
    Mean Squared Error loss
    """

    def loss(self, actual, predicted):
        return np.square(predicted - actual).mean()

    def grad(self, actual, predicted):
        return 2 * (predicted - actual)
