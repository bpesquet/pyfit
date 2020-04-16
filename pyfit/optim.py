"""
Optimization algorithms for gradient descent
"""

# pylint: disable=too-few-public-methods

from pyfit.engine import Vector


class Optimizer:
    """Base class for optimizers"""

    def __init__(self, parameters: Vector):
        self.parameters: Vector = parameters

    def zero_grad(self) -> None:
        """Reset gradients for all parameters"""

        for p in self.parameters:
            p.grad = 0


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, parameters: Vector, learning_rate: float = 0.01):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def step(self) -> None:
        """Update model parameters in the opposite direction of their gradient"""

        for p in self.parameters:
            p.data -= self.learning_rate * p.grad
