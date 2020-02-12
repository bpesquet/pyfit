"""
Layer classes
"""

# Temporarily disable annoying Pylint check
# pylint: disable=too-few-public-methods

from typing import Dict, Callable
import numpy as np
from pyfit import Tensor
from ._activations import tanh, tanh_prime

F = Callable[[Tensor], Tensor]


class Differentiable:
    """
    Base class for all differentiable computations
    """

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the output corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, gradients: Tensor) -> Tensor:
        """
        Backpropagate these gradients through the layer
        """
        raise NotImplementedError


class ParametersMixin:
    """
    Mixin for adding parameters and gradient values to a class
    """

    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}


class Linear(Differentiable, ParametersMixin):
    """
    Linear layer
    """

    def __init__(self, *, in_features: int, out_features: int) -> None:
        super().__init__()
        # Randomly init weights and bias
        self.params["w"] = np.random.randn(in_features, out_features)
        self.params["b"] = np.random.randn(out_features)
        # Init inputs
        self.inputs: Tensor = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, gradients: Tensor) -> Tensor:
        self.grads["b"] = np.sum(gradients, axis=0)
        self.grads["w"] = self.inputs.T @ gradients
        return gradients @ self.params["w"].T


class Activation(Differentiable):
    """
    Layer corresponding to an activation function
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
        # Init inputs
        self.inputs: Tensor = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, gradients: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * gradients


class Tanh(Activation):
    """
    Tanh activation
    """

    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)
