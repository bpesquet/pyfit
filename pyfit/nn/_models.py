"""
Model classes
"""

from typing import Sequence
from pyfit import Tensor
from ._layers import Differentiable


class Sequential(Differentiable):
    """
    Neural network defined as a linear stack of layers
    """

    def __init__(self, layers: Sequence[Differentiable]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradients: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
        return gradients
