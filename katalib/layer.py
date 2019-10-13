"""
Layers are the building blocks of models.
Each layer passes its inputs forward and propagates gradients backward.
"""

import numpy as np


class Layer():
    """
    Base class for all layers
    """

    def __init__(self):
        self.inputs = []

    def forward(self, inputs):
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad):
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    Linear layer: y = w.x + b
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.params = {}
        self.grads = {}

        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)
        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T
