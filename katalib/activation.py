"""
Activation functions apply a function to their inputs.
"""

import numpy as np
from katalib.layer import Layer


class Activation(Layer):
    """
    Base class for all activation functions
    """

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.func_prime(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def func_prime(self, x):
        raise NotImplementedError


class Tanh(Activation):
    """
    Hyperbolic tangent activation function
    """

    def func(self, x):
        return np.tanh(x)

    def func_prime(self, x):
        y = self.func(x)
        return 1 - y ** 2
