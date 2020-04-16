"""
Neural network providing a PyTorch-like API.

Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
"""

import random
from typing import List
from pyfit.engine import Scalar, Vector


class Module:
    """A differentiable computation"""

    def zero_grad(self) -> None:
        """Reset gradients for all parameters"""

        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> Vector:
        """Return parameters"""

        raise NotImplementedError


class Neuron(Module):
    """A single neuron"""

    def __init__(self, in_features: int, nonlin: bool = True):
        self.w: Vector = [Scalar(random.uniform(-1, 1)) for _ in range(in_features)]
        self.b: Scalar = Scalar(0)
        self.nonlin = nonlin

    def __call__(self, x: Vector) -> Scalar:
        act: Scalar = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> Vector:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """A layer of neurons"""

    def __init__(self, in_features: int, out_features: int, nonlin: bool = True):
        self.neurons = [Neuron(in_features, nonlin) for _ in range(out_features)]

    def __call__(self, x: Vector) -> Vector:
        return [n(x) for n in self.neurons]

    def parameters(self) -> Vector:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A Multi-Layer Perceptron, aka shallow neural network"""

    def __init__(self, input_features: int, layers: List[int]):
        sizes: List[int] = [input_features] + layers
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(layers) - 1)
            for i in range(len(layers))
        ]

    def __call__(self, x: Vector) -> Vector:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> Vector:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
