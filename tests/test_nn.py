"""
Unit tests for neural nets API
"""

# pylint: disable=missing-docstring

from pyfit.engine import Scalar
from pyfit.nn import Neuron, Layer, MLP


def test_neuron() -> None:
    n_features = 3
    neuron = Neuron(n_features, nonlin=False)

    # Check parameter count
    assert len(neuron.parameters()) == n_features + 1  # Including bias

    # Overwite parameters for output prediction
    neuron.w = list(Scalar(x) for x in range(n_features))
    neuron.b = Scalar(-0.75)

    # Check output computation (weighted sum + bias)
    assert (
        neuron([Scalar(1), Scalar(-0.5), Scalar(1.5)]).data == 1.75
    )  # 1*0 + (-0.5)*1 + 1.5*2 + 1*(-0.75)
    assert (
        neuron([Scalar(1), Scalar(-3), Scalar(1.5)]).data == -0.75
    )  # 1*0 + (-3)*1 + 1.5*2 + 1*(-0.75)

    # Add ReLU = max(0,x) activation function
    neuron.nonlin = True

    assert neuron([Scalar(1), Scalar(-0.5), Scalar(1.5)]).data == 1.75
    assert neuron([Scalar(1), Scalar(-3), Scalar(1.5)]).data == 0


def test_layer() -> None:
    layer = Layer(3, 2)

    assert len(layer.neurons) == 2
    assert len(layer.parameters()) == 8  # 3*2 + 2


def test_mlp() -> None:
    model = MLP(2, [3, 1])  # 1 hidden layer with 3 neurons

    assert len(model.layers) == 2
    assert len(model.parameters()) == 13  # 2*3+3 + 3*1+1
