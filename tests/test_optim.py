"""
Unit tests for optimizers
"""

# pylint: disable=missing-docstring

from pyfit.engine import Vector, Scalar
from pyfit.optim import SGD


def test_sgd() -> None:
    params: Vector = [Scalar(3), Scalar(-1), Scalar(0.5)]
    sgd = SGD(parameters=params, learning_rate=0.5)

    # Simulate gradient update (normally done through backprop)
    for i, p in enumerate(params):
        p.grad = i

    sgd.step()
    assert params[0].data == 3  # 3 - 0.5 * 0
    assert params[1].data == -1.5  # -1 - 0.5 * 1
    assert params[2].data == -0.5  # 0.5 - 0.5 * 2

    sgd.zero_grad()
    for _, p in enumerate(params):
        assert p.grad == 0
