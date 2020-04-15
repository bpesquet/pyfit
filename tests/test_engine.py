"""
Unit tests for autodiff engine
"""

# pylint: disable=missing-docstring

from pyfit.engine import Scalar


def test_scalar() -> None:
    x = Scalar(1.0)
    y = (x * 2 + 1).relu()
    assert y.data == 3
    y.backward()
    assert x.grad == 2  # dy/dx


def test_scalar_sub() -> None:
    x = Scalar(2.5)
    y = Scalar(3.5)
    z = x - y
    assert z.data == -1
    z.backward()
    assert x.grad == 1  # dz/dx
    assert y.grad == -1  # dz/dy


def test_scalar_div() -> None:
    x = Scalar(1.0)
    y = Scalar(4.0)
    z = x / y
    assert z.data == 0.25
    z.backward()
    assert x.grad == 0.25  # dz/dx
    assert y.grad == -0.0625  # dz/dy
