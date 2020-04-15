"""
Unit tests for autodiff engine
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

from pyfit.engine import Scalar


def test_engine() -> None:
    x = Scalar(1.0)
    z = 2 * x + 2 + x
    q = z + z * x
    h = z * z
    y = h + q + q * x
    assert y.data == 45.0
    y.backward()
    assert x.grad == 62  # The numerical value of dy / dx
