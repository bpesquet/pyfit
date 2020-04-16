"""
Autograd engine implementing reverse-mode autodifferentiation, aka backpropagation.

Heavily inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
"""

# https://stackoverflow.com/questions/35701624/pylint-w0212-protected-access
# pylint: disable=protected-access

from typing import Union, Tuple, List, Set, Callable


class Scalar:
    """Stores a single scalar value and its gradient"""

    def __init__(
        self, data: float, children: Tuple["Scalar", ...] = (), op: str = ""
    ) -> None:
        self.data: float = data
        self.grad: float = 0

        # Internal variables used for autograd graph construction
        self._backward: Callable = lambda: None
        self._prev: Set[Scalar] = set(children)
        self._op = (
            op  # The operation that produced this node, for graphviz / debugging / etc
        )

    def __add__(self, other: Union["Scalar", float]) -> "Scalar":
        _other: Scalar = other if isinstance(other, Scalar) else Scalar(other)
        out: Scalar = Scalar(self.data + _other.data, (self, _other), "+")

        def _backward() -> None:
            self.grad += out.grad  # d(out)/d(self) = 1
            _other.grad += out.grad  # d(out)/d(other) = 1

        out._backward = _backward

        return out

    def __sub__(self, other: Union["Scalar", float]) -> "Scalar":
        _other: Scalar = other if isinstance(other, Scalar) else Scalar(other)
        out: Scalar = Scalar(self.data - _other.data, (self, _other), "-")

        def _backward() -> None:
            self.grad += out.grad  # d(out)/d(self) = 1
            _other.grad -= out.grad  # d(out)/d(other) = -1

        out._backward = _backward

        return out

    def __mul__(self, other: Union["Scalar", float]) -> "Scalar":
        _other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * _other.data, (self, _other), "*")

        def _backward() -> None:
            self.grad += out.grad * _other.data  # d(out)/d(self) = other
            _other.grad += out.grad * self.data  # d(out)/d(other) = self

        out._backward = _backward

        return out

    def __truediv__(self, other: Union["Scalar", float]) -> "Scalar":
        _other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data / _other.data, (self, _other), "/")

        def _backward() -> None:
            self.grad += out.grad / _other.data  # d(out)/d(self) = 1/other
            # d(out)/d(other) = -self/(other*other)
            _other.grad += out.grad * (-self.data / (_other.data * _other.data))

        out._backward = _backward

        return out

    def relu(self) -> "Scalar":
        """Compute ReLU"""

        out = Scalar(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        """Compute gradients through backpropagation"""

        # Topological order all of the children in the graph
        topo: Vector = []
        visited: Set[Scalar] = set()

        def build_topo(node: Scalar) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other: Union["Scalar", float]) -> "Scalar":
        return self.__add__(other)

    def __rsub__(self, other: Union["Scalar", float]) -> "Scalar":
        _other = other if isinstance(other, Scalar) else Scalar(other)
        return _other.__sub__(self)

    def __rmul__(self, other: Union["Scalar", float]) -> "Scalar":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union["Scalar", float]) -> "Scalar":
        _other = other if isinstance(other, Scalar) else Scalar(other)
        return _other.__truediv__(self)

    def __repr__(self) -> str:
        return f"Scalar(data={self.data}, grad={self.grad})"


Vector = List[Scalar]
