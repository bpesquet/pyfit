# pyfit

A minimalist neural networks library built on a tiny autograd engine. Very much inspired by the [micrograd](https://github.com/karpathy/micrograd) library created by Andrej Karpathy.

## Overview

This project aims to:

- demonstrate [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), a core concept of modern Deep Learning frameworks like [PyTorch](https://pytorch.org) and [TensorFlow](https://www.tensorflow.org/);
- define a simple API for training neural nets, somehow mimicking [Keras](https://keras.io/) and [PyTorch Ignite](https://pytorch.org/ignite/);
- follow good coding practices, including [type annotations](https://www.python.org/dev/peps/pep-0484/) and unit tests.

## Demonstration

The [demo notebook](demo.ipynb) showcases what **pyfit** is all about.

## Features

- Autograd engine [ [source](pyfit/engine.py) | [tests](tests/test_engine.py) ]
- Neural networks API [ [source](pyfit/nn.py) | [tests](tests/test_nn.py) ]
- Metrics [ [source](pyfit/metrics.py) | [tests](tests/test_metrics.py) ]
- Optimizers [ [source](pyfit/optim.py) | [tests](tests/test_optim.py) ]
- Data utilities [ [source](pyfit/data.py) | [tests](tests/test_data.py) ]
- Training API [ [source](pyfit/train.py) | [tests](tests/test_train.py) ]

## Development Notes

### Checking the code

**pyfit** uses the following tools:

- [black](https://github.com/psf/black) for code formatting.
- [pylint](https://www.pylint.org/) and [mypy](http://mypy-lang.org/) for linting.
- [pytest](https://pytest.org) for testing.

Run the following commands in project root folder to check the codebase.

```bash
> python -m pylint pyfit tests  # linting (including type checks)
> python -m mypy .              # type checks only
> python -m pytest              # test suite
```

### Uploading the package to PyPI

```bash
> python setup.py sdist bdist_wheel
> python -m twine upload dist/* --skip-existing
```
