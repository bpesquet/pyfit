[![Build Status](https://travis-ci.org/bpesquet/pyfit.svg?branch=master&logo=travis-ci&logoColor=white)](https://travis-ci.org/bpesquet/pyfit)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyfit.svg)](https://pypi.org/project/pyfit)
[![PyPI Version](https://img.shields.io/pypi/v/pyfit.svg)](https://pypi.org/project/pyfit)
[![PyPI status](https://img.shields.io/pypi/status/pyfit.svg)](https://pypi.python.org/project/pyfit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# pyfit

**pyfit** is a minimalist neural networks library written from scratch in Python for educational purposes.

## Overview

This project aims to:

- help Machine Learning students and enthusiasts get a deeper understanding of neural networks ;
- demonstrate [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), a core concept of modern Deep Learning frameworks like PyTorch and TensorFlow ;
- define a clean, pythonic API and follow good coding practices, including [type annotations](https://www.python.org/dev/peps/pep-0484/) and unit tests.

> This material is used in the Machine Learning course taught at [ENSC](https://ensc.bordeaux-inp.fr). [ENSEIRB-MATMECA](https://enseirb-matmeca.bordeaux-inp.fr) and [IOGS](https://www.institutoptique.fr). See also [Acknowledgments](ACKNOWLEDGMENTS.md).

## Demonstration

See the [demo notebook](demo.ipynb).

## Features

- Autograd [ [source](pyfit/engine.py) | [tests](tests/test_engine.py) ]
- Neural Networks API [ [source](pyfit/nn.py) | tests (soon!) ]
- Losses [ [source](pyfit/loss.py) | [tests](tests/test_loss.py) ]
- Optimizers [ [source](pyfit/optim.py) | tests (soon!) ]
- Data Utilities [ [source](pyfit/data.py) | tests (soon!) ]
- Metrics (soon!)
- Training (soon!)

## Development Notes

### Checking the code

**pyfit** uses the following tools:

- [black](https://github.com/psf/black) for code formatting.
- [pylint](https://www.pylint.org/) and [mypy](http://mypy-lang.org/) for linting.
- [pytest](https://pytest.org) for testing.

Run the following commands in project root folder to check the codebase.

```bash
> python -m pylint ./pyfit # linting (including type checks)
> python -m mypy .         # type checks only
> python -m pytest         # test suite
```

### Uploading the package to PyPI

```bash
> python setup.py sdist bdist_wheel
> python -m twine upload dist/* --skip-existing
```
