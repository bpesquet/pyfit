[![Build Status](https://travis-ci.org/bpesquet/pyfit.svg?branch=master&logo=travis-ci&logoColor=white)](https://travis-ci.org/bpesquet/pyfit)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyfit.svg)](https://pypi.org/project/pyfit)
[![PyPI Version](https://img.shields.io/pypi/v/pyfit.svg)](https://pypi.org/project/pyfit)
[![PyPI status](https://img.shields.io/pypi/status/pyfit.svg)](https://pypi.python.org/project/pyfit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# pyfit

**pyfit** is a simple Machine Learning library built with Python and [NumPy](https://numpy.org/) for educational purposes.

## Overview

This project's main goal is to help ML students and enthusiasts get a deeper understanding of the Machine Learning workflow and main algorithms, by implementing them from scratch.

As a Python package, it also strives to define a clean, pythonic API and follow good coding practices, including [type annotations](https://www.python.org/dev/peps/pep-0484/).

## Status

**pyfit** is currently in alpha status. See [Progress](https://github.com/bpesquet/pyfit/projects/1) for details.

## Content

- Data Preprocessing [ [source](pyfit/preprocessing.py) | [tests](tests/test_preprocessing.py) ]
- Metrics [ [source](pyfit/metrics/) | [tests](tests/test_metrics.py) ]
- Plotting [ [source](pyfit/plot.py) ]
- K-Nearest Neighbors [ [source](pyfit/neighbors.py) | [tests](tests/test_neighbors.py) ]
- Neural Networks [ [source](pyfit/nn.py) ]
- ... More to come!

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
