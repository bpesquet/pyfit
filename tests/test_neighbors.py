"""
Unit tests for neighbors-based algorithms
"""

# Docstrings are superfluous for test functions
# pylint: disable=missing-docstring

import numpy as np
from pyfit.neighbors import KNeighborsClassifier, KNeighborsRegressor


def test_knn_classifier_int() -> None:
    samples = np.array([[0], [1], [2], [3]])
    targets = np.array([0, 0, 1, 1])

    clf = KNeighborsClassifier()
    clf.fit(samples, targets)

    predicted = clf.predict(np.array([[1.1]]))
    assert predicted[0] == 0

    predicted = clf.predict(np.array([[2.2], [0.9]]))
    assert predicted[0] == 1
    assert predicted[1] == 0


def test_knn_classifier_str() -> None:
    samples = np.array([[0], [1], [2], [3]])
    targets = np.array(["a", "a", "b", "b"])

    clf = KNeighborsClassifier()
    clf.fit(samples, targets)

    predicted = clf.predict(np.array([[1.1]]))
    assert predicted[0] == "a"

    predicted = clf.predict(np.array([[2.2], [0.9]]))
    assert predicted[0] == "b"
    assert predicted[1] == "a"


def test_knn_regressor() -> None:
    samples = np.array([[0], [1], [2], [3]])
    targets = np.array([0, 0, 1, 1])

    reg = KNeighborsRegressor(k=2)
    reg.fit(samples, targets)

    predicted = reg.predict(np.array([[1.5]]))
    assert predicted[0] == 0.5
