"""
Demos of neighbors-based algorithms
"""


# Docstring are superfluous for demo functions
# pylint: disable=missing-docstring

from sklearn.datasets import make_classification
from pyfit.preprocessing import train_test_split
from pyfit.neighbors import KNeighborsClassifier
from pyfit.metrics import accuracy


def demo_knn_classifier() -> None:
    x, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        random_state=1111,
        class_sep=1.5,
    )

    x_train, x_test = train_test_split(x, test_ratio=0.1)
    y_train, y_test = train_test_split(y, test_ratio=0.1)

    clf = KNeighborsClassifier(k=4)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(f'K-NN classifier accuracy: {accuracy(y_test, y_pred) * 100}%')


if __name__ == "__main__":
    demo_knn_classifier()
