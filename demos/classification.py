"""
Demos of neighbors-based algorithms
"""


# Docstring are superfluous for demo functions
# pylint: disable=missing-docstring

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from pyfit.neighbors import KNeighborsClassifier


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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=1111)

    clf = KNeighborsClassifier(k=4)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(predictions)


if __name__ == "__main__":
    demo_knn_classifier()
