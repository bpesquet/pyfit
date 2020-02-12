"""
K-Nearest Neighbors algorithms
"""

# Temporarily disable annoying Pylint check
# pylint: disable=too-few-public-methods

from collections import Counter
from typing import List, Union, Tuple
import numpy as np
from pyfit import Tensor, BaseEstimator
from pyfit.metrics import Distance, euclidean_distance


class KNeighborsEstimator(BaseEstimator):
    """
    Abstract K-NN estimator
    """

    def __init__(self, *, k: int = 5, distance: Distance = euclidean_distance) -> None:
        super().__init__()
        self.k = k
        self.distance = distance

    def _sort_neighbors(self, x_new: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """
        Sort all samples by (increasing) distance to new sample
        """
        # Associate training data and targets in a list of tuples
        training_samples: List[Tuple[Tensor, Tensor]] = list(
            zip(self.x_train, self.y_train)
        )

        # Sort samples by their distance to new sample
        sorted_neighbors: List[Tuple[Tensor, Tensor]] = sorted(
            ((x, y) for (x, y) in training_samples),
            key=lambda sample: self.distance(x_new, sample[0]),
        )

        return sorted_neighbors


class KNeighborsClassifier(KNeighborsEstimator):
    """
    K-NN classifier
    """

    def predict(self, x_test: Tensor) -> List[Union[int, str]]:
        """
        Predict results for test samples
        """
        # Return a list containing the prediction for each new sample
        return [self._predict_one(x_new) for x_new in x_test]

    def _predict_one(self, x_new: Tensor) -> Union[int, str]:
        """
        Predict result for a single new sample
        """
        neighbors = self._sort_neighbors(x_new)

        # Keep targets of the k nearest neighbors
        k_nearest_targets: List[Union[int, str]] = [
            target for (_, target) in neighbors[: self.k]
        ]

        # Get most frequent target in nearest neighbors
        winner_target: Union[int, str] = Counter(k_nearest_targets).most_common(1)[0][0]

        return winner_target


class KNeighborsRegressor(KNeighborsEstimator):
    """
    K-NN regressor
    """

    def predict(self, x_test: Tensor) -> List[float]:
        """
        Predict results for test samples
        """
        # Return a list containing the prediction for each new sample
        return [self._predict_one(x_new) for x_new in x_test]

    def _predict_one(self, x_new: Tensor) -> float:
        """
        Predict result for a single new sample
        """
        neighbors = self._sort_neighbors(x_new)

        # Keep targets of the k nearest neighbors
        k_nearest_targets: List[float] = [target for (_, target) in neighbors[: self.k]]

        # Compute the mean target value
        target_mean = np.mean(k_nearest_targets)

        return target_mean
