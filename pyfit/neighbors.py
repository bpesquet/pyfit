"""
K-Nearest Neighbors algorithms
"""

from collections import Counter
from typing import List, Any
import numpy as np
from pyfit.base import BaseEstimator
from pyfit.metrics import Distance, euclidean_distance


class KNeighborsEstimator(BaseEstimator):
    """
    Abstract K-NN estimator
    """

    def __init__(self, k: int = 5, distance: Distance = euclidean_distance) -> None:
        super().__init__()
        self.k = k
        self.distance = distance

    def predict(self, new_samples: np.ndarray) -> List[Any]:
        """
        Predict results for new samples
        """
        # Return a list containing the prediction for each new sample
        return [self._predict_one(s) for s in new_samples]

    def _predict_one(self, new_sample: np.ndarray) -> Any:
        """
        Predict result for a single new sample
        """
        # Associate samples and targets
        samples_and_targets = zip(self.samples, self.targets)

        # Sort all samples by their distance to x
        neighbors = sorted(((samples, target) for (samples, target) in samples_and_targets),
                           key=lambda tuple: self.distance(new_sample, tuple[0]))

        # Keep targets of the k nearest neighbors
        k_nearest_targets = [target for (_, target) in neighbors[:self.k]]

        # Let derived class process the nearest neighbors
        return self._compute_prediction(k_nearest_targets)

    def _compute_prediction(self, nearest_targets: List[Any]) -> Any:
        """
        Process nearest targets to compute the prediction
        """
        raise NotImplementedError()


class KNeighborsClassifier(KNeighborsEstimator):
    """
    K-NN classifier
    """

    def _compute_prediction(self, nearest_targets: List[Any]) -> Any:
        # Get most frequent target in nearest neighbors
        winner_target = Counter(nearest_targets).most_common(1)[0][0]
        return winner_target


class KNeighborsRegressor(KNeighborsEstimator):
    """
    K-NN regressor
    """

    def _compute_prediction(self, nearest_targets: List[float]) -> float:
        # Compute the mean target value
        return np.mean(nearest_targets)
