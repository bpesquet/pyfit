"""
Core definitions for models and training
"""

import numpy as np


class BaseEstimator:
    """
    Base abstract class for all estimators
    """

    samples: np.ndarray = None
    targets: np.ndarray = None

    def fit(self, samples: np.ndarray, targets: np.ndarray) -> None:
        """
        Stores samples and targets for use in derived classes
        """
        self.samples = samples
        self.targets = targets
