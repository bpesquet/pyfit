"""
K-Nearest Neighbors algorithms
"""

from enum import Enum, auto


class Weights(Enum):
    """

    """
    UNIFORM = auto()
    DISTANCE = auto()


class KNeighborsClassifier():
    """
    K-NN classification algorithm
    """

    def __init__(self, k: int = 5, weights: Weights = Weights.UNIFORM):
        super().__init__()
        self.k = k
        self.weights = weights
