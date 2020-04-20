"""
Unit tests for training API
"""

# pylint: disable=missing-docstring

from typing import List
from pyfit.engine import Scalar, Vector
from pyfit.nn import MLP
from pyfit.optim import SGD
from pyfit.metrics import mean_squared_error
from pyfit.data import BatchIterator
from pyfit.train import Trainer


def test_trainer() -> None:
    # dataset for AND logical function
    x_train: List[Vector] = [
        list(map(Scalar, x)) for x in [[0, 0], [0, 1], [1, 0], [1, 1]]
    ]
    y_train: Vector = [Scalar(0), Scalar(0), Scalar(0), Scalar(1)]

    model = MLP(2, [1])  # Logistic regression
    optimizer = SGD(model.parameters(), learning_rate=0.1)
    data_iterator = BatchIterator(x_train, y_train)

    trainer = Trainer(model, optimizer, loss=mean_squared_error)
    loss, acc = trainer.fit(data_iterator, num_epochs=50, verbose=False)

    assert loss < 0.1
    assert acc == 1.0  # 100%
