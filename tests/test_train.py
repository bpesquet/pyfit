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
from pyfit.train import Trainer, History


def test_trainer() -> None:
    # dataset for AND logical function
    x_train: List[Vector] = [
        list(map(Scalar, x)) for x in [[0, 0], [0, 1], [1, 0], [1, 1]]
    ]
    y_train: Vector = [Scalar(0), Scalar(0), Scalar(0), Scalar(1)]

    model = MLP(2, [1])  # Logistic regression
    optimizer = SGD(model.parameters(), learning_rate=0.1)
    data_iterator = BatchIterator(x_train, y_train)

    num_epochs = 50
    trainer = Trainer(model, optimizer, loss=mean_squared_error)
    history: History = trainer.fit(data_iterator, num_epochs=num_epochs, verbose=False)

    # Training metrics are recorded for each epoch
    assert len(history["loss"]) == len(history["acc"]) == num_epochs

    # Access final values for metrics
    loss = history["loss"][-1]
    acc = history["acc"][-1]
    assert loss < 0.1
    assert acc == 1.0  # 100%
