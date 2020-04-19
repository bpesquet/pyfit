"""
Training API
"""

from pyfit.engine import Scalar
from pyfit.nn import Module
from pyfit.optim import Optimizer
from pyfit.loss import Loss
from pyfit.data import BatchIterator


class Trainer:
    """Encapsulates the model training loop"""

    def __init__(self, model: Module, optimizer: Optimizer, loss: Loss):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def fit(
        self, data_iterator: BatchIterator, num_epochs=500, verbose=False
    ) -> Tuple(float, float):
        """Fits the model to the data"""

        epoch_loss = Scalar(0)
        for epoch in range(num_epochs):
            # Reset the gradients of model parameters
            self.optimizer.zero_grad()
            epoch_loss = Scalar(0)

            for batch in data_iterator():
                # Forward pass
                outputs = list(map(self.model, batch.inputs))

                # Loss computation
                y_pred = [item for sublist in outputs for item in sublist]
                batch_loss = self.loss(y_pred, batch.targets)
                epoch_loss += batch_loss

                # Backprop and gradient descent
                batch_loss.backward()
                self.optimizer.step()

            if verbose:
                print(f"Epoch {epoch}, loss = {epoch_loss.data}")

        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return epoch_loss.data, sum(accuracy) / len(accuracy)
