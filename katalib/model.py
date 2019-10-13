"""
Models combine layers to make predictions on data.
"""

from katalib.loss import MSE
from katalib.optimizer import SGD


class Model:
    """
    A model is a linear stack of layers.
    """

    def __init__(self, layers):
        self.layers = [layers]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def train(self,
              inputs,
              targets,
              num_epochs=5000,
              loss=MSE(),
              optimizer=SGD()):
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in iterator(inputs, targets):
                predicted = self.forward(batch.inputs)
                epoch_loss += loss.loss(predicted, batch.targets)
                grad = loss.grad(predicted, batch.targets)
                self.backward(grad)
                optimizer.step(net)
            print(epoch, epoch_loss)
