from functools import partial
from random import gauss

from optimizer.GD import GD


class LinearRegression:
    """
    Gradient descent linear regression.
    :ivar _optimizer: Optimizer instance.
    :ivar _loss: Loss function.
    :ivar _weights: Regression coefficients.
    """

    def __init__(self, inputs):
        """
        Create a LinearRegression object with standard normal distribution weights.
        :param inputs: The input shape of the model.
        """
        self._optimizer = None
        self._loss = None
        self._weights = [gauss(0, 1) for _ in range(inputs)]

    def compile(self, optimizer=GD(), loss=None):
        """
        Configures the model for training.
        :param optimizer: Optimizer instance.
        :param loss: Loss function.
        """
        self._optimizer = optimizer
        self._loss = loss

    def fit(self, x, y):
        """
        Train the model by optimizer and return loss history.
        :param x: Input features.
        :param y: Input labels.
        :return: Loss history.
        """
        epoch = 0

        while True:
            epoch += 1

            self._optimizer.minimize(partial(self._loss, x, y), self._weights)

            print("Epoch {} loss: {}".format(epoch, self._optimizer.loss[-1]))

            if len(self._optimizer.loss) >= 2 and abs(
                    self._optimizer.loss[-1] - self._optimizer.loss[-2]) < 1e-16:
                break

        return self._optimizer.loss

    def evaluate(self, x, y):
        """
        Return the loss value for the model in test mode.
        :param x: Input data.
        :param y: Target data.
        :return: Test loss.
        """
        return self._loss(x, y, self._weights)

    def predict(self, x):
        """
        Generates output predictions for the input samples.
        :param x: Input samples.
        :return: Prediction.
        """
        return [sum(attr * weight for attr, weight in zip(attrs, self._weights)) for attrs in x]
