class GD:
    """
    Optimizer that implements the Gradient descent algorithm.
    :ivar lr: Learning rate.
    :ivar loss: loss history.
    """

    def __init__(self, lr=0.01):
        """
        Create a GD object with learning rate and loss history.
        :param lr: Learning rate.
        """
        self.lr = lr
        self.loss = []

    def minimize(self, loss, var_list=None):
        """
        Minimize loss by updating var_list.
        :param loss: Loss function.
        :param var_list: Trainable variables.
        """
        grads_and_vars = self.compute_gradients(loss, var_list)

        var_list.clear()
        var_list.extend(self.apply_gradients(grads_and_vars))

    def compute_gradients(self, loss, var_list=None):
        """
        Compute gradients of loss on trainable variables.
        :param loss: Loss function.
        :param var_list: Trainable variables.
        :return: A list of (gradient, variable) pairs.
        """
        self.loss.append(loss(var_list))

        return list(zip(gradients(loss, var_list), var_list))

    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients to variables.
        :param grads_and_vars: List of (gradient, variable) pairs.
        :return: Representing the current iteration.
        """
        return [var - self.lr * grad for grad, var in grads_and_vars]


def gradients(ys, xs):
    """
    Calculate ∂y/∂x for each x and y in xs and ys.
    :param ys: Dependent variables.
    :param xs: Independent variables.
    :return: Gradients.
    """
    features, targets = ys.args

    return [sum((label - sum(x * weight for x, weight in zip(feature, xs))) * -feature[i] for feature, label in
                zip(features, targets)) * 2 / len(targets) for i in range(len(xs))]
