import numpy as np


class LinearRegressionGD:
    """ Linear regression
    Parameters
    ----------
    eta: float, learning rate [0, 1]
    n_iter: int,

    Attributes
    ----------
    w_: 1d-array, weights after fitting
    cost_: list, cost in each epoch
    """

    def __init__(self, eta: float = 0.001, n_iter: int = 20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data
        Parameters
        ----------
        X : [array-like], shape=[n_samples, n_features]
            Training vectors
        y : [array-like], shape=[n_samples]
            target values.
        Returns
        -------
        self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input
        Parameters
        ----------
        X : [array-like]
            [description]
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step
        Parameters
        ----------
        X : [type]
            [description]
        """
        return self.net_input(X)

