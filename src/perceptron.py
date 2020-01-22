import numpy as np


class Perceptron:
    """ Perception classifier
    Parameters
    ----------
    eta: float, learning rate [0, 1]
    n_iter: int,
    random_state: int
    Attributes
    ----------
    w_: 1d-array, weights after fitting
    erros: list, number of misclassification in each epoch
    """

    def __init__(self, eta: float, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

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
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
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
        return np.where(self.net_input(X) >= 0.0, 1, -1)

