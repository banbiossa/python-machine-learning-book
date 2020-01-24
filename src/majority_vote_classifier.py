from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import six


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote="classlabel", weights=None):
        """A majority vote ensemble classifier
        
        Parameters
        ----------
        classifiers : [array-like], shape=[n_classifiers]
            Different classifiers for enemble
            [description]
        vote : str, {'classlabel', 'probability'}
            by default 'classlabel'
            'classlabel': prediciton based on argmax of class labels.
            'probability': argmax of sum or probabilities
        weights : [array-like], shape= [n_classifiers]
            by default None
        """
        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """Fit classifiers
        
        Parameters
        ----------
        X : {array-like, sparse matrix}
            shape = [n_samples, n_features]
            Matrix of training examples
        y : array-like, shape=[n_samples]
            Vector of target class examples
        
        Returns
        -------
        self: object
        """
        # use label encoder to ensure class labels star
        # with 0, which is important for np.argmax
        # call in self.predict
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """Predict class labels for X
        
        Parameters
        ----------
        X : {array-like, sparse_matrix},
            Shape = [n_samples, n_features]
            Matrix of training samples
        
        Returns
        -------
        maj_vote: array-like, shape=[n_samples]
            Predicted class labels
        """
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
            maj_vote = self.labelenc_.inverse_transform(maj_vote)
            return maj_vote

        if self.vote == "classlabel":  # class label vote
            # collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions,
            )
            maj_vote = self.labelenc_.inverse_transform(maj_vote)
            return maj_vote

        # not implemtented
        raise NotImplementedError(f"Vote type {self.vote} is not implemtend")

    def predict_proba(self, X):
        """Predict class probabilities for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}
            Training vectors, shape = [n_samples, n_features]
        
        Returns
        -------
        avg_proba: array-like,
            shape = [n_samples, n_classes]
            Weighted average probability of each class for each sample
        """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """Get classifier parameters names for Gridsearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)

        # else
        out = self.named_classifiers.copy()
        for name, step in six.iteritems(self.named_classifiers):
            for key, value in six.iteritems(step.get_params(deep=True)):
                out[f"{name}__{key}"] = value
        return out

