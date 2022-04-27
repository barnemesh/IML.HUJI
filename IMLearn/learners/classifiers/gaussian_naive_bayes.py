from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / n_samples

        self.mu_ = np.zeros((self.classes_.shape[0], n_features))
        for i, c in enumerate(self.classes_):
            self.mu_[i] = np.mean(X[(y == c)], axis=0)

        self.vars_ = np.zeros((counts.shape[0], n_features))
        for i, c in enumerate(self.classes_):
            self.vars_[i] = np.var(X[(y == c)], ddof=1, axis=0)

        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.take(self.classes_, np.argmax(self.likelihood(X), axis=1))

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        n_samples, n_features = X.shape

        log_pi = np.log(self.pi_)
        log_sig = 0.5 * np.sum(np.log(self.vars_), axis=1)
        log_s2pi = n_features * 0.5 * np.log(2 * np.pi)
        diag_var = np.asarray([np.diagflat(var) for var in self.vars_])

        z = np.zeros((n_samples, self.classes_.shape[0]))
        for i, mu in enumerate(self.mu_):
            inv_var = np.linalg.inv(diag_var[i])
            z[:, i] = 0.5 * np.einsum("bi,ij,bj->b", X-mu, inv_var, X-mu)
        y = log_pi - log_sig - log_s2pi - z

        return y

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
