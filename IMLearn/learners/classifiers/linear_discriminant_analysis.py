from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
            self.mu_[i] = np.sum(X[(y == c)], axis=0) / counts[i]

        z = np.copy(X)
        for i in range(n_samples):  # TODO :?????
            a = np.take(self.classes_, y[i])
            z[i] -= self.mu_[a]

        self.cov_ = np.cov(z, rowvar=False, ddof=counts.shape[0])
        self._cov_inv = inv(self.cov_)

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
        # TODO: if likelihood adds back normalization -
        #  then recalculate without?
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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # log(pi) + X @ _cov_inv @ mu.T - diag(0.5 * mu @ cov_inv @ mu)
        log_pi = np.log(self.pi_)
        a = np.einsum("mn,nj,ji->mi", X, self._cov_inv, self.mu_.T)
        b = 0.5 * np.einsum("ij,jk,ki->i", self.mu_, self._cov_inv, self.mu_.T)
        # TODO: add back the 1/Z normalization factor?????
        # z = np.log(np.sqrt(np.pow(2 * np.pi, n_features) * det(self.cov_)))
        return log_pi + a - b  # - z

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
