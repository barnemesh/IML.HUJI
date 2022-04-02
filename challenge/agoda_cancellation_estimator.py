from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

PROB_LIMIT = 0.002

class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self):
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge
        Parameters
        ----------
        Attributes
        ----------
        """
        super().__init__()
        # classification mode
        self.forest = RandomForestClassifier()
        self.logistic = LogisticRegression(max_iter=100000)
        self.neural = MLPClassifier()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Notes
        -----
        """
        # classification mode
        self.forest.fit(X, y)
        self.logistic.fit(X, y)
        self.neural.fit(X, y)

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

        # classification mode
        # pred1 = self.forest.predict(X)
        # pred2 = self.logistic.predict(X)
        # pred3 = self.neural.predict(X)
        pred1 = self.forest.predict_proba(X)
        pred2 = self.logistic.predict_proba(X)
        pred3 = self.neural.predict_proba(X)
        result = []

        for (bol, bol1, bol2) in zip(pred1, pred2, pred3):
            if bol[1] > PROB_LIMIT:
                vote1 = False
            else:
                vote1 = True

            if bol1[1] > PROB_LIMIT:
                vote2 = False
            else:
                vote2 = True

            if bol2[1] > PROB_LIMIT:
                vote3 = False
            else:
                vote3 = True

            result.append((vote1 and (vote2 or vote3) or (vote2 and vote3)))

        return np.array(result)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under loss function
        """
        y_pred = self.predict(X)
        result = 0
        for (pred, true) in zip(y_pred, y):
            if pred == true:
                result += 1

        return result / y.size