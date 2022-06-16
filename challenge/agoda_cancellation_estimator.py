from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    HistGradientBoostingClassifier, \
    ExtraTreesClassifier, BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, balanced: bool = False, voting="soft"):
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge
        Parameters
        ----------
        Attributes
        ----------
        """
        super().__init__()
        self.balanced = balanced
        self.PROB_LIMIT = 0.5
        self.PROB_LIMIT1 = 0.5
        self.PROB_LIMIT2 = 0.5

        if balanced:
            # TODO: try grid search
            self.estimator = RandomForestClassifier(class_weight="balanced", ccp_alpha=0.0001)
            # self.estimator2 = AdaBoostClassifier(n_estimators=75, algorithm="SAMME")
            self.estimator2 = BaggingClassifier(n_estimators=20, max_samples=0.75, max_features=0.75, bootstrap_features=True)
            self.estimator3 = ExtraTreesClassifier(class_weight="balanced", ccp_alpha=0.0001)
        else:
            estimators = [('rf', RandomForestClassifier(class_weight="balanced", ccp_alpha=0.0001)),
                          ('bag', BaggingClassifier(n_estimators=20, max_samples=0.75, max_features=0.75, bootstrap_features=True)),
                          ('et', ExtraTreesClassifier(class_weight="balanced", ccp_alpha=0.0001))
                          ]
            self.estimator = VotingClassifier(estimators=estimators, voting=voting)

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
        self.estimator.fit(X, y)
        if self.balanced:
            self.estimator2.fit(X, y)
            self.estimator3.fit(X, y)

    def _predict(self, X: np.ndarra) -> np.ndarray:
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
        if not self.balanced:
            return self.estimator.predict(X)
        pred1 = self.estimator.predict_proba(X)  # [:, 1] >= self.PROB_LIMIT
        pred2 = self.estimator2.predict_proba(X)  # [:, 1] >= self.PROB_LIMIT1
        pred3 = self.estimator3.predict_proba(X)  # [:, 1] >= self.PROB_LIMIT2

        result = []
        for (bol, bol1, bol2) in zip(pred1, pred2, pred3):
            if bol[1] > self.PROB_LIMIT:
                vote1 = True
            else:
                vote1 = False

            if bol1[1] > self.PROB_LIMIT1:
                vote2 = True
            else:
                vote2 = False

            if bol2[1] > self.PROB_LIMIT2:
                vote3 = True
            else:
                vote3 = False

            result.append((vote1 and (vote2 or vote3) or (vote2 and vote3)))

        return np.array(result)

    def set_probs(self, p1, p2, p3):
        self.PROB_LIMIT = p1
        self.PROB_LIMIT1 = p2
        self.PROB_LIMIT2 = p3

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
