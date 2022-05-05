from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self):
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape

        cur_err = 2
        for i in range(n_features):
            for s in {-1, 1}:
                thr, thr_err = self._find_threshold(X[:, i], y, s)
                if thr_err < cur_err:
                    self.threshold_ = thr
                    self.j_ = i
                    self.sign_ = s
                    cur_err = thr_err

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        feature = X[:, self.j_] - self.threshold_
        return self.sign_ * (np.sign(feature) + (feature == 0))

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # sort the values, and rearrange the labels accordingly
        sort_idx = np.argsort(values)
        values, labels = values[sort_idx], labels[sort_idx]

        #  calculate the loss for each threshold=values[i]
        threshes = np.concatenate(
            [[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        minimal_theta_loss = np.sum(np.abs(labels[np.sign(labels) != sign]))
        losses = np.append(minimal_theta_loss,
                           minimal_theta_loss - np.cumsum(labels * sign))

        # return the loss-minimizer threshold and its loss
        min_loss_idx = np.argmin(losses)
        return threshes[min_loss_idx], losses[min_loss_idx]
        #
        # # ind = np.argsort(values)
        # # x = values[ind]
        # cur_thr = 0
        # cur_thr_err = 2
        # for i in range(values.shape[0]):
        #     feature = values - values[i]
        #     miss = np.abs(labels) * (np.sign(labels) != sign * (
        #             np.sign(feature) + (feature == 0)))
        #     thr_err = np.sum(miss)
        #     # thr_err = misclassification_error(labels, sign * (np.sign(feature) + (feature == 0)))
        #     if thr_err < cur_thr_err:
        #         cur_thr = values[i]
        #         cur_thr_err = thr_err
        # return cur_thr, cur_thr_err

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
        return misclassification_error(y, self._predict(X))
