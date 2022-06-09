from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # n_samples, n_features = X.shape
    folds = np.array_split(X, cv)  # TODO: split indices instead, and use them?
    labels = np.array_split(y, cv)
    train_scores = []
    validation_scores = []
    for i in range(cv):
        folds_without_i = folds[:i] + folds[i+1:]
        labels_without_i = labels[:i] + labels[i+1:]
        x_i = np.concatenate(folds_without_i)
        y_i = np.concatenate(labels_without_i)
        estimator.fit(x_i, y_i)
        train_scores.append(scoring(y_i, estimator.predict(x_i)))
        validation_scores.append(scoring(labels[i], estimator.predict(folds[i])))

    return np.mean(train_scores), np.mean(validation_scores)
