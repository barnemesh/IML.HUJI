from __future__ import annotations
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
    indexes = np.arange(0, X.shape[0], step=1)
    np.random.shuffle(indexes)
    train_score = []
    validation_score = []

    for fold_indexes in np.array_split(indexes, cv):
        X_train = np.delete(X, fold_indexes, axis=0)
        y_train = np.delete(y, fold_indexes, axis=0)

        estimator.fit(X_train, y_train)
        train_score.append(scoring(y_train, estimator.predict(X_train)))
        validation_score.append(scoring(y[fold_indexes], estimator.predict(X[fold_indexes])))

    return np.mean(train_score), np.mean(validation_score)