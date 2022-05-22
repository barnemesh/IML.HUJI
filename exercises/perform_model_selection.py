from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    def f(x):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    samples = np.linspace(-1.2, 2, n_samples)
    labels = f(samples)
    eps = np.random.normal(loc=0, scale=noise, size=n_samples)
    labels += eps

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(samples), pd.Series(labels), 2.0 / 3.0)
    X_train, y_train = X_train.to_numpy()[:, 0], y_train.to_numpy()
    X_test, y_test = X_test.to_numpy()[:, 0], y_test.to_numpy()

    fig = go.Figure(
        data=[
            go.Scatter(
                x=X_train,
                y=f(X_train),
                mode="markers",
                name="Train Samples",
                marker=dict(color=custom[0][0])
            ),
            go.Scatter(
                x=X_test,
                y=f(X_test),
                mode="markers",
                name="Test Samples",
                marker=dict(color=custom[0][-1])
            )
        ],
        layout=go.Layout(
            title="Q2.1.1",
            xaxis=dict(title="samples"),
            yaxis=dict(title="Noiseless")
        )
    )
    fig.write_image(f"./Plots/Ex5/TrueNoiselessModelSplitNoise{noise}.png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors = []
    validation_errors = []
    for k in range(11):
        est = PolynomialFitting(k)
        t_error, v_error = cross_validate(est, X_train, y_train, mean_square_error)
        train_errors.append(t_error)
        validation_errors.append(v_error)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=[k for k in range(11)],
                y=train_errors,
                mode="markers",
                name="Train Error Average",
                marker=dict(color=custom[0][0])
            ),
            go.Scatter(
                x=[k for k in range(11)],
                y=validation_errors,
                mode="markers",
                name="Validation Error Average",
                marker=dict(color=custom[0][-1])
            )
        ],
        layout=go.Layout(
            title="Q2.1.2",
            xaxis=dict(title="K deg of polynomial"),
            yaxis=dict(title="MSE")
        )
    )
    fig.write_image(f"./Plots/Ex5/Polynomial5FoldErrorsNoise{noise}.png")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = np.argmin(validation_errors)
    est = PolynomialFitting(k)
    est.fit(X_train, y_train)
    test_error = mean_square_error(y_test, est.predict(X_test))
    print(f"\nFor n_samples={n_samples}, noise={noise}:")
    print(f"Minimum validation error was for deg {k}.")
    print(f"Deg {k} MSE: {np.round(test_error, 2)}.")
    print(f"Deg {k} Validation MSE: {np.round(validation_errors[k], 2)}.")



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    np.random.seed(0)
    select_polynomial_degree(noise=0)
    np.random.seed(0)
    select_polynomial_degree(n_samples=1500, noise=10)
