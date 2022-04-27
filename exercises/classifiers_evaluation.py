import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        # callback = lambda p, s, r: losses.append(p.loss(X, y))
        def callback(p: Perceptron, s: np.ndarray, r: int):
            losses.append(p.loss(X, y))

        model = Perceptron(callback=callback)
        model.fit(X, y)

        fig = go.Figure(
            layout=go.Layout(
                margin=dict(t=80),
                title=f"Perceptron loss as function of iterations<br>{n} data",
                xaxis=dict(title="Iteration"),
                yaxis=dict(title="Loss"),
                width=800, height=600
            ),
            data=[
                go.Scatter(
                    y=losses,
                    mode="lines",
                    showlegend=False
                )
            ])
        fig.write_image("./Plots/Ex3/" + n + ".png")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_predict = lda.predict(X)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        gnb_predict = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"predictions using GNB<br>accuracy score={np.round(accuracy(y, gnb_predict), 3)}",
                f"predictions using LDA<br>accuracy score={np.round(accuracy(y, lda_predict), 3)}"],
        )
        fig.add_traces(
            [
                go.Scatter(x=X[:, 0],
                           y=X[:, 1],
                           showlegend=False,
                           mode="markers",
                           marker=dict(color=gnb_predict,
                                       symbol=y,
                                       colorscale="sunsetdark")
                           ),
                go.Scatter(x=X[:, 0],
                           y=X[:, 1],
                           showlegend=False,
                           mode="markers",
                           marker=dict(color=lda_predict,
                                       symbol=y,
                                       colorscale="sunsetdark")
                           )
            ],
            rows=1, cols=[1, 2]
        )

        # Add traces for data-points setting symbols and colors
        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces(
            [
                go.Scatter(x=gnb.mu_[:, 0],
                           y=gnb.mu_[:, 1],
                           mode="markers",
                           showlegend=False,
                           marker=dict(color="Black",
                                       symbol="x"),
                           ),
                go.Scatter(x=lda.mu_[:, 0],
                           y=lda.mu_[:, 1],
                           mode="markers",
                           showlegend=False,
                           marker=dict(color="Black",
                                       symbol="x"),
                           )
            ],
            rows=1, cols=[1, 2]
        )

        # Add ellipses depicting the covariances of the fitted Gaussians
        traces = []
        for mu in lda.mu_:
            traces.append(get_ellipse(mu, lda.cov_))
        fig.add_traces(traces, rows=1, cols=2)
        traces = []
        for var, mu in zip(gnb.vars_, gnb.mu_):
            traces.append(get_ellipse(mu, np.diagflat(var)))
        fig.add_traces(traces, rows=1, cols=1)

        fig.update_layout(title_text=f,
                          margin=dict(t=100),
                          width=1200,
                          height=600)
        fig.show()
        fig.write_image(f"./Plots/Ex3/lda_gauss_predict_{f}.png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
