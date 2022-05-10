import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y) = generate_data(train_size, noise)
    (test_X, test_y) = generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(lambda: DecisionStump(), n_learners)
    model.fit(train_X, train_y)

    test_errors = []
    train_errors = []

    for i in range(1, n_learners + 1):
        test_errors.append(model.partial_loss(test_X, test_y, i))
        train_errors.append(model.partial_loss(train_X, train_y, i))

    fig = go.Figure(
        data=[
            go.Scatter(y=test_errors,
                       x=list(range(1, n_learners + 1)),
                       mode="lines",
                       name="Test loss"
                       ),
            go.Scatter(y=train_errors,
                       x=list(range(1, n_learners + 1)),
                       mode="lines",
                       name="Train loss"
                       )
        ],
        layout=go.Layout(
            width=1000,
            height=600,
            title=f"AdaBoost loss based on number of learners, noise={noise}",
            xaxis=dict(title="Number of learners"),
            yaxis=dict(title="loss")
        )
    )

    # fig.write_image(f"./Plots/Ex4/AdaLossLearnerNumberNoise{noise}.png")
    fig.show()

    # Question 2: Plotting decision surfaces
    symbols = np.array(["circle", "x"])
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{i} learners" for i in T],
                        horizontal_spacing=0.05, vertical_spacing=0.05
                        )

    m = go.Scatter(x=test_X[:, 0],
                   y=test_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   name="Label 1",
                   marker=dict(color=(test_y == 1).astype(int),
                               symbol=class_symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black",
                                         width=1)))
    for i, t in enumerate(T):
        fig.add_traces(
            [
                decision_surface(lambda x: model.partial_predict(x, t),
                                 lims[0],
                                 lims[1],
                                 showscale=False), m
            ],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        width=800,
        height=900,
        title=rf"$\textbf{{Decision Boundaries Of AdaBoost based on number of learners: noise={noise}}}$",
        margin=dict(t=100)
    ).update_xaxes(
        matches='x',
        range=[-1, 1],
        constrain="domain"
    ).update_yaxes(
        matches='y',
        range=[-1, 1],
        constrain="domain",
        scaleanchor="x",
        scaleratio=1
    )

    # fig.write_image(f"./Plots/Ex4/AdaBoostDecisionBoundariesNoise{noise}.png")
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    i = np.argmin(test_errors)
    from IMLearn.metrics import accuracy
    fig = go.Figure(
        data=[
            decision_surface(lambda x: model.partial_predict(x, i + 1),
                             lims[0],
                             lims[1],
                             showscale=False), m
        ],
        layout=go.Layout(
            width=600,
            height=600,
            title=f"AdaBoost lowest loss ensemble. noise={noise}.<br>"
                  f"Ensemble size={i + 1} , "
                  f"Accuracy={accuracy(test_y, model.partial_predict(test_X, i + 1)):.3f}",
            margin=dict(t=100)
        )
    )
    fig.update_xaxes(
        range=[-1, 1],
        constrain="domain"
    ).update_yaxes(
        range=[-1, 1],
        constrain="domain",
        scaleanchor="x",
        scaleratio=1
    )
    # fig.write_image(f"./Plots/Ex4/AdaBoostBestEnsembleNoise{noise}.png")
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure(
        data=[
            decision_surface(model.predict,
                             lims[0],
                             lims[1],
                             showscale=False),
            go.Scatter(x=train_X[:, 0],
                       y=train_X[:, 1],
                       mode="markers",
                       showlegend=False,
                       marker=dict(color=(train_y == 1).astype(int),
                                   size=(model.D_ / np.max(model.D_)) * 6,
                                   # size=model.D_,
                                   # sizemode='area',
                                   # sizeref=2.*np.max(model.D_)/(6**2),
                                   symbol=class_symbols[train_y.astype(int)],
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=1))
                       )
        ],
        layout=go.Layout(
            width=600,
            height=600,
            title=f"AdaBoost train set with sample sized by weights, noise={noise}.",
        )
    )
    fig.update_xaxes(
        range=[-1, 1],
        constrain="domain"
    ).update_yaxes(
        range=[-1, 1],
        constrain="domain",
        scaleanchor="x",
        scaleratio=1
    )
    # fig.write_image(f"./Plots/Ex4/AdaBoostFullTrainWithWeights{noise}.png")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
