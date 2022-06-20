import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

from utils import custom


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    gd_values = []
    gd_weights = []

    def callback(solver: GradientDescent,
                 weights: np.ndarray,
                 val: np.ndarray,
                 grad: np.ndarray,
                 t: int, eta: float, delta: float):
        gd_values.append(val)
        gd_weights.append(weights)

    return callback, gd_values, gd_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for mod, name in [(L1, "L1"), (L2, "L2")]:
        lowest = []
        mod_fig = go.Figure()
        for eta in etas:
            module = mod(init)
            callback, gd_values, gd_weights = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta), callback=callback)
            last = gd.fit(module, None, None)
            fig = plot_descent_path(
                mod,
                np.array(gd_weights),
                rf"$\textbf{{Q.3.1.1.1}}-\text{{Decent Path For {name} With }}\eta = {eta}$"
            )
            fig.write_image(f"./Plots/Ex6/DecentPathFor{name}WithEta{eta}.png", scale=2)
            trace = go.Scatter(
                x=list(range(len(gd_values))),
                y=gd_values,
                mode="markers+lines",
                name=rf"$\text{{Convergence Rate of {name} With }}\eta = {eta}$"
            )
            fig = go.Figure(
                data=[trace],
                layout=go.Layout(
                    title=rf"$\textbf{{Q.3.1.1.2}}-\text{{Convergence Rate of {name} With }}\eta = {eta}$",
                    xaxis=dict(title=r"$\text{Iteration}$"),
                    yaxis=dict(title=rf"$\text{{Norm {name}}}$")
                )
            )
            fig.write_image(f"./Plots/Ex6/ConvergenceRateFor{name}WithEta{eta}.png", scale=2)
            mod_fig.add_trace(trace)
            lowest.append(np.min(gd_values))
            print(f"{name} achieved lowest loss: {lowest[-1]}. for eta={eta}. iteration={len(gd_values)}")
        print(f"{name} achieved lowest loss overall: {np.min(lowest)}")
        mod_fig.update_layout(
            title=rf"$\textbf{{Q.3.1.1.2}}-\text{{Convergence Rate of {name}}}$",
            xaxis=dict(title=r"$\text{Iteration}$"),
            yaxis=dict(title=rf"$\text{{Norm {name}}}$")
        )
        mod_fig.write_image(f"./Plots/Ex6/ConvergenceRateFor{name}.png", scale=2)
        mod_fig.update_layout(
            xaxis=dict(title=r"$\text{Iteration - Log Scale}$"),
        ).update_xaxes(
            type="log",
            range=(0, np.log10(1000)),
            constrain="domain"
        )
        mod_fig.write_image(f"./Plots/Ex6/ConvergenceRateFor{name}LogScale.png", scale=2)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    weights = dict()
    mod_fig = go.Figure()
    for gamma in reversed(gammas):
        module = L1(init)
        callback, gd_values, gd_weights = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        last = gd.fit(module, None, None)
        weights[gamma] = gd_weights.copy()
        trace = go.Scatter(
            x=list(range(len(gd_values))),
            y=gd_values,
            mode="markers+lines",
            name=rf"$\gamma = {gamma}$"
        )
        mod_fig.add_trace(trace)

    # Plot algorithm's convergence for the different values of gamma
    mod_fig.update_layout(
        title=r"$\textbf{Q.3.1.2.5}-\text{Convergence Rate of L1 over different }\gamma$",
        xaxis=dict(title=r"$\text{Iteration}$"),
        yaxis=dict(title=r"$\text{L1 Norm}$"),
    )
    mod_fig.write_image(f"./Plots/Ex6/ExponentialConvergenceRateForL1.png", scale=2)
    mod_fig.update_layout(
        xaxis=dict(title=r"$\text{Iteration - Log Scale}$"),
    ).update_xaxes(
        type="log",
        range=(0, np.log10(1000)),
        constrain="domain"
    )
    mod_fig.write_image(f"./Plots/Ex6/ExponentialConvergenceRateForL1LogScale.png", scale=2)

    # Plot descent path for gamma=0.95
    callback, gd_values, gd_weights = get_gd_state_recorder_callback()
    gd = GradientDescent(ExponentialLR(eta, gammas[1]), callback=callback)
    last = gd.fit(L2(init), None, None)
    plot_descent_path(
        L1,
        np.array(weights[gammas[1]]),
        rf"$\textbf{{Q.3.1.2.7}}-\text{{Decent Path For L1 With }}\gamma = {gammas[1]}$"
    ).write_image(f"./Plots/Ex6/ExponentialDecentPathForL1WithGamma{gammas[1]}.png", scale=2)
    plot_descent_path(
        L2,
        np.array(gd_weights),
        rf"$\textbf{{Q.3.1.2.7}}-\text{{Decent Path For L2 With }}\gamma = {gammas[1]}$"
    ).write_image(f"./Plots/Ex6/ExponentialDecentPathForL2WithGamma{gammas[1]}.png", scale=2)


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    c = [custom[0], custom[-1]]
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    module = LogisticRegression(
        include_intercept=True,
        solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    )
    module.fit(X_train.to_numpy(), y_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_train, module.predict_proba(X_train.to_numpy()))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="black", dash='dash'),
                name="Random Class Assignment"
            ),
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='markers+lines',
                text=thresholds,
                name="",
                showlegend=False,
                marker_color=c[1][1],
                hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")
        )
    )
    fig.write_image(f"./Plots/Ex6/ROCCurveLogisticRegressionOverTrain.png", scale=2)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
