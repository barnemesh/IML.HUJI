import pandas

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import datetime as dt

# from sklearn.metrics import r2_score

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    # Drop unused values here---
    full_data = full_data.drop(["id", "date"], axis=1)
    full_data = full_data.dropna()

    # remove outlier that are likely to be errors.
    full_data = full_data.drop(
        full_data[
            (full_data["price"] <= 0) |
            (full_data["sqft_living"] <= 0) |
            (full_data["sqft_lot"] <= 0) |
            (full_data["sqft_above"] <= 0) |
            (full_data["yr_built"] <= 0) |
            (full_data["sqft_lot15"] <= 0) |

            (full_data["bathrooms"] < 0) |
            (full_data["bedrooms"] < 0) |
            (full_data["floors"] < 0) |
            (full_data["sqft_basement"] < 0) |
            (full_data["yr_renovated"] < 0) |

            (~full_data["waterfront"].isin([0, 1])) |
            (~full_data["view"].isin(range(5))) |
            (~full_data["condition"].isin(range(1, 6))) |
            (~full_data["grade"].isin(range(1, 15))) |

            (full_data["bedrooms"] > 30)
            ].index
    )
    # full_data = full_data.drop(
    #     full_data[(full_data["bedrooms"] == 0) & (full_data["bathrooms"] == 0)].index
    # )

    # create new features from existing features:
    full_data["zipcode"] = full_data["zipcode"].astype(int)
    full_data = pd.get_dummies(full_data,
                               prefix="zipcode_",
                               columns=["zipcode"])

    full_data["yr_last_worked_on"] = full_data["yr_renovated"].astype(int)
    full_data["yr_last_worked_on"].mask(
        full_data["yr_renovated"] <= full_data["yr_built"],
        full_data["yr_built"],
        axis=0,
        inplace=True
    )
    # drop data used only for new features:
    full_data = full_data.drop(["yr_built", "yr_renovated"], axis=1)
    return full_data.drop("price", axis=1), full_data["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = np.std(y)
    for name, values in X.items():
        array = values.to_numpy()
        p_cor = np.cov(array, y)[0, 1] / (np.std(array) * std_y)
        go.Figure(
            data=go.Scatter(
                x=array,
                y=y,
                mode="markers",
            )
        ).update_layout(
            title=f"feature {name} correlation with response. pearson = {p_cor}",
            xaxis_title=f"{name}",
            yaxis_title=f"response"
        ).write_image(f"{output_path}/{name}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, responses = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(features, responses)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(features, responses)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_X, test_y = test_X.to_numpy(), test_y.to_numpy()
    model = LinearRegression()
    p_arr = np.arange(10, 101)
    mean_loss_array = np.zeros(91)
    std_loss_array = np.zeros(91)
    for i in range(10, 101):
        losses = []
        for _ in range(10):
            ipX_train, ipy_train, ipX_test, ipy_test = \
                split_train_test(train_X, train_y, i / 100)
            ipX_train, ipy_train = ipX_train.to_numpy(), ipy_train.to_numpy()
            losses.append(model.fit(ipX_train, ipy_train).loss(test_X, test_y))

        mean_loss_array[i - 10], std_loss_array[i - 10] = \
            np.mean(losses, axis=0), np.std(losses, axis=0)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=p_arr,
                y=mean_loss_array,
                name="Mean loss",
                mode="markers+lines",
                line=dict(dash="dash"),
                marker=dict(color="green", opacity=.7)
            ),
            go.Scatter(
                x=p_arr,
                y=mean_loss_array - 2 * std_loss_array,
                name="confidence",
                fill=None, mode="lines",
                line=dict(color="lightgrey"),
                showlegend=False),
            go.Scatter(
                x=p_arr,
                y=mean_loss_array + 2 * std_loss_array,
                name="confidence",
                fill='tonexty',
                mode="lines",
                line=dict(color="lightgrey"),
                showlegend=False)],
        layout=go.Layout(
            title="Mean loss as function of p% of the training set",
            xaxis=dict(title="p% of training set"),
            yaxis=dict(title="Mean loss")
        )
    )
    fig.show()
    fig.write_image(f"./Plots/MeanLoss.png")
