import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename,
                            parse_dates=["Date"]).dropna().drop_duplicates()
    full_data.drop(
        full_data[
            (~full_data["Day"].isin(range(1, 32))) |
            (~full_data["Month"].isin(range(1, 13))) |
            (full_data["Temp"] < -70)
            ].index,
        inplace=True
    )
    date_from_col = pd.to_datetime(full_data[['Year', 'Month', 'Day']])
    full_data.drop(
        full_data[full_data["Date"] != date_from_col].index,
        inplace=True
    )
    full_data["DayOfYear"] = full_data["Date"].dt.dayofyear
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    il_df = df.drop(df[df["Country"] != "Israel"].index)
    il_df["Year"] = il_df["Year"].astype(str)

    plot = px.scatter(
        data_frame=il_df,
        x="DayOfYear",
        y="Temp",
        color="Year",
        title="Temp as function of DayOfYear in Israel"
    )
    plot.show()

    grouped_months = il_df.groupby(["Month"])["Temp"].agg("std")
    plot = px.bar(
        data_frame=grouped_months,
        y="Temp",
        title="std of Temp per month in Israel",
        labels={"Temp": "STD of Temp"}
    )
    plot.show()

    # Question 3 - Exploring differences between countries
    cm_df = df.groupby(["Country", "Month"])["Temp"].agg(
        ["mean", "std"]).reset_index()
    plot = px.line(
        data_frame=cm_df,
        x="Month",
        y="mean",
        error_y="std",
        title="average monthly temperature",
        color="Country",
        labels={"mean": "Mean of Temp"}
    )
    plot.show()

    # Question 4 - Fitting model for different values of `k`
    loss = []
    train_X, train_y, test_X, test_y = split_train_test(
        il_df["DayOfYear"], il_df["Temp"]
    )
    test_X, test_y = test_X.to_numpy(), test_y.to_numpy()
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss.append([k, np.round(model.loss(test_X, test_y), 2)])

    loss_df = pd.DataFrame.from_records(loss, columns=["k", "loss"])
    print(loss_df.to_string(index=False))
    plot = px.bar(
        data_frame=loss_df,
        x="k",
        y="loss",
        text="loss",
        title="Test error recorded based on degree of Polyfit k",
        labels={"loss": "Test error recorded"}
    )
    plot.show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5)
    model.fit(il_df["DayOfYear"], il_df["Temp"])
    no_il_df = df.drop(df[df["Country"] == "Israel"].index)
    ctr_df = [y for x, y in no_il_df.groupby(["Country"])]
    ctr_loss = []
    for c in ctr_df:
        X = c["DayOfYear"]
        y = c["Temp"]
        country = c["Country"].unique()[0]
        ctr_loss.append([country, np.round(model.loss(X, y), 2)])

    ctr_loss_df = pd.DataFrame.from_records(ctr_loss,
                                            columns=["Country", "Loss"])
    plot = px.bar(
        data_frame=ctr_loss_df,
        x="Country",
        y="Loss",
        title="Error of model fitted over Israel by country",
        labels={"Loss": "Test error recorded"},
        text="Loss",
        color="Country"
    )
    plot.show()
