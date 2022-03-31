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
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    full_data["Date"] = pd.to_datetime(full_data["Date"])
    full_data.drop(  # todo: days in month based on the month (and year in feb)
        full_data[(full_data["Day"] <= 0) | (full_data["Day"] > 31)].index,
        inplace=True
    )
    full_data.drop(
        full_data[(full_data["Month"] <= 0) | (full_data["Month"] > 12)].index,
        inplace=True
    )
    date_from_col = pd.to_datetime(full_data[['Year', 'Month', 'Day']])
    full_data.drop(
        full_data[full_data["Date"] != date_from_col].index,
        inplace=True
    )
    full_data.drop(
        full_data[full_data["Temp"] < -72].index,
        inplace=True
    )
    full_data["DayOfYear"] = full_data["Date"].dt.dayofyear
    features = full_data.drop(["Temp"], axis=1)
    # return features, full_data["Temp"]
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    il_df = df.drop(df[df["Country"] != "Israel"].index)
    il_df["Year"] = il_df["Year"].astype(str)
    plot = px.scatter(data_frame=il_df, x="DayOfYear", y="Temp", color="Year",
                      title="Temp as function of DayOfYear in israel")
    plot.show()
    plot.write_image("./Plots/IsraelTemp.png")  # TODO : remove
    # TODO: x^3 - x^2 ?
    grouped_months = il_df.groupby(["Month"])["Temp"].agg("std")
    plot = px.bar(data_frame=grouped_months, y="Temp",
                  title="std of Temp per month in Israel",
                  labels={"Temp": "STD of Temp"})
    plot.show()
    plot.write_image("./Plots/IsraelMothStd.png")  # TODO : remove
    # TODO: better in months with lower STD?

    # Question 3 - Exploring differences between countries
    px.line()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
