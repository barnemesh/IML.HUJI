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
    cm_df = df.groupby(["Country", "Month"])["Temp"].agg(
        ["mean", "std"]).reset_index()
    plot = px.line(
        data_frame=cm_df,
        x="Month",
        y="mean",
        error_y="std",
        title="average monthly temperature",
        color="Country",
        labels={"MeanTemp": "Mean of Temp", "StdTemp": "Std of Temp"}
    )
    plot.show()
    plot.write_image("./Plots/AvgTemp.png")  # TODO : remove
    # TODO: SA has a different pattern, will not do good with the IL model.
    # TODO: NTL will also not do as good as JRD - the model is likely to have
    #  strong bias.

    # Question 4 - Fitting model for different values of `k`
    loss = []
    train_X, train_y, test_X, test_y = split_train_test(
        il_df["DayOfYear"], il_df["Temp"]
    )
    test_X, test_y = test_X.to_numpy(), test_y.to_numpy()
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss.append(
            [k,
             np.round(model.loss(test_X, test_y), 2),
             # np.mean((test_y-y_hat)**2)
             ])
        print("deg {} : loss {}".format(*loss[-1]))

    loss_df = pd.DataFrame.from_records(loss, columns=["k", "loss"])
    plot = px.bar(data_frame=loss_df,
                  x="k",
                  y="loss",
                  title="Test error recorded based on degree of Polyfit k",
                  labels={"loss": "Test error recorded"})
    plot.show()
    plot.write_image("./Plots/PolyfitError.png")  # TODO : remove
    # TODO: 5 or 6 are the best - 6 slightly better, marginal, but 5 less
    #  complex. higher values will cause over fit -
    #  we see that 9 and 10 already have the same loss, so adding higher k will
    #  only complicate the model and will not get better results.

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
        ctr_loss.append(
            [country,
             np.round(model.loss(X, y), 2),
             # np.mean((test_y-y_hat)**2)
             ])
        # TODO: remove
        plot = px.scatter(
            x=X,
            y=[y, model.predict(X)],
            title=f"{country} predict and true",
        ).write_image(f"./Plots/{country}Predict.png")

    ctr_loss_df = pd.DataFrame.from_records(ctr_loss,
                                            columns=["Country", "Loss"])
    plot = px.bar(data_frame=ctr_loss_df,
                  x="Country",
                  y="Loss",
                  title="Error of model fitted over Israel by country",
                  labels={"Loss": "Test error recorded"},
                  text="Loss",
                  color="Country")
    plot.show()
    plot.write_image("./Plots/CountryPolyfit.png")
    # TODO: we have great fit in jordan - that had very similar cross with the
    #  area of the israel data, but with the netherlands, even if the
    #  polynomial is in similar shape, we had consistent bias, and that
    #  increased our loss. In south africa we had the opposite mistake -
    #  since some of the data did match, the loss was lower, even when the
    #  the polynomial was completely wrong - as we saw in the months that had
    #  some cross in the values .
    #  summary:
    #  netherlands : we saw no crossing with israel, big loss.
    #  jordan : lots of cross, big good.
    #  south africa : some cross, some not, not so good, misleading results!
