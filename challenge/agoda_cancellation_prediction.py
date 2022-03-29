from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
# from IMLearn.base import BaseEstimator
from IMLearn.utils import split_train_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import datetime as dt


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename,
                            parse_dates=["cancellation_datetime",
                                         "booking_datetime",
                                         "checkin_date",
                                         "checkout_date",
                                         "hotel_live_date"]).drop_duplicates()
    full_data["cancellation_datetime"] = full_data["cancellation_datetime"].notna()
    full_data.dropna()
    full_data["booking_datetime"] = full_data["booking_datetime"].map(dt.datetime.toordinal).fillna(0)
    full_data["checkin_date"] = full_data["checkin_date"].map(dt.datetime.toordinal).fillna(0)
    full_data["checkout_date"] = full_data["checkout_date"].map(dt.datetime.toordinal).fillna(0)
    full_data["hotel_live_date"] = full_data["hotel_live_date"].map(dt.datetime.toordinal).fillna(0)
    features = full_data.drop(["hotel_country_code",
                               "accommadation_type_name",
                               "cancellation_policy_code",
                               "charge_option",
                               "customer_nationality",
                               "guest_nationality_country_name",
                               "language",
                               "origin_country_code",
                               "original_payment_type",
                               "original_payment_method",
                               "original_payment_type",
                               "original_payment_currency"
                               ], axis=1)
    # features = full_data[[
    #     "booking_datetime",
    #     "checkin_date",
    #     "checkout_date",
    #     "hotel_live_date",
    #     "is_first_booking",
    #     "hotel_star_rating",
    #     "no_of_ad"]]
    labels = full_data["cancellation_datetime"]

    return features, labels


def load_test(filename: str):
    full_data = pd.read_csv(filename,
                            parse_dates=["booking_datetime",
                                         "checkin_date",
                                         "checkout_date",
                                         "hotel_live_date"]).drop_duplicates()
    full_data.dropna()
    full_data["booking_datetime"] = full_data["booking_datetime"].map(dt.datetime.toordinal).fillna(0)
    full_data["checkin_date"] = full_data["checkin_date"].map(dt.datetime.toordinal).fillna(0)
    full_data["checkout_date"] = full_data["checkout_date"].map(dt.datetime.toordinal).fillna(0)
    full_data["hotel_live_date"] = full_data["hotel_live_date"].map(dt.datetime.toordinal).fillna(0)
    features = full_data.drop(["hotel_country_code",
                               "accommadation_type_name",
                               "cancellation_policy_code",
                               "charge_option",
                               "customer_nationality",
                               "guest_nationality_country_name",
                               "language",
                               "origin_country_code",
                               "original_payment_type",
                               "original_payment_method",
                               "original_payment_type",
                               "original_payment_currency"
                               ], axis=1)
    return features


def evaluate_and_export(estimator  #: BaseEstimator,
                        , X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, responses = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, responses)

    # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    model = LogisticRegression(max_iter=1000)
    estimator = model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    acc = predictions == test_y
    print(acc.value_counts())
    print(accuracy_score(test_y, predictions))

    std_y = np.std(responses)
    for name, values in df.items():
        array = values.to_numpy()
        p_cor = np.cov(array, responses)[0, 1] / (np.std(array) * std_y)
        print(p_cor)
    # Store model predictions over test set
    real = load_test("../datasets/test_set_week_1.csv")
    evaluate_and_export(estimator, real, "id1_id2_id3.csv")
