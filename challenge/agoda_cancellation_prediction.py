from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
import numpy as np
import pandas as pd
import datetime as dt
import re
from sklearn.metrics import confusion_matrix, classification_report
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier


def sample_together(n, X, y):
    rows = random.sample(np.arange(0, len(X.index)).tolist(), n)
    return X.iloc[rows,], y.iloc[rows,]


def undersample(X, y, under=1):
    y_min = y.loc[y == under]
    y_max = y.loc[y != under]
    X_min = X.filter(y_min.index, axis=0)
    X_max = X.filter(y_max.index, axis=0)

    X_under, y_under = sample_together(len(y_min.index), X_max, y_max)

    X = pd.concat([X_under, X_min])
    y = pd.concat([y_under, y_min])
    return X, y


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
    full_data = pd.read_csv(filename,
                            parse_dates=["cancellation_datetime",
                                         "booking_datetime",
                                         "checkin_date",
                                         "checkout_date",
                                         "hotel_live_date"]).drop_duplicates()

    # month_cancellation = (full_data['cancellation_datetime'].dt.month).fillna(50)
    # booking_month = (full_data['booking_datetime'].dt.month).fillna(30)
    # cancel_after_booking_month = month_cancellation - booking_month
    # cancel_after_booking_month = cancel_after_booking_month.mask(
    #     cancel_after_booking_month < 0, cancel_after_booking_month + 12
    # )
    # checkin_day = full_data['checkin_date'].dt.day
    #
    # full_data["cancel_month_day"] = (full_data['cancellation_datetime'].dt.day).fillna(60)
    # full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
    #     cancel_after_booking_month != 1, 100)
    # full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
    #     checkin_day <= 15, 100)
    # full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
    #     full_data["cancel_month_day"] < 7, 100)
    #
    # full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
    #     (full_data["cancel_month_day"] <= 13), True)
    #
    # full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
    #     full_data["cancel_month_day"] > 13, False)

    full_data["cancellation_days_after_booking"] = \
        (full_data['cancellation_datetime'].fillna(full_data["booking_datetime"]) -
         full_data['booking_datetime']).dt.days

    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        full_data["cancellation_days_after_booking"] < 7, 100)

    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        (full_data["cancellation_days_after_booking"] < 31), True)

    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        full_data["cancellation_days_after_booking"] > 30, False)

    full_data = full_data.drop(['cancellation_datetime'], axis=1)
    features, p_full_data, encoder = preprocessing(full_data)
    features = features.drop([
        "cancellation_days_after_booking",
        # "cancel_month_day"
    ], axis=1)

    # labels = p_full_data["cancel_month_day"]
    labels = p_full_data["cancellation_days_after_booking"]

    # features_under, labels_under = undersample(features, labels)

    # features_leftovers = features.drop(features_under.index)
    # labels_leftover = labels.drop(labels_under.index)

    return features, labels, encoder


def load_prev_data(filename: str):
    full_data = pd.read_csv(filename,
                            parse_dates=["booking_datetime",
                                         "checkin_date",
                                         "checkout_date",
                                         "hotel_live_date"]).drop_duplicates()

    features, p_full_data, encoder = preprocessing(full_data)

    labels = p_full_data["cancellation_bool"]

    return features, labels, encoder


def load_prev_data_separate(test: str, label: str):
    full_data = pd.read_csv(test, parse_dates=["booking_datetime",
                                               "checkin_date",
                                               "checkout_date",
                                               "hotel_live_date"])

    labels_only = pd.read_csv(label)
    full_data["cancellation_bool"] = labels_only["cancel"]
    full_data.drop_duplicates()

    return full_data


def load_all_weeks(encoder=None):
    df1 = load_prev_data_separate("./Test_sets/week_1_test_data.csv", "./Labels/week_1_labels.csv")
    df2 = load_prev_data_separate("./Test_sets/week_2_test_data.csv", "./Labels/week_2_labels.csv")
    df3 = load_prev_data_separate("./Test_sets/week_3_test_data.csv", "./Labels/week_3_labels.csv")
    df4 = load_prev_data_separate("./Test_sets/week_4_test_data.csv", "./Labels/week_4_labels.csv")
    df5 = load_prev_data_separate("./Test_sets/week_5_test_data.csv", "./Labels/week_5_labels.csv")
    df_combined = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    features, p_full_data, encoder = preprocessing(df_combined, encoder)
    labels = p_full_data["cancellation_bool"]
    features = features.drop(["cancellation_bool"], axis=1)
    return features, labels, encoder


def load_weeks_3(encoder=None):
    df1 = load_prev_data_separate("./Test_sets/week_1_test_data.csv", "./Labels/week_1_labels.csv")
    df2 = load_prev_data_separate("./Test_sets/week_2_test_data.csv", "./Labels/week_2_labels.csv")
    df3 = load_prev_data_separate("./Test_sets/week_3_test_data.csv", "./Labels/week_3_labels.csv")
    df4 = load_prev_data_separate("./Test_sets/week_4_test_data.csv", "./Labels/week_4_labels.csv")
    df_combined = pd.concat([df1, df2, df3, df4], ignore_index=True)
    features, p_full_data, encoder = preprocessing(df_combined, encoder)
    labels = p_full_data["cancellation_bool"]
    features = features.drop(["cancellation_bool"], axis=1)
    return features, labels, encoder


def preprocessing(full_data, encoder=None):
    # full_data["charge_option_numbered"] = full_data["charge_option"].map({"Pay Now": 2, "Pay Later": 1,
    #                                                                       'Pay at Check-in': 0})

    # TODO: this give "value" to the type, should be dummies, but what do we do with missing dummies?
    full_data["guest_nationality_country_name_processed"] = full_data["guest_nationality_country_name"].map({
        'China': 7, 'South Africa': 6, 'South Korea': 7, 'Singapore': 7, 'Thailand': 7, 'Argentina': 4,
        'Taiwan': 7, 'Saudi Arabia': 2, 'Mexico': 3, 'Malaysia': 7, 'Germany': 0, 'New Zealand': 5,
        'Hong Kong': 7, 'Vietnam': 7, 'Indonesia': 5, 'Australia': 5, 'Norway': 0, 'United Kingdom': 0,
        'Peru': 4, 'Japan': 7, 'Philippines': 7, 'United States': 3, 'India': 7, 'Sri Lanka': 6,
        'Czech Republic': 0, 'Finland': 0, 'United Arab Emirates': 2, 'Brazil': 4, 'Bangladesh': 7,
        'France': 0, 'Cambodia': 7, 'Russia': 0, 'Belgium': 0, 'Bahrain': 2, 'Macau': 7, 'Switzerland': 0,
        'Hungary': 0, 'Italy': 0, 'Austria': 0, 'Oman': 2, 'Spain': 0, 'Ukraine': 0, 'Slovakia': 0, 'Canada': 3,
        'Kuwait': 1, 'Denmark': 0, 'Pakistan': 2, 'Ireland': 0, 'Brunei Darussalam': 7, 'Poland': 0,
        'Sweden': 0, 'Morocco': 6, 'Israel': 1, 'Egypt': 1, 'Netherlands': 0, 'Myanmar': 7, 'Angola': 6,
        'Romania': 0, 'Mauritius': 6, 'Kenya': 6, 'Mongolia': 7, 'Laos': 7, 'Nepal': 7, 'Chile': 4, 'Turkey': 1,
        'Qatar': 2, 'Jordan': 1, 'Puerto Rico': 3, 'Uruguay': 4, 'Algeria': 6, 'Portugal': 0, 'UNKNOWN': 8,
        'Jersey': 0, 'Colombia': 3, 'Greece': 0, 'Yemen': 2, 'Slovenia': 0, 'Botswana': 6, 'Estonia': 0,
        'Reunion Island': 6, 'Palestinian Territory': 1, 'Cyprus': 1, 'Papua New Guinea': 5,
        'Fiji': 5, 'Azerbaijan': 2, 'Somalia': 6, 'French Guiana': 4, 'French Polynesia': 5,
        'Tunisia': 6, 'Madagascar': 6, 'Iraq': 2, 'Northern Mariana Islands': 5, 'Gambia': 6,
        'Guatemala': 3, 'Zambia': 6, 'Guam': 5, 'Senegal': 6, 'Kazakhstan': 2, "Cote D'ivoire": 6,
        'Monaco': 0, 'Nigeria': 6, 'Curacao': 3, 'Malta': 1, 'Lithuania': 0, 'Bahamas': 3, 'Uzbekistan': 2,
        'Zimbabwe': 6, 'Luxembourg': 0, 'Albania': 0, 'Ghana': 6, 'Bulgaria': 0, 'Costa Rica': 3,
        'Mozambique': 6, 'Montenegro': 0, 'Maldives': 0, 'Guinea': 6,
        'Sint Maarten (Netherlands)': 0, 'Central African Republic': 6,
        'Democratic Republic of the\xa0Congo': 6, 'Uganda': 6, 'Kyrgyzstan': 2, 'Afghanistan': 2,
        'Mali': 6, 'Lebanon': 1, 'Eswatini': 6, 'Faroe Islands': 0, 'Barbados': 3, 'Benin': 6,
        'Venezuela': 4, 'Georgia': 2, 'South Sudan': 6, 'Gabon': 6, 'Aruba': 4, 'Latvia': 0,
        'British Indian Ocean Territory': 7, 'Andorra': 0, 'Bhutan': 7, 'Togo': 6, 'Belarus': 0,
        'New Caledonia': 5, 'Isle Of Man': 0, 'Burkina Faso': 6, 'Iceland': 0, 'Croatia': 0,
        'Namibia': 6, 'Cameroon': 6, 'Trinidad & Tobago': 4}).fillna(8)

    categoricals = [
        # "hotel_chain_code",  # TODO drop this?
        # "hotel_brand_code",  # TODO: hotel identity features seemed to be important - should try to keep these 2?
        "guest_nationality_country_name_processed",
        "charge_option",
        "original_payment_type",
        # "language",
        # "customer_nationality",
        # "hotel_country_code",
        # "original_payment_method",
        # "original_payment_currency",
        # "origin_country_code",
        "accommadation_type_name"
    ]
    full_data = full_data.drop(
        [
            "hotel_chain_code",  # TODO drop this?
            "hotel_brand_code",  # TODO: hotel identity features seemed to be important - should try to keep these 2?
            # "guest_nationality_country_name_processed",
            # "charge_option",
            # "original_payment_type",
            "language",
            "customer_nationality",
            "hotel_country_code",
            "original_payment_method",
            "original_payment_currency",
            "origin_country_code",
            # "accommadation_type_name"
        ], axis=1)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoder.fit(full_data[categoricals])

    transformed = encoder.transform(full_data[categoricals])
    ohe_df = pd.DataFrame(transformed,
                          columns=encoder.get_feature_names_out(categoricals))
    full_data = pd.concat([full_data, ohe_df], axis=1).drop(categoricals, axis=1)

    full_data["special_requests"] = full_data["request_nonesmoke"].fillna(0) + full_data["request_latecheckin"].fillna(
        0) \
                                    + full_data["request_highfloor"].fillna(0) + full_data["request_largebed"].fillna(0) \
                                    + full_data["request_twinbeds"].fillna(0) + full_data["request_airport"].fillna(0) \
                                    + full_data["request_earlycheckin"].fillna(0)

    full_data = full_data.drop([
        "request_nonesmoke",
        "request_latecheckin",
        "request_highfloor",
        "request_largebed",
        "request_twinbeds",
        "request_airport",
        "request_earlycheckin"
    ], axis=1)

    # TODO: this one is important!
    full_data['TimeDiff'] = (full_data['checkin_date'] - full_data['booking_datetime']).dt.days

    # TODO: this one is important!
    full_data["cancellation_policy_numbered"] = \
        full_data.apply(lambda x: transform_policy(x["cancellation_policy_code"],
                                                   x["TimeDiff"],
                                                   x["original_selling_amount"]), axis=1)

    full_data["booking_datetime_day"] = full_data["booking_datetime"].dt.day
    full_data["booking_datetime_month_delta"] = full_data["checkin_date"].dt.month - full_data["booking_datetime"].dt.month
    full_data["booking_datetime_month_delta"] = full_data["booking_datetime_month_delta"].mask(
            full_data["booking_datetime_month_delta"] < 0, full_data["booking_datetime_month_delta"] + 12
        )
    full_data["checkin_date_day"] = full_data["checkin_date"].dt.day
    full_data["checkout_date_delta"] = (full_data["checkout_date"] - full_data["checkin_date"]).dt.days
    # full_data["hotel_live_date"] = full_data["hotel_live_date"].dt.dayofyear
    # full_data["checkout_date"] = full_data["checkout_date"].map(dt.datetime.toordinal)

    # full_data["booking_datetime"] = full_data["booking_datetime"].map(dt.datetime.toordinal)
    # full_data["checkin_date"] = full_data["checkin_date"].map(dt.datetime.toordinal)
    # full_data["checkout_date"] = full_data["checkout_date"].map(dt.datetime.toordinal)

    full_data["hotel_live_date"] = full_data["hotel_live_date"].map(dt.datetime.toordinal)
    full_data = full_data.drop([
        "booking_datetime",
        "checkin_date",
        "checkout_date"
    ], axis=1)
    p_full_data = full_data

    # features = full_data[[
    #     "TimeDiff",
    #     "cancellation_policy_numbered",
    #     "hotel_star_rating",
    #     "no_of_children",
    #     "no_of_adults",
    #     "is_first_booking",
    #     "special_requests",
    #     "hotel_area_code",
    #     "original_selling_amount",
    #     "charge_option_numbered",
    #     "accommadation_type_name_proccessed",
    #     "guest_nationality_country_name_processed"
    # ]]
    full_data = full_data.drop([
        "cancellation_policy_code",
        "guest_nationality_country_name",
        "h_booking_id",
        "hotel_id",
        "h_customer_id",
    ], axis=1)
    features = full_data
    return features, p_full_data, encoder


def load_test(filename: str, encoder):
    full_data = pd.read_csv(filename,
                            parse_dates=["booking_datetime",
                                         "checkin_date",
                                         "checkout_date",
                                         "hotel_live_date"]).drop_duplicates()

    features, p_full_data, encoder = preprocessing(full_data, encoder)

    return features


regex = r"""([\d])([D|P])([\d])([N|P])"""


def transform_policy(policy, nights, cost):
    matches = re.findall(regex, policy)

    result = 0

    for match in matches:
        if len(match) == 2:
            result += (int(match[0]) / 100) * cost
        else:
            if match[0] == "0":
                divider = 1
            else:
                divider = int(match[0])

            if match[3] == 'N':

                if nights == 0:
                    nights_divider = 1
                else:
                    nights_divider = nights
                policy_cost = divider * (int(match[2]) / nights_divider) * cost

            else:
                policy_cost = divider * (int(match[2]) / 100) * cost

            if policy_cost > result:
                result = policy_cost

    return result


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
    df, responses, encoder = load_data("../datasets/agoda_cancellation_train.csv")
    # df_prev, responses_prev, encoder = load_weeks_3(encoder)
    df_prev, responses_prev, encoder = load_all_weeks(encoder)
    X_train, X_test, y_train, y_test = train_test_split(df, responses, test_size=0.25)
    # X_train_wk, X_test_wk, y_train_wk, y_test_wk = train_test_split(df_prev, responses_prev, test_size=0.25)
    # df_all = pd.concat([df, df_prev], ignore_index=True)
    # # TODO: fill na here is shadowing bad preprocessing
    # df_all = df_all.fillna(0)
    # responses_all = pd.concat([responses, responses_prev], ignore_index=True)
    # X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(df_all, responses_all, test_size=0.25)

    # forest = DecisionTreeClassifier(class_weight={0: 0.52, 1: 14})
    # a = forest.cost_complexity_pruning_path(X_train_all, y_train_all.astype(bool))
    # forest2 = DecisionTreeClassifier()
    # b = forest2.cost_complexity_pruning_path(X_train_all, y_train_all.astype(bool))
    # dict_a = np.array([a.ccp_alphas, a.impurities])
    # dict_b = np.array([b.ccp_alphas, b.impurities])
    og_est_b = AgodaCancellationEstimator(balanced=True)
    og_est_b.fit(X_train, y_train.astype(bool))

    # a = np.array([og_est_b.estimator.feature_importances_, og_est_b.estimator.feature_names_in_])
    #
    og_est = AgodaCancellationEstimator()
    og_est.fit(X_train, y_train.astype(bool))
    #
    # wk_est_b = AgodaCancellationEstimator(balanced=True)
    # wk_est_b.fit(X_train_wk, y_train_wk.astype(bool))
    #

    # wk_est = AgodaCancellationEstimator()
    # wk_est.fit(X_train_wk, y_train_wk.astype(bool))

    # all_est_b = AgodaCancellationEstimator(balanced=True)
    # all_est_b.fit(X_train_all, y_train_all.astype(bool))

    # all_est = AgodaCancellationEstimator()
    # all_est.fit(X_train_all, y_train_all.astype(bool))

    # df4 = load_prev_data_separate("./Test_sets/test_set_week_4.csv", "./Labels/test_set_week_4_labels.csv")
    # features, p_full_data = preprocessing(df4)
    # features = features.drop(["cancellation_bool"], axis=1)
    # labels = p_full_data["cancellation_bool"]
    # df3, resp3 = load_weeks_3()
    # est_wk_3 = AgodaCancellationEstimator()
    # est_wk_3.fit(df3, resp3.astype(bool))
    # est_prev = AgodaCancellationEstimator(balanced=True)
    # est_prev.fit(df, responses.astype(bool))

    # X_split = pd.concat([X_test, X_test_wk], ignore_index=True)
    # y_split = pd.concat([y_test, y_test_wk], ignore_index=True)
    # pred = (est_prev.predict(features) + est_wk_3.predict(features)) >= 1
    # print("############ Split Data ################")
    # print("%% On Split test %%")
    # print(confusion_matrix(labels.astype(bool), pred))
    # print(classification_report(labels.astype(bool), pred))

    print("############ Original Data Balanced ################")
    print("%% On Original test %%")
    print(confusion_matrix(y_test.astype(bool), og_est_b.predict(X_test)))
    print(classification_report(y_test.astype(bool), og_est_b.predict(X_test)))
    print("%% On New Data %%")
    print(confusion_matrix(responses_prev.astype(bool), og_est_b.predict(df_prev)))
    print(classification_report(responses_prev.astype(bool), og_est_b.predict(df_prev)))

    print("############ Original Data UnBalanced ################")
    print("%% On Original test %%")
    print(confusion_matrix(y_test.astype(bool), og_est.predict(X_test)))
    print(classification_report(y_test.astype(bool), og_est.predict(X_test)))
    print("%% On New Data %%")
    print(confusion_matrix(responses_prev.astype(bool), og_est.predict(df_prev)))
    print(classification_report(responses_prev.astype(bool), og_est.predict(df_prev)))

    # print("############ New Data Balanced ################")
    # print("%% On New Data Test %%")
    # print(confusion_matrix(y_test_wk.astype(bool), wk_est_b.predict(X_test_wk)))
    # print(classification_report(y_test_wk.astype(bool), wk_est_b.predict(X_test_wk)))

    # print("############ New Data UnBalanced ################")
    # print("%% On New Data Test %%")
    # print(confusion_matrix(y_test_wk.astype(bool), wk_est.predict(X_test_wk)))
    # print(classification_report(y_test_wk.astype(bool), wk_est.predict(X_test_wk)))

    # print("############ All Data Balanced ################")
    # print("%% On All Data Test %%")
    # print(confusion_matrix(y_test_all.astype(bool), all_est_b.predict(X_test_all)))
    # print(classification_report(y_test_all.astype(bool), all_est_b.predict(X_test_all)))

    # print("############ All Data UnBalanced ################")
    # print("%% On All Data Test %%")
    # print(confusion_matrix(y_test_all.astype(bool), all_est.predict(X_test_all)))
    # print(classification_report(y_test_all.astype(bool), all_est.predict(X_test_all)))
    #
    # est = AgodaCancellationEstimator(balanced=True)
    # est.fit(df, responses.astype(bool))
    # df1 = load_prev_data_separate("./Test_sets/test_set_week_1.csv", "./Labels/test_set_week_1_labels.csv")
    # df2 = load_prev_data_separate("./Test_sets/test_set_week_2.csv", "./Labels/test_set_labels_week_2.csv")
    # df3 = load_prev_data_separate("./Test_sets/test_set_week_3.csv", "./Labels/test_set_week_3_labels.csv")
    # df4 = load_prev_data_separate("./Test_sets/test_set_week_4.csv", "./Labels/test_set_week_4_labels.csv")
    # features, p_full_data = preprocessing(df1)
    # labels = p_full_data["cancellation_bool"]
    # features = features.drop(["cancellation_bool"], axis=1)
    # print("############ Original One Week At A TIme ################")
    # print(confusion_matrix(labels.astype(bool), est.predict(features)))
    # print(classification_report(labels.astype(bool), est.predict(features)))
    # features, p_full_data = preprocessing(df2)
    # labels = p_full_data["cancellation_bool"]
    # features = features.drop(["cancellation_bool"], axis=1)
    # print(confusion_matrix(labels.astype(bool), est.predict(features)))
    # print(classification_report(labels.astype(bool), est.predict(features)))
    # features, p_full_data = preprocessing(df3)
    # labels = p_full_data["cancellation_bool"]
    # features = features.drop(["cancellation_bool"], axis=1)
    # print(confusion_matrix(labels.astype(bool), est.predict(features)))
    # print(classification_report(labels.astype(bool), est.predict(features)))
    # features, p_full_data = preprocessing(df4)
    # labels = p_full_data["cancellation_bool"]
    # features = features.drop(["cancellation_bool"], axis=1)
    # print(confusion_matrix(labels.astype(bool), est.predict(features)))
    # print(classification_report(labels.astype(bool), est.predict(features)))
    #
    # print(confusion_matrix(responses_prev.astype(bool), est.predict(df_prev)))
    # print(classification_report(responses_prev.astype(bool), est.predict(df_prev)))
    #
    # df3, resp3 = load_weeks_3()
    # df3 = pd.concat([df, df3], ignore_index=True)
    # # TODO: fill na here is shadowing bad preprocessing
    # df3 = df3.fillna(0)
    # resp3 = pd.concat([responses, resp3], ignore_index=True)
    # df5 = load_prev_data_separate("./Test_sets/week_5_test_data.csv", "./Labels/week_5_labels.csv")
    # features, p_full_data, encoder = preprocessing(df5, encoder)
    # labels = p_full_data["cancellation_bool"]
    # features = features.drop(["cancellation_bool"], axis=1)
    #
    # est_wk_3 = AgodaCancellationEstimator(balanced=True)
    # est_wk_3.fit(df_all, responses_all.astype(bool))
    # print(confusion_matrix(labels.astype(bool), est_wk_3.predict(features)))
    # print(classification_report(labels.astype(bool), est_wk_3.predict(features)))
    # a = np.array([est_wk_3.estimator.feature_importances_, est_wk_3.estimator.feature_names_in_])

    # est = AgodaCancellationEstimator(balanced=True)
    # est.fit(df, responses.astype(bool))
    # # Store model predictions over test set
    # real = load_test("./Test_sets/week_6_test_data.csv", encoder)
    # # a = all_est_b.estimator.predict_proba(real)
    # evaluate_and_export(est, real, "312245087_312162464_316514314.csv")
