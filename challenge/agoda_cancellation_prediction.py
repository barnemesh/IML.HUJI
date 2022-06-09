import itertools

import sklearn.ensemble
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from IMLearn.model_selection.cross_validate import cross_validate
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
import numpy as np
import pandas as pd
import datetime as dt
import re
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import random
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, BaggingClassifier, \
    ExtraTreesClassifier


# from sklearn.model_selection import cross_validate


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

    month_cancellation = (full_data['cancellation_datetime'].dt.month).fillna(50)
    booking_month = (full_data['booking_datetime'].dt.month).fillna(30)
    cancel_after_booking_month = month_cancellation - booking_month
    cancel_after_booking_month = cancel_after_booking_month.mask(
        cancel_after_booking_month < 0, cancel_after_booking_month + 12
    )
    checkin_day = full_data['checkin_date'].dt.day

    full_data["cancel_month_day"] = (full_data['cancellation_datetime'].dt.day).fillna(60)
    full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
        cancel_after_booking_month != 1, 100)
    # full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
    #     checkin_day <= 15, 100)
    full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
        full_data["cancel_month_day"] < 6, 100)

    full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
        (full_data["cancel_month_day"] <= 14), True)

    full_data["cancel_month_day"] = full_data["cancel_month_day"].mask(
        full_data["cancel_month_day"] > 14, False)

    full_data["cancellation_days_after_booking"] = \
        (full_data['cancellation_datetime'].fillna(full_data["booking_datetime"]) -  # TODO::
         full_data['booking_datetime']).dt.days

    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        full_data["cancellation_days_after_booking"] < 7, 100)

    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        (full_data["cancellation_days_after_booking"] < 40), True)

    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        full_data["cancellation_days_after_booking"] > 39, False)

    full_data = full_data.drop(['cancellation_datetime'], axis=1)

    features, p_full_data, encoder = preprocessing(full_data)
    features = features.drop([
        "cancellation_days_after_booking",
        "cancel_month_day"
    ], axis=1)
    # labels = p_full_data["cancel_month_day"]
    labels = p_full_data["cancellation_days_after_booking"]
    # labels = (p_full_data["cancellation_days_after_booking"] | p_full_data["cancel_month_day"])

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
    full_data = full_data.drop_duplicates()

    return full_data


def load_all_weeks(encoder=None):
    df1 = load_prev_data_separate("./Test_sets/week_1_test_data.csv", "./Labels/week_1_labels.csv")
    df2 = load_prev_data_separate("./Test_sets/week_2_test_data.csv", "./Labels/week_2_labels.csv")
    df3 = load_prev_data_separate("./Test_sets/week_3_test_data.csv", "./Labels/week_3_labels.csv")
    df4 = load_prev_data_separate("./Test_sets/week_4_test_data.csv", "./Labels/week_4_labels.csv")
    df5 = load_prev_data_separate("./Test_sets/week_5_test_data.csv", "./Labels/week_5_labels.csv")
    df6 = load_prev_data_separate("./Test_sets/week_6_test_data.csv", "./Labels/week_6_labels.csv")
    df7 = load_prev_data_separate("./Test_sets/week_7_test_data.csv", "./Labels/week_7_labels.csv")
    df_combined = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)
    features, p_full_data, encoder = preprocessing(df_combined, encoder)
    labels = p_full_data["cancellation_bool"]
    features = features.drop(["cancellation_bool"], axis=1)
    return features, labels, encoder


def load_weeks_3(encoder=None):
    df1 = load_prev_data_separate("./Test_sets/week_1_test_data.csv", "./Labels/week_1_labels.csv")
    df2 = load_prev_data_separate("./Test_sets/week_2_test_data.csv", "./Labels/week_2_labels.csv")
    df3 = load_prev_data_separate("./Test_sets/week_3_test_data.csv", "./Labels/week_3_labels.csv")
    df4 = load_prev_data_separate("./Test_sets/week_4_test_data.csv", "./Labels/week_4_labels.csv")
    df5 = load_prev_data_separate("./Test_sets/week_5_test_data.csv", "./Labels/week_5_labels.csv")
    df6 = load_prev_data_separate("./Test_sets/week_6_test_data.csv", "./Labels/week_6_labels.csv")
    df_combined = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    features, p_full_data, encoder = preprocessing(df_combined, encoder)
    labels = p_full_data["cancellation_bool"]
    features = features.drop(["cancellation_bool"], axis=1)
    return features, labels, encoder


def preprocessing(full_data, encoder=None):
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
        "guest_nationality_country_name_processed",
        "charge_option",
        "original_payment_type",
        "accommadation_type_name"
    ]
    full_data = full_data.drop(
        [
            # "accommadation_type_name",
            # "guest_nationality_country_name_processed",
            # "charge_option",
            # "original_payment_type",

            "hotel_chain_code",
            "hotel_brand_code",
            "language",
            "customer_nationality",
            "hotel_country_code",
            "original_payment_method",
            "original_payment_currency",
            "origin_country_code",
        ], axis=1)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoder.fit(full_data[categoricals])

    transformed = encoder.transform(full_data[categoricals])
    ohe_df = pd.DataFrame(transformed,
                          columns=encoder.get_feature_names_out(categoricals))
    full_data = pd.concat([full_data, ohe_df], axis=1).drop(categoricals, axis=1)

    # full_data["hotel_chain_code"] = full_data["hotel_chain_code"].fillna(-1)
    full_data["special_requests"] = (full_data["request_nonesmoke"].fillna(0) +
                                     full_data["request_latecheckin"].fillna(0) +
                                     full_data["request_highfloor"].fillna(0) +
                                     full_data["request_largebed"].fillna(0) +
                                     full_data["request_twinbeds"].fillna(0) +
                                     full_data["request_airport"].fillna(0) +
                                     full_data["request_earlycheckin"].fillna(0))

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
    full_data["TimeDiff"] = full_data["TimeDiff"].mask(
        full_data["TimeDiff"] < 0, 0
    )
    full_data["Nights_delta"] = (full_data["checkout_date"] - full_data["checkin_date"]).dt.days - 1
    full_data["Nights_delta"] = full_data["Nights_delta"].mask(
        full_data["Nights_delta"] < 0, 0
    )

    # full_data["booking_datetime_day"] = full_data["booking_datetime"].dt.day
    full_data["booking_datetime_month_delta"] = full_data["checkin_date"].dt.month - full_data[
        "booking_datetime"].dt.month
    full_data["booking_datetime_month_delta"] = full_data["booking_datetime_month_delta"].mask(
        full_data["booking_datetime_month_delta"] < 0, full_data["booking_datetime_month_delta"] + 12
    )
    # full_data["checkin_date_day"] = full_data["checkin_date"].dt.day
    # full_data["days_to_checkin"] = (full_data["checkin_date"] - full_data["booking_datetime"]).dt.days

    # full_data["booking_datetime"] = full_data["booking_datetime"].dt.dayofyear
    # full_data["checkin_date"] = full_data["checkin_date"].dt.dayofyear
    # full_data["checkout_date"] = full_data["checkout_date"].dt.dayofyear
    # full_data["booking_datetime"] = full_data["booking_datetime"].map(dt.datetime.toordinal)
    # full_data["checkin_date"] = full_data["checkin_date"].map(dt.datetime.toordinal)
    # full_data["checkout_date"] = full_data["checkout_date"].map(dt.datetime.toordinal)

    full_data["before_period_fine"] = np.zeros(full_data.shape[0], dtype=int)
    full_data["in_period_fine"] = np.zeros(full_data.shape[0], dtype=int)
    full_data["after_period_fine"] = np.zeros(full_data.shape[0], dtype=int)
    full_data = full_data.apply(transform_policy, axis=1)
    full_data["before_period_fine_amount"] = full_data["before_period_fine"] * full_data["original_selling_amount"]
    full_data["in_period_fine_amount"] = full_data["in_period_fine"] * full_data["original_selling_amount"]
    full_data["after_period_fine_amount"] = full_data["after_period_fine"] * full_data["original_selling_amount"]

    full_data["week_day_checkin"] = full_data["checkin_date"].apply(date_time_to_day_in_week)
    full_data["week_day_checkout"] = full_data["checkout_date"].apply(date_time_to_day_in_week)

    full_data["hotel_live_date"] = full_data["hotel_live_date"].map(dt.datetime.toordinal)
    full_data = full_data.drop([
        "booking_datetime",
        "checkin_date",
        "checkout_date"
    ], axis=1)
    p_full_data = full_data
    full_data = full_data.drop([
        "cancellation_policy_code",
        "guest_nationality_country_name",
        "h_booking_id",
        "hotel_id",
        "h_customer_id",
        "is_user_logged_in"
    ], axis=1)
    features = full_data
    return features, p_full_data, encoder


def date_time_to_day_in_week(data):
    return data.to_pydatetime().weekday()


def load_test(filename: str, encoder):
    full_data = pd.read_csv(filename,
                            parse_dates=["booking_datetime",
                                         "checkin_date",
                                         "checkout_date",
                                         "hotel_live_date"]).drop_duplicates()

    features, p_full_data, encoder = preprocessing(full_data, encoder)

    return features


def calculate_fine(nights_percent, num, nights):
    if nights_percent == "N":
        nights = nights if nights != 0 else 1
        return np.floor(100 * num / nights)

    elif nights_percent == "P":
        return num

    else:
        return 0


regex = r"""([\d]+)([D|P|N])([\d]*)([N|P]?)"""


def transform_policy(data):
    matches = re.findall(regex, data["cancellation_policy_code"])

    closest_after = np.NINF

    for match in matches:
        match = list(filter(lambda x: x != "", match))

        if len(match) == 2:
            match = [0, "D"] + match

        if len(match) == 4:
            if data["TimeDiff"] - int(match[0]) < 7:
                data["before_period_fine"] = np.max([calculate_fine(match[3], int(match[2]), data["Nights_delta"]),
                                                     data["before_period_fine"]])

            elif (data["TimeDiff"] - int(match[0])) >= 7 and (data["TimeDiff"] - int(match[0]) <= 30):
                data["in_period_fine"] = np.max([calculate_fine(match[3], int(match[2]), data["Nights_delta"]),
                                                 data["in_period_fine"]])

            else:
                if int(match[0]) >= closest_after:
                    data["after_period_fine"] = np.max([calculate_fine(match[3], int(match[2]), data["Nights_delta"]),
                                                        data["after_period_fine"]])

                    closest_after = int(match[0])

    data["in_period_fine"] = np.max([data["in_period_fine"],
                                     data["before_period_fine"]])

    data["after_period_fine"] = np.max([data["in_period_fine"],
                                        data["after_period_fine"],
                                        data["before_period_fine"]])

    return data


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


def testings(df, responses, encoder):
    # df_prev, responses_prev, encoder = load_weeks_3(encoder)
    df_prev, responses_prev, encoder = load_all_weeks(encoder)
    # X_train, X_test, y_train, y_test = train_test_split(df, responses, test_size=0.25)
    # X_train_wk, X_test_wk, y_train_wk, y_test_wk = train_test_split(df_prev, responses_prev, test_size=0.25)
    df_all = pd.concat([df, df_prev], ignore_index=True)
    # df_all = df_all.fillna(0)
    responses_all = pd.concat([responses, responses_prev], ignore_index=True)
    # X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(df_all, responses_all, test_size=0.25)

    est1 = RandomForestClassifier(class_weight="balanced", ccp_alpha=0.0001)
    est2 = BaggingClassifier(n_estimators=20, max_samples=0.75, max_features=0.75, bootstrap_features=True)
    est3 = ExtraTreesClassifier(class_weight="balanced", ccp_alpha=0.0001)
    est = VotingClassifier(
        estimators=[('rf', est1), ('bag', est2), ('et', est3)],
        voting="soft"
    )  #
    # gscv = GridSearchCV(est,
    #                     param_grid={"weights": list(
    #                         itertools.product(np.linspace(start=0, stop=1, num=5, endpoint=False), repeat=3)
    #                     )
    #                     },
    #                     scoring="f1_macro",
    #                     cv=KFold(shuffle=True))
    # gscv.fit(df_all, responses_all.astype(bool))
    # print(gscv.cv_results_["best_params_"])
    # print(gscv.cv_results_["best_score_"])
    # est_stack = StackingClassifier(estimators=[('rf', est1), ('bag', est2), ('et', est3)],
    #                                cv=KFold(shuffle=True))

    # all_est_b = AgodaCancellationEstimator(balanced=True)
    # all_est_b.set_probs(0.6, 0.4, 0.7)
    # est.fit(df_all, responses_all.astype(bool))
    # all_est_b.fit(df_all, responses_all.astype(bool))

    def f1_macro(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")
    # print("All Balanced")
    # print(cross_validate(all_est_b, df_all.to_numpy(), responses_all.to_numpy().astype(bool), f1_macro, cv=3))
    print("Voting")
    print(cross_validate(est, df_all.to_numpy(), responses_all.to_numpy().astype(bool), f1_macro, cv=3))
    # print("Stacking")
    # print(cross_validate(est_stack, df_all.to_numpy(), responses_all.to_numpy().astype(bool), f1_macro, cv=3))

    # df7 = load_prev_data_separate("./Test_sets/week_7_test_data.csv", "./Labels/week_7_labels.csv")
    # features, p_full_data, encoder = preprocessing(df7, encoder)
    # labels = p_full_data["cancellation_bool"]
    # features = features.drop(["cancellation_bool"], axis=1)

    # print("############ All Data Balanced ################")
    # print("%% On week 7 %%")
    # print(confusion_matrix(labels.astype(bool), all_est_b.predict(features)))
    # print(classification_report(labels.astype(bool), all_est_b.predict(features), digits=5))
    #
    # print("############ Voting Classifier ################")
    # print("%% On week 7 %%")
    # print(confusion_matrix(labels.astype(bool), est.predict(features)))
    # print(classification_report(labels.astype(bool), est.predict(features), digits=5))

    # for i in range(5):
    #     np.random.seed(i)
    #     scores = cross_validate(est, df_all, responses_all.astype(bool),
    #                    scoring="f1_macro", cv=KFold(shuffle=True), return_train_score=True)
    #     print(f"######## All Data Balanced , seed={i} ###########")
    #     test_score = scores["test_score"]
    #     print(f"avg test: {np.mean(test_score)}")
    #     print("scores:")
    #     print(test_score)
    #     train_score = scores["train_score"]
    #     print(f"avg test: {np.mean(train_score)}")
    #     print("scores:")
    #     print(train_score)


if __name__ == '__main__':
    np.random.seed(0)
    # Load data
    df, responses, encoder = load_data("./agoda_cancellation_train.csv")
    df_prev, responses_prev, encoder = load_all_weeks(encoder)
    # X_train, X_test, y_train, y_test = train_test_split(df, responses, test_size=0.25)
    # X_train_wk, X_test_wk, y_train_wk, y_test_wk = train_test_split(df_prev, responses_prev, test_size=0.25)
    df_all = pd.concat([df, df_prev], ignore_index=True)
    responses_all = pd.concat([responses, responses_prev], ignore_index=True)
    # testings(df, responses, encoder)  # TODO: Uncomment this to test

    est = AgodaCancellationEstimator()
    # est = AgodaCancellationEstimator(balanced=True)  #  TODO: decide if we want this or the new one?
    est.fit(df_all, responses_all.astype(bool))

    # Store model predictions over test set
    real = load_test("./Test_sets/week_8_test_data.csv", encoder)
    evaluate_and_export(est, real, "312245087_312162464_316514314.csv")
