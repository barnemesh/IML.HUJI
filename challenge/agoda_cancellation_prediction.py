from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
# from IMLearn.base import BaseEstimator
from IMLearn.utils import split_train_test
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import datetime as dt
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor


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


    full_data["cancellation_days_after_booking"] = \
        (full_data['cancellation_datetime'].fillna(full_data["booking_datetime"]) -
         full_data['booking_datetime']).dt.days


    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        full_data["cancellation_days_after_booking"] < 7, 100)

    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        (full_data["cancellation_days_after_booking"] < 31), True)

    full_data["cancellation_days_after_booking"] = full_data["cancellation_days_after_booking"].mask(
        full_data["cancellation_days_after_booking"] > 30, False)


    full_data["guest_nationality_country_name_processed"] = full_data["guest_nationality_country_name"].map({
        'China': 7, 'South Africa': 6, 'South Korea': 7, 'Singapore': 7, 'Thailand': 7, 'Argentina': 4,
        'Taiwan': 7, 'Saudi Arabia': 2, 'Mexico': 3, 'Malaysia': 7, 'Germany': 0, 'New Zealand': 5,
        'Hong Kong': 7, 'Vietnam': 7, 'Indonesia': 5, 'Australia': 5, 'Norway': 0, 'United Kingdom': 0,
        'Peru': 4, 'Japan': 7, 'Philippines': 7, 'United States': 3, 'India': 7, 'Sri Lanka': 6,
        'Czech Republic': 0, 'Finland': 0, 'United Arab Emirates': 2, 'Brazil': 4, 'Bangladesh': 7,
        'France': 0, 'Cambodia': 7, 'Russia': 0, 'Belgium': 0, 'Bahrain': 2, 'Macau': 7, 'Switzerland': 0,
        'Hungary': 0, 'Italy': 0, 'Austria': 0, 'Oman': 2, 'Spain': 0, 'Ukraine': 0, 'Slovakia': 0, 'Canada': 3,
        'Kuwait': 1, 'Denmark': 0, 'Pakistan': 2, 'Ireland': 0, 'Brunei Darussalam': 7,  'Poland': 0,
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
        'Namibia': 6, 'Cameroon': 6, 'Trinidad & Tobago': 4})

    full_data["original_payment_type_proccessed"] = full_data["original_payment_type"].map({
        'Invoice': 1, 'Credit Card': 0, 'Gift Card': 2})

    full_data["accommadation_type_name_proccessed"] = full_data["accommadation_type_name"].map({
        'Hotel': 0, 'Resort': 1, 'Serviced Apartment': 2, 'Guest House / Bed & Breakfast': 3,
        'Hostel': 4, 'Capsule Hotel': 5, 'Home': 6, 'Apartment': 7, 'Bungalow': 8, 'Motel': 9, 'Ryokan': 10,
        'Tent': 11, 'Resort Villa': 12, 'Love Hotel': 13, 'Holiday Park / Caravan Park': 14,
        'Private Villa': 15, 'Boat / Cruise': 16, 'UNKNOWN': 21, 'Inn': 17, 'Lodge': 18, 'Homestay': 19,
        'Chalet': 20})

    full_data["charge_option_numbered"] = full_data["charge_option"].map({"Pay Now": 2, "Pay Later": 1,
                                                                          'Pay at Check-in': 0})

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

    full_data['TimeDiff'] = (full_data['checkin_date'] - full_data['booking_datetime']).dt.days

    full_data["cancellation_policy_numbered"] = \
        full_data.apply(lambda x: transform_policy(x["cancellation_policy_code"],
                                                   x["TimeDiff"],
                                                   x["original_selling_amount"]), axis=1)

    full_data["booking_datetime"] = full_data["booking_datetime"].map(dt.datetime.toordinal)  # .fillna(0)
    full_data["checkin_date"] = full_data["checkin_date"].map(dt.datetime.toordinal)  # .fillna(0)
    full_data["checkout_date"] = full_data["checkout_date"].map(dt.datetime.toordinal)  # .fillna(0)
    full_data["hotel_live_date"] = full_data["hotel_live_date"].map(dt.datetime.toordinal)  # .fillna(0)

    labels = full_data["cancellation_days_after_booking"]
    features = full_data[[
        "TimeDiff",
        "cancellation_policy_numbered",
        "hotel_star_rating",
        "no_of_children",
        "no_of_adults",
        "is_first_booking",
        "special_requests",
        "hotel_area_code",
        "original_selling_amount",
        "charge_option_numbered",
        "accommadation_type_name_proccessed",
        "guest_nationality_country_name_processed",
    ]]

    return features, labels


def load_test(filename: str):
    full_data = pd.read_csv(filename,
                            parse_dates=["booking_datetime",
                                         "checkin_date",
                                         "checkout_date",
                                         "hotel_live_date"]).drop_duplicates()

    full_data["charge_option_numbered"] = full_data["charge_option"].map({"Pay Now": 2, "Pay Later": 1,
                                                                          'Pay at Check-in': 0})

    full_data["original_payment_type_proccessed"] = full_data["original_payment_type"].map({
        'Invoice': 1, 'Credit Card': 0, 'Gift Card': 2})

    full_data["cancellation_days_after_booking"] = \
        (full_data['cancellation_datetime'].fillna(full_data["booking_datetime"]) -
         full_data['booking_datetime']).dt.days

    full_data["guest_nationality_country_name_processed"] = full_data["guest_nationality_country_name"].map({
        'China': 7, 'South Africa': 6, 'South Korea': 7, 'Singapore': 7, 'Thailand': 7, 'Argentina': 4,
        'Taiwan': 7, 'Saudi Arabia': 2, 'Mexico': 3, 'Malaysia': 7, 'Germany': 0, 'New Zealand': 5,
        'Hong Kong': 7, 'Vietnam': 7, 'Indonesia': 5, 'Australia': 5, 'Norway': 0, 'United Kingdom': 0,
        'Peru': 4, 'Japan': 7, 'Philippines': 7, 'United States': 3, 'India': 7, 'Sri Lanka': 6,
        'Czech Republic': 0, 'Finland': 0, 'United Arab Emirates': 2, 'Brazil': 4, 'Bangladesh': 7,
        'France': 0, 'Cambodia': 7, 'Russia': 0, 'Belgium': 0, 'Bahrain': 2, 'Macau': 7, 'Switzerland': 0,
        'Hungary': 0, 'Italy': 0, 'Austria': 0, 'Oman': 2, 'Spain': 0, 'Ukraine': 0, 'Slovakia': 0, 'Canada': 3,
        'Kuwait': 1, 'Denmark': 0, 'Pakistan': 2, 'Ireland': 0, 'Brunei Darussalam': 7,  'Poland': 0,
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
        'Namibia': 6, 'Cameroon': 6, 'Trinidad & Tobago': 4})

    full_data["special_requests"] = full_data["request_nonesmoke"].fillna(0) + full_data["request_latecheckin"].fillna(
        0) \
                                    + full_data["request_highfloor"].fillna(0) + full_data["request_largebed"].fillna(0) \
                                    + full_data["request_twinbeds"].fillna(0) + full_data["request_airport"].fillna(0) \
                                    + full_data["request_earlycheckin"].fillna(0)

    full_data["accommadation_type_name_proccessed"] = full_data["accommadation_type_name"].map({
        'Hotel': 0, 'Resort': 1, 'Serviced Apartment': 2, 'Guest House / Bed & Breakfast': 3,
        'Hostel': 4, 'Capsule Hotel': 5, 'Home': 6, 'Apartment': 7, 'Bungalow': 8, 'Motel': 9, 'Ryokan': 10,
        'Tent': 11, 'Resort Villa': 12, 'Love Hotel': 13, 'Holiday Park / Caravan Park': 14,
        'Private Villa': 15, 'Boat / Cruise': 16, 'UNKNOWN': 21, 'Inn': 17, 'Lodge': 18, 'Homestay': 19,
        'Chalet': 20})

    full_data = full_data.drop([
        "request_nonesmoke",
        "request_latecheckin",
        "request_highfloor",
        "request_largebed",
        "request_twinbeds",
        "request_airport",
        "request_earlycheckin",
        "hotel_chain_code",
    ], axis=1)
    full_data = full_data.dropna()
    full_data['TimeDiff'] = (full_data['checkin_date'] - full_data['booking_datetime']).dt.days
    full_data["cancellation_policy_numbered"] = \
        full_data.apply(lambda x: transform_policy(x["cancellation_policy_code"],
                                                   x["TimeDiff"],
                                                   x["original_selling_amount"]), axis=1)
    full_data["booking_datetime"] = full_data["booking_datetime"].map(dt.datetime.toordinal)  # .fillna(0)
    full_data["checkin_date"] = full_data["checkin_date"].map(dt.datetime.toordinal)  # .fillna(0)
    full_data["checkout_date"] = full_data["checkout_date"].map(dt.datetime.toordinal)  # .fillna(0)
    full_data["hotel_live_date"] = full_data["hotel_live_date"].map(dt.datetime.toordinal)  # .fillna(0)

    features = full_data[[
        "TimeDiff",
        "cancellation_policy_numbered",
        "hotel_star_rating",
        "no_of_children",
        "no_of_adults",
        "original_selling_amount",
        "is_first_booking",
        "special_requests",
        "hotel_area_code",
        "original_selling_amount",
        "charge_option_numbered",
        "accommadation_type_name_proccessed",
        "guest_nationality_country_name_processed"
    ]]
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
    df, responses = load_data(
        "C:/Users/yuval/Desktop/second_year/semester_B/IML.HUJI/datasets/agoda_cancellation_train.csv")

    train_X, train_y, test_X, test_y = split_train_test(df, responses)

    # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    # model = KNeighborsClassifier()
    # model = LogisticRegression(max_iter=100000)
    # # pipe = make_pipeline(StandardScaler(), model)
    # # estimator = pipe.fit(train_X, train_y)  # apply scaling on training data
    # model.fit(train_X, train_y)
    # predictions = model.predict(test_X)
    # std_y = np.std(responses)
    # for name, values in df.items():
    #     array = values.to_numpy()
    #     p_cor = np.cov(array, responses)[0, 1] / (np.std(array) * std_y)
    #     print(f"{name} : {p_cor}")

    # print("coef:")
    # for name, coef in zip(model., np.transpose(model.coef_)):
    #     print(f"{name}: {coef}")
    #
    # print()
    # print(roc_auc_score(model.predict(test_X), test_y))
    print("----classifiers----\n\n")
    print("logistic")
    # print(confusion_matrix(test_y, predictions))
    # print(classification_report(test_y, predictions))
    # Store model predictions over test set
    # real = load_test("../datasets/test_set_week_1.csv")
    # evaluate_and_export(model, real, "id1_id2_id3.csv")

    # print("forest")
    # forest = RandomForestClassifier(n_estimators=1000)
    # forest.fit(train_X, train_y)
    # print(confusion_matrix(test_y, forest.predict(test_X)))
    # print(classification_report(test_y, forest.predict(test_X)))
    #
    # print("neural")
    # neural = MLPClassifier()
    # neural.fit(train_X, train_y)
    # print(confusion_matrix(test_y, neural.predict(test_X)))
    # print(classification_report(test_y, neural.predict(test_X)))
    #
    # print("voting estimator")
    # our_estimator = AgodaCancellationEstimator()
    # our_estimator.fit(np.array(train_X), np.array(train_y))
    # result_est = our_estimator.predict(np.array(test_X))
    # print(confusion_matrix(test_y, result_est))
    # print(classification_report(test_y, result_est))

    # print("----regressions----\n\n")
    #
    # print()

    # print("forest")
    # forest_reg = RandomForestRegressor(n_estimators=1000)
    # forest_reg.fit(train_X, train_y)
    # print(confusion_matrix(test_y, forest_reg.predict(test_X)))
    # print(classification_report(test_y, forest_reg.predict(test_X)))
    #

    est = AgodaCancellationEstimator()
    est.fit(np.array(train_X), np.array(train_y.astype(bool)))
    print(confusion_matrix(test_y.astype(bool), est.predict(np.array(test_X))))
    print(classification_report(test_y.astype(bool), est.predict(np.array(test_X))))