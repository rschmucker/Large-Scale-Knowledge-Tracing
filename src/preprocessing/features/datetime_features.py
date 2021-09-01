import numpy as np
import pandas as pd

DFORMAT = '%d-%b-%Y %H:%M:%S'

WEEKEND_ENCODER = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 1
}

POD = {  # part of day
    0: 6,
    1: 6,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 3,
    11: 3,
    12: 3,
    13: 3,
    14: 4,
    15: 4,
    16: 4,
    17: 4,
    18: 5,
    19: 5,
    20: 5,
    21: 5,
    22: 6,
    23: 6,
}


def extract_datetime(df, name):
    if name == "squirrel":
        return pd.to_datetime(df['date_time'], format=DFORMAT)
    elif name == "ednet_kt3":
        return pd.to_datetime(df['unix_time'], unit='s')
    elif name == "eedi":
        return pd.to_datetime(df['unix_time'], unit='s')
    elif name == "junyi_15":
        return pd.to_datetime(df['unix_time'], unit='s')
    else:
        raise ValueError("Dataset is not supported: " + name)


###############################################################################
# Datetime features
###############################################################################

def month_one_hot(data_dict):
    """Create a sparse dataframe containing month one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information
    Output:
        month_df (pandas DataFrame): sparse one-hot month embedding
    """
    data = data_dict["interaction_df"]
    date_df = extract_datetime(data, data_dict["dataset"])
    month_df = data[["U_ID"]].copy()
    month_df["month"] = date_df.dt.month
    month_df = pd.get_dummies(month_df, columns=['month'], sparse=True)
    return month_df


def week_one_hot(data_dict):
    """Create a sparse dataframe containing week one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information
    Output:
        week_df (pandas DataFrame): sparse one-hot week embedding
    """
    data = data_dict["interaction_df"]
    date_df = extract_datetime(data, data_dict["dataset"])
    week_df = data[["U_ID"]].copy()
    week_df["week"] = date_df.dt.isocalendar().week
    week_df = pd.get_dummies(week_df, columns=['week'], sparse=True)
    return week_df


def day_one_hot(data_dict):
    """Create a sparse dataframe containing day one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information
    Output:
        day_df (pandas DataFrame): sparse one-hot day embedding
    """
    data = data_dict["interaction_df"]
    date_df = extract_datetime(data, data_dict["dataset"])
    day_df = data[["U_ID"]].copy()
    day_df["day"] = date_df.dt.weekday
    day_df = pd.get_dummies(day_df, columns=['day'], sparse=True)
    return day_df


def hour_one_hot(data_dict):
    """Create a sparse dataframe containing hour one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information
    Output:
        hour_df (pandas DataFrame): sparse one-hot hour embedding
    """
    data = data_dict["interaction_df"]
    date_df = extract_datetime(data, data_dict["dataset"])
    hour_df = data[["U_ID"]].copy()
    hour_df["hour"] = date_df.dt.hour
    hour_df = pd.get_dummies(hour_df, columns=['hour'], sparse=True)
    return hour_df


def weekend_one_hot(data_dict):
    """Create a sparse dataframe containing weekend one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information
    Output:
        weekend_df (pandas DataFrame): sparse one-hot weekend embedding
    """
    data = data_dict["interaction_df"]
    date_df = extract_datetime(data, data_dict["dataset"])
    wend_df = data[["U_ID"]].copy()
    wend_df["weekend"] = date_df.dt.weekday
    wend_df["weekend"] = \
        np.array([WEEKEND_ENCODER[d] for d in wend_df['weekend']])
    wend_df = pd.get_dummies(wend_df, columns=['weekend'], sparse=True)
    return wend_df


def part_of_day_one_hot(data_dict):
    """Create a sparse dataframe containing part of day one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information
    Output:
        pod_df (pandas DataFrame): sparse one-hot part of day embedding
    """
    data = data_dict["interaction_df"]
    date_df = extract_datetime(data, data_dict["dataset"])
    pod_df = data[["U_ID"]].copy()
    pod_df["part_of_day"] = date_df.dt.hour
    pod_df["part_of_day"] = np.array([POD[h] for h in pod_df["part_of_day"]])
    pod_df = pd.get_dummies(pod_df, columns=['part_of_day'], sparse=True)
    return pod_df
