import os
import numpy as np
import pandas as pd
from scipy import sparse
from config.constants import DATASET_PATH
import src.preprocessing.features.feature_util as feature_util


###############################################################################
# Interaction time features
###############################################################################

# The computation of categorical features follows: SAINT+
# https://arxiv.org/abs/2010.12042

def user_response_time(p_dict):
    """Create a dataframe containing response time for every user interaction

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, partition_df, p_path, = \
        p_dict["p_id"], p_dict["partition_df"], p_dict["p_path"]
    df = partition_df[['U_ID', 'user_id', 'item_id', 'timestamp']].copy()

    if p_dict["dataset"] == "squirrel":
        partition_raw = p_dict["partition_raw"]
        df_raw = partition_raw[partition_raw["event"] == 2]

        print("Processing partition ", p_id)
        response_times, order_verifier = np.empty(0), np.empty(0)
        for i, user_id in enumerate(df["user_id"].unique()):
            feature_util.ping(p_id, i)
            df_user = df[df["user_id"] == user_id].copy()
            df_user_raw = df_raw[df_raw["user_id"] == user_id].copy()

            user_times = []
            vals = df_user[["item_id", "timestamp"]].values
            for item_id, answer_time in vals:
                rel_data = df_user_raw[
                    (df_user_raw["question_ids"] == item_id) &
                    (df_user_raw["timestamp"] <= answer_time)
                ]
                if len(rel_data) == 0:
                    user_times.append(-1)
                else:
                    ask_time = np.max(rel_data["timestamp"].values)
                    resp_time = answer_time - ask_time
                    user_times.append(resp_time)
            response_times = np.concatenate((response_times,
                                            np.array(user_times)))
            order_verifier = np.concatenate((order_verifier,
                                            np.ones(len(df_user)) * user_id))
        assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
            "IDs are not aligned for p_id " + str(p_id)
    elif p_dict["dataset"] in ["ednet_kt3", "junyi_15"]:
        response_times = partition_df["response_time"].values
    else:
        raise ValueError("response time unvailable for: " + p_dict["dataset"])

    df.drop(columns=['user_id', 'item_id', 'timestamp'], inplace=True)
    df["response_time"] = response_times

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def user_response_time_cat(p_dict):
    """Create a dataframe containing categorical response time for every user
    interaction as well as phi values.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    feature_path = DATASET_PATH[p_dict["dataset"]] + "features/"
    if not os.path.isfile(feature_path + "resp_time.pkl"):
        raise RuntimeError("Compute basic response feature first")
    resp_time = pd.read_pickle(feature_path + "resp_time.pkl")
    resp_time.isnull().sum().sum() == 0, "Error, NaN values in resp_time data."

    p_id, partition_df, p_path = \
        p_dict["p_id"], p_dict["partition_df"], p_dict["p_path"]
    df = partition_df[['U_ID', 'user_id']].copy()

    # Combine frames and assert that length is the same
    old_len = len(df)
    df = pd.merge(df, resp_time, how='left', left_on='U_ID', right_on='U_ID')
    assert len(df) == old_len, "Length change is not inteded"

    print("Processing partition ", p_id)
    # to check user order
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        tmp = np.zeros((len(df_user), 302))
        for i, t in enumerate(df_user["response_time"].values):
            if t < 0:  # Flag missing values
                tmp[i][-1] = 1
            else:
                tmp[i][-2] = feature_util.phi(t)
                t = int(min(t, 300))
                tmp[i][t] = 1
        tmps.append(sparse.csr_matrix(tmp))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'response_time'], inplace=True)

    # combine with U_Id frame
    resp_cat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    print("Response cat", resp_cat.shape)
    cols = ["resp_cat_" + str(i) for i in range(resp_cat.shape[1])]
    resp_df = pd.DataFrame.sparse.from_spmatrix(resp_cat, columns=cols)
    resp_df["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, resp_df)
    return 1


def user_prev_response_time_cat(p_dict):
    """Create a dataframe containing categorical response time for every user
    interaction as well as phi values shifted by one interaction.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    feature_path = DATASET_PATH[p_dict["dataset"]] + "features/"
    if not os.path.isfile(feature_path + "resp_time.pkl"):
        raise RuntimeError("Compute basic response feature first")
    resp_time = pd.read_pickle(feature_path + "resp_time.pkl")
    resp_time.isnull().sum().sum() == 0, "Error, NaN values in resp_time data."

    p_id, partition_df, p_path = \
        p_dict["p_id"], p_dict["partition_df"], p_dict["p_path"]
    df = partition_df[['U_ID', 'user_id']].copy()

    # Combine frames and assert that length is the same
    old_len = len(df)
    df = pd.merge(df, resp_time, how='left', left_on='U_ID', right_on='U_ID')
    assert len(df) == old_len, "Length change is not inteded"

    print("Processing partition ", p_id)
    # to check user order
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        tmp = np.zeros((len(df_user), 302))

        # shift interaction times by one
        resp_times = np.concatenate([[-1],
                                     df_user["response_time"].values[:-1]])
        for i, t in enumerate(resp_times):
            if t < 0:  # Flag missing values
                tmp[i][-1] = 1
            else:
                tmp[i][-2] = feature_util.phi(t)
                t = int(min(t, 300))
                tmp[i][t] = 1
        tmps.append(sparse.csr_matrix(tmp))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'response_time'], inplace=True)

    # combine with U_Id frame
    resp_cat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    print("Prev Response cat", resp_cat.shape)
    cols = ["prev_resp_cat_" + str(i) for i in range(resp_cat.shape[1])]
    resp_df = pd.DataFrame.sparse.from_spmatrix(resp_cat, columns=cols)
    resp_df["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, resp_df)
    return 1


def user_lag_time(p_dict):
    """Create a dataframe containing lag time for every user interaction

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, partition_df, p_path = \
        p_dict["p_id"], p_dict["partition_df"], p_dict["p_path"]
    df = partition_df[['U_ID', 'user_id', 'item_id', 'timestamp']].copy()

    if p_dict["dataset"] == "squirrel":
        partition_raw = p_dict["partition_raw"]
        df_raw = partition_raw[partition_raw["event"] == 2]

        print("Processing partition ", p_id)
        lag_times, order_verifier = np.empty(0), np.empty(0)
        for i, user_id in enumerate(df["user_id"].unique()):
            feature_util.ping(p_id, i)
            df_user = df[df["user_id"] == user_id].copy()
            df_user_raw = df_raw[df_raw["user_id"] == user_id].copy()

            user_times = [-1]
            vals = df_user[["item_id", "timestamp"]].values
            for item_id, answer_time in vals:
                rel_data = df_user_raw[
                    (df_user_raw["timestamp"] >= answer_time)
                ]
                if len(rel_data) == 0:
                    user_times.append(-1)
                else:
                    new_question_time = np.min(rel_data["timestamp"].values)
                    lag_time = new_question_time - answer_time
                    user_times.append(lag_time)
            lag_times = np.concatenate((lag_times, np.array(user_times[:-1])))
            order_verifier = np.concatenate((order_verifier,
                                            np.ones(len(df_user)) * user_id))
        assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
            "IDs are not aligned for p_id " + str(p_id)
    elif p_dict["dataset"] in ["ednet_kt3", "junyi_15"]:
        lag_times = partition_df["lag_time"].values
    else:
        raise ValueError("lag time unavailable for: " + p_dict["dataset"])

    df.drop(columns=['user_id', 'item_id', 'timestamp'], inplace=True)
    df["lag_time"] = lag_times

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def user_lag_time_cat(p_dict):
    """Create a dataframe containing categorical lag time for every user
    interaction as well as phi values.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    feature_path = DATASET_PATH[p_dict["dataset"]] + "features/"
    if not os.path.isfile(feature_path + "lag_time.pkl"):
        raise RuntimeError("Compute basic lag_time feature first")
    lag_time = pd.read_pickle(feature_path + "lag_time.pkl")
    lag_time.isnull().sum().sum() == 0, "Error, Nan values in lag_time data."

    p_id, partition_df, p_path = \
        p_dict["p_id"], p_dict["partition_df"], p_dict["p_path"]
    df = partition_df[['U_ID', 'user_id']].copy()

    # Combine frames and assert that length is the same
    old_len = len(df)
    df = pd.merge(df, lag_time, how='left', left_on='U_ID', right_on='U_ID')
    assert len(df) == old_len, "Length change is not inteded"

    print("Processing partition ", p_id)
    # to check user order
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        tmp = np.zeros((len(df_user), 152))

        lag_times = df_user["lag_time"].values / 60.0  # convert to minutes
        for i, t in enumerate(lag_times):
            if t < 0:  # Flag missing values
                tmp[i][-1] = 1
            else:
                tmp[i][-2] = feature_util.phi(t)
                if t < 6:
                    tmp[i][int(t % 6)] = 1
                else:  # t >= 6
                    t = min(t, 1440)
                    tmp[i][int(6 + (t // 10))] = 1
        tmps.append(sparse.csr_matrix(tmp))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'lag_time'], inplace=True)

    # combine with U_Id frame
    lag_cat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    print("Lag cat", lag_cat.shape)
    cols = ["lag_cat_" + str(i) for i in range(lag_cat.shape[1])]
    lag_df = pd.DataFrame.sparse.from_spmatrix(lag_cat, columns=cols)
    lag_df["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, lag_df)
    return 1


def user_prev_lag_time_cat(p_dict):
    """Create a dataframe containing categorical lag time for every user
    interaction as well as phi values shifted by one interaction.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    feature_path = DATASET_PATH[p_dict["dataset"]] + "features/"
    if not os.path.isfile(feature_path + "lag_time.pkl"):
        raise RuntimeError("Compute basic lag_time feature first")
    lag_time = pd.read_pickle(feature_path + "lag_time.pkl")
    lag_time.isnull().sum().sum() == 0, "Error, Nan values in lag_time data."

    p_id, partition_df, p_path = \
        p_dict["p_id"], p_dict["partition_df"], p_dict["p_path"]
    df = partition_df[['U_ID', 'user_id']].copy()

    # Combine frames and assert that length is the same
    old_len = len(df)
    df = pd.merge(df, lag_time, how='left', left_on='U_ID', right_on='U_ID')
    assert len(df) == old_len, "Length change is not inteded"

    print("Processing partition ", p_id)
    # to check user order
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        tmp = np.zeros((len(df_user), 152))

        lag_times = df_user["lag_time"].values / 60.0  # convert to minutes
        lag_times = np.concatenate([[-1], lag_times[:-1]])  # shift by one
        for i, t in enumerate(lag_times):
            if t < 0:  # Flag missing values
                tmp[i][-1] = 1
            else:
                tmp[i][-2] = feature_util.phi(t)
                if t < 6:
                    tmp[i][int(t % 6)] = 1
                else:  # t >= 6
                    t = min(t, 1440)
                    tmp[i][int(6 + (t // 10))] = 1
        tmps.append(sparse.csr_matrix(tmp))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'lag_time'], inplace=True)

    # combine with U_Id frame
    lag_cat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    print("Prev Lag cat", lag_cat.shape)
    cols = ["prev_lag_cat_" + str(i) for i in range(lag_cat.shape[1])]
    lag_df = pd.DataFrame.sparse.from_spmatrix(lag_cat, columns=cols)
    lag_df["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, lag_df)
    return 1
