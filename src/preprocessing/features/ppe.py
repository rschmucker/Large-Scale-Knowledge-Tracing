"""Implements features from PPE paper."""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from config.constants import DATASET_PATH
from src.preprocessing.features import feature_util

# fixed choices in PPE paper
PPE_C = 0.1
PPE_X = 0.6


def Q_mat_to_dict(q_mat):
    """Little helper function to speed things up."""
    q_dict = {}
    for i, vec in enumerate(q_mat):
        indices = []
        for j, e in enumerate(vec):
            if e == 1:
                indices.append(j)
        q_dict[i] = indices
    return q_dict


def ppe_val(times, ppe_b, ppe_m):
    """Compute value of ppe feature."""
    if times.shape[0] < 2:
        return 0
    times = times / (24 * 3600)  # normalize to day
    ts = 0.0001 + times[-1] - times[:-1]  # last entry is current time
    lags = times[1:] - times[:-1]
    ti_x = ts ** (-PPE_X)
    normalizer = np.sum(1 / ti_x)
    wi = ti_x * normalizer

    Nc = lags.shape[0] ** PPE_C
    T = np.sum(ts * wi)
    d = ppe_b + (ppe_m * np.sum(1 / np.log(np.e + lags)) / lags.shape[0])
    val = Nc * (T ** (-d))
    return val


def ppe_feature(p_dict):
    """Create a dataframe containing ppe counts for user's skill attempts.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path, ppe_b, ppe_m = p_dict["p_id"], p_dict["p_path"], \
        p_dict["ppe_b"], p_dict["ppe_m"]
    print("ppe_b", ppe_b, "ppe_m", ppe_m)

    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    Q_dict = Q_mat_to_dict(Q_mat)
    cs = ['U_ID', 'user_id', 'item_id', 'unix_time']
    df = p_dict["partition_df"][cs].copy()
    num_skills = Q_mat.shape[1]

    print("processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        # skills = Q_mat[df_user["item_id"].astype(int)].copy()
        items = df_user["item_id"].values
        timestamps = df_user["unix_time"].values
        tmp = np.zeros((len(timestamps), num_skills))
        ts_dict = defaultdict(lambda: [])

        for i, item_id in enumerate(items):
            for j in Q_dict[item_id]:
                ts_dict[j].append(timestamps[i])
                tmp[i][j] = ppe_val(np.array(ts_dict[j]), ppe_b, ppe_m)

        tmps.append(sparse.csr_matrix(tmp))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id', 'unix_time'], inplace=True)

    # combine with U_Id frame
    skill_fail_mat = sparse.vstack(tmps)
    cols = ["RPFA_F_" + str(i) for i in range(num_skills)]
    skill_fail_mat = pd.DataFrame.sparse.from_spmatrix(skill_fail_mat,
                                                       columns=cols)
    skill_fail_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, skill_fail_mat)
    return 1
