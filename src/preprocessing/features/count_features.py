import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from config.constants import DATASET_PATH
from src.preprocessing.features import feature_util


###############################################################################
# Count features
###############################################################################

def total_count_attempts(p_dict):
    """Create a dataframe containing counts for user's total previous attempts.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    df = p_dict["partition_df"][['U_ID', 'user_id']].copy()

    print("processing partition ", p_id)
    order_verifier = np.empty(0)
    attempt_vec = np.empty(0)
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        counts = np.arange(len(df_user))
        attempt_vec = np.concatenate((attempt_vec, feature_util.phi(counts)))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df = df.drop(columns=['user_id'], inplace=False)
    df["tcA"] = attempt_vec

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def total_count_wins(p_dict):
    """Create a dataframe containing counts for a user's total previous wins.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    df = p_dict["partition_df"][['U_ID', 'user_id', 'correct']].copy()

    print("processing partition ", p_id)
    order_verifier = np.empty(0)
    win_vec = np.empty(0)
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        user_vec = np.concatenate((np.zeros(1),
                                  np.cumsum(df_user["correct"])[:-1]))
        win_vec = np.concatenate((win_vec, feature_util.phi(user_vec)))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df = df.drop(columns=['user_id', 'correct'], inplace=False)
    df["tcW"] = win_vec

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def skill_count_attempts(p_dict):
    """Create a dataframe containing counts for a user's attempts on a skill.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    df = p_dict["partition_df"][['U_ID', 'user_id', 'item_id']].copy()
    num_skills = Q_mat.shape[1]

    print("processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        skills = Q_mat[df_user["item_id"].astype(int)].copy()
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        tmp_m = feature_util.phi(np.cumsum(tmp, 0) * skills)
        tmps.append(sparse.csr_matrix(tmp_m))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df.drop(columns=['user_id', 'item_id'], inplace=True)
    # combine with U_Id frame
    skill_attempt_mat = sparse.vstack(tmps)
    cols = ["scA_" + str(i) for i in range(num_skills)]
    skill_attempt_mat = pd.DataFrame.sparse.from_spmatrix(skill_attempt_mat,
                                                          columns=cols)
    skill_attempt_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, skill_attempt_mat)
    return 1


def skill_count_wins(p_dict):
    """Create a dataframe containing counts for a user's wins on a skill.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    cs = ['U_ID', 'user_id', 'item_id', 'correct']
    df = p_dict["partition_df"][cs].copy()
    num_skills = Q_mat.shape[1]

    print("processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        skills = Q_mat[df_user["item_id"].astype(int)].copy()
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        corrects = np.hstack((np.array(0),
                             df_user["correct"])).reshape(-1, 1)[:-1]
        wins = feature_util.phi(np.cumsum(tmp * corrects, 0) * skills)
        tmps.append(sparse.csr_matrix(wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id', 'correct'], inplace=True)

    # combine with U_Id frame
    skill_win_mat = sparse.vstack(tmps)
    cols = ["scW_" + str(i) for i in range(num_skills)]
    skill_win_mat = pd.DataFrame.sparse.from_spmatrix(skill_win_mat,
                                                      columns=cols)
    skill_win_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, skill_win_mat)
    return 1


def item_count_attempts(p_dict):
    """Create a dataframe containing counts for a user's attempts on an item.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    df = p_dict["partition_df"][['U_ID', 'user_id', 'item_id']].copy()

    print("processing partition ", p_id)
    item_attempt_vec, order_verifier = np.empty(0), np.empty(0)
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        item_ids = df_user["item_id"].values
        onehot = OneHotEncoder(categories=[range(max(item_ids) + 1)])
        item_ids_oh = onehot.fit_transform(item_ids.reshape(-1, 1)).toarray()
        tmp = np.vstack((np.zeros(item_ids_oh.shape[1]),
                        np.cumsum(item_ids_oh, 0)))[:-1]
        u_attempts = feature_util.phi(tmp[np.arange(len(item_ids)), item_ids])
        item_attempt_vec = np.concatenate((item_attempt_vec, u_attempts))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id'], inplace=True)
    df["icA"] = item_attempt_vec

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def item_count_wins(p_dict):
    """Create a dataframe containing counts for a user's wins on an item.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    cs = ['U_ID', 'user_id', 'item_id', 'correct']
    df = p_dict["partition_df"][cs].copy()

    print("processing partition ", p_id)
    item_attempt_vec, order_verifier = np.empty(0), np.empty(0)
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        it_ids = df_user["item_id"].values
        labels = df_user['correct'].values.reshape(-1, 1)
        onehot = OneHotEncoder(categories=[range(max(it_ids) + 1)])
        item_ids_oh = onehot.fit_transform(it_ids.reshape(-1, 1)).toarray()
        tmp = np.vstack((np.zeros(item_ids_oh.shape[1]),
                        np.cumsum(item_ids_oh * labels, 0)))[:-1]
        user_attempts = feature_util.phi(tmp[np.arange(len(it_ids)), it_ids])

        item_attempt_vec = np.concatenate((item_attempt_vec, user_attempts))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id', 'correct'], inplace=True)
    df["icW"] = item_attempt_vec

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def part_count_attempts(p_dict):
    """Create a dataframe containing counts for a user's attempts on a part.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    Part_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                               'preparation/part_mat.npz').toarray()
    df = p_dict["partition_df"][['U_ID', 'user_id', 'item_id']].copy()
    num_parts = Part_mat.shape[1]

    print("processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        parts = Part_mat[df_user["item_id"].astype(int)].copy()
        tmp = np.vstack((np.zeros(num_parts), parts))[:-1]
        tmp_m = feature_util.phi(np.cumsum(tmp, 0) * parts)
        tmps.append(sparse.csr_matrix(tmp_m))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df.drop(columns=['user_id', 'item_id'], inplace=True)
    # combine with U_Id frame
    part_attempt_mat = sparse.vstack(tmps)
    cols = ["partcA_" + str(i) for i in range(num_parts)]
    part_attempt_mat = pd.DataFrame.sparse.from_spmatrix(part_attempt_mat,
                                                         columns=cols)
    part_attempt_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, part_attempt_mat)
    return 1


def part_count_wins(p_dict):
    """Create a dataframe containing counts for a user's wins on a part.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    Part_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                               'preparation/part_mat.npz').toarray()
    cs = ['U_ID', 'user_id', 'item_id', 'correct']
    df = p_dict["partition_df"][cs].copy()
    num_skills = Part_mat.shape[1]

    print("processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        parts = Part_mat[df_user["item_id"].astype(int)].copy()
        tmp = np.vstack((np.zeros(num_skills), parts))[:-1]
        corrects = np.hstack((np.array(0),
                             df_user["correct"])).reshape(-1, 1)[:-1]
        wins = feature_util.phi(np.cumsum(tmp * corrects, 0) * parts)
        tmps.append(sparse.csr_matrix(wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id', 'correct'], inplace=True)

    # combine with U_Id frame
    part_win_mat = sparse.vstack(tmps)
    cols = ["partcW_" + str(i) for i in range(num_skills)]
    part_win_mat = pd.DataFrame.sparse.from_spmatrix(part_win_mat,
                                                     columns=cols)
    part_win_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, part_win_mat)
    return 1
