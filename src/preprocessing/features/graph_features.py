import numpy as np
import pandas as pd
from scipy import sparse
from config.constants import DATASET_PATH
from src.preprocessing.features.feature_util import phi


###############################################################################
# Graph features
###############################################################################


def pre_skill_count_attempts(p_dict):
    """Create a dataframe with counts for a user's attempts on a prereq skill.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    path = DATASET_PATH[p_dict["dataset"]] + 'preparation/q_mat.npz'
    Q_mat = sparse.load_npz(path).toarray()
    path = DATASET_PATH[p_dict["dataset"]] + 'preparation/pre_mat.npz'
    Pre_mat = sparse.load_npz(path).toarray()
    cs = ['U_ID', 'user_id', 'item_id']
    df = p_dict["partition_df"][cs].copy()

    if not p_dict["dataset"] == "junyi_15":
        QP_mat = Q_mat @ Pre_mat
    else:  # Junyi 15 has item level prereques
        QP_mat = Pre_mat
    num_skills = QP_mat.shape[1]

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        if i % 1000 == 0:
            print("Ping", p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        skills = QP_mat[df_user["item_id"].astype(int)].copy()
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        tmp_m = phi(np.cumsum(tmp, 0) * skills)
        tmps.append(sparse.csr_matrix(tmp_m))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id'], inplace=True)

    # combine with U_Id frame
    pre_skill_attempt_mat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    cs = ["scA_" + str(i) for i in range(num_skills)]
    pre_skill_attempt_mat = \
        pd.DataFrame.sparse.from_spmatrix(pre_skill_attempt_mat, columns=cs)
    pre_skill_attempt_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    print("STORING SDF", p_id)
    pre_skill_attempt_mat.to_pickle(p_path)
    print("Completed partition ", p_id, df.shape, pre_skill_attempt_mat.shape)
    return 1


def pre_skill_count_wins(p_dict):
    """Create a dataframe with counts for a user's wins on a prereq skills.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    path = DATASET_PATH[p_dict["dataset"]] + 'preparation/q_mat.npz'
    Q_mat = sparse.load_npz(path).toarray()
    path = DATASET_PATH[p_dict["dataset"]] + 'preparation/pre_mat.npz'
    Pre_mat = sparse.load_npz(path).toarray()
    cs = ['U_ID', 'user_id', 'item_id', 'correct']
    df = p_dict["partition_df"][cs].copy()

    if not p_dict["dataset"] == "junyi_15":
        QP_mat = Q_mat @ Pre_mat
    else:  # Junyi 15 has item level prereques
        QP_mat = Pre_mat
    num_skills = QP_mat.shape[1]

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        skills = QP_mat[df_user["item_id"].astype(int)].copy()
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        corrects = np.hstack((np.array(0),
                             df_user["correct"])).reshape(-1, 1)[:-1]
        wins = phi(np.cumsum(tmp * corrects, 0) * skills)
        tmps.append(sparse.csr_matrix(wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id', 'correct'], inplace=True)

    # combine with U_Id frame
    pre_skill_win_mat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    cs = ["scW_" + str(i) for i in range(num_skills)]
    pre_skill_win_mat = \
        pd.DataFrame.sparse.from_spmatrix(pre_skill_win_mat, columns=cs)
    pre_skill_win_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    print("STORING SDF", p_id)
    pre_skill_win_mat.to_pickle(p_path)
    print("Completed partition ", p_id, df.shape, pre_skill_win_mat.shape)
    return 1


def post_skill_count_attempts(p_dict):
    """Create a dataframe with counts for a user's attempts on a skill.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    path = DATASET_PATH[p_dict["dataset"]] + 'preparation/q_mat.npz'
    Q_mat = sparse.load_npz(path).toarray()
    path = DATASET_PATH[p_dict["dataset"]] + 'preparation/pre_mat.npz'
    Pre_mat = sparse.load_npz(path).toarray()
    cs = ['U_ID', 'user_id', 'item_id']
    df = p_dict["partition_df"][cs].copy()

    # Pre_mat transpose captures post reqs
    if not p_dict["dataset"] == "junyi_15":
        QP_mat = Q_mat @ (Pre_mat.T)
    else:  # Junyi 15 has item level prereques
        QP_mat = Pre_mat.T
    num_skills = QP_mat.shape[1]

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        if i % 1000 == 0:
            print("Ping", p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        skills = QP_mat[df_user["item_id"].astype(int)].copy()
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        tmp_m = phi(np.cumsum(tmp, 0) * skills)
        tmps.append(sparse.csr_matrix(tmp_m))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id'], inplace=True)

    # combine with U_Id frame
    post_skill_att_mat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    cs = ["scA_" + str(i) for i in range(num_skills)]
    post_skill_att_mat = \
        pd.DataFrame.sparse.from_spmatrix(post_skill_att_mat, columns=cs)
    post_skill_att_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    print("STORING SDF", p_id)
    post_skill_att_mat.to_pickle(p_path)
    print("Completed partition ", p_id, df.shape, post_skill_att_mat.shape)
    return 1


def post_skill_count_wins(p_dict):
    """Create a dataframe with counts for a user's wins on a postreq skills.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    path = DATASET_PATH[p_dict["dataset"]] + 'preparation/q_mat.npz'
    Q_mat = sparse.load_npz(path).toarray()
    path = DATASET_PATH[p_dict["dataset"]] + 'preparation/pre_mat.npz'
    Pre_mat = sparse.load_npz(path).toarray()
    cs = ['U_ID', 'user_id', 'item_id', 'correct']
    df = p_dict["partition_df"][cs].copy()

    # Pre_mat transpose captures post reqs
    if not p_dict["dataset"] == "junyi_15":
        QP_mat = Q_mat @ (Pre_mat.T)
    else:  # Junyi 15 has item level prereques
        QP_mat = Pre_mat.T
    num_skills = QP_mat.shape[1]

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        skills = QP_mat[df_user["item_id"].astype(int)].copy()
        tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
        corrects = np.hstack((np.array(0),
                             df_user["correct"])).reshape(-1, 1)[:-1]
        wins = phi(np.cumsum(tmp * corrects, 0) * skills)
        tmps.append(sparse.csr_matrix(wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id', 'correct'], inplace=True)

    # combine with U_Id frame
    post_skill_win_mat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    cs = ["scW_" + str(i) for i in range(num_skills)]
    post_skill_win_mat = \
        pd.DataFrame.sparse.from_spmatrix(post_skill_win_mat, columns=cs)
    post_skill_win_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    print("STORING SDF", p_id)
    post_skill_win_mat.to_pickle(p_path)
    print("Completed partition ", p_id, df.shape, post_skill_win_mat.shape)
    return 1
