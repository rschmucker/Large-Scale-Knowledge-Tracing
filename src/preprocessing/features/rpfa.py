"""Implements features from RPFA paper."""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from config.constants import DATASET_PATH
from src.preprocessing.features import feature_util

FAILURE_DECAY = 0.1
PROP_DECAY = 0.6
N_GHOST = 3


def recency_count_failures(p_dict):
    """Create a dataframe containing weighted counts for user's skill failures.

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
        corrects = df_user["correct"].values
        tmp = np.zeros((len(corrects), num_skills))
        fail_dict = defaultdict(lambda: 0)

        for i, s_vec in enumerate(skills):
            for j, e in enumerate(s_vec):
                if e == 1:
                    tmp[i][j] = fail_dict[j]
                    fail_dict[j] = FAILURE_DECAY * \
                        ((1 - corrects[i]) + fail_dict[j])

        tmps.append(sparse.csr_matrix(tmp))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id', 'correct'], inplace=True)

    # combine with U_Id frame
    skill_fail_mat = sparse.vstack(tmps)
    cols = ["RPFA_F_" + str(i) for i in range(num_skills)]
    skill_fail_mat = pd.DataFrame.sparse.from_spmatrix(skill_fail_mat,
                                                       columns=cols)
    skill_fail_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, skill_fail_mat)
    return 1


def recency_count_proportion(p_dict):
    """Create a dataframe containing weighted proportion for skill correctness.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """

    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    cs = ['U_ID', 'user_id', 'item_id', 'correct']
    df = p_dict["partition_df"][cs].copy()
    num_skills = Q_mat.shape[1]

    # ghost init
    ghost_init = sum([PROP_DECAY ** i for i in range(1, N_GHOST + 1)])

    print("processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id].copy()
        skills = Q_mat[df_user["item_id"].astype(int)].copy()
        corrects = df_user["correct"].values
        tmp = np.zeros((len(corrects), num_skills))

        prop_dict = defaultdict(lambda: 0)
        weight_acc = defaultdict(lambda: ghost_init)

        for i, s_vec in enumerate(skills):
            for j, e in enumerate(s_vec):
                if e == 1:
                    tmp[i][j] = prop_dict[j] / weight_acc[j]
                    prop_dict[j] = PROP_DECAY * (corrects[i] + prop_dict[j])
                    weight_acc[j] = PROP_DECAY * (1 + weight_acc[j])

        tmps.append(sparse.csr_matrix(tmp))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'item_id', 'correct'], inplace=True)

    # combine with U_Id frame
    skill_prop_mat = sparse.vstack(tmps)
    cols = ["RPFA_R_" + str(i) for i in range(num_skills)]
    skill_prop_mat = pd.DataFrame.sparse.from_spmatrix(skill_prop_mat,
                                                       columns=cols)
    skill_prop_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, skill_prop_mat)
    return 1
