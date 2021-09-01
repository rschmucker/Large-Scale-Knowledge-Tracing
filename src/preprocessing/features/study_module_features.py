import numpy as np
import pandas as pd
from collections import defaultdict
from config.constants import N_SM
import src.preprocessing.features.feature_util as feature_util


def smodule_attempts(p_dict):
    """Create a dataframe containing counts for a user's study module attempts.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    df = p_dict["partition_df"]
    if "hashed_skill_id" in df:
        df["skill_id"] = df["hashed_skill_id"]
    cs = ['U_ID', 'user_id', 's_module', 'skill_id']
    df = df[cs].copy()
    n_sm = N_SM[p_dict["dataset"]]

    print("Processing partition ", p_id)
    vec, order_verifier = np.empty((0, n_sm)), np.empty(0)
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        smodule_wins = get_smodule_attempts(df_user, n_sm)
        vec = np.concatenate((vec, smodule_wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))

    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df = df.drop(columns=['user_id', 's_module', 'skill_id'], inplace=False)
    cs = ['smA' + str(i) for i in range(n_sm)]
    df[cs] = pd.DataFrame(vec, index=df.index)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def get_smodule_attempts(df_user: pd.DataFrame, n_sm: int) -> np.ndarray:
    skill2smc = defaultdict(lambda: [0] * n_sm)
    smodule_attempts = []

    for sm, skill in df_user[["s_module", "skill_id"]].values:
        attempts = skill2smc[skill]
        smodule_attempts.append(attempts.copy())
        attempts[sm - 1] += 1
        skill2smc[skill] = attempts

    return feature_util.phi(np.array(smodule_attempts))


def smodule_wins(p_dict):
    """Create a dataframe containing counts for a user's study module wins.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    df = p_dict["partition_df"]
    if "hashed_skill_id" in df:
        df["skill_id"] = df["hashed_skill_id"]
    cs = ['U_ID', 'user_id', 's_module', 'skill_id', 'correct']
    df = df[cs].copy()
    n_sm = N_SM[p_dict["dataset"]]

    print("Processing partition ", p_id)
    vec, order_verifier = np.empty((0, n_sm)), np.empty(0)
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        smodule_wins = get_smodule_wins(df_user, n_sm)
        vec = np.concatenate((vec, smodule_wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))

    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    cs = ['user_id', 's_module', 'skill_id', 'correct']
    df = df.drop(columns=cs, inplace=False)
    cs = ['smW' + str(i) for i in range(n_sm)]
    df[cs] = pd.DataFrame(vec, index=df.index)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def get_smodule_wins(df_user: pd.DataFrame, n_sm: int) -> np.ndarray:
    skill2smc = defaultdict(lambda: [0] * n_sm)
    smodule_corrects = []

    cs = ["s_module", "skill_id", "correct"]
    for sm, skill, cor in df_user[cs].values:
        corrects = skill2smc[skill]
        smodule_corrects.append(corrects.copy())
        corrects[sm - 1] += cor
        skill2smc[skill] = corrects

    return feature_util.phi(np.array(smodule_corrects))
