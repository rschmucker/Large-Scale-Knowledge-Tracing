# This file will continue features describing the reading behaviour of students
import numpy as np
from collections import defaultdict
import config.constants as c
import src.preprocessing.features.feature_util as feature_util


###############################################################################
# Reading features
###############################################################################

def user_reading_count(p_dict):
    """Create a dataframe containing reading count features. We count how many
    explanations a student has read overall and per skill at a certain point.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]

    if p_dict["dataset"] == "elemmath_2021":
        cs = ['U_ID', 'user_id', 'skill_id', 'timestamp']
        df = p_dict["partition_df"][cs].copy()
        df_raw = p_dict["partition_raw"]

        print("Processing partition ", p_id)
        tcr_count, scr_count, o_verif = np.empty(0), np.empty(0), np.empty(0)
        for i, user_id in enumerate(df["user_id"].unique()):
            feature_util.ping(p_id, i)
            df_user = df[df["user_id"] == user_id].copy()
            df_user_raw = df_raw[df_raw["user_id"] == user_id].copy()
            user_tcr_count, user_scr_count = [], []

            ri = 0
            tcr_counter = 0
            scr_dict = defaultdict(lambda: 0)
            for s_id, a_time in df_user[["skill_id", "timestamp"]].values:
                while df_user_raw["timestamp"].values[ri] < a_time:
                    if df_user_raw["event"].values[ri] == c.READING:
                        tcr_counter += 1
                        scr_dict[int(df_user_raw["tag_code"].values[ri])] += 1
                    ri += 1
                user_tcr_count.append(tcr_counter)
                user_scr_count.append(scr_dict[s_id])

            tcr_count = np.concatenate((tcr_count, np.array(user_tcr_count)))
            scr_count = np.concatenate((scr_count, np.array(user_scr_count)))
            o_verif = np.concatenate((o_verif,
                                     np.ones(len(df_user)) * user_id))
        assert np.count_nonzero(df["user_id"].values - o_verif) == 0, \
            "IDs are not aligned for p_id " + str(p_id)
        df.drop(columns=['user_id', 'skill_id', 'timestamp'], inplace=True)
    elif p_dict["dataset"] == "ednet_kt3":
        tcr_count = p_dict["partition_df"]["rc_total"].values
        scr_count = p_dict["partition_df"]["rc_part"].values
        df = p_dict["partition_df"][["U_ID"]].copy()
    elif p_dict["dataset"] == "junyi_15":
        tcr_count = p_dict["partition_df"]["rc_total"].values
        scr_count = p_dict["partition_df"]["rc_skill"].values
        df = p_dict["partition_df"][["U_ID"]].copy()
    else:
        raise ValueError("reading count unvailable for: " + p_dict["dataset"])

    df["rc_total"] = feature_util.phi(tcr_count)
    df["rc_skill"] = feature_util.phi(scr_count)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def user_reading_time(p_dict):
    """Create a dataframe containing reading time features. We count much time
    has a student spent on reading overall and per skill at a certain point.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]

    if p_dict["dataset"] == "elemmath_2021":
        cs = ['U_ID', 'user_id', 'skill_id', 'timestamp']
        df = p_dict["partition_df"][cs].copy()
        df_raw = p_dict["partition_raw"]

        print("Processing partition ", p_id)
        ttr_time, str_time, o_verif = np.empty(0), np.empty(0), np.empty(0)
        for i, user_id in enumerate(df["user_id"].unique()):
            feature_util.ping(p_id, i)
            df_user = df[df["user_id"] == user_id].copy()
            df_user_raw = df_raw[df_raw["user_id"] == user_id].copy()
            user_ttr_time, user_str_time = [], []

            ri = 0
            ttr_timer = 0
            str_dict = defaultdict(lambda: 0)
            for s_id, a_time in df_user[["skill_id", "timestamp"]].values:
                while df_user_raw["timestamp"].values[ri] < a_time:
                    if df_user_raw["event"].values[ri] == c.READING:
                        cur_time = df_user_raw["timestamp"].values[ri]
                        next_time = df_user_raw["timestamp"].values[ri + 1]
                        gap = (next_time - cur_time) / 60.0
                        ttr_timer += gap
                        skills = int(df_user_raw["tag_code"].values[ri])
                        str_dict[skills] += gap
                    ri += 1
                user_ttr_time.append(ttr_timer)
                user_str_time.append(str_dict[s_id])

            ttr_time = np.concatenate((ttr_time, np.array(user_ttr_time)))
            str_time = np.concatenate((str_time, np.array(user_str_time)))
            o_verif = np.concatenate((o_verif,
                                     np.ones(len(df_user)) * user_id))
        assert np.count_nonzero(df["user_id"].values - o_verif) == 0, \
            "IDs are not aligned for p_id " + str(p_id)
        df.drop(columns=['user_id', 'skill_id', 'timestamp'], inplace=True)
    elif p_dict["dataset"] == "ednet_kt3":
        ttr_time = p_dict["partition_df"]["rt_total"].values / 60.0
        str_time = p_dict["partition_df"]["rt_part"].values / 60.0
        df = p_dict["partition_df"][["U_ID"]].copy()
    elif p_dict["dataset"] == "junyi_15":
        ttr_time = p_dict["partition_df"]["rt_total"].values / 60.0
        str_time = p_dict["partition_df"]["rt_skill"].values / 60.0
        df = p_dict["partition_df"][["U_ID"]].copy()
    else:
        raise ValueError("reading count unvailable for: " + p_dict["dataset"])

    df["rt_total"] = feature_util.phi(ttr_time)
    df["rt_skill"] = feature_util.phi(str_time)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1
