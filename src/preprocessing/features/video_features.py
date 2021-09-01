import numpy as np
from collections import defaultdict
import config.constants as c
import src.preprocessing.features.feature_util as feature_util


###############################################################################
# Video features
###############################################################################

def videos_watched(p_dict):
    """Create a dataframe containing video watched counts. We count how many
    videos a student has watched overall and per skill at a certain point.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]

    if p_dict["dataset"] == "squirrel":
        cs = ['U_ID', 'user_id', 'skill_id', 'timestamp']
        df = p_dict["partition_df"][cs].copy()
        df_raw = p_dict["partition_raw"]

        print("Processing partition ", p_id)
        tcw_count, scw_count, o_ver = np.empty(0), np.empty(0), np.empty(0)
        for i, user_id in enumerate(df["user_id"].unique()):
            feature_util.ping(p_id, i)
            df_user = df[df["user_id"] == user_id].copy()
            df_user_raw = df_raw[df_raw["user_id"] == user_id].copy()
            user_tcw_count, user_scw_count = [], []

            ri, tcw_counter = 0, 0
            scw_dict = defaultdict(lambda: 0)
            for (s_id, a_time) in df_user[["skill_id", "timestamp"]].values:
                while df_user_raw["timestamp"].values[ri] < a_time:
                    if df_user_raw["event"].values[ri] == c.WATCH_VIDEO:
                        if np.isnan(df_user_raw["tag_code"].values[ri]):
                            print("FOUND NAN")
                            tcw_counter += 1
                        else:
                            tcw_counter += 1
                            skills = int(df_user_raw["tag_code"].values[ri])
                            scw_dict[skills] += 1
                    ri += 1
                user_tcw_count.append(tcw_counter)
                user_scw_count.append(scw_dict[s_id])

            tcw_count = np.concatenate((tcw_count, np.array(user_tcw_count)))
            scw_count = np.concatenate((scw_count, np.array(user_scw_count)))
            o_ver = np.concatenate((o_ver, np.ones(len(df_user)) * user_id))
        assert np.count_nonzero(df["user_id"].values - o_ver) == 0, \
            "IDs are not aligned for p_id " + str(p_id)
        df.drop(columns=['user_id', 'skill_id', 'timestamp'], inplace=True)
    elif p_dict["dataset"] == "ednet_kt3":
        tcw_count = p_dict["partition_df"]["vc_total"].values
        scw_count = p_dict["partition_df"]["vc_part"].values
        df = p_dict["partition_df"][["U_ID"]].copy()
    else:
        raise ValueError("reading count unvailable for: " + p_dict["dataset"])

    df["vw_total"] = feature_util.phi(tcw_count)
    df["vw_skill"] = feature_util.phi(scw_count)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def videos_time_watched(p_dict):
    """Create a dataframe containing time spent on videos. We sum how much
    time a student has spent on watching videos overall and per skill.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]

    if p_dict["dataset"] == "squirrel":
        cs = ['U_ID', 'user_id', 'skill_id', 'timestamp']
        df = p_dict["partition_df"][cs].copy()
        df_raw = p_dict["partition_raw"]

        print("Processing partition ", p_id)
        ttw_time, stw_time, o_ver = np.empty(0), np.empty(0), np.empty(0)
        for i, user_id in enumerate(df["user_id"].unique()):
            feature_util.ping(p_id, i)
            df_user = df[df["user_id"] == user_id].copy()
            df_user_raw = df_raw[df_raw["user_id"] == user_id].copy()
            user_ttw_time, user_stw_time = [], []

            ri = 0
            ttw_timer = 0
            stw_dict = defaultdict(lambda: 0)
            for (s_id, a_time) in df_user[["skill_id", "timestamp"]].values:
                while df_user_raw["timestamp"].values[ri] < a_time:
                    if df_user_raw["event"].values[ri] == c.WATCH_VIDEO:
                        cur_time = df_user_raw["timestamp"].values[ri]
                        next_time = df_user_raw["timestamp"].values[ri + 1]
                        gap = (next_time - cur_time) / 60.0
                        ttw_timer += gap
                        if np.isnan(df_user_raw["tag_code"].values[ri]):
                            print("FOUND NAN")
                        else:
                            skills = int(df_user_raw["tag_code"].values[ri])
                            stw_dict[skills] += gap
                    ri += 1
                user_ttw_time.append(ttw_timer)
                user_stw_time.append(stw_dict[s_id])

            ttw_time = np.concatenate((ttw_time, np.array(user_ttw_time)))
            stw_time = np.concatenate((stw_time, np.array(user_stw_time)))
            o_ver = np.concatenate((o_ver, np.ones(len(df_user)) * user_id))
        assert np.count_nonzero(df["user_id"].values - o_ver) == 0, \
            "IDs are not aligned for p_id " + str(p_id)
        df.drop(columns=['user_id', 'skill_id', 'timestamp'], inplace=True)
    elif p_dict["dataset"] == "ednet_kt3":
        ttw_time = p_dict["partition_df"]["vt_total"].values
        stw_time = p_dict["partition_df"]["vt_part"].values
        df = p_dict["partition_df"][["U_ID"]].copy()
    else:
        raise ValueError("reading count unvailable for: " + p_dict["dataset"])

    df["vt_total"] = feature_util.phi(ttw_time)
    df["vt_skill"] = feature_util.phi(stw_time)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def videos_skipped(p_dict):
    """Create a dataframe containing video skipped counts. We count how many
    videos a student has skipped overall and per skill at a certain point.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]

    if p_dict["dataset"] == "squirrel":
        cs = ['U_ID', 'user_id', 'skill_id', 'timestamp']
        df = p_dict["partition_df"][cs].copy()
        df_raw = p_dict["partition_raw"]

        print("Processing partition ", p_id)
        tcs_count, scs_count, o_ver = np.empty(0), np.empty(0), np.empty(0)
        for i, user_id in enumerate(df["user_id"].unique()):
            feature_util.ping(p_id, i)
            df_user = df[df["user_id"] == user_id].copy()
            df_user_raw = df_raw[df_raw["user_id"] == user_id].copy()
            user_tcs_count, user_scs_count = [], []

            ri = 0
            tcs_counter = 0
            scs_dict = defaultdict(lambda: 0)
            for (s_id, a_time) in df_user[["skill_id", "timestamp"]].values:
                while df_user_raw["timestamp"].values[ri] < a_time:
                    if df_user_raw["event"].values[ri] == c.SKIP_VIDEO:
                        if np.isnan(df_user_raw["tag_code"].values[ri]):
                            print("FOUND NAN")
                            tcs_counter += 1
                        else:
                            tcs_counter += 1
                            skills = int(df_user_raw["tag_code"].values[ri])
                            scs_dict[skills] += 1
                    ri += 1
                user_tcs_count.append(tcs_counter)
                user_scs_count.append(scs_dict[s_id])

            tcs_count = np.concatenate((tcs_count, np.array(user_tcs_count)))
            scs_count = np.concatenate((scs_count, np.array(user_scs_count)))
            o_ver = np.concatenate((o_ver, np.ones(len(df_user)) * user_id))
        assert np.count_nonzero(df["user_id"].values - o_ver) == 0, \
            "IDs are not aligned for p_id " + str(p_id)
        df.drop(columns=['user_id', 'skill_id', 'timestamp'], inplace=True)
    elif p_dict["dataset"] == "ednet_kt3":
        tcs_count = p_dict["partition_df"]["vs_total"].values
        scs_count = p_dict["partition_df"]["vs_part"].values
        df = p_dict["partition_df"][["U_ID"]].copy()
    else:
        raise ValueError("reading count unvailable for: " + p_dict["dataset"])

    df["vs_total"] = feature_util.phi(tcs_count)
    df["vs_skill"] = feature_util.phi(scs_count)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1
