import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from config.constants import DATASET_PATH
from src.preprocessing.features import feature_util


# [month, week, day, hour]
WINDOW_LENGTHS = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
N_WINDOW = len(WINDOW_LENGTHS) + 1  # last window is total time


class TimeWindowQueue:
    """A queue for counting efficiently the number of events in time windows.
    Complexity:
        All operators in amortized O(W) time where W is the number of windows.

    From JJ's KTM repository: https://github.com/jilljenn/ktm.
    """
    def __init__(self, window_lengths):
        self.queue = []
        self.window_lengths = window_lengths
        self.cursors = [0] * len(self.window_lengths)

    def __len__(self):
        return len(self.queue)

    def get_counters(self, t):
        self.update_cursors(t)
        v = [len(self.queue)] + \
            [len(self.queue) - cursor for cursor in self.cursors]
        return v

    def push(self, time):
        self.queue.append(time)

    def update_cursors(self, t):
        for pos, length in enumerate(self.window_lengths):
            while (self.cursors[pos] < len(self.queue) and
                   t - self.queue[self.cursors[pos]] >= length):
                self.cursors[pos] += 1


def time_window_total_count_attempts(p_dict):
    """Create a dataframe containing counts for a user's total previous
    attempts in time windows.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    cs = ['U_ID', 'user_id', 'timestamp']
    df = p_dict["partition_df"][cs].copy()

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    counters = defaultdict(lambda: TimeWindowQueue(WINDOW_LENGTHS))
    attempt_vec = np.empty((0, N_WINDOW))
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        attempts = np.zeros((len(df_user), N_WINDOW))
        for i, time in enumerate(df_user['timestamp']):
            vs = np.array(counters[user_id].get_counters(time))
            counts = feature_util.phi(vs)
            attempts[i, :] = counts
            counters[user_id].push(time)
        attempt_vec = np.concatenate((attempt_vec, attempts))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df = df.drop(columns=['user_id', 'timestamp'], inplace=False)
    df[["tcA_TW0", "tcA_TW1", "tcA_TW2", "tcA_TW3", "tcA_TW4"]] = \
        pd.DataFrame(attempt_vec, index=df.index)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def time_window_total_count_wins(p_dict):
    """Create a dataframe containing counts for a user's total previous wins
    in time windows.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    cs = ['U_ID', 'user_id', 'timestamp', 'correct']
    df = p_dict["partition_df"][cs].copy()

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    counters = defaultdict(lambda: TimeWindowQueue(WINDOW_LENGTHS))
    win_vec = np.empty((0, N_WINDOW))
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        wins = np.zeros((len(df_user), N_WINDOW))
        cs = ['timestamp', 'correct']
        for i, (time, correct) in enumerate(df_user[cs].values):
            vals = np.array(counters[user_id, 'correct'].get_counters(time))
            counts = feature_util.phi(vals)
            wins[i, :] = counts
            if correct:
                counters[user_id, 'correct'].push(time)
        win_vec = np.concatenate((win_vec, wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df = df.drop(columns=['user_id', 'timestamp', 'correct'], inplace=False)
    df[["tcW_TW0", "tcW_TW1", "tcW_TW2", "tcW_TW3", "tcW_TW4"]] = \
        pd.DataFrame(win_vec, index=df.index)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def time_window_skill_count_attempts(p_dict):
    """Create a dataframe containing counts for a user's attempts on a skill
    in time window.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    cs = ['U_ID', 'user_id', 'timestamp', 'item_id']
    df = p_dict["partition_df"][cs].copy()
    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    num_skills = Q_mat.shape[1]
    Q_mat_dict = feature_util.get_Q_mat_dict(Q_mat)

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    counters = defaultdict(lambda: TimeWindowQueue(WINDOW_LENGTHS))
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        attempts = np.zeros((len(df_user), num_skills * N_WINDOW))
        cs = ['timestamp', 'item_id']
        for i, (time, item) in enumerate(df_user[cs].values):
            for s_id in Q_mat_dict[int(item)]:
                s_id = int(s_id)
                vals = counters[user_id, s_id, "skill"].get_counters(time)
                counts = feature_util.phi(np.array(vals))
                attempts[i, s_id * N_WINDOW:(s_id + 1) * N_WINDOW] = counts
                counters[user_id, s_id, "skill"].push(time)
        tmps.append(sparse.csr_matrix(attempts))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df = df.drop(columns=['user_id', 'timestamp', 'item_id'], inplace=False)
    # combine with U_Id frame
    tws_attempt_mat = sparse.vstack(tmps)
    cols = []
    for i in range(N_WINDOW):
        cols += ["scA_TW" + str(i) + "_" + str(j) for j in range(num_skills)]
    tws_attempt_mat = pd.DataFrame.sparse.from_spmatrix(tws_attempt_mat,
                                                        columns=cols)
    tws_attempt_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, tws_attempt_mat)
    return 1


def time_window_skill_count_wins(p_dict):
    """Create a dataframe containing counts for a user's wins on a skill in
    time window.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    cs = ['U_ID', 'user_id', 'timestamp', 'item_id', 'correct']
    df = p_dict["partition_df"][cs].copy()
    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    num_skills = Q_mat.shape[1]
    Q_mat_dict = feature_util.get_Q_mat_dict(Q_mat)

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    counters = defaultdict(lambda: TimeWindowQueue(WINDOW_LENGTHS))
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        wins = np.zeros((len(df_user), num_skills * N_WINDOW))
        cs = ['timestamp', 'item_id', 'correct']
        for i, (ts, item, correct) in enumerate(df_user[cs].values):
            for s_id in Q_mat_dict[int(item)]:
                s_id = int(s_id)
                vals = counters[user_id, s_id, "skill", "cor"].get_counters(ts)
                counts = feature_util.phi(np.array(vals))
                wins[i, s_id * N_WINDOW:(s_id + 1) * N_WINDOW] = counts
                if correct:
                    counters[user_id, s_id, "skill", "cor"].push(ts)
        tmps.append(sparse.csr_matrix(wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    cs = ['user_id', 'timestamp', 'item_id', 'correct']
    df = df.drop(columns=cs, inplace=False)
    # combine with U_Id frame
    tws_win_mat = sparse.vstack(tmps)
    cols = []
    for i in range(N_WINDOW):
        cols += ["scW_TW" + str(i) + "_" + str(j) for j in range(num_skills)]
    tws_win_mat = pd.DataFrame.sparse.from_spmatrix(tws_win_mat, columns=cols)
    tws_win_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, tws_win_mat)
    return 1


def time_window_item_count_attempts(p_dict):
    """Create a dataframe containing counts for a user's attempts on an item
    in time window.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    cs = ['U_ID', 'user_id', 'timestamp', 'item_id']
    df = p_dict["partition_df"][cs].copy()

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    counters = defaultdict(lambda: TimeWindowQueue(WINDOW_LENGTHS))
    attempt_vec = np.empty((0, N_WINDOW))
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        attempts = np.zeros((len(df_user), N_WINDOW))
        cs = ['timestamp', 'item_id']
        for i, (time, item) in enumerate(df_user[cs].values):
            vals = counters[user_id, item, "item"].get_counters(time)
            counts = feature_util.phi(np.array(vals))
            attempts[i, :] = counts
            counters[user_id, item, "item"].push(time)
        attempt_vec = np.concatenate((attempt_vec, attempts))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    df.drop(columns=['user_id', 'timestamp', 'item_id'], inplace=True)
    df[["icA_TW0", "icA_TW1", "icA_TW2", "icA_TW3", "icA_TW4"]] = \
        pd.DataFrame(attempt_vec, index=df.index)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1


def time_window_item_count_wins(p_dict):
    """Create a dataframe containing counts for a user's wins on an item in
    time window.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path = p_dict["p_id"], p_dict["p_path"]
    cs = ['U_ID', 'user_id', 'timestamp', 'item_id', 'correct']
    df = p_dict["partition_df"][cs].copy()

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    counters = defaultdict(lambda: TimeWindowQueue(WINDOW_LENGTHS))
    win_vec = np.empty((0, N_WINDOW))
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        wins = np.zeros((len(df_user), N_WINDOW))

        cs = ['timestamp', 'item_id', 'correct']
        for i, (time, it, correct) in enumerate(df_user[cs].values):
            vals = counters[user_id, it, 'item', 'correct'].get_counters(time)
            counts = feature_util.phi(np.array(vals))
            wins[i, :] = counts
            if correct:
                counters[user_id, it, 'item', 'correct'].push(time)
        win_vec = np.concatenate((win_vec, wins))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)

    cs = ['user_id', 'timestamp', 'item_id', 'correct']
    df.drop(columns=cs, inplace=True)
    df[["icW_TW0", "icW_TW1", "icW_TW2", "icW_TW3", "icW_TW4"]] = \
        pd.DataFrame(win_vec, index=df.index)

    # safe for later combination
    feature_util.store_partial_df(p_id, p_path, df)
    return 1
