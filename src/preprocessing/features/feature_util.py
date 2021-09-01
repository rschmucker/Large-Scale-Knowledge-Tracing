# Some helper functions that are shared between multiple features
import numpy as np


def phi(x):
    return np.log(1 + x)


def phi_inv(x):
    return np.exp(x) - 1


def ping(p_id, i):
    if i % 1000 == 0:
        print("Ping", p_id, i)


def store_partial_df(p_id, p_path, df):
    print("STORING DF", p_id)
    df.to_pickle(p_path)
    print("Completed partition ", p_id, df.shape)


def get_Q_mat_dict(Q_mat):
    # Transform q-matrix into dictionary for fast lookup
    num_items = Q_mat.shape[0]
    Q_mat_dict = {i: set() for i in range(num_items)}
    for i, j in np.argwhere(Q_mat == 1):
        Q_mat_dict[i].add(j)
    return Q_mat_dict


def one_vector(data_dict):
    """Create a dataframe containing a one vector.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        one_df (pandas DataFrame): df containing one vector
    """
    one_df = data_dict["interaction_df"][['U_ID']].copy()
    one_df["ones"] = np.ones(len(one_df))
    return one_df
