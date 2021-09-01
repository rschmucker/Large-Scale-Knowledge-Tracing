import numpy as np
import pandas as pd
from scipy import sparse


###############################################################################
# n-gram features
###############################################################################

def sequence_n_gram(p_dict):
    """Create a dataframe containing ids for different correct patterns

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    p_id, p_path, n_steps = p_dict["p_id"], p_dict["p_path"], p_dict["n_steps"]
    df = p_dict["partition_df"][['U_ID', 'user_id', 'correct']].copy()
    cap = 2 ** n_steps

    print("Processing partition ", p_id)
    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        if i % 1000 == 0:
            print("Ping", p_id, i)
        df_user = df[df["user_id"] == user_id].copy()
        labels = df_user['correct'].values

        grams = np.zeros((df_user.shape[0], cap + 1))
        c = 0
        for i, l in enumerate(labels):
            if i < n_steps:
                grams[i][cap] = 1
            else:
                grams[i][c] = 1
            c = ((2 * c) % cap) + l

        tmps.append(sparse.csr_matrix(grams))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_id)
    df.drop(columns=['user_id', 'correct'], inplace=True)

    # combine with U_Id frame
    gram_mat = sparse.vstack(tmps)
    print("Creating SDF", p_id)
    cols = [str(n_steps) + "-gram_" + str(i) for i in range(cap + 1)]
    gram_mat = pd.DataFrame.sparse.from_spmatrix(gram_mat, columns=cols)
    gram_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    print("STORING SDF", p_id)
    gram_mat.to_pickle(p_path)
    print("Completed partition ", p_id, df.shape, gram_mat.shape)
    return 1
