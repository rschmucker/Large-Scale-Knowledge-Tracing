import numpy as np
import pandas as pd
from scipy import sparse
from config.constants import DATASET_PATH
from src.preprocessing.features import feature_util

MIN_SCHOOL_CNT = 1
MIN_TEACH_CNT = 1


###############################################################################
# One-hot features
###############################################################################

def user_one_hot(data_dict):
    """Create a sparse dataframe containing user one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        user_df (pandas DataFrame): sparse one-hot user embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    columns.remove('U_ID')
    columns.remove('user_id')
    user_df = data.drop(columns=columns, inplace=False)
    user_df = pd.get_dummies(user_df, columns=['user_id'], sparse=True)
    return user_df


def user_skill_one_hot(data_dict):
    """Create a sparse dataframe containing user one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        user_df (pandas DataFrame): sparse one-hot user embedding
    """
    data = data_dict["interaction_df"]
    data['user_id'] = \
        1000000 * data['user_id'].values + data['skill_id'].values
    columns = list(data.columns)
    columns.remove('U_ID')
    columns.remove('user_id')
    user_df = data.drop(columns=columns, inplace=False)
    user_df = pd.get_dummies(user_df, columns=['user_id'], sparse=True)
    return user_df



def item_one_hot(data_dict):
    """Create a sparse dataframe containing item one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information
    Output:
        item_df (pandas DataFrame): sparse one-hot item embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    columns.remove('U_ID')
    columns.remove('item_id')
    item_df = data.drop(columns=columns, inplace=False)
    item_df = pd.get_dummies(item_df, columns=['item_id'], sparse=True)
    return item_df


def skill_one_hot(p_dict):
    """Create a sparse dataframe containing skill one-hot encodings.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    columns = [c for c in list(p_dict["partition_df"].columns)
               if c not in ['U_ID', 'user_id', 'item_id']]
    df = p_dict["partition_df"].drop(columns=columns, inplace=False)

    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_dict["p_id"], i)
        df_user = df[df["user_id"] == user_id].copy()
        tmp_m = Q_mat[df_user["item_id"].astype(int)].copy()
        tmps.append(sparse.csr_matrix(tmp_m))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_dict["p_id"])
    df.drop(columns=['user_id', 'item_id'], inplace=True)

    # combine with U_Id frame
    print("Creating SDF")
    skill_oh_mat = sparse.vstack(tmps)
    cs = ["skill_" + str(i) for i in range(Q_mat.shape[1])]
    skill_oh_mat = pd.DataFrame.sparse.from_spmatrix(skill_oh_mat, columns=cs)
    skill_oh_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_dict["p_id"], p_dict["p_path"],
                                  skill_oh_mat)
    return 1


def school_one_hot(data_dict):
    """Create a sparse dataframe containing school one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        school_df (pandas DataFrame): sparse one-hot school embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    columns.remove('U_ID')
    columns.remove('school_id')
    school_df = data.drop(columns=columns, inplace=False)

    counts = pd.value_counts(school_df["school_id"])
    mask = school_df["school_id"].isin(counts[counts < MIN_SCHOOL_CNT].index)
    school_df["school_id"][mask] = -1

    school_df = pd.get_dummies(school_df, columns=['school_id'], sparse=True)
    return school_df


def teacher_one_hot(data_dict):
    """Create a sparse dataframe containing teacher one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        teacher_df (pandas DataFrame): sparse one-hot teacher embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    columns.remove('U_ID')
    columns.remove('teacher_id')
    teacher_df = data.drop(columns=columns, inplace=False)

    counts = pd.value_counts(teacher_df["teacher_id"])
    mask = teacher_df["teacher_id"].isin(counts[counts < MIN_TEACH_CNT].index)
    teacher_df["teacher_id"][mask] = -1

    teacher_df = pd.get_dummies(teacher_df, columns=['teacher_id'],
                                sparse=True)
    return teacher_df


def study_module_one_hot(data_dict):
    """Create a sparse dataframe containing study module one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        s_module_df (pandas DataFrame): sparse one-hot study module embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    assert "s_module" in columns, "The interaction_df must contain s_module."
    columns.remove('U_ID')
    columns.remove('s_module')
    s_module_df = data.drop(columns=columns, inplace=False)
    s_module_df = pd.get_dummies(s_module_df, columns=['s_module'],
                                 sparse=True)
    return s_module_df


def course_one_hot(data_dict):
    """Create a sparse dataframe containing course id one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        course_df (pandas DataFrame): sparse one-hot course id embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    assert "course_id" in columns, "The interaction_df must contain course_id."
    columns.remove('U_ID')
    columns.remove('course_id')
    course_df = data.drop(columns=columns, inplace=False)
    course_df = pd.get_dummies(course_df, columns=['course_id'], sparse=True)
    return course_df


def difficulty_one_hot(data_dict):
    """Create a sparse dataframe containing item difficulty one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        difficulty_df (pandas DataFrame): sparse one-hot difficulty embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    assert "item_difficulty" in columns, "Interaction_df needs difficulty."
    columns.remove('U_ID')
    columns.remove('item_difficulty')
    difficulty_df = data.drop(columns=columns, inplace=False)
    difficulty_df = pd.get_dummies(difficulty_df, columns=['item_difficulty'],
                                   sparse=True)
    return difficulty_df


def apptype_one_hot(data_dict):
    """Create a sparse dataframe containing item apptype one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        apptype_df (pandas DataFrame): sparse one-hot item apptype embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    assert "app_type" in columns, "The interaction_df must contain app_type."
    columns.remove('U_ID')
    columns.remove('app_type')
    apptype_df = data.drop(columns=columns, inplace=False) - 1
    apptype_df = pd.get_dummies(apptype_df, columns=['app_type'], sparse=True)
    print(apptype_df)
    return apptype_df


def topic_one_hot(data_dict):
    """Create a sparse dataframe containing item topic one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        topic_df (pandas DataFrame): sparse one-hot item topic embedding
    """
    data = data_dict["interaction_df"]
    columns = list(data.columns)
    assert "topic_id" in columns, "The interaction_df must contain topic_id."
    columns.remove('U_ID')
    columns.remove('topic_id')
    topic_df = data.drop(columns=columns, inplace=False)
    topic_df = pd.get_dummies(topic_df, columns=['topic_id'], sparse=True)
    print(topic_df)
    return topic_df


def bundle_one_hot(data_dict):
    """Create a sparse dataframe containing item bundle one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        bundle_df (pandas DataFrame): sparse one-hot item bundle embedding
    """
    data = data_dict["interaction_df"]
    assert "bundle_id" in list(data.columns), \
        "The interaction_df must contain bundle_id."
    bundle_df = data[["U_ID", "bundle_id"]].copy()
    bundle_df = pd.get_dummies(bundle_df, columns=['bundle_id'], sparse=True)
    print(bundle_df)
    return bundle_df


def social_support_one_hot(data_dict):
    """Create a sparse dataframe containing social support one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        bundle_df (pandas DataFrame): sparse one-hot item social support
    """
    data = data_dict["interaction_df"]
    assert "premium" in list(data.columns), \
        "The interaction_df must contain premium."
    premium_df = data[["U_ID", "premium"]].copy()
    premium_df = pd.get_dummies(premium_df, columns=['premium'], sparse=True)
    print(premium_df)
    return premium_df


def age_one_hot(data_dict):
    """Create a sparse dataframe containing user age one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        bundle_df (pandas DataFrame): sparse one-hot user age
    """
    data = data_dict["interaction_df"]
    assert "user_age" in list(data.columns), \
        "The interaction_df must contain user_age."
    age_df = data[["U_ID", "user_age"]].copy()
    age_df = pd.get_dummies(age_df, columns=['user_age'], sparse=True)
    print(age_df)
    return age_df


def gender_one_hot(data_dict):
    """Create a sparse dataframe containing gender one-hot encodings.

    Arguments:
        data_dict (dict): contains all relevant information

    Output:
        bundle_df (pandas DataFrame): sparse one-hot item gender
    """
    data = data_dict["interaction_df"]
    assert "gender" in list(data.columns), \
        "The interaction_df must contain gender."
    gender_df = data[["U_ID", "gender"]].copy()
    gender_df = pd.get_dummies(gender_df, columns=['gender'], sparse=True)
    print(gender_df)
    return gender_df


def part_one_hot(p_dict):
    """Create a sparse dataframe containing part one-hot encodings. The feature
    indicates in which of the 7 TOIEC categories an item falls into.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    Part_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                               'preparation/part_mat.npz').toarray()
    df = p_dict["partition_df"][['U_ID', 'user_id', 'item_id']].copy()

    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        feature_util.ping(p_dict["p_id"], i)
        df_user = df[df["user_id"] == user_id].copy()
        tmp_m = Part_mat[df_user["item_id"].astype(int)].copy()
        tmps.append(sparse.csr_matrix(tmp_m))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_dict["p_id"])
    df.drop(columns=['user_id', 'item_id'], inplace=True)

    # combine with U_Id frame
    print("Creating SDF")
    part_oh_mat = sparse.vstack(tmps)
    cs = ["part_" + str(i) for i in range(Part_mat.shape[1])]
    part_oh_mat = pd.DataFrame.sparse.from_spmatrix(part_oh_mat, columns=cs)
    part_oh_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    feature_util.store_partial_df(p_dict["p_id"], p_dict["p_path"],
                                  part_oh_mat)
    return 1


def prereq_one_hot(p_dict):
    """Create a sparse dataframe containing prereq one-hot encodings.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    Pre_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                              'preparation/pre_mat.npz').toarray()
    columns = [c for c in list(p_dict["partition_df"].columns)
               if c not in ['U_ID', 'user_id', 'item_id']]
    df = p_dict["partition_df"].drop(columns=columns, inplace=False)

    if not p_dict["dataset"] == "junyi_15":
        QP_mat = Q_mat @ Pre_mat
    else:  # Junyi 15 has item level prereques
        QP_mat = Pre_mat

    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        if i % 1000 == 0:
            print("Ping", p_dict["p_id"], i)
        df_user = df[df["user_id"] == user_id].copy()
        tmp_m = QP_mat[df_user["item_id"].astype(int)].copy()
        tmps.append(sparse.csr_matrix(tmp_m))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_dict["p_id"])
    df.drop(columns=['user_id', 'item_id'], inplace=True)

    # combine with U_Id frame
    print("Creating SDF")
    pre_oh_mat = sparse.vstack(tmps)
    cs = ["prereq_" + str(i) for i in range(QP_mat.shape[1])]
    pre_oh_mat = pd.DataFrame.sparse.from_spmatrix(pre_oh_mat, columns=cs)
    pre_oh_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    print("STORING SDF", p_dict["p_id"])
    pre_oh_mat.to_pickle(p_dict["p_path"])
    print("Completed partition ", p_dict["p_id"], df.shape, pre_oh_mat.shape)
    return 1


def postreq_one_hot(p_dict):
    """Create a sparse dataframe containing postreq one-hot encodings.

    Arguments:
        p_dict (dict): contains relevant information for parallel computation
    """
    Q_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                            'preparation/q_mat.npz').toarray()
    Pre_mat = sparse.load_npz(DATASET_PATH[p_dict["dataset"]] +
                              'preparation/pre_mat.npz').toarray()
    columns = [c for c in list(p_dict["partition_df"].columns)
               if c not in ['U_ID', 'user_id', 'item_id']]
    df = p_dict["partition_df"].drop(columns=columns, inplace=False)

    if not p_dict["dataset"] == "junyi_15":
        QP_mat = Q_mat @ (Pre_mat.T)
    else:  # Junyi 15 has item level prereques
        QP_mat = Pre_mat

    order_verifier = np.empty(0)
    tmps = []
    for i, user_id in enumerate(df["user_id"].unique()):
        if i % 1000 == 0:
            print("Ping", p_dict["p_id"], i)
        df_user = df[df["user_id"] == user_id].copy()
        tmp_m = QP_mat[df_user["item_id"].astype(int)].copy()
        tmps.append(sparse.csr_matrix(tmp_m))
        order_verifier = np.concatenate((order_verifier,
                                        np.ones(len(df_user)) * user_id))
    assert np.count_nonzero(df["user_id"].values - order_verifier) == 0, \
        "IDs are not aligned for p_id " + str(p_dict["p_id"])
    df.drop(columns=['user_id', 'item_id'], inplace=True)

    # combine with U_Id frame
    print("Creating SDF")
    post_oh_mat = sparse.vstack(tmps)
    cs = ["postreq_" + str(i) for i in range(QP_mat.shape[1])]
    post_oh_mat = pd.DataFrame.sparse.from_spmatrix(post_oh_mat, columns=cs)
    post_oh_mat["U_ID"] = df["U_ID"]

    # safe for later combination
    print("STORING SDF", p_dict["p_id"])
    post_oh_mat.to_pickle(p_dict["p_path"])
    print("Completed partition ", p_dict["p_id"], df.shape, post_oh_mat.shape)
    return 1
