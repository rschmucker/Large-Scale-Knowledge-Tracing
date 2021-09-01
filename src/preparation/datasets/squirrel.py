import os
import numpy as np
import pandas as pd
from scipy import sparse
from config.constants import DATASET_PATH, MIN_INTERACTIONS_PER_USER


INT_COLUMNS = [
    "user_id",
    "item_id",
    "correct",
    "skill_id",
    "school_id",
    "item_difficulty",
    "s_module",
    "app_type",
    "teacher_id",
    "topic_id",
    "unix_time"
]
# Outlier user that breaks sorted order
OT = 608105


def standardize_df_squirrel(df):
    """Standardize the data to the conventional column format
    Arguments:
        df (pandas dataframe): ElemMATHdata.csv format data

    Outputs:
        df (pandas dataFrame): save data with standardized column names
    """
    renamed_df = df.rename(columns={
        'question_ids': 'item_id',
        'is_right': 'correct',
        'tag_code': 'skill_id',
        'q_difficulty': 'item_difficulty'}
    )
    # Convert time from ms to seconds
    renamed_df["timestamp"] = renamed_df["server_time"] // 1000
    for col in INT_COLUMNS:
        if col in renamed_df:
            renamed_df[col] = renamed_df[col].astype(np.int64)
    return renamed_df


def prepare_squirrel(n_splits):
    # import data
    df = pd.read_csv(os.path.join(DATASET_PATH["squirrel"],
                     "ElemMATHdata_03_2021.csv"))
    df.drop(df[df['s_module'] == 7].index, inplace=True)
    print('# of data imported: ' + str(df.shape[0]))
    print('min server time: ' + str(df["server_time"].min()))
    # Unix time in seconds adjusted for pytz "Asia/Shanghai"
    df["unix_time"] = (df["server_time"].values / 1000 + (13 * 3600))
    df["server_time"] = df["server_time"] - df["server_time"].min()
    print(df.head(3))

    # core df: user, question interaction with meta info
    core_df = df[['user_id', 'question_ids', 'server_time', 'is_right',
                  'tag_code', 'school_id', 'q_difficulty', 's_module',
                  'app_type', 'course_id', 'teacher_id', 'topic_id',
                  'unix_time', "date_time"]].dropna()
    core_df = standardize_df_squirrel(core_df)
    print('# of core data: ' + str(core_df.shape[0]))
    print(core_df.head(3))

    # video df: user, video interaction
    video_df = df[['user_id', 'video_id', 'video_type', 'video_duration',
                   'event_begin', 'event_end', 'server_time', 'tag_code',
                   'event']].fillna(value={'event_end': 0}).dropna()
    video_df = standardize_df_squirrel(video_df)
    print('# of video data: ' + str(video_df.shape[0]))
    print(video_df.head(3))

    # if_active_check df: user, questions expanation checking activity
    active_check_df = df[['user_id', 'server_time', 'tag_code',
                          'if_active_check']].dropna()
    active_check_df = standardize_df_squirrel(active_check_df)
    print('# of active check data: ' + str(active_check_df.shape[0]))
    print(active_check_df.head(3))

    # Filter out too short sequences
    # We filter out 608105 because it interupts the sequence of user 83817
    core_df = \
        core_df.groupby("user_id").filter(lambda x:
                                          (len(x) >= MIN_INTERACTIONS_PER_USER)
                                          and (x["user_id"].unique()[0] != OT))

    # Build Q-matrix
    num_items = core_df["item_id"].max() + 1
    num_skills = core_df["skill_id"].max() + 1
    Q_mat = np.zeros((num_items, num_skills))
    for item_id, skill_id in core_df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Build Pre-matrix
    structure_df = pd.read_csv(os.path.join(DATASET_PATH["squirrel"],
                                            "ElemMATH_KnowledgeStructure.csv"))
    Pre_mat = np.zeros((num_skills, num_skills))
    cs = ["PREtag_code", "POSTtag_code"]
    for pre_id, post_id in structure_df[cs].values:
        Pre_mat[post_id, pre_id] = 1
    assert len(structure_df) - np.sum(Pre_mat) == 0, "Only one entry per edge"
    Pre_mat = sparse.csr_matrix(Pre_mat)

    # Data is already sorted by users and temporally for each user
    video_df = video_df[['user_id', 'timestamp', 'video_id', 'video_type',
                         'video_duration', 'event_begin', 'event_end',
                         'skill_id', 'event']]
    active_check_df = active_check_df[['user_id', 'timestamp', 'skill_id',
                                       'if_active_check']]
    core_df.reset_index(inplace=True, drop=True)
    video_df.reset_index(inplace=True, drop=True)
    active_check_df.reset_index(inplace=True, drop=True)

    # Prepare splits for cross-validation
    from src.preparation.prepare_data import determine_splits
    determine_splits(core_df, "squirrel", n_splits=n_splits)

    # Save data
    prep_path = os.path.join(DATASET_PATH["squirrel"], "preparation")
    Q_mat = sparse.csr_matrix(Q_mat)
    Pre_mat = sparse.csr_matrix(Pre_mat)
    sparse.save_npz(os.path.join(prep_path, "q_mat.npz"), Q_mat)
    sparse.save_npz(os.path.join(prep_path, "pre_mat.npz"), Pre_mat)
    core_path = os.path.join(prep_path, "preprocessed_data.csv")
    video_path = os.path.join(prep_path, "preprocessed_video_data.csv")
    active_path = os.path.join(prep_path, "preprocessed_active_check_data.csv")
    core_df.to_csv(core_path, sep="\t", index=False)
    video_df.to_csv(video_path, sep="\t", index=False)
    active_check_df.to_csv(active_path, sep="\t", index=False)
