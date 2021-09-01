import os
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz, csr_matrix
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils.queue import TimeWindowQueue
import constants as c


def phi(x):
    return np.log(1 + x)


WINDOW_LENGTHS = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
NUM_WINDOWS = len(WINDOW_LENGTHS) + 1


def get_difficulty_feature(df_user: np.ndarray) -> np.ndarray:
    """
    Get the difficulty feature for this user
    1. difficulty associated with each item
    2. average difficulty correct so far at this skill
    3. average difficulty incorrect so far at this skill

    Arguments:
        df_user: the dataframe of only this user
    
    Output:
        difficulty features for this user with each row corresponding to an item
    """
    # 2 & 3
    # running_record: <skill_id : correct_difficulty_sum, correct_num, incorrect_difficulty_sum, incorrect_num>
    CORRECT_DIFFICULTY_SUM_INDEX = 0
    CORRECT_NUM_INDEX = 1
    INCORRECT_DIFFICULTY_SUM_INDEX = 2
    INCORRECT_NUM_INDEX = 3
    running_record = {}

    user_ave_difficulties = []
    for row in df_user:
        skill = row[c.SKILL_ID]
        if skill not in running_record:
            running_record[skill] = [0, 0, 0, 0]

        record = running_record[skill]
        average_correct = record[CORRECT_DIFFICULTY_SUM_INDEX] / float(record[CORRECT_NUM_INDEX]) if record[CORRECT_NUM_INDEX] != 0 else 0
        average_incorrect = record[INCORRECT_DIFFICULTY_SUM_INDEX] / float(record[INCORRECT_NUM_INDEX]) if record[INCORRECT_NUM_INDEX] != 0 else 0
        user_ave_difficulties.append([average_correct, average_incorrect])

        if row[c.CORRECT] == 1:
            running_record[skill][CORRECT_DIFFICULTY_SUM_INDEX] += row[c.ITEM_DIFFICULTY]
            running_record[skill][CORRECT_NUM_INDEX] += 1
        else:
            running_record[skill][INCORRECT_DIFFICULTY_SUM_INDEX] += row[c.ITEM_DIFFICULTY]
            running_record[skill][INCORRECT_NUM_INDEX] += 1


    # combine past difficulties count with the current difficulty
    difficulty_feature = np.hstack((user_ave_difficulties, df_user[:, c.ITEM_DIFFICULTY].reshape(-1, 1))).reshape(-1, 3)
    return difficulty_feature


def get_video_feature(user_id: int, df_user: np.ndarray, video_df: pd.DataFrame) -> np.ndarray:
    """Get the video interaction feature for this user
    1. # videos this user has watched on this skill
    2. # videos this user has skipped on this skill

    Arguments:
        user_id: the specified user
        df_user: the core dataframe of only this user
        video_df: dataframe with all the video interaction info
    
    Output:
        video features for this user with each row corresponding to an item (skill)
    """
    video_df_user = video_df[video_df["user_id"] == user_id].reset_index(drop=True)

    video_counts = []
    for row in df_user:
        video_history_df = video_df_user[video_df_user["timestamp"] < row[c.TIMESTAMP]]
        watched = video_history_df[(video_history_df.event == c.WATCH_VIDEO) & (video_history_df.skill_id == row[c.SKILL_ID])].shape[0]
        skipped = video_history_df[(video_history_df.event == c.SKIP_VIDEO) & (video_history_df.skill_id == row[c.SKILL_ID])].shape[0]
        video_counts.append([watched, skipped])
    return phi(np.array(video_counts))

def get_smodule_feature(df_user: pd.DataFrame) -> np.ndarray:
    """Get the student performance on past smodules
    1. # attempt on each smodule for this skill (6)
    2. # correct on each smodule for this skill (6)
    3. current smodule onehot (6)

    Arguments:
        df_user: the core dataframe of only this user
    
    Output:
        module features for this user with each row corresponding to an item (skill)
    """

    skill2smcounts = {}
    smodule_attempts = []
    smodule_corrects = []
    onehots = []
    
    for row in df_user:
        cur_smodule = row[c.S_MODULE]
        cur_skill = row[c.SKILL_ID]
        attempts, corrects = None, None
        if cur_skill in skill2smcounts:
            attempts, corrects = skill2smcounts[cur_skill]
        else:
            attempts, corrects = [0] * 6, [0] * 6
        smodule_attempts.append(attempts.copy())
        smodule_corrects.append(corrects.copy())
        attempts[cur_smodule - 1] += 1
        if row[c.CORRECT]:
            corrects[cur_smodule - 1] += 1
        skill2smcounts[cur_skill] = (attempts, corrects)
        
        onehot = [0] * 6
        onehot[cur_smodule - 1] = 1
        onehots.append(onehot)
    smodule_attempts = smodule_attempts
    smodule_corrects = smodule_corrects
    
    return np.hstack((phi(np.array(smodule_attempts)), phi(np.array(smodule_corrects)), onehots))


def get_active_check_feature(user_id: int, df_user: np.ndarray, active_check_df: pd.DataFrame) -> np.ndarray:
    """Get active check features
    1. count for how many times this student checked question explanation on this skill

    Arguments:
        user_id: the specified user
        df_user: the core dataframe of only this user
        active_check_df: dataframe with all the checking behaviors
    
    Output:
        features for this user with each row corresponding to an item (skill)
    """
    active_check_df_user = active_check_df[active_check_df["user_id"] == user_id]
    check_counts = []
    for row in df_user:
        check_history_df = active_check_df_user[
            (active_check_df_user["timestamp"] < row[c.TIMESTAMP]) & 
            (active_check_df_user["if_active_check"] == 1) &
            (active_check_df_user["skill_id"] == row[c.SKILL_ID])
            ]
        checked = check_history_df.shape[0]
        check_counts.append([checked])
    return phi(np.array(check_counts))


def get_difficulty_buckets(difficulties: np.ndarray) -> np.ndarray:
    """Get difficulty buckets
    1. difficulty 10-20: easy [1,0,0], 30-70: medium [0,1,0], 80-90: difficulty [0,0,1]

    Arguments:
        difficulties: difficulties vector
    
    Output:
        difficulty buckets
    """
    buckets = []
    for diff in difficulties:
        if diff < 30:
            buckets.append([1, 0, 0])
        elif diff > 70:
            buckets.append([0, 0, 1])
        else:
            buckets.append([0, 1, 0])
    return np.array(buckets)


def get_global_item_correctness(df: np.ndarray) -> np.ndarray:
    """Get global item correctness
    1. global item correctness up to the current timestamp

    Arguments:
        df: dataframe
    
    Output:
        global item correctness for the current item/question

    Note:
        To avoid overfitting for low-frequency items (from the report https://dqanonymousdata.blob.core.windows.net/neurips-public/papers/aidemy/report_task_1_2.pdf):
        GI = (count(item_id) * mean_correctness(item_id) + w_smoothing * mean_correctness(global)) / (count(item_id) + w_smoothing)
    """

    prior = 0.5
    w_smoothing = 5

    # <item_id : [count, average correctness]>
    item_record = {}

    gi = []

    for row in df:
        correct = row[c.CORRECT]
        cur_item_id = row[c.ITEM_ID]

        global_average = None
        if 'global' not in item_record:
            global_average = prior

            item_record['global'] = [1, correct]
        else:
            cur_global_record = item_record['global']

            global_average = cur_global_record[1]

            new_global_average = ((cur_global_record[1] * cur_global_record[0]) + float(correct)) / (cur_global_record[0] + 1)
            item_record['global'] = [cur_global_record[0] + 1, new_global_average]

        cur_gi = None
        if cur_item_id not in item_record:
            cur_gi = global_average

            item_record[cur_item_id] = [1, correct]
        else:
            cur_item_record = item_record[cur_item_id]

            cur_gi = ((cur_item_record[0] * cur_item_record[1]) + (w_smoothing * global_average)) / (cur_item_record[0] + w_smoothing)

            new_average = ((cur_item_record[1] * cur_item_record[0]) + float(correct)) / (cur_item_record[0] + 1)
            item_record[cur_item_id] = [cur_item_record[0] + 1, new_average]
        gi.append([cur_gi])
    return np.array(gi)
        

def get_global_skill_correctness(df: np.ndarray) -> np.ndarray:
    """Get global skill correctness
    1. global skill correctness up to the current timestamp

    Arguments:
        df: dataframe
    
    Output:
        global skill correctness for the current skill/tagcode

    Note:
        To avoid overfitting for low-frequency items (from the report https://dqanonymousdata.blob.core.windows.net/neurips-public/papers/aidemy/report_task_1_2.pdf):
        GS = (count(skill_id) * mean_correctness(skill_id) + w_smoothing * mean_correctness(global)) / (count(skill_id) + w_smoothing)
    """

    prior = 0.5
    w_smoothing = 5

    # <skill_id : [count, average correctness]>
    skill_record = {}

    gs = []

    for row in df:
        correct = row[c.CORRECT]
        cur_skill_id = row[c.SKILL_ID]

        global_average = None
        if 'global' not in skill_record:
            global_average = prior

            skill_record['global'] = [1, correct]
        else:
            cur_global_record = skill_record['global']

            global_average = cur_global_record[1]

            new_global_average = ((cur_global_record[1] * cur_global_record[0]) + float(correct)) / (cur_global_record[0] + 1)
            skill_record['global'] = [cur_global_record[0] + 1, new_global_average]

        cur_gs = None
        if cur_skill_id not in skill_record:
            cur_gs = global_average

            skill_record[cur_skill_id] = [1, correct]
        else:
            cur_skill_record = skill_record[cur_skill_id]

            cur_gs = ((cur_skill_record[0] * cur_skill_record[1]) + (w_smoothing * global_average)) / (cur_skill_record[0] + w_smoothing)

            new_average = ((cur_skill_record[1] * cur_skill_record[0]) + float(correct)) / (cur_skill_record[0] + 1)
            skill_record[cur_skill_id] = [cur_skill_record[0] + 1, new_average]
        gs.append([cur_gs])
    return np.array(gs)
    

def get_global_school_correctness(df: np.ndarray) -> np.ndarray:
    """Get global school correctness
    1. global school correctness up to the current timestamp

    Arguments:
        df: dataframe
    
    Output:
        global school correctness for the current school

    Note:
        To avoid overfitting for low-frequency items (from the report https://dqanonymousdata.blob.core.windows.net/neurips-public/papers/aidemy/report_task_1_2.pdf):
        GSCH = (count(school_id) * mean_correctness(school_id) + w_smoothing * mean_correctness(global)) / (count(school_id) + w_smoothing)
    """

    prior = 0.5
    w_smoothing = 5

    # <school_id : [count, average correctness]>
    school_record = {}

    gsch = []

    for row in df:
        correct = row[c.CORRECT]
        cur_school_id = row[c.SCHOOL_ID]

        global_average = None
        if 'global' not in school_record:
            global_average = prior

            school_record['global'] = [1, correct]
        else:
            cur_global_record = school_record['global']

            global_average = cur_global_record[1]

            new_global_average = ((cur_global_record[1] * cur_global_record[0]) + float(correct)) / (cur_global_record[0] + 1)
            school_record['global'] = [cur_global_record[0] + 1, new_global_average]

        cur_gsch = None
        if cur_school_id not in school_record:
            cur_gsch = global_average

            school_record[cur_school_id] = [1, correct]
        else:
            cur_school_record = school_record[cur_school_id]

            cur_gsch = ((cur_school_record[0] * cur_school_record[1]) + (w_smoothing * global_average)) / (cur_school_record[0] + w_smoothing)

            new_average = ((cur_school_record[1] * cur_school_record[0]) + float(correct)) / (cur_school_record[0] + 1)
            school_record[cur_school_id] = [cur_school_record[0] + 1, new_average]
        gsch.append([cur_gsch])
    return np.array(gsch)


def get_response_time(df: np.ndarray) -> np.ndarray:
    """Get response time for the current data
    1. response time

    Arguments:
        df: dataframe
    
    Output:
        current question response time

    Note:
        some response time is not avaialable (-1), and will be replaced by global average of response time
    """

    response_time_np = df[:, c.RESPONSE_TIME].reshape(-1, 1)
    running_ave = 0
    count = 0
    for i, rt in enumerate(response_time_np):
        if rt == -1:
            response_time_np[i] = running_ave
        else:
            running_ave = ((running_ave * count) + rt) / (count + 1)
            count += 1
    return response_time_np


def get_item_based_response_time(df: np.ndarray) -> np.ndarray:
    """Get response time for the current data with reference to the item average response time
    way 1:
    1. current response time
    2. item average response time
    way 2:
    1. current / average
    way 3:
    1. |current - average|

    Arguments:
        df: dataframe
    
    Output:
        current question response time with reference to the item average response time

    Note:
        some response time is not avaialable (-1), and will be replaced by global average of response time
    """
    # first row response time
    prior = 26215
    w_smoothing = 5
    # <item_id : [count, average response time]>
    item_record = {}

    feature = []

    for row in df:
        response_time = row[c.RESPONSE_TIME]
        item_id = row[c.ITEM_ID]

        global_average = None
        if 'global' not in item_record:
            global_average = prior

            item_record['global'] = [1, response_time]
        else:
            cur_global_record = item_record['global']

            global_average = cur_global_record[1]

            if response_time != -1:
                new_global_average = ((cur_global_record[1] * cur_global_record[0]) + float(response_time)) / (cur_global_record[0] + 1)
                item_record['global'] = [cur_global_record[0] + 1, new_global_average]

        cur_feature = None
        if item_id not in item_record:
            cur_feature = global_average

            if response_time != -1:
                item_record[item_id] = [1, response_time]
        else:
            cur_item_record = item_record[item_id]

            cur_feature = ((cur_item_record[0] * cur_item_record[1]) + (w_smoothing * global_average)) / (cur_item_record[0] + w_smoothing)

            if response_time != -1:
                new_average = ((cur_item_record[1] * cur_item_record[0]) + float(response_time)) / (cur_item_record[0] + 1)
                item_record[item_id] = [cur_item_record[0] + 1, new_average]

        if response_time == -1:
            response_time = cur_feature
        # way 1
        # feature.append([response_time, cur_feature])
        # way 2
        # feature.append([response_time / float(cur_feature)])
        # way 3
        feature.append([abs(response_time - float(cur_feature))])
    return np.array(feature)


def get_user_based_response_time(df: np.ndarray) -> np.ndarray:
    """Get response time for the current data with reference to the user average response time
    way 1:
    1. current response time
    2. user average response time
    way 2:
    1. current / average
    way 3:
    1. |current - average|

    Arguments:
        df: dataframe
    
    Output:
        current question response time with reference to the user average response time

    Note:
        some response time is not avaialable (-1), and will be replaced by global average of response time
    """
    # first row response time
    prior = 26215
    w_smoothing = 5
    # <user_id : [count, average response time]>
    user_record = {}

    feature = []

    for row in df:
        response_time = row[c.RESPONSE_TIME]
        user_id = row[c.USER_ID]

        global_average = None
        if 'global' not in user_record:
            global_average = prior

            user_record['global'] = [1, response_time]
        else:
            cur_global_record = user_record['global']
            global_average = cur_global_record[1]

            if response_time != -1:
                new_global_average = ((cur_global_record[1] * cur_global_record[0]) + float(response_time)) / (cur_global_record[0] + 1)
                user_record['global'] = [cur_global_record[0] + 1, new_global_average]

        cur_feature = None
        if user_id not in user_record:
            cur_feature = global_average

            if response_time != -1:
                user_record[user_id] = [1, response_time]
        else:
            cur_user_record = user_record[user_id]

            cur_feature = ((cur_user_record[0] * cur_user_record[1]) + (w_smoothing * global_average)) / (cur_user_record[0] + w_smoothing)

            if response_time != -1:
                new_average = ((cur_user_record[1] * cur_user_record[0]) + float(response_time)) / (cur_user_record[0] + 1)
                user_record[user_id] = [cur_user_record[0] + 1, new_average]

        if response_time == -1:
            response_time = cur_feature
        # way 1
        # feature.append([response_time, cur_feature])
        # way 2
        # feature.append([response_time / float(cur_feature)])
        # way 3
        feature.append([abs(response_time - float(cur_feature))])
    return np.array(feature)


def df_to_sparse(df, video_df, active_check_df, Q_mat, active_features, X_base):
    """Build sparse dataset from dense dataset and q-matrix.

    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        video_df (pandas DataFrame): output by prepare_data.py
        active_check_df (pandas DataFrame): output by prepare_data.py
        Q_mat (sparse array): q-matrix, output by prepare_data.py. item-skill matrix
        active_features (list of str): features
        X_base (sparse): the base X file to build on top of

    Output:
        sparse_df (sparse array): sparse dataset where first 7 columns are the same as in df
    """
    print('active features: ')
    print(active_features)
    if X_base != None:
        print('X_base shape: ' + str(X_base.shape))

    _, num_skills = Q_mat.shape
    features = {}

    # Keep track of original dataset
    features['df'] = np.empty((0, len(df.keys())))
    # Keep track of base X file
    features['base'] = X_base

    # Skill features
    # num_skills length
    if 's' in active_features:
        features["s"] = sparse.csr_matrix(np.empty((0, num_skills)))

    # Past attempts and wins features
    # each num_skills (past item count for relevent skill) + 1 (past count for this item_id) + 1 (past count for all items)
    for key in ['a', 'w']:
        if key in active_features:
            features[key] = sparse.csr_matrix(np.empty((0, num_skills + 2)))

    # Past item difficulties
    # 1 (average difficulty correct so far on this skill) + 1 (average difficulty incorrecct so far on this skill) + 1 (difficulty of current item)
    if 'd' in active_features:
        # difficulty feature: [current item difficulty, average difficulties completed]
        features["d"] = sparse.csr_matrix(np.empty((0, 3)))

    # Past video interaction histories
    # 1 (videos on this skill watched) + 1 (videos on this skill skipped)
    if 'v' in active_features:
        # video feature: [videos on this skill watched, videos on this skill skipped]
        features["v"] = sparse.csr_matrix(np.empty((0, 2)))

    # Past counts for s_modules
    # S_MODULE_NUM (attempted count for this particular skill_id) + S_MODULE_NUM (correct count for this particular skill_id) + S_MODULE_NUM (onehot encoding)
    if 'sm' in active_features:
        # video feature: [videos on this skill watched, videos on this skill skipped]
        features["sm"] = sparse.csr_matrix(np.empty((0, 3 * c.S_MODULE_NUM)))

    # Past counts for active_check question explanations on this skill
    # 1 (count for active checks on this skill)
    if 'ac' in active_features:
        features["ac"] = sparse.csr_matrix(np.empty((0, 1)))

    # Build feature rows by iterating over users
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id][['user_id', 'item_id', 'timestamp', 'correct', 'skill_id', 'school_id', 'item_difficulty', 's_module', 'app_type', 'response_time']].copy()
        df_user = df_user.values
        num_items_user = df_user.shape[0]

        skills = Q_mat[df_user[:, c.ITEM_ID].astype(int)].copy()

        features['df'] = np.vstack((features['df'], df_user))

        item_ids = df_user[:, c.ITEM_ID].reshape(-1, 1)
        labels = df_user[:, c.CORRECT].reshape(-1, 1)

        # Current skills one hot encoding
        if 's' in active_features:
            features['s'] = sparse.vstack([features["s"], sparse.csr_matrix(skills)])

        if 'd' in active_features:
            features['d'] = sparse.vstack([features['d'], sparse.csr_matrix(get_difficulty_feature(df_user))])

        if 'v' in active_features:
            features['v'] = sparse.vstack([features['v'], sparse.csr_matrix(get_video_feature(user_id, df_user, video_df))])
        
        if 'sm' in active_features:
            features['sm'] = sparse.vstack([features['sm'], sparse.csr_matrix(get_smodule_feature(df_user))])
        
        if 'ac' in active_features:
            features['ac'] = sparse.vstack([features['ac'], sparse.csr_matrix(get_active_check_feature(user_id, df_user, active_check_df))])

        # Attempts
        if 'a' in active_features:
            attempts = np.zeros((num_items_user, num_skills + 2))

            # Past attempts for relevant skills
            if 'sc' in active_features:
                tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
                # attempts record the count of attempts on each skill at each time step (a row represent a timestamp)
                # total of 4 skills, this student did 3 items (2 with skill 0, 1 with skill 1)
                # [1, 0, 0, 0]
                # [1, 0, 0, 0]
                # [0, 1, 0, 0]
                # attempts:
                # [1, 0, 0, 0]
                # [2, 0, 0, 0]
                # [2, 1, 0, 0]
                # each cell is taken log
                attempts[:, :num_skills] = phi(np.cumsum(tmp, 0) * skills)

            # Past attempts for item
            if 'ic' in active_features:
                onehot = OneHotEncoder(n_values=df_user[:, c.ITEM_ID].max() + 1)
                item_ids_onehot = onehot.fit_transform(item_ids).toarray()
                tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot, 0)))[:-1]
                attempts[:, num_skills] = phi(tmp[np.arange(num_items_user), df_user[:, c.ITEM_ID]])

            # Past attempts for all items
            if 'tc' in active_features:
                attempts[:, -1] = phi(np.arange(num_items_user))

            features['a'] = sparse.vstack([features['a'], sparse.csr_matrix(attempts)])

        # Wins
        if "w" in active_features:
            wins = np.zeros((num_items_user, num_skills + 2))

            # Past wins for relevant skills
            if 'sc' in active_features:
                tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
                corrects = np.hstack((np.array(0), df_user[:, c.CORRECT])).reshape(-1, 1)[:-1]
                wins[:, :num_skills] = phi(np.cumsum(tmp * corrects, 0) * skills)

            # Past wins for item
            if 'ic' in active_features:
                onehot = OneHotEncoder(n_values=df_user[:, c.ITEM_ID].max() + 1)
                item_ids_onehot = onehot.fit_transform(item_ids).toarray()
                tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot * labels, 0)))[:-1]
                wins[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, c.ITEM_ID]])

            # Past wins for all items
            if 'tc' in active_features:
                wins[:, -1] = phi(np.concatenate((np.zeros(1), np.cumsum(df_user[:, c.CORRECT])[:-1])))

            features['w'] = sparse.vstack([features['w'], sparse.csr_matrix(wins)])


    # User and item one hot encodings
    onehot = OneHotEncoder()
    if 'u' in active_features:
        features['u'] = onehot.fit_transform(features["df"][:, c.USER_ID].reshape(-1, 1))
    if 'i' in active_features:
        features['i'] = onehot.fit_transform(features["df"][:, c.ITEM_ID].reshape(-1, 1))

    # School one hot encoding
    if 'sch' in active_features:
        features['sch'] = onehot.fit_transform(features["df"][:, c.SCHOOL_ID].reshape(-1, 1))

    # app_type one got encoding
    if 'at' in active_features:
        features['at'] = onehot.fit_transform(features["df"][:, c.APP_TYPE].reshape(-1, 1) - 1)

    # Difficulty buckets for each item
    # length: 3 (onehot for easy, medium, and difficult)
    if 'db' in active_features:
        features['db'] = sparse.csr_matrix(get_difficulty_buckets(features["df"][:, c.ITEM_DIFFICULTY].reshape(-1, 1)))

    # Global features
    # global question/item correctness
    # length: 1 (correctness conditioned on question)
    if 'gi' in active_features:
        features['gi'] = sparse.csr_matrix(get_global_item_correctness(features["df"]))

    # global skill/tagcode correctness
    # length: 1 (correctness conditioned on skill)
    if 'gs' in active_features:
        features['gs'] = sparse.csr_matrix(get_global_skill_correctness(features["df"]))

    # global school correctness
    # length: 1 (correctness conditioned on school)
    if 'gsch' in active_features:
        features['gsch'] = sparse.csr_matrix(get_global_school_correctness(features["df"]))

    # Response time features
    # current response time
    # length: 1 (current question response time)
    if 'rt' in active_features:
        features['rt'] = sparse.csr_matrix(get_response_time(features["df"]))

    # current response time with reference of item average response time
    # length: 1-2
    if 'ti' in active_features:
        features['ti'] = sparse.csr_matrix(get_item_based_response_time(features["df"]))

    # current response time with reference of user average response time
    # length: 1-2
    if 'tu' in active_features:
        features['tu'] = sparse.csr_matrix(get_user_based_response_time(features["df"]))

    print(features.keys())

    if X_base != None:
        return sparse.hstack([features['base'], sparse.hstack([features[x] for x in features.keys() if x != 'df' and x != 'base'])]).tocsr()
    else:
        return sparse.hstack([sparse.csr_matrix(features['df']),
                       sparse.hstack([features[x] for x in features.keys() if x != 'df' and x != 'base'])]).tocsr()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode sparse feature matrix for logistic regression.')
    parser.add_argument('-u', action='store_true',
                        help='If True, include user one hot encoding.')
    parser.add_argument('-i', action='store_true',
                        help='If True, include item one hot encoding.')
    parser.add_argument('-s', action='store_true',
                        help='If True, include skills many hot encoding .')
    parser.add_argument('-ic', action='store_true',
                        help='If True, include item historical counts.')
    parser.add_argument('-sc', action='store_true',
                        help='If True, include skills historical counts.')
    parser.add_argument('-mc', action='store_true',
                        help='If True, include module historical counts.')
    parser.add_argument('-tc', action='store_true',
                        help='If True, include total historical counts.')
    parser.add_argument('-w', action='store_true',
                        help='If True, historical counts include wins.')
    parser.add_argument('-a', action='store_true',
                        help='If True, historical counts include attempts.')
    parser.add_argument('-tw', action='store_true',
                        help='If True, historical counts are encoded as time windows.')
    parser.add_argument('-v', action='store_true',
                        help='If True, historical counts include videos.')
    parser.add_argument('-sch', action='store_true',
                        help='If True, include school one hot encoding.')
    parser.add_argument('-d', action='store_true',
                        help='If True, historical difficulties of items.')
    parser.add_argument('-sm', action='store_true',
                        help='If True, historical counts for s_modules.')
    parser.add_argument('-at', action='store_true',
                        help='If True, include app_type information.')
    parser.add_argument('-ac', action='store_true',
                        help='If True, historical counts for active checks on this skill_id.')
    parser.add_argument('-db', action='store_true',
                        help='If True, difficulty buckets for items (10-20 easy, 30-70 medium, 80-90 difficult).')
    parser.add_argument('-gi', action='store_true',
                        help='If True, global item/question correctness.')
    parser.add_argument('-gs', action='store_true',
                        help='If True, global skill/tagcode correctness.')
    parser.add_argument('-gsch', action='store_true',
                        help='If True, global school correctness.')
    parser.add_argument('-rt', action='store_true',
                        help='If True, current question response time.')
    parser.add_argument('-ti', action='store_true',
                        help='If True, current question response time with reference to the item average response time.')
    parser.add_argument('-tu', action='store_true',
                        help='If True, current question response time with reference to the user average response time.')
    parser.add_argument('-exp', action='store_true',
                        help='If True, encode experiment data')
    parser.add_argument('-dynamic', action='store_true',
                        help='If True, encode features dynamically as an increment to encoded files')
    parser.add_argument('--num_users_to_train', type=int, default=-1)
    args = parser.parse_args()

    suffix = "" if args.num_users_to_train < 0 else "_" + str(args.num_users_to_train)
    data_path = os.path.join('data')
    data_file = f'exp_preprocessed_data{suffix}.csv' if args.exp else f'preprocessed_data{suffix}.csv'
    df = pd.read_csv(os.path.join(data_path, data_file), sep="\t")
    df = df[['user_id', 'item_id', 'timestamp', 'correct', 'skill_id', 'school_id', 'item_difficulty', 's_module', 'app_type' ,'response_time']]
    Q_mat = sparse.load_npz(os.path.join(data_path, f'q_mat{suffix}.npz')).toarray()
    video_df = pd.read_csv(os.path.join(data_path, f'preprocessed_video_data{suffix}.csv'), sep="\t")
    active_check_df = pd.read_csv(os.path.join(data_path, f'preprocessed_active_check_data{suffix}.csv'), sep="\t")
    print('encoding ' + data_file)

    all_features = ['u', 'i', 's', 'ic', 'sc', 'tc', 'mc', 'w', 'a', 'tw', 'v', 'sch', 'd', 'sm', 'at', 'ac', 'db', 'gi', 'gs', 'gsch', 'rt', 'ti', 'tu']
    active_features = [features for features in all_features if vars(args)[features]]
    features_suffix = '_'.join(active_features) + suffix
    if args.exp:
        features_suffix += '_exp'
    X_base = None
    if args.dynamic:
        X_base = csr_matrix(load_npz(f'./data/X-isicsctcwa{suffix}.npz'))
        active_features = list(set(active_features) - set(['i', 's', 'ic', 'sc', 'tc', 'w', 'a']))
        print('base file: ')
        print(f'./data/X-isicsctcwa{suffix}.npz')

    X = df_to_sparse(df, video_df, active_check_df, Q_mat, active_features, X_base)
    print('final X file shape: ' + str(X.shape))
    print('file name: ' + f"X-{features_suffix}")
    sparse.save_npz(os.path.join(data_path, f"X-{features_suffix}"), X)

