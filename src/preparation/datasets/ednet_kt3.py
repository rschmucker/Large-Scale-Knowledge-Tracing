""" This file contains code to combine EdNet KT3 data into standardized format.
"""
import os
import numpy as np
import pandas as pd
from enum import Enum
from scipy import sparse
from collections import defaultdict
from config.constants import DATASET_PATH, MIN_INTERACTIONS_PER_USER


class KT3State(Enum):
    NEUTRAL = 1
    IN_LECTURE = 2
    IN_EXPLANATION = 3
    IN_BUNDLE = 4


SOURCE_MAPPING = {
    'tutor': 1,
    'my_note': 2,
    'diagnosis': 3,
    'sprint': 4,
    'adaptive_offer': 5,
    'review': 6,
    'review_quiz': 7,
    'archive': 8
}

PLATFORM_MAPPING = {'mobile': 1, 'web': 2}

# These users answer question during lectures
ILL_BEHAVED = [378881, 501798, 823503, 4586, 388703, 830559, 536644, 5319,
               530086, 318, 329, 7350, 518884, 18609]


###############################################################################
# Process question meta-data
###############################################################################

def hash_q_mat(Q_mat):
    hash_id = 1
    item_dic = {}
    indices_dic = {}
    for row_idx, row in enumerate(Q_mat):
        indices = tuple(row.indices.tolist())
        if indices not in indices_dic:
            indices_dic[indices] = hash_id
            hash_id += 1
        item_dic[row_idx] = indices_dic[indices]
    return item_dic


def prepare_question_meta_data():
    question_df = pd.read_csv(os.path.join(DATASET_PATH["ednet_kt3"],
                                           "contents/questions.csv"))

    # Prepare answer and bundle dictionary
    answer_key = {}
    question_to_bundle = {}
    q2part = {}
    e2part = {}
    cs = ["question_id", "bundle_id", "correct_answer", "part"]
    for question_id, bundle_id, answer, part in question_df[cs].values:
        question_to_bundle[question_id] = bundle_id
        q2part[question_id] = part
        e2part['e' + bundle_id[1:]] = part
        answer_key[question_id] = answer

    # Prepare question ids to build Q-Matrix
    question_df['question_id'] = \
        np.array([int(i[1:]) for i in question_df['question_id']])
    assert min(question_df['question_id']) > 0, "Sanity check failed"

    # Build Q-Matrix
    num_items = max(question_df['question_id']) + 1
    num_skills = max([max([int(s) for s in tag.split(';')])
                      for tag in question_df["tags"]]) + 1
    Q_mat = np.zeros((num_items, num_skills))
    for question_id, tag in question_df[["question_id", "tags"]].values:
        for skill_id in set([int(s) for s in tag.split(';')]):
            skill_id = max(skill_id, 0)  # Handle -1 as skill 0
            Q_mat[question_id, skill_id] = 1

    # Build Part-Matrix
    num_parts = max(question_df['part']) + 1
    assert (min(question_df['part']) == 1) and \
           (max(question_df['part']) == 7), "TOIEC parts are in [1, 7]"
    Part_mat = np.zeros((num_items, num_parts))
    for question_id, part_id in question_df[["question_id", "part"]].values:
        Part_mat[question_id, part_id] = 1

    return Q_mat, Part_mat, answer_key, question_to_bundle, q2part, e2part


###############################################################################
# Process lecture meta-data
###############################################################################

def prepare_lecture_meta_data():
    lectures_df = pd.read_csv(os.path.join(DATASET_PATH["ednet_kt3"],
                                           "contents/lectures.csv"))
    l2part = {}
    l2dur = {}
    cs = ['lecture_id', 'part', 'video_length']
    for lecture_id, part, duration in lectures_df[cs].values:
        l2part[lecture_id] = part
        l2dur[lecture_id] = duration / 1000.0  # convert ms to s
    return l2part, l2dur


###############################################################################
# Process user data
###############################################################################

def extract_row_data(user_df, answer_dict, q2part, e2part, l2part, l2dur):
    row_data = []
    response_buffer = {}
    # reading features
    trc, prc = 0, defaultdict(lambda: 0)
    trt, prt = 0, defaultdict(lambda: 0)
    # video features
    tvc, pvc = 0, defaultdict(lambda: 0)
    tvt, pvt = 0, defaultdict(lambda: 0)
    tvs, pvs = 0, defaultdict(lambda: 0)

    row = 0
    state = KT3State.NEUTRAL
    last_answer_time, lag_time, enter_time = -1, -1, -1
    cs = ["timestamp", "action_type", "item_id", "user_answer"]
    for time, action, item_id, answer in user_df[cs].values:
        item_type = item_id[0]

        if state == KT3State.NEUTRAL:
            assert action == "enter", "Did expect enter in neutral state"
            enter_time = time
            if last_answer_time != -1:
                lag_time = time - last_answer_time
            if item_type == "b":
                state = KT3State.IN_BUNDLE
            elif item_type == "e":
                state = KT3State.IN_EXPLANATION
                trc += 1
                prc[e2part[item_id]] += 1
            elif item_type == "l":
                state = KT3State.IN_LECTURE
                tvc += 1
                pvc[l2part[item_id]] += 1
            else:
                raise ValueError("The specified  type is unknown")

        elif state == KT3State.IN_LECTURE:  # Watching a lecture
            assert item_type == "l" and action == "quit", \
                "In explanation unexpected item or action " + \
                str(item_id) + " " + str(action)
            gap = time - enter_time
            tvt += gap
            pvt[l2part[item_id]] += gap
            if gap <= (0.9 * l2dur[item_id]):
                tvs += 1
                pvs[l2part[item_id]] += 1
            state = KT3State.NEUTRAL

        elif state == KT3State.IN_EXPLANATION:  # Reading expert commentary
            if item_type == "q":
                assert action == "respond", "Did not expect action " + \
                                            action + " found q inside bundle."
                flag = (answer == answer_dict[item_id])
                resp_time = time - enter_time
                p_rc = prc[q2part[item_id]]
                p_rt = prt[q2part[item_id]]
                p_vc = pvc[q2part[item_id]]
                p_vt = pvt[q2part[item_id]]
                p_vs = pvs[q2part[item_id]]
                response_buffer[item_id] = (row, flag, lag_time, item_id,
                                            resp_time, trc, p_rc, trt, p_rt,
                                            tvc, p_vc, tvt, p_vt, tvs, p_vs)
            elif item_type == "e":
                assert action == "quit", "In expl unexpected item/action " + \
                                         str(item_id) + " " + str(action)
                if response_buffer:
                    row_data += [response_buffer[k] for k in response_buffer]
                    response_buffer = {}
                    last_answer_time = time
                    gap = time - enter_time
                    trt += gap
                    prt[e2part[item_id]] += gap
                state = KT3State.NEUTRAL
            else:
                RuntimeError("Unexpected '" + item_id + "' in explanation.")

        elif state == KT3State.IN_BUNDLE:
            if item_type == "q":
                assert action == "respond", "Did not expect action " + \
                                            action + " found q inside bundle."
                flag = (answer == answer_dict[item_id])
                resp_time = time - enter_time
                p_rc = prc[q2part[item_id]]
                p_rt = prt[q2part[item_id]]
                p_vc = pvc[q2part[item_id]]
                p_vt = pvt[q2part[item_id]]
                p_vs = pvs[q2part[item_id]]
                response_buffer[item_id] = (row, flag, lag_time, item_id,
                                            resp_time, trc, p_rc, trt, p_rt,
                                            tvc, p_vc, tvt, p_vt, tvs, p_vs)
            elif item_type == "b":
                assert action == "submit", "Did not expect action " + \
                                           action + " found row inside bundle."
                if response_buffer:
                    row_data += [response_buffer[k] for k in response_buffer]
                    response_buffer = {}
                    last_answer_time = time
                state = KT3State.NEUTRAL
            else:
                raise RuntimeError("Unexpected '" + item_id + "' in bundle.")

        else:
            raise RuntimeError("Ended up in invalid state")
        row += 1
    row_data.sort()
    return row_data


def process_user(file, answer_dict, bundle_dict, q2p, e2p, l2p, l2dur):
    user_df_path = os.path.join(DATASET_PATH["ednet_kt3"], "KT3/" + file)
    user_df = pd.read_csv(user_df_path)

    # user_id
    user_id = int(file[1:-4])
    user_df['user_id'] = np.ones(len(user_df), dtype=int) * user_id
    # timestamp
    user_df["timestamp"] = user_df["timestamp"] / 1000
    user_df["unix_time"] = np.copy(user_df["timestamp"].values)
    user_df["timestamp"] = user_df["timestamp"] - user_df["timestamp"].min()

    # correct, lag_time, bundle_id
    row_data = extract_row_data(user_df, answer_dict, q2p, e2p, l2p, l2dur)

    rel_rows = [r[0] for r in row_data]
    rel_correct = np.array([int(r[1]) for r in row_data])
    rel_lag_time = np.array([r[2] for r in row_data])
    rel_resp_time = np.array([r[4] for r in row_data])
    rel_item_id = [r[3] for r in row_data]
    rel_bundle = [int(bundle_dict[r[3]][1:]) for r in row_data]
    rel_rc_total = np.array([int(r[5]) for r in row_data])
    rel_rc_part = np.array([int(r[6]) for r in row_data])
    rel_rt_total = np.array([int(r[7]) for r in row_data])
    rel_rt_part = np.array([int(r[8]) for r in row_data])
    rel_vc_total = np.array([int(r[9]) for r in row_data])
    rel_vc_part = np.array([int(r[10]) for r in row_data])
    rel_vt_total = np.array([int(r[11]) for r in row_data])
    rel_vt_part = np.array([int(r[12]) for r in row_data])
    rel_vs_total = np.array([int(r[13]) for r in row_data])
    rel_vs_part = np.array([int(r[14]) for r in row_data])

    user_df = user_df.iloc[rel_rows]  # only select rows of final response
    user_df.reset_index(inplace=True, drop=True)
    assert (np.all(user_df["item_id"].values == rel_item_id)), "Error item_id"
    assert (np.all(user_df["action_type"].values == "respond")), \
        "Mismatch found non-respond action in answers"
    user_df["part_id"] = np.array([q2p[q] for q in rel_item_id])
    user_df["correct"] = rel_correct
    user_df["lag_time"] = rel_lag_time
    user_df["response_time"] = rel_resp_time
    user_df["bundle_id"] = rel_bundle
    user_df["rc_total"] = rel_rc_total
    user_df["rc_part"] = rel_rc_part
    user_df["rt_total"] = rel_rt_total
    user_df["rt_part"] = rel_rt_part
    user_df["vc_total"] = rel_vc_total
    user_df["vc_part"] = rel_vc_part
    user_df["vt_total"] = rel_vt_total
    user_df["vt_part"] = rel_vt_part
    user_df["vs_total"] = rel_vs_total
    user_df["vs_part"] = rel_vs_part

    # s_module
    user_df["source"].replace(SOURCE_MAPPING, inplace=True)
    # app_type
    user_df["platform"].replace(PLATFORM_MAPPING, inplace=True)

    # item_id
    user_df = user_df.rename(columns={
        'question_id': 'item_id',
        'source': 's_module',
        'platform': 'app_type'
    })
    user_df['item_id'] = np.array([int(i[1:]) for i in user_df['item_id']])

    user_df = user_df[['user_id', 'item_id', 'timestamp', 'correct',
                       's_module', 'app_type', 'lag_time', 'response_time',
                       'bundle_id', 'unix_time', 'rc_total', 'rc_part',
                       'rt_total', 'rt_part', 'vc_total', 'vc_part',
                       'vt_total', 'vt_part', 'vs_total', 'vs_part', 
                       'part_id']]
    assert user_df.isnull().sum().sum() == 0, \
        "Missing value in df for user" + str(user_id)
    return user_df


###############################################################################
# Prepare dataset
###############################################################################

def prepare_ednet_kt3(n_splits, train_split=0.8):
    print("Preparing question data\n----------------------------------------")
    Q_mat, Part_mat, answer_key, question_to_bundle, q2part, e2part = \
        prepare_question_meta_data()
    l2part, l2dur = prepare_lecture_meta_data()

    print("Preparing user data\n----------------------------------------")
    files = os.listdir(os.path.join(DATASET_PATH["ednet_kt3"], "KT3/"))

    dfs = []
    for i, f in enumerate(files):
        if i % 100 == 0:
            print("Num processed: ", i)
        if int(f[1:-4]) in ILL_BEHAVED:
            continue

        df = process_user(f, answer_key, question_to_bundle, q2part, e2part,
                          l2part, l2dur)
        if len(df) > MIN_INTERACTIONS_PER_USER:
            dfs.append(df)
    interaction_df = pd.concat(dfs)

    # create splits for cross validation
    from src.preparation.prepare_data import determine_splits
    determine_splits(interaction_df, "ednet_kt3", n_splits=n_splits)

    # save results
    preparation_path = os.path.join(DATASET_PATH["ednet_kt3"], "preparation")
    Q_mat = sparse.csr_matrix(Q_mat)
    sparse.save_npz(os.path.join(preparation_path, "q_mat.npz"), Q_mat)
    Part_mat = sparse.csr_matrix(Part_mat)
    sparse.save_npz(os.path.join(preparation_path, "part_mat.npz"), Part_mat)
    interaction_path = os.path.join(preparation_path, "preprocessed_data.csv")

    hashed_q = hash_q_mat(Q_mat)
    s_id = [hashed_q[item_id] for item_id in interaction_df.item_id]
    interaction_df.assign(hashed_skill_id=s_id).to_csv(interaction_path,
                                                       sep="\t", index=False)
