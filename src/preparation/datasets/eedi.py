import os
import datetime
import numpy as np
import pandas as pd
from scipy import sparse
from dateutil.relativedelta import relativedelta
from config.constants import DATASET_PATH, MIN_INTERACTIONS_PER_USER

TIME_REGEX = "%Y-%m-%d %H:%M:%S.%f"


###############################################################################
# Process student meta-data
###############################################################################

ENCODING = {
    0: 0,
    1: 1,
    2: 2,
    3: 0
}


def prepare_student_meta_data():
    path = DATASET_PATH["eedi"] + "data/metadata/student_metadata_task_1_2.csv"
    user_df = pd.read_csv(path)
    user_df['PremiumPupil'].fillna(-1, inplace=True)

    i = 0
    user_to_id = {}
    user_to_gender = {}
    user_to_date = {}
    user_to_premium = {}

    cols = ["UserId", "Gender", "DateOfBirth", "PremiumPupil"]
    for user, gen, date, premium in user_df[cols].values:
        user_to_id[user] = i
        user_to_gender[user] = ENCODING[gen]
        if str(date) == "nan":
            user_to_date[user] = -1
        else:
            user_to_date[user] = datetime.datetime.strptime(date, TIME_REGEX)
        user_to_premium[user] = premium
        i += 1

    return user_to_id, user_to_gender, user_to_date, user_to_premium


###############################################################################
# Process subject meta-data
###############################################################################

def prepare_subject_meta_data():
    path = DATASET_PATH["eedi"] + "data/metadata/subject_metadata.csv"
    subject_df = pd.read_csv(path)

    # Build Pre-matrix
    i = 0
    subject_to_id = {}
    subject_to_lv = {}
    for k in [0, 1, 2, 3]:
        for sid, lv in subject_df[["SubjectId", "Level"]].values:
            subject_to_lv[sid] = lv
            if lv != k:
                continue
            if sid not in subject_to_id:
                subject_to_id[sid] = i
                i += 1

    num_skills = i
    Pre_mat = np.eye(num_skills, dtype=int)
    cols = ["SubjectId", "ParentId", "Level"]
    for sid, pid, lv in subject_df[cols].values:
        if np.isnan(pid):
            assert lv == 0, "Only level 0 skill have no parent"
            continue
        Pre_mat[subject_to_id[sid], subject_to_id[pid]] = 1
    assert (2 * num_skills) - np.sum(Pre_mat) == 2, "Assert one one ancestor"
    Pre_mat = sparse.csr_matrix(Pre_mat)

    return num_skills, subject_to_id, subject_to_lv, Pre_mat


###############################################################################
# Process Question meta-data
###############################################################################

def prepare_question_meta_data(num_skills, subject_to_id, subject_to_lv):
    path = DATASET_PATH["eedi"] + \
        "data/metadata/question_metadata_task_1_2.csv"
    question_df = pd.read_csv(path)

    i = 0
    question_to_id = {}
    for q in question_df["QuestionId"]:
        question_to_id[i] = i
        i += 1

    # Build Q-matrix
    num_items = i
    Q_mat = np.zeros((num_items, num_skills))
    for item_id, skills in question_df[["QuestionId", "SubjectId"]].values:
        line = [int(s) for s in skills[1:-1].split(',')]
        line = [subject_to_id[s] for s in line if subject_to_lv[s] == 3]
        for skill_id in line:
            Q_mat[question_to_id[item_id], skill_id] = 1

    return question_to_id, Q_mat


###############################################################################
# Process answer meta-data
###############################################################################

def prepare_answer_meta_data():
    path = DATASET_PATH["eedi"] + "data/metadata/answer_metadata_task_1_2.csv"
    answer_df = pd.read_csv(path)
    answer_df['SchemeOfWorkId'].fillna(-1, inplace=True)
    answer_df['SchemeOfWorkId'] = answer_df['SchemeOfWorkId'].astype(int)
    answer_df['Confidence'].fillna(-1, inplace=True)

    i = 0
    quiz_to_id = {}
    for quiz in answer_df["QuizId"]:
        if quiz not in quiz_to_id:
            quiz_to_id[quiz] = i
            i += 1

    i = 0
    group_to_id = {}
    for group in answer_df["GroupId"]:
        if group not in group_to_id:
            group_to_id[group] = i
            i += 1

    i = 0
    scheme_to_id = {}
    scheme_to_id[-1] = -1
    for scheme in answer_df["SchemeOfWorkId"]:
        if scheme not in scheme_to_id:
            scheme_to_id[scheme] = i
            i += 1

    i = 0
    answer_to_date = {}
    answer_to_quiz = {}
    answer_to_group = {}
    answer_to_scheme = {}
    answer_confidence = {}
    cols = [
        "AnswerId", "DateAnswered", "Confidence",
        "GroupId", "QuizId", "SchemeOfWorkId"
    ]
    for aid, date, confidence, group, quiz, scheme in answer_df[cols].values:
        if i % 10000 == 0:
            print(i)
        answer_to_date[aid] = datetime.datetime.strptime(date, TIME_REGEX)
        answer_to_quiz[aid] = quiz_to_id[quiz]
        answer_to_group[aid] = group_to_id[group]
        answer_to_scheme[aid] = scheme_to_id[scheme]
        answer_confidence[aid] = confidence
        i += 1

    return answer_to_date, answer_to_quiz, answer_to_group, \
        answer_to_scheme, answer_confidence


###############################################################################
# Prepare dataset
###############################################################################

def prepare_eedi(n_splits):
    print("\nPrepare user meta-data")
    print("-----------------------------------------------")
    u_to_id, u_to_gender, u_to_date, u_to_premium = prepare_student_meta_data()

    print("Prepare subject meta-data")
    print("-----------------------------------------------")
    num_skills, s_to_id, s_to_lv, Pre_mat = prepare_subject_meta_data()

    print("Prepare question meta-data")
    print("-----------------------------------------------")

    q_to_id, Q_mat = prepare_question_meta_data(num_skills, s_to_id, s_to_lv)

    print("Prepare answer meta-data")
    print("-----------------------------------------------")
    a_to_date, a_to_quiz, a_to_group, a_to_scheme, a_confidence = \
        prepare_answer_meta_data()

    print("\nPreparing interaction data")
    print("-----------------------------------------------")
    # drop CorrectAnswer, AnswerValue, not contained in the test files
    base = DATASET_PATH["eedi"] + "data/"
    a_order = base + "task_1_answer_id_ordering.npy"
    a_order = np.load(a_order, allow_pickle=True)
    a_order = a_order.item()
    train_df = pd.read_csv(base + "train_data/train_task_1_2.csv")
    train_df.drop(columns=['CorrectAnswer', 'AnswerValue'], inplace=True)
    base += "test_data/"
    public_test_df = pd.read_csv(base + "test_public_answers_task_1.csv")
    private_test_df = pd.read_csv(base + "test_private_answers_task_1.csv")
    raw_df = pd.concat([train_df, public_test_df, private_test_df])

    # create empty dataframe
    interaction_df = pd.DataFrame()

    # user information
    interaction_df["user_id"] = \
        np.array([u_to_id[user] for user in raw_df["UserId"]])

    ages = []
    for user, answer in raw_df[["UserId", "AnswerId"]].values:
        if u_to_date[user] == -1:
            ages.append(-1)
        else:
            x = relativedelta(a_to_date[answer], u_to_date[user]).years
            x = max(-1, x)
            x = min(25, x)
            ages.append(x)
    interaction_df["user_age"] = np.array(ages)
    interaction_df["gender"] = \
        np.array([u_to_gender[u] for u in raw_df["UserId"]])
    interaction_df["premium"] = \
        np.array([u_to_premium[u] for u in raw_df["UserId"]])

    # item_id
    interaction_df["item_id"] = \
        np.array([q_to_id[q] for q in raw_df["QuestionId"]])

    # timestamp
    times = np.array([a_to_date[a].timestamp() for a in raw_df["AnswerId"]])
    interaction_df["unix_time"] = np.copy(times)
    times = times - np.min(times)
    interaction_df["timestamp"] = times

    # NOTE: This value is only for debugging purposes
    interaction_df["AnswerId"] = raw_df["AnswerId"].values

    # answer information
    interaction_df["bundle_id"] = \
        np.array([a_to_quiz[a] for a in raw_df["AnswerId"]])
    interaction_df["teacher_id"] = \
        np.array([a_to_group[a] for a in raw_df["AnswerId"]])
    interaction_df["s_module"] = \
        np.array([a_to_scheme[a] for a in raw_df["AnswerId"]])
    interaction_df["confidence"] = \
        np.array([a_confidence[a] for a in raw_df["AnswerId"]])

    # ordered answer id
    interaction_df["sequence_number"] = \
        np.array([a_order[a] for a in raw_df["AnswerId"]])

    # correct
    interaction_df["correct"] = raw_df["IsCorrect"].values

    print(interaction_df.head())

    print("Storing interaction data")
    print("------------------------------------------------------------------")

    # remove users with to little interactions
    def f(x): return (len(x) >= MIN_INTERACTIONS_PER_USER)
    interaction_df = interaction_df.groupby("user_id").filter(f)
    interaction_df = interaction_df.sort_values(["user_id", "timestamp",
                                                 "sequence_number"])
    interaction_df.reset_index(inplace=True, drop=True)

    # create splits for cross validation
    from src.preparation.prepare_data import determine_splits
    determine_splits(interaction_df, "eedi", n_splits=n_splits)

    # save results
    pp = os.path.join(DATASET_PATH["eedi"], "preparation")
    Q_mat = sparse.csr_matrix(Q_mat)
    sparse.save_npz(os.path.join(pp, "q_mat.npz"), Q_mat)
    Pre_mat = sparse.csr_matrix(Pre_mat)
    sparse.save_npz(os.path.join(pp, "pre_mat.npz"), Pre_mat)

    hashed_q = hash_q_mat(Q_mat)
    skill_id = [hashed_q[item_id] for item_id in interaction_df.item_id]
    pp = os.path.join(pp, "preprocessed_data.csv")
    interaction_df.assign(hashed_skill_id=skill_id).to_csv(pp, sep="\t",
                                                           index=False)


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
