import os
import datetime
import numpy as np
import pandas as pd
from scipy import sparse
from config.constants import DATASET_PATH, MIN_INTERACTIONS_PER_USER

REG = '%Y-%m-%d %H:%M:%S'
DIFFICULTY = {
    'unset': 0,
    'easy': 1,
    'normal': 2,
    'hard': 3
}


# NOTE: Junyi20 rounds timestamps to the nearest 15 min which prevents exact
# reconstruction of the interaction sequence

###############################################################################
# Process user meta-data
###############################################################################

def prepare_user_meta_data():
    path = os.path.join(DATASET_PATH["junyi_20"], "Info_UserData.csv")
    user_df = pd.read_csv(path)

    city_to_number = {}
    for i, city in enumerate(user_df["user_city"].unique()):
        city_to_number[city] = i

    i = 0
    user_to_id = {}
    user_to_grade = {}
    user_to_city = {}
    cs = ["uuid", "user_grade", "user_city"]
    for uuid, grade, city in user_df[cs].values:
        user_to_id[uuid] = i
        user_to_grade[uuid] = grade
        user_to_city[uuid] = city_to_number[city]
        i += 1

    u_has_teacher = {}
    u_self_coached = {}
    u_has_students = {}
    u_belongs_to_class = {}
    u_has_class = {}
    cs = ["uuid", "has_teacher_cnt", "is_self_coach", "has_student_cnt",
          "belongs_to_class_cnt", "has_class_cnt"]
    for uuid, has_t, self_c, has_s, belongs_c, has_c in user_df[cs].values:
        u_has_teacher[uuid] = int(has_t > 0)
        u_self_coached[uuid] = int(self_c)
        u_has_students[uuid] = int(has_s > 0)
        u_belongs_to_class[uuid] = int(belongs_c > 0)
        u_has_class[uuid] = int(has_c > 0)

    return user_to_id, user_to_grade, user_to_city, u_has_teacher, \
        u_self_coached, u_has_students, u_belongs_to_class, u_has_class


###############################################################################
# Process content meta-data
###############################################################################

def prepare_question_meta_data():
    path = os.path.join(DATASET_PATH["junyi_20"], "Info_Content.csv")
    content_df = pd.read_csv(path)

    # Build Pre-matrix
    i = 0
    level_to_id = {}
    for k in ["level1_id", "level2_id", "level3_id", "level4_id"]:
        for lv in content_df[k]:
            if lv not in level_to_id:
                level_to_id[lv] = i
                i += 1

    num_skills = i
    Pre_mat = np.eye(num_skills, dtype=int)
    cs = ["level1_id", "level2_id", "level3_id", "level4_id"]
    for lv1, lv2, lv3, lv4 in content_df[cs].values:
        Pre_mat[level_to_id[lv2], level_to_id[lv1]] = 1
        Pre_mat[level_to_id[lv3], level_to_id[lv2]] = 1
        Pre_mat[level_to_id[lv4], level_to_id[lv3]] = 1
    assert (2 * num_skills) - np.sum(Pre_mat) == 1, "Assert single node parent"
    Pre_mat = sparse.csr_matrix(Pre_mat)

    # Extract content specific information
    i = 0
    content_to_id = {}
    content_to_difficulty = {}
    content_to_stage = {}
    content_to_skill = {}
    cs = ["ucid", "difficulty", "learning_stage", "level4_id"]
    for ucid, diff, stage, level in content_df[cs].values:
        content_to_id[ucid] = i
        content_to_difficulty[ucid] = DIFFICULTY[diff]
        content_to_stage[ucid] = stage
        content_to_skill[ucid] = level_to_id[level]
        i += 1

    return Pre_mat, content_to_id, content_to_difficulty, content_to_stage, \
        content_to_skill


###############################################################################
# Prepare dataset
###############################################################################

def prepare_junyi_20(n_splits):
    print("Preparing question data\n-----------------------------------------")
    Pre_mat, c_to_id, c_to_dif, c_to_st, c_to_sk = prepare_question_meta_data()

    print("Preparing user data\n---------------------------------------------")
    u_to_id, u_to_gr, u_to_city, u_h_t, u_self_c, u_has_s, u_bt_c, u_h_c = \
        prepare_user_meta_data()

    print("Preparing interaction data\n--------------------------------------")
    path = os.path.join(DATASET_PATH["junyi_20"], "Log_Problem.csv")
    raw_df = pd.read_csv(path)

    # create empty dataframe
    interaction_df = pd.DataFrame()

    # user information
    interaction_df["user_id"] = \
        np.array([u_to_id[uuid] for uuid in raw_df["uuid"].values])
    interaction_df["city_id"] = \
        np.array([u_to_city[uuid] for uuid in raw_df["uuid"].values])
    interaction_df["user_grade"] = \
        np.array([u_to_gr[uuid] for uuid in raw_df["uuid"].values])
    interaction_df["has_teacher"] = \
        np.array([u_h_t[uuid] for uuid in raw_df["uuid"].values])
    interaction_df["self_coached"] = \
        np.array([u_self_c[uuid] for uuid in raw_df["uuid"].values])
    interaction_df["has_students"] = \
        np.array([u_has_s[uuid] for uuid in raw_df["uuid"].values])
    interaction_df["belongs_to_class"] = \
        np.array([u_bt_c[uuid] for uuid in raw_df["uuid"].values])
    interaction_df["has_class"] = \
        np.array([u_h_c[uuid] for uuid in raw_df["uuid"].values])

    # item_id
    i = 0
    problem_to_id = {}
    for upid in raw_df["upid"].values:
        if upid not in problem_to_id:
            problem_to_id[upid] = i
            i += 1
    interaction_df["item_id"] = \
        np.array([problem_to_id[upid] for upid in raw_df["upid"].values])

    # skill_id
    interaction_df["skill_id"] = \
        np.array([c_to_sk[ucid] for ucid in raw_df["ucid"].values])

    # timestamp
    print("parsing timestamps...")
    times = [datetime.datetime.strptime(t[:-4], REG) for t
             in raw_df["timestamp_TW"].values]
    times = np.array([t.timestamp() for t in times])
    times = times - np.min(times)
    interaction_df["timestamp"] = times
    print("completed timestamps")

    # correct
    interaction_df["correct"] = raw_df["is_correct"].values

    # Build Q-matrix
    num_items = interaction_df["item_id"].max() + 1
    num_skills = interaction_df["skill_id"].max() + 1
    Q_mat = np.zeros((num_items, num_skills))
    for item_id, skill_id in interaction_df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1
    # NOTE: SOME PROBLEMS BELONG TO MULTIPLE SKILLS
    # assert num_items == np.sum(Q_mat) + 1, "Assert one skill per item"

    # item_difficulty
    interaction_df["item_difficulty"] = \
        np.array([c_to_dif[ucid] for ucid in raw_df["ucid"].values])
    # problem number
    interaction_df["problem_number"] = raw_df["problem_number"]
    # seconds taken
    interaction_df["total_sec_taken"] = raw_df["total_sec_taken"]
    # number of attempts on question
    interaction_df["attempt_cnt"] = raw_df["total_attempt_cnt"]
    # number of hints taken
    interaction_df["hint_cnt"] = raw_df["used_hint_cnt"]
    # how often encountered in excercise
    interaction_df["encounter_cnt"] = raw_df["exercise_problem_repeat_session"]
    # bundle_id
    interaction_df["bundle_id"] = \
        np.array([c_to_id[ucid] for ucid in raw_df["ucid"].values])
    print("completed interaction_df")

    # remove users with to little interactions

    def f(x): return (len(x) >= MIN_INTERACTIONS_PER_USER)
    interaction_df = interaction_df.groupby("user_id").filter(f)
    cs = ["user_id", "timestamp", "problem_number"]
    interaction_df = interaction_df.sort_values(cs)
    interaction_df.reset_index(inplace=True, drop=True)

    # create splits for cross validation
    from src.preparation.prepare_data import determine_splits
    determine_splits(interaction_df, "junyi_20", n_splits=n_splits)

    # save results
    pp = os.path.join(DATASET_PATH["junyi_20"], "preparation")
    path = os.path.join(pp, "q_mat.npz")
    sparse.save_npz(path, sparse.csr_matrix(Q_mat))
    path = os.path.join(pp, "pre_mat.npz")
    sparse.save_npz(path, sparse.csr_matrix(Pre_mat))
    path = os.path.join(pp, "preprocessed_data.csv")
    interaction_df.to_csv(path, sep="\t", index=False)
