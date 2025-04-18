import os
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from config.constants import DATASET_PATH, MIN_INTERACTIONS_PER_USER

SM = {
    (False, False, False): 0,
    (False, False, True): 1,
    (False, True, False): 2,
    (False, True, True): 3,
    (True, False, False): 4,
    (True, False, True): 5,
    (True, True, False): 6,
    (True, True, True): 7
}


###############################################################################
# Process Question meta-data
###############################################################################

def prepare_question_meta_data():

    df = pd.read_csv(DATASET_PATH["junyi_15"] + "junyi_Exercise_table.csv")

    i, j, k = 0, 0, 0
    name_to_id = {}
    topic_to_id = {}
    area_to_id = {}
    name_to_skill = {}
    name_to_area = {}
    cols = ["name", "topic", "area"]
    for n, t, a in df[cols].values:
        # there are two known duplicated rows in the data
        if n not in ["matrix_mul_two", "matrix_app_fruit_oil"]:
            assert n not in name_to_id, "duplicated name " + str(n)
            assert n not in name_to_skill, "duplicated name " + str(n)
            assert n not in name_to_area, "duplicated name " + str(n)

        if t not in topic_to_id:
            topic_to_id[t] = j
            j += 1

        if a not in area_to_id:
            area_to_id[a] = k
            k += 1

        name_to_id[n] = i
        name_to_skill[n] = topic_to_id[t]
        name_to_area[n] = area_to_id[a]
        i += 1

    num_items = i
    num_skills = j
    num_parts = k
    print("completed assignments")

    # Build Q-mat
    Q_mat = np.zeros((num_items, num_skills))
    for n, t in df[["name", "topic"]].values:
        i_id = name_to_id[n]
        s_id = topic_to_id[t]
        Q_mat[i_id, s_id] = 1
    print("completed q_mat")

    # Build Pre-mat
    Pre_mat = np.zeros((num_items, num_items))
    for n, pre in df[["name", "prerequisites"]].values:
        i_id = name_to_id[n]
        Pre_mat[i_id, i_id] = 1
        if isinstance(pre, float):  # Skipping nan values
            continue
        pres = [n.strip() for n in pre.split(",")]
        for p in pres:
            p_id = name_to_id[p]
            Pre_mat[i_id, p_id] = 1
    print("completed pre_mat")

    # Build Part-mat
    Part_mat = np.zeros((num_items, num_parts))
    for n, a in df[["name", "area"]].values:
        i_id = name_to_id[n]
        p_id = area_to_id[a]
        Part_mat[i_id, p_id] = 1
    print("completed part_mat")

    return name_to_id, name_to_skill, name_to_area, Q_mat, Pre_mat, Part_mat


###############################################################################
# Process Interaction data
###############################################################################

def prepare_junyi_15(n_splits):

    print("Prepare question meta-data")
    print("-----------------------------------------------")

    n_to_id, n_to_skill, n_to_area, Q_mat, Pre_mat, Part_mat = \
        prepare_question_meta_data()

    print("\nPreparing interaction data")
    print("-----------------------------------------------")

    #df = pd.read_csv(DATASET_PATH["junyi_15"] + "junyi_ProblemLog_original.csv")
    
    
    #df = df[df["exercise"].isin(n_to_id)]  # Filter out rows with unknown exercises
    #df = df.dropna(subset=["exercise"])    # Drop rows where 'exercise' is NaN
    #df = df.reset_index(drop=True)         # Reset the index to fix row alignment
    
    df = pd.read_csv(DATASET_PATH["junyi_15"] + "junyi_ProblemLog_original.csv", low_memory=False)

    # Drop rows with missing exercise field (which causes lookup to fail)
    df = df[df["exercise"].notna()]  # Ensures no NaN in "exercise"

    # Drop rows where the exercise is not in the ID dictionary
    df = df[df["exercise"].isin(n_to_id)]

    # Reset the index to ensure row counts align
    df = df.reset_index(drop=True)



    # create empty dataframe
    interaction_df = pd.DataFrame()

    interaction_df["user_id"] = df["user_id"]  # User IDs are 0-max
    
    df = df[df["exercise"].isin(n_to_id)]
    
    df = df.reset_index(drop=True)

    
    interaction_df["item_id"] = np.array([n_to_id[n] for n in df["exercise"]])
    interaction_df["skill_id"] = \
        np.array([n_to_skill[n] for n in df["exercise"]])
    interaction_df["part_id"] = \
        np.array([n_to_area[n] for n in df["exercise"]])
    interaction_df["correct"] = df["correct"].astype(int)

    # add timestamp information
    times = df["time_done"].values / 1000000  # convert microsec to sec
    interaction_df["unix_time"] = np.copy(times)
    times = times - np.min(times)
    interaction_df["timestamp"] = times

    # study module
    print("Adding sm information")
    cs = ['topic_mode', 'suggested', 'review_mode']
    sm_vals, i = [], 0
    for a, b, c in df[cs].values:
        if i % 100000 == 0:
            print(i)
        sm_vals.append(SM[(a, b, c)])
        i += 1
    interaction_df["s_module"] = np.array(sm_vals)

    # response time based on first attempt
    response_time = []
    for i, s in enumerate(df["time_taken_attempts"]):
        if isinstance(s, str):
            ts = s.split("&")
            time = max(0, int(ts[0]))
            response_time.append(time)
        elif isinstance(s, int):
            response_time.append(s)
        else:
            response_time.append(0)
    interaction_df["response_time"] = np.array(response_time)

    # Add some temporary information
    interaction_df["time_taken"] = np.clip(df["time_taken"].values, 0, None)
    interaction_df["hint_used"] = df["hint_used"].astype(int)
    df['hint_time_taken_list'] = df['hint_time_taken_list'].fillna(0).values
    hint_time_taken = []
    for i, s in enumerate(df["hint_time_taken_list"]):
        if isinstance(s, str):
            ts = s.split("&")
            time = max(0, sum([int(t) for t in ts]))
            hint_time_taken.append(time)
        elif isinstance(s, int):
            hint_time_taken.append(s)
        else:
            hint_time_taken.append(0)
    interaction_df["hint_time_taken"] = np.array(hint_time_taken)

    # Filter and sort users
    def f(x): return (len(x) >= MIN_INTERACTIONS_PER_USER)
    interaction_df = interaction_df.groupby("user_id").filter(f)
    interaction_df = interaction_df.sort_values(["user_id", "timestamp"])

    lag_times = []
    rcs_total, rts_total = [], []
    rcs_skill, rts_skill = [], []
    print("Adding hint and lag time information")
    for i, user in enumerate(interaction_df["user_id"].unique()):
        if i % 1000 == 0:
            print(i)
        rc, rt = 0, 0
        rcs, rts = defaultdict(lambda: 0), defaultdict(lambda: 0)
        lt = -1
        user_df = interaction_df[interaction_df["user_id"] == user].copy()
        cols = ["timestamp", "time_taken", "hint_used", "hint_time_taken",
                "skill_id"]

        for t, tt, h, ht, s_id in user_df[cols].values:
            assert ht >= 0, "issue with hint time computation"
            if lt == -1:
                lag_times.append(-1)
            else:
                lag = (t - tt) - lt
                lag_times.append(lag)
            lt = t
            # hint features
            rcs_total.append(rc)
            rts_total.append(rt)
            rcs_skill.append(rcs[s_id])
            rts_skill.append(rts[s_id])
            rc += h
            rt += ht
            rcs[s_id] += h
            rts[s_id] += ht

    interaction_df["lag_time"] = np.clip(np.array(lag_times), 0, None)
    interaction_df["rc_total"] = np.array(rcs_total)
    interaction_df["rt_total"] = np.array(rts_total)
    interaction_df["rc_skill"] = np.array(rcs_skill)
    interaction_df["rt_skill"] = np.array(rts_skill)

    # remove tmp information
    cs = ["time_taken", "hint_used", "hint_time_taken"]
    interaction_df.drop(columns=cs, inplace=True)

    print("Storing interaction data")
    print("------------------------------------------------------------------")

    from src.preparation.prepare_data import determine_splits
    determine_splits(interaction_df, "junyi_15", n_splits=n_splits)

    # save results
    preparation_path = os.path.join(DATASET_PATH["junyi_15"], "preparation")
    Q_mat = sparse.csr_matrix(Q_mat)
    sparse.save_npz(os.path.join(preparation_path, "q_mat.npz"), Q_mat)
    Pre_mat = sparse.csr_matrix(Pre_mat)
    sparse.save_npz(os.path.join(preparation_path, "pre_mat.npz"), Pre_mat)
    Part_mat = sparse.csr_matrix(Part_mat)
    sparse.save_npz(os.path.join(preparation_path, "part_mat.npz"), Part_mat)
    interaction_path = os.path.join(preparation_path, "preprocessed_data.csv")
    interaction_df.to_csv(interaction_path, sep="\t", index=False)
