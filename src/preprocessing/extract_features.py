"""This file contains functions for rapid feature preprocessing.
"""
import os
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
import src.utils.data_loader as data_loader
import src.utils.prepare_parser as prepare_parser
# Import feature functions
from config.constants import ALL_FEATURES, DATASET_PATH
import src.preprocessing.features.one_hot_features as oh_features
import src.preprocessing.features.count_features as c_features
import src.preprocessing.features.graph_features as gr_features
import src.preprocessing.features.time_window_features as tw_features
import src.preprocessing.features.rpfa as rpfa_features
import src.preprocessing.features.video_features as v_features
import src.preprocessing.features.study_module_features as sm_features
import src.preprocessing.features.interaction_time_features as it_features
import src.preprocessing.features.datetime_features as dt_features
import src.preprocessing.features.reading_features as r_features
from src.preprocessing.features.average_user_correct import user_avg_correct
from src.preprocessing.features.n_gram_feature import sequence_n_gram
from src.preprocessing.features import feature_util

N_STEPS = 10  # default for n-gram feature

FEATURE_FUNCTIONS = {
    # One-hot features
    'u': ("SERIAL", oh_features.user_one_hot),
    'i': ("SERIAL", oh_features.item_one_hot),
    's': ("PARALLEL", oh_features.skill_one_hot),
    'sch': ("SERIAL", oh_features.school_one_hot),
    'tea': ("SERIAL", oh_features.teacher_one_hot),
    'sm': ("SERIAL", oh_features.study_module_one_hot),
    'c': ("SERIAL", oh_features.course_one_hot),
    'd': ("SERIAL", oh_features.difficulty_one_hot),
    'at': ("SERIAL", oh_features.apptype_one_hot),
    't': ("SERIAL", oh_features.topic_one_hot),
    'bundle': ("SERIAL", oh_features.bundle_one_hot),
    'part': ("PARALLEL", oh_features.part_one_hot),
    'ss': ("SERIAL", oh_features.social_support_one_hot),
    'age': ("SERIAL", oh_features.age_one_hot),
    'gender': ("SERIAL", oh_features.gender_one_hot),
    'user_skill': ("SERIAL", oh_features.user_skill_one_hot),
    # User count features
    'tcA': ("PARALLEL", c_features.total_count_attempts),
    'tcW': ("PARALLEL", c_features.total_count_wins),
    'scA': ("PARALLEL", c_features.skill_count_attempts),
    'scW': ("PARALLEL", c_features.skill_count_wins),
    'icA': ("PARALLEL", c_features.item_count_attempts),
    'icW': ("PARALLEL", c_features.item_count_wins),
    'partcA': ("PARALLEL", c_features.part_count_attempts),
    'partcW': ("PARALLEL", c_features.part_count_wins),
    # Skill graph features
    'pre': ("PARALLEL", oh_features.prereq_one_hot),
    'post': ("PARALLEL", oh_features.postreq_one_hot),
    'precA': ("PARALLEL", gr_features.pre_skill_count_attempts),
    'precW': ("PARALLEL", gr_features.pre_skill_count_wins),
    'postcA': ("PARALLEL", gr_features.post_skill_count_attempts),
    'postcW': ("PARALLEL", gr_features.post_skill_count_wins),
    # Video features
    'vw': ("PARALLEL", v_features.videos_watched),
    'vs': ("PARALLEL", v_features.videos_skipped),
    'vt': ("PARALLEL", v_features.videos_time_watched),
    # Reading features
    'rc': ("PARALLEL", r_features.user_reading_count),
    'rt': ("PARALLEL", r_features.user_reading_time),
    # Study Module features
    'smA': ("PARALLEL", sm_features.smodule_attempts),
    'smW': ("PARALLEL", sm_features.smodule_wins),
    # Interaction time features
    'resp_time': ("PARALLEL", it_features.user_response_time),
    'resp_time_cat': ("PARALLEL", it_features.user_response_time_cat),
    'prev_resp_time_cat': ("PARALLEL",
                           it_features.user_prev_response_time_cat),
    'lag_time': ("PARALLEL", it_features.user_lag_time),
    'lag_time_cat': ("PARALLEL", it_features.user_lag_time_cat),
    'prev_lag_time_cat': ("PARALLEL", it_features.user_prev_lag_time_cat),
    # Date time features
    'month': ("SERIAL", dt_features.month_one_hot),
    'week': ("SERIAL", dt_features.week_one_hot),
    'day': ("SERIAL", dt_features.day_one_hot),
    'hour': ("SERIAL", dt_features.hour_one_hot),
    'weekend': ("SERIAL", dt_features.weekend_one_hot),
    'part_of_day': ("SERIAL", dt_features.part_of_day_one_hot),
    # Time Window features
    'tcA_TW': ("PARALLEL", tw_features.time_window_total_count_attempts),
    'tcW_TW': ("PARALLEL", tw_features.time_window_total_count_wins),
    'scA_TW': ("PARALLEL", tw_features.time_window_skill_count_attempts),
    'scW_TW': ("PARALLEL", tw_features.time_window_skill_count_wins),
    'icA_TW': ("PARALLEL", tw_features.time_window_item_count_attempts),
    'icW_TW': ("PARALLEL", tw_features.time_window_item_count_wins),
    # RPFA features
    'rpfa_F': ("PARALLEL", rpfa_features.recency_count_failures),
    'rpfa_R': ("PARALLEL", rpfa_features.recency_count_proportion),
    # Higher order
    'user_avg_correct': ("SERIAL", user_avg_correct),
    'n_gram': ("PARALLEL", sequence_n_gram),
    # Other features
    "ones":  ("SERIAL", feature_util.one_vector),
}


def extract_features(features, data_dict, recompute=False):
    print("\nComputing features:")
    print("----------------------------------------")
    for f in features:
        assert f in FEATURE_FUNCTIONS, "Identifier '" + f + \
            "' is no valid feature."

        file_path = DATASET_PATH[data_dict["dataset"]] + \
            "features/" + f + ".pkl"
        # Check if feature already exists
        if os.path.isfile(file_path) and not recompute:
            print("Feature '" + f + "' already exists.")
            continue

        print("Extracting feature '" + f + "'...")
        start = time.perf_counter()
        func = FEATURE_FUNCTIONS[f][1]
        if FEATURE_FUNCTIONS[f][0] == "SERIAL":
            feature_df = func(data_dict)
        elif FEATURE_FUNCTIONS[f][0] == "PARALLEL":
            feature_df = parallel_feature_computation(data_dict, f, func)
        feature_df = feature_df.astype(pd.SparseDtype("float", 0))
        print("DF shape: ", feature_df.shape)
        print("Computed feature '" + f + "' in " +
              str(round(time.perf_counter() - start, 2)) + " seconds")
        feature_df.to_pickle(file_path)
    print("----------------------------------------")
    print("Completed feature computation\n")


def extract_target_values(data_dict, recompute=False):
    print("")
    target_path = DATASET_PATH[data_dict["dataset"]] + \
        "features/" + "target.npy"
    if os.path.isfile(target_path) and not recompute:
        print("Target values already exist.")
        return

    print("Preparing target values...")
    interaction_df = data_dict["interaction_df"]
    target_values = interaction_df["correct"].values
    np.save(target_path, target_values)
    print("Stored target values")


def parallel_feature_computation(data_dict, fname, ffunc):
    n_threads = data_dict["n_threads"]
    interaction_df = data_dict["interaction_df"]
    unique_ids = interaction_df["user_id"].unique()

    tmp_path = DATASET_PATH[data_dict["dataset"]] + "tmp/"
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)

    # Prepare jobs
    print("\nPreparing jobs...")
    workload = []
    user_partition = np.array_split(unique_ids, 2 * n_threads)
    for i, partition in enumerate(user_partition):
        print("Partition: " + str(i))
        interaction_selector = interaction_df['user_id'].isin(partition)
        w = {
            "p_id": i,
            "partition_df": interaction_df.loc[interaction_selector].copy(),
            "p_path": tmp_path + fname + "_p" + str(i) + ".pkl",
            "dataset": data_dict["dataset"],
            "n_steps": data_dict.get('n_steps', N_STEPS),  # for n-gram
        }
        if data_dict["dataset"] == "elemmath_2021":
            raw_df = data_dict["raw_df"]
            raw_selector = raw_df['user_id'].isin(partition)
            w["partition_raw"] = raw_df.loc[raw_selector].copy()
        workload.append(w)
    print("Completed job preparations")

    # Run jobs in parallel
    print("\nParallel data encoding...")
    pool = Pool(n_threads)
    res = pool.map(ffunc, workload)
    pool.close()
    pool.join()
    assert sum(res) == len(workload), "Detected issue in parallel processing"
    print("\nCompleted parallel encoding")

    tmp_frames = []
    for w in workload:
        tmp_frames.append(pd.read_pickle(w["p_path"]))

    # Combine frames
    print("Combining parallel frames...")
    feature_frame = pd.concat(tmp_frames)
    print("Completed frame combination")

    # Remove TMP files
    shutil.rmtree(tmp_path)
    return feature_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features.')
    prepare_parser.add_feature_arguments(parser)
    parser.add_argument('-recompute', action='store_true',
                        help='If set, recompute previously computed features.')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Thread number for feature preparation.')
    args = parser.parse_args()

    active_features = [features for features in ALL_FEATURES
                       if vars(args)[features]]
    active_features.sort()
    recompute = args.recompute
    dataset = args.dataset
    print("Dataset name:", dataset)
    assert dataset in DATASET_PATH, "The specified dataset is not supported"
    print("Selected features: ", active_features)
    print("Recomputing existing features: " + str(recompute))
    print("Num Threads: " + str(args.num_threads))
    print("Split ID: " + str(args.split_id))
    data_dict = data_loader.load_preprocessed_data(dataset)
    data_dict["n_threads"] = args.num_threads
    data_dict["split_id"] = args.split_id

    feature_path = DATASET_PATH[dataset] + "features/"
    if not os.path.isdir(feature_path):
        os.mkdir(feature_path)
    extract_target_values(data_dict, recompute=recompute)
    extract_features(active_features, data_dict, recompute=recompute)
