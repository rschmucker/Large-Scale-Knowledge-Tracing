"""This file contains function to load preprocessed data and combine features
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from config.constants import ALL_FEATURES, DATASET_PATH, SEED


def load_preprocessed_data(dataset):
    print("\nLoading preprocessed data:")
    print("----------------------------------------")
    data_dict = {}
    data_dict["dataset"] = dataset
    start = time.perf_counter()
    prep_path = os.path.join(DATASET_PATH[dataset], "preparation")

    print("Reading files...")
    if dataset == "elemmath_2021":
        pp = os.path.join(DATASET_PATH[dataset], "ElemMath2021_data.csv")
        raw_df = pd.read_csv(pp)
        raw_df["timestamp"] = \
            (raw_df["server_time"] - raw_df["server_time"].min()) // 1000
        data_dict["raw_df"] = raw_df
        pp = os.path.join(prep_path, "preprocessed_video_data.csv")
        data_dict["video_df"] = pd.read_csv(pp, sep="\t")
        pp = os.path.join(prep_path, "preprocessed_active_check_data.csv")
        data_dict["active_check_df"] = pd.read_csv(pp, sep="\t")
        assert data_dict["video_df"].isnull().sum().sum() == 0, \
            "Error, found NaN values in the preprocessed data."
        assert data_dict["active_check_df"].isnull().sum().sum() == 0, \
            "Error, found NaN values in the preprocessed data."

    pp = os.path.join(prep_path, "preprocessed_data.csv")
    data_dict["interaction_df"] = pd.read_csv(pp, sep="\t")
    data_dict["Q_mat"] = \
        load_npz(os.path.join(prep_path, 'q_mat.npz')).toarray()

    print("Checking consistency...")
    assert data_dict["interaction_df"].isnull().sum().sum() == 0, \
        "Error, found NaN values in the preprocessed data."

    # Add unique identifier to each interaction in the main df for later joins
    print("Adding unique identifier to interactions...")
    data_dict["interaction_df"]['U_ID'] = \
        np.array(range(1, len(data_dict["interaction_df"]) + 1))

    print('Loaded in ' + str(round(time.perf_counter() - start, 2)) + ' sec')
    print("----------------------------------------")
    print("Completed data loading\n")
    return data_dict


def combine_features(features, dataset):
    print("\nCombining feature frames:")
    print("----------------------------------------")
    start = time.perf_counter()
    frames = []
    id_col = None
    for f in features:
        assert f in ALL_FEATURES, "'" + f + "' is no valid feature."

        # Check if file exists
        file_path = DATASET_PATH[dataset] + "features/" + f + ".pkl"
        if not os.path.isfile(file_path):
            raise ValueError("DF for feature '" + f + "' not computed yet.")

        print("Loading feature '" + f + "'...")
        df = pd.read_pickle(file_path)
        df.isnull().sum().sum() == 0, "Error, found NaN in feature data."
        if id_col is None:
            u_ids = df["U_ID"]
        else:
            assert np.count_nonzero(df["U_ID"] - u_ids) == 0, \
                "U_IDs are not aligned"
        df.drop(columns=["U_ID"], inplace=True)

        print(f, df.shape)
        df = df.sparse.to_coo()
        frames.append(sparse.csr_matrix(df))

    print("Combining frames")
    sparse_matrix = sparse.csr_matrix(sparse.hstack(frames))
    print("combined shape", sparse_matrix.shape)
    span = round(time.perf_counter() - start, 2)
    print("Feature combination took " + str(span) + " seconds")
    print("----------------------------------------")
    print("Completed feature combination\n")
    return sparse_matrix


def load_split(split_id, dataset):
    suf = str(SEED) + "_" + str(split_id) + ".pkl"
    path = DATASET_PATH[dataset] + "preparation/split_s" + suf
    with open(path, "rb") as file_object:
        split = pickle.load(file_object)
    return split


def get_combined_features_and_split(features, split_id, dataset):
    X = combine_features(features, dataset)
    y = np.load(DATASET_PATH[dataset] + 'features/target.npy')
    split = load_split(split_id, dataset)
    return X, y, split

def load_interaction_df(dataset):
    return load_preprocessed_data(dataset)["interaction_df"]
