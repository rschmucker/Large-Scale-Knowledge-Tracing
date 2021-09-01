"""Merges multiple features stored on disk for training."""
import argparse
import numpy as np
import pandas as pd
from os import path
from src.utils.misc import set_random_seeds
from config.constants import SEED, DATASET_PATH

SUPPORTED = ["squirrel", "ednet_kt3", 'eedi', 'junyi_15']


def merge_features(dataset, feature_list):
    p_path = path.join(DATASET_PATH[dataset], "preparation")
    f_path = path.join(DATASET_PATH[dataset], "features")
    dfs = []

    # Interaction df
    p = p_path + "/preprocessed_data_saint.csv"
    print("Loading: ", p)
    if path.exists(p):
        p_df = pd.read_csv(p, sep="\t")
        u_ids = np.array(range(1, len(p_df) + 1))
        assert p_df.isnull().sum().sum() == 0, "Error, NaN values."
        dfs.append(p_df)
    else:
        raise RuntimeError("Run data preparation first.")

    # Categorical lag time feature
    for feature_name in feature_list:
        p = f_path + f"/{feature_name}.pkl"
        print("Loading: ", p)
        if path.exists(p):
            df = pd.read_pickle(p).reset_index(drop=True)
            assert df.isnull().sum().sum() == 0, "Error, NaN values."
            assert np.count_nonzero(df["U_ID"] - u_ids) == 0, "U_ID error"
            df.drop(columns=["U_ID"], inplace=True)
            df.columns = [f'feature_{c}' for i, c in enumerate(df.columns)]
            dfs.append(df)
        else:
            raise RuntimeError("Compute lag_time_cat feature first.")

    print("Combining frames")
    saint_df = pd.concat(dfs, axis=1, ignore_index=False)
    save_path = p_path + "/preprocessed_data_saint_features.csv"
    print("Storing data")
    saint_df.to_csv(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare SAINT data.')
    parser.add_argument('--dataset', type=str, help="Dataset to prepare.")
    args = parser.parse_args()
    set_random_seeds(SEED)

    dataset = args.dataset
    print("\nDataset: " + dataset)
    assert dataset in SUPPORTED, "Provided data unknown: " + dataset

    merge_features(dataset, ['sm'])

    print("----------------------------------------")
    print("Completed SAINT preparation\n")
