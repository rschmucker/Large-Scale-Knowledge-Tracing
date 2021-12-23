"""Preparation step for Saint model."""
import argparse
import numpy as np
import pandas as pd
from os import path
from src.utils.misc import set_random_seeds
from config.constants import SEED, DATASET_PATH

SUPPORTED = ["elemmath_2021", "ednet_kt3", 'eedi', 'junyi_15']


def prepare_saint(dataset):

    p_path = path.join(DATASET_PATH[dataset], "preparation")
    f_path = path.join(DATASET_PATH[dataset], "features")
    dfs = []

    # Interaction df
    p = p_path + "/preprocessed_data.csv"
    print("Loading: ", p)
    if path.exists(p):
        p_df = pd.read_csv(p, sep="\t")
        u_ids = np.array(range(1, len(p_df) + 1))
        assert p_df.isnull().sum().sum() == 0, "Error, NaN values."
        dfs.append(p_df)
    else:
        raise RuntimeError("Run data preparation first.")

    # Categorical lag time feature
    p = f_path + "/lag_time_cat.pkl"
    print("Loading: ", p)
    if path.exists(p):
        lag_df = pd.read_pickle(p).reset_index(drop=True)
        assert lag_df.isnull().sum().sum() == 0, "Error, NaN values."
        assert np.count_nonzero(lag_df["U_ID"] - u_ids) == 0, "U_ID error"
        lag_df.drop(columns=["U_ID"], inplace=True)
        lag_df = \
            lag_df.assign(lag_time_cat=np.argmax(lag_df.to_numpy(), axis=1))
        lag_df = lag_df[['lag_time_cat']]
        dfs.append(lag_df)
    elif dataset == "eedi":
        pass
    else:
        raise RuntimeError("Compute lag_time_cat feature first.")

    # Categorical response time feature
    p = f_path + "/resp_time_cat.pkl"
    print("Loading: ", p)
    if path.exists(p):
        resp_df = pd.read_pickle(p).reset_index(drop=True)
        assert resp_df.isnull().sum().sum() == 0, "Error, NaN values."
        assert np.count_nonzero(resp_df["U_ID"] - u_ids) == 0, "U_ID error"
        resp_df.drop(columns=["U_ID"], inplace=True)
        resp_df = \
            resp_df.assign(resp_time_cat=np.argmax(resp_df.to_numpy(), axis=1))
        resp_df = resp_df[['resp_time_cat']]
        dfs.append(resp_df)
    elif dataset == "eedi":
        pass
    else:
        raise RuntimeError("Compute resp_time_cat feature first.")

    # One-hot part encoding for EdNet dataset
    # if dataset == "ednet_kt3":
    #     p = f_path + "/part.pkl"
    #     print("Loading: ", p)
    #     if path.exists(p):
    #         part_df = pd.read_pickle(p).reset_index(drop=True)
    #         assert part_df.isnull().sum().sum() == 0, "Error, NaN values."
    #         assert np.count_nonzero(resp_df["U_ID"] - u_ids) == 0, "U_ID err"
    #         part_df.drop(columns=["U_ID"], inplace=True)
    #         dfs.append(part_df)
    #     else:
    #         raise RuntimeError("Compute part 1-hot encoding feature first.")

    # Combine and store data
    print("Combining frames")
    saint_df = pd.concat(dfs, axis=1, ignore_index=False)
    save_path = p_path + "/preprocessed_data_saint.csv"
    print("Storing data")
    saint_df.to_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare SAINT data.')
    parser.add_argument('--dataset', type=str, help="Dataset to prepare.")
    args = parser.parse_args()
    set_random_seeds(SEED)

    dataset = args.dataset
    print("\nDataset: " + dataset)
    assert dataset in SUPPORTED, "Provided data unknown: " + dataset

    prepare_saint(dataset)

    print("----------------------------------------")
    print("Completed SAINT preparation\n")
