"""Converts raw data into a standardized format."""
import os
import pickle
import argparse
import numpy as np

from sklearn.model_selection import KFold
from config.constants import SEED, DATASET_PATH
from src.utils.misc import set_random_seeds
from src.preparation.datasets.squirrel import prepare_squirrel
from src.preparation.datasets.ednet_kt3 import prepare_ednet_kt3
from src.preparation.datasets.junyi_20 import prepare_junyi_20
from src.preparation.datasets.junyi_15 import prepare_junyi_15
from src.preparation.datasets.eedi import prepare_eedi


def determine_splits(interaction_df, dataset, n_splits=5):
    user_ids = interaction_df["user_id"]
    unique_ids = interaction_df["user_id"].unique()
    np.random.shuffle(unique_ids)

    print("\nPreparing cross-validation splits...")
    kf = KFold(n_splits=n_splits)
    split_id = 0
    for train, test in kf.split(unique_ids):
        split = {
            "train_ids": unique_ids[train],
            "test_ids": unique_ids[test],
            "selector_train": np.isin(user_ids,  unique_ids[train]),
            "selector_test": np.isin(user_ids, unique_ids[test])
        }
        path = DATASET_PATH[dataset] + "preparation/split_s" + str(SEED) + \
            "_" + str(split_id) + ".pkl"
        with open(path, "wb") as file_object:
            pickle.dump(split, file_object)
        split_id += 1
    print("Completed cross-validation splits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('--dataset', type=str, help="Dataset to prepare.")
    parser.add_argument('--n_splits', type=int, default=5)
    args = parser.parse_args()
    set_random_seeds(SEED)

    dataset = args.dataset
    print("Dataset:", dataset + "\n")
    preparation_path = os.path.join(DATASET_PATH[dataset], "preparation")
    if not os.path.isdir(preparation_path):
        os.mkdir(preparation_path)

    if dataset == "squirrel":
        prepare_squirrel(args.n_splits)
    elif dataset == "ednet_kt3":
        prepare_ednet_kt3(args.n_splits)
    elif dataset == "junyi_15":
        prepare_junyi_15(args.n_splits)
    elif dataset == "junyi_20":
        prepare_junyi_20(args.n_splits)
    elif dataset == "eedi":
        prepare_eedi(args.n_splits)
    else:
        raise ValueError("The provided dataset name is unknown.")

    print("----------------------------------------")
    print("Completed data preparation\n")
