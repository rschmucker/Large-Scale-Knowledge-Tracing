"""Computes dataset secific sequence entropy and predictability."""
import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import defaultdict
from config.constants import DATASET_PATH


def list_entropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def max_prob(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    prob = max(counts) / len(labels)
    return prob


def measure_entropy(dataset):
    path = os.path.join(DATASET_PATH[dataset], "preparation")
    path = os.path.join(path, "preprocessed_data.csv")
    df = pd.read_csv(path, sep="\t")

    # sequence entropy for different types of instances
    key = "skill_id"  # hashed_skill_id # skill_id" # "item_id"
    print("Key", key)

    # compute counts for every item
    buffer = defaultdict(list)
    for i, user in enumerate(df["user_id"].unique()):
        if i % 1000 == 0:
            print(i)
        user_df = df[df["user_id"] == user].copy()
        last = -1
        for item in user_df[key]:
            buffer[last].append(item)
            last = item

    # compute overall entropy
    ent, n = 0, 0
    for k in buffer:
        ent += (entropy(buffer[k]) * len(buffer[k]))
        n += len(buffer[k])
    assert n == len(df), "length error"

    print("Sum Entropy: ", ent)
    ent = ent / n
    print("Normalized Entropy: ", ent)
    print("\n------------------------------------------------\n")

    # compute overall prediction power
    prob, n = 0, 0
    for k in buffer:
        p = max_prob(buffer[k])
        assert 0 <= p and p <= 1, "Range violated"
        prob += (p * len(buffer[k]))
        n += len(buffer[k])
    assert n == len(df), "length error"

    print("Sum prob: ", prob)
    prob = prob / n
    print("Normalized prob: ", prob)
    print("\n------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure dataset entropy.')
    parser.add_argument('--dataset', type=str, help="Relevant dataset.")
    args = parser.parse_args()

    dataset = args.dataset
    print("Dataset name:", dataset)
    assert dataset in DATASET_PATH, "The specified dataset is not supported"

    measure_entropy(dataset)
