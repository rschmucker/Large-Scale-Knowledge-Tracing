import pickle
import argparse
import numpy as np
import pandas as pd
from config.constants import ALL_FEATURES, DATASET_PATH, SEED
import src.utils.data_loader as loader
from src.utils.metrics import compute_metrics
from src.training.compute_lr import train_func
import src.utils.prepare_parser as prepare_parser
from src.preprocessing.features.feature_util import phi_inv

BOUNDS = [(0, 10), (10, 50), (50, 100), (100, 250),
          (250, 500), (500, np.inf)]


def train_time_specialized(X, y, split, args):

    print("Loading tcA for filtering")
    tcA = pd.read_pickle(DATASET_PATH[args.dataset] + "features/tcA.pkl")
    tcA = phi_inv(tcA["tcA"].values)  # undo prior transformation

    s = 0
    tr_accs, tr_aucs, te_accs, te_aucs = [], [], [], []
    y_pred_tr, y_pred_te = np.empty(0), np.empty(0)
    y_truth_tr, y_truth_te = np.empty(0), np.empty(0)

    for lower, upper in BOUNDS:
        print("\nBetween " + str(lower) + " and " + str(upper))
        selector = ((lower <= tcA) & (tcA < upper))
        s += np.sum(selector)  # check if we include all values once

        print("Preparing partition for training")
        X_train = X[split["selector_train"] & selector]
        y_train = y[split["selector_train"] & selector]
        X_test = X[split["selector_test"] & selector]
        y_test = y[split["selector_test"] & selector]

        m_train, m_test, lr_model = \
            train_func(X_train, y_train, X_test, y_test, args)

        pred_tr = lr_model.predict_proba(X_train)[:, 1]
        pred_te = lr_model.predict_proba(X_test)[:, 1]

        # Accumulate values for overall performance
        y_pred_tr = np.concatenate((y_pred_tr, pred_tr))
        y_pred_te = np.concatenate((y_pred_te, pred_te))
        y_truth_tr = np.concatenate((y_truth_tr, y_train))
        y_truth_te = np.concatenate((y_truth_te, y_test))

        print("\nbound performance")
        acc_train, auc_train, nll_train, mse_train, f1 = \
            compute_metrics(pred_tr, y_train)
        acc_test, auc_test, nll_test, mse_test, f1 = \
            compute_metrics(pred_te, y_test)

        tr_accs.append(acc_train)
        tr_aucs.append(auc_train)
        te_accs.append(acc_test)
        te_aucs.append(auc_test)

    # assert s == len(y), "Issue during filtering"
    print("\n------------------------")
    print("Completed bracket computations")

    print("\nOverall performance:")
    acc_train, auc_train, nll_train, mse_train, f1 = \
        compute_metrics(y_pred_tr, y_truth_tr)
    acc_test, auc_test, nll_test, mse_test, f1 = \
        compute_metrics(y_pred_te, y_truth_te)

    tr_accs.append(acc_train)
    tr_aucs.append(auc_train)
    te_accs.append(acc_test)
    te_aucs.append(auc_test)

    return tr_accs, te_accs, tr_aucs, te_aucs


def evaluate_generalist_over_time(X, y, split, lr_model):
    print("Loading tcA for filtering")
    tcA = pd.read_pickle(DATASET_PATH[args.dataset] + "features/tcA.pkl")
    tcA = phi_inv(tcA["tcA"].values)  # undo prior transformation

    s = 0
    tr_accs, tr_aucs, te_accs, te_aucs = [], [], [], []
    y_pred_tr, y_pred_te = np.empty(0), np.empty(0)
    y_truth_tr, y_truth_te = np.empty(0), np.empty(0)

    for lower, upper in BOUNDS:
        print("\nBetween " + str(lower) + " and " + str(upper))
        selector = ((lower <= tcA) & (tcA < upper))
        s += np.sum(selector)  # check if we include all values once

        print("Preparing partition for training")
        X_train = X[split["selector_train"] & selector]
        y_train = y[split["selector_train"] & selector]
        X_test = X[split["selector_test"] & selector]
        y_test = y[split["selector_test"] & selector]

        pred_tr = lr_model.predict_proba(X_train)[:, 1]
        pred_te = lr_model.predict_proba(X_test)[:, 1]

        # Accumulate values for overall performance
        y_pred_tr = np.concatenate((y_pred_tr, pred_tr))
        y_pred_te = np.concatenate((y_pred_te, pred_te))
        y_truth_tr = np.concatenate((y_truth_tr, y_train))
        y_truth_te = np.concatenate((y_truth_te, y_test))

        print("\nbound performance")
        acc_train, auc_train, nll_train, mse_train, f1 = \
            compute_metrics(pred_tr, y_train)
        acc_test, auc_test, nll_test, mse_test, f1 = \
            compute_metrics(pred_te, y_test)

        tr_accs.append(acc_train)
        tr_aucs.append(auc_train)
        te_accs.append(acc_test)
        te_aucs.append(auc_test)
    # assert s == len(y), "Issue during filtering"
    print("\n------------------------")
    print("Completed bracket computations")

    print("\nOverall performance:")
    acc_train, auc_train, nll_train, mse_train, f1 = \
        compute_metrics(y_pred_tr, y_truth_tr)
    acc_test, auc_test, nll_test, mse_test, f1 = \
        compute_metrics(y_pred_te, y_truth_te)

    tr_accs.append(acc_train)
    tr_aucs.append(auc_train)
    te_accs.append(acc_test)
    te_aucs.append(auc_test)

    return tr_accs, te_accs, tr_aucs, te_aucs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train specialized models.')
    prepare_parser.add_feature_arguments(parser)
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads for feature preparation.')
    parser.add_argument('--num_iterations', type=int, default=5000)
    parser.add_argument('--generalist', type=bool)

    args = parser.parse_args()
    max_splits = args.split_id
    selected_features = [features for features in ALL_FEATURES
                         if vars(args)[features]]
    selected_features.sort()

    print("Selected features: ", selected_features)
    print("Num Threads: " + str(args.num_threads))
    print("Cross-validation splits: " + str(max_splits))
    print("Iterations: " + str(args.num_iterations))
    dataset = args.dataset
    print("Dataset name:", dataset)
    print("Generalist: " + str(args.generalist))
    assert dataset in DATASET_PATH, "The specified dataset is not supported"
    suf = '_'.join(selected_features)

    cross_tr_accs = []
    cross_te_accs = []
    cross_tr_aucs = []
    cross_te_aucs = []

    for split_id in range(max_splits):
        print("\n------------------------------------------")
        print("Split ID: " + str(split_id))
        print("------------------------------------------\n")
        X, y, split = loader.get_combined_features_and_split(selected_features,
                                                             split_id, dataset)

        if args.generalist:  # load and evaluate generalist model
            print("evaluating generalist\n")
            mod_path = "./artifacts/lr_baselines/" + dataset + "/models/"
            mod_path += "s" + str(SEED) + "-" + str(split_id) + "_" + \
                suf + ".pkl"
            lr_model = pickle.load(open(mod_path, "rb"))
            tr_accs, te_accs, tr_aucs, te_aucs = \
                evaluate_generalist_over_time(X, y, split, lr_model)
            print("\nACCs: ", tr_accs, "\n")
            assert len(tr_accs) == (len(BOUNDS) + 1), "Error in computation"
            name = dataset + "_generalist_" + suf

        else:  # Train specialized models for different point in time
            print("training and evaluating time specialized lr models\n")
            tr_accs, te_accs, tr_aucs, te_aucs = \
                train_time_specialized(X, y, split, args)
            name = dataset + "_time_specialized_" + suf

            print("\nACCs: ", tr_accs, "\n")

        cross_tr_accs.append(tr_accs)
        cross_te_accs.append(te_accs)
        cross_tr_aucs.append(tr_aucs)
        cross_te_aucs.append(te_aucs)

    print(cross_tr_accs)

    res_dict = {
        "cross_tr_accs": np.vstack(cross_tr_accs),
        "cross_te_accs": np.vstack(cross_te_accs),
        "cross_tr_aucs": np.vstack(cross_tr_aucs),
        "cross_te_aucs": np.vstack(cross_te_aucs)
    }
    print(res_dict)

    with open('notebooks/analysis/temporal/' + name + '.pkl', 'wb') as f:
        pickle.dump(res_dict, f)

    print("\n----------------------------------------")
    print("Completed logistic regression analysis")
