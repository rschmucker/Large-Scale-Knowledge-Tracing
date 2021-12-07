import os
import json
import time
import pickle
import datetime
import argparse
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from config.constants import ALL_FEATURES, DATASET_PATH, SEED
from src.utils.metrics import compute_metrics
import src.utils.prepare_parser as prepare_parser
from src.utils.data_loader import load_preprocessed_data, \
    get_combined_features_and_split
from src.preprocessing.extract_features import extract_features


def train_func(X_train, y_train, X_test, y_test, args):
    print("\nPerforming logistic regression:")
    print("----------------------------------------")
    print("Fitting LR model...")
    start = time.perf_counter()
    model = LogisticRegression(solver="liblinear",  # "saga" "lbfgs"
                               max_iter=args.num_iterations,
                               n_jobs=args.num_threads,
                               verbose=1)

    model.fit(X_train, y_train)
    now = time.perf_counter()
    print("Fitted LR model in " + str(round(now - start, 2)) + " seconds")

    print("Evaluating LR model...")
    start = time.perf_counter()
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    acc_train, auc_train, nll_train, mse_train, f1 = \
        compute_metrics(y_pred_train, y_train)
    acc_test, auc_test, nll_test, mse_test, f1 = \
        compute_metrics(y_pred_test, y_test)

    metrics_train = {
        "acc": acc_train,
        "auc": auc_train,
        "nll": nll_train,
        "mse": mse_train,
        "rmse": np.sqrt(mse_train),
        "f1": f1
    }

    metrics_test = {
        "acc": acc_test,
        "auc": auc_test,
        "nll": nll_test,
        "mse": mse_test,
        "rmse": np.sqrt(mse_test),
        "f1": f1
    }

    now = time.perf_counter()
    print("Evaluated LR model in " + str(round(now - start, 2)) + " seconds")

    print("----------------------------------------")
    print("Completed logistic regression\n")
    return metrics_train, metrics_test, model


def store_results(met_train, met_test, features, spl_id, model, path,
                  f_dec, r_dec):
    print("\nStoring results...")
    suf = '_'.join(features)

    res_path = path + "s" + str(SEED) + "-" + str(spl_id) + "_" + suf + \
        "_" + str(f_dec) + "_" + str(r_dec) + ".json"
    res_dict = {
        "time": datetime.datetime.now().isoformat(),
        "features": features,
        "seed": SEED,
        "split_id": spl_id,
        "metrics_train": met_train,
        "metrics_test": met_test
    }
    with open(res_path, "w") as f:
        json.dump(res_dict, f)

    mod_path = path + "/models/" + "s" + str(SEED) + "-" \
        + str(spl_id) + "_" + suf + ".pkl"
    with open(mod_path, "w") as f:
        pickle.dump(model, open(mod_path, 'wb'))
    print("Stored results under: " + res_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LR model.')
    prepare_parser.add_feature_arguments(parser)
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads for feature preparation.')
    parser.add_argument('--num_iterations', type=int, default=5000)
    parser.add_argument('--exp_name', type=str,
                        help="Experiment name", default="other")

    args = parser.parse_args()
    split_id = args.split_id
    selected_features = [features for features in ALL_FEATURES
                         if vars(args)[features]]
    selected_features.sort()

    print("Selected features: ", selected_features)
    print("Experiment name:", args.exp_name)
    print("Num Threads: " + str(args.num_threads))
    print("Cross-validation split: " + str(split_id))
    print("Iterations: " + str(args.num_iterations))
    print("Dataset name:", args.dataset)
    assert args.dataset in DATASET_PATH, "The specified dataset not supported"

    res_path = "./artifacts/" + args.exp_name + "/"
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    res_path += args.dataset + "/"
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    if not os.path.isdir(res_path + "models/"):
        os.mkdir(res_path + "models/")

    feature_path = DATASET_PATH[args.dataset] + "features/"
    if not os.path.isdir(feature_path):
        os.mkdir(feature_path)

    f_decays = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    r_decays = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # only prepare one of them at a time
    for f_dec in f_decays:
        data_dict = load_preprocessed_data(args.dataset)
        data_dict["n_threads"] = args.num_threads
        data_dict["split_id"] = args.split_id
        data_dict["rpfa_fail_decay"] = f_dec
        extract_features(["rpfa_F"], data_dict, recompute=True)

        for r_dec in r_decays:
            data_dict["rpfa_prop_decay"] = r_dec
            extract_features(["rpfa_R"], data_dict, recompute=True)

            # Combine and evaluate
            # --------------------------------------------------------------
            # retrieve combined data and train/test-split
            X, y, split = get_combined_features_and_split(selected_features,
                                                          split_id,
                                                          args.dataset)
            X_train = X[split["selector_train"]]
            y_train = y[split["selector_train"]]
            X_test = X[split["selector_test"]]
            y_test = y[split["selector_test"]]

            m_train, m_test, lr_model = \
                train_func(X_train, y_train, X_test, y_test, args)

            print(f"\nfeatures = {selected_features}, "
                f"\nacc_train = {m_train['acc']}, acc_test = {m_test['acc']}, "
                f"\nauc_train = {m_train['auc']}, auc_test = {m_test['auc']}, "
                f"\nf1_train = {m_train['f1']}, f1_test = {m_test['f1']}, "
                f"\nrmse_tr = {m_train['rmse']}, rmse_te = {m_test['rmse']},")

            store_results(m_train, m_test, selected_features,
                        split_id, lr_model, res_path, f_dec, r_dec)
            print("\n----------------------------------------")
            print("Completed logistic regression")
