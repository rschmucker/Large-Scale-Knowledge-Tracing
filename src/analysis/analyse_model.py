"""This file contains code to evaluate existing models
"""
import pickle
import argparse
import numpy as np
import pandas as pd
from config.constants import ALL_FEATURES, DATASET_PATH, SEED
import src.utils.data_loader as data_loader
import src.utils.prepare_parser as prepare_parser
from src.utils.metrics import compute_metrics


def eval_train_test_performance(data_dict, X, y, split, model):
    print("\nTrain/Test performance:")
    print("----------------------------------------")
    predict_proba = model.predict_proba(X)[:, 1]

    y_pred_tr = predict_proba[split["selector_train"]]
    # y_pred_tr = np.ones(len(y_pred_tr))
    tr_acc, tr_auc, _, _ = \
        compute_metrics(y_pred_tr, y[split["selector_train"]])

    y_pred_te = predict_proba[split["selector_test"]]
    # y_pred_te = np.ones(len(y_pred_te))
    te_acc, te_auc, _, _ = \
        compute_metrics(y_pred_te, y[split["selector_test"]])

    print(f"\nfeatures = {sel_features}, "
          f"\nauc_train = {tr_auc}, auc_test = {te_auc}, "
          f"\nacc_train = {tr_acc}, acc_test = {te_acc}, ")


def eval_pre_post_performance(data_dict, X, y, split, model):
    print("\nPerformance different modules:")
    print("----------------------------------------")
    interaction_df = data_dict["interaction_df"]
    predict_proba = model.predict_proba(X)[:, 1]

    print("\nEffective Learning")
    selector_eff = (interaction_df["s_module"] == 2)
    y_test = y[split["selector_test"] & selector_eff]
    y_pred = predict_proba[split["selector_test"] & selector_eff]
    acc, auc, _, _ = compute_metrics(y_pred, y_test)
    print("AUC", auc)
    print("ACC", acc)

    print("\nPre-test")
    selector_pre = (interaction_df["s_module"] == 4)
    y_test = y[split["selector_test"] & selector_pre]
    y_pred = predict_proba[split["selector_test"] & selector_pre]
    acc, auc, _, _ = compute_metrics(y_pred, y_test)
    print("AUC", auc)
    print("ACC", acc)

    print("\nPost-test")
    selector_post = (interaction_df["s_module"] == 3)
    y_test = y[split["selector_test"] & selector_post]
    y_pred = predict_proba[split["selector_test"] & selector_post]
    acc, auc, _, _ = compute_metrics(y_pred, y_test)
    print("AUC", auc)
    print("ACC", acc)


def eval_performance_over_time(data_dict, X, y, split, model):
    print("\nPerformance different times:")
    print("----------------------------------------")
    predict_proba = model.predict_proba(X)[:, 1]

    print("Loading tcA for filtering")
    tcA = \
        pd.read_pickle(DATASET_PATH[data_dict["dataset"]] + "features/tcA.pkl")
    tcA = np.exp(tcA["tcA"].values) - 1

    bounds = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500),
              (500, 1000), (1000, np.inf)]
    s = 0
    tr_accs, tr_aucs, te_accs, te_aucs = [], [], [], []
    for lower, upper in bounds:
        print("\nBetween " + str(lower) + " and " + str(upper))
        selector = ((lower <= tcA) & (tcA < upper))
        s += np.sum(selector)

        y_train = y[split["selector_train"] & selector]

        y_pred = predict_proba[split["selector_train"] & selector]
        # y_pred = np.ones(len(y_train))  # always correct
        tr_acc, tr_auc, _, _ = compute_metrics(y_pred, y_train)

        y_test = y[split["selector_test"] & selector]

        y_pred = predict_proba[split["selector_test"] & selector]
        # y_pred = np.ones(len(y_test))  # always correct
        te_acc, te_auc, _, _ = compute_metrics(y_pred, y_test)

        print(f"\nfeatures = {sel_features}, "
              f"\nauc_train = {tr_auc}, auc_test = {te_auc}, "
              f"\nacc_train = {tr_acc}, acc_test = {te_acc}, ")

        tr_aucs.append(round(tr_auc * 100, 2))
        te_aucs.append(round(te_auc * 100, 2))
        tr_accs.append(round(tr_acc * 100, 2))
        te_accs.append(round(te_acc * 100, 2))
    assert s == len(y), "Issue during filtering"

    print("\ntr_auc string: " + " & ".join([str(x) for x in tr_aucs]))
    print("\nte_auc string: " + " & ".join([str(x) for x in te_aucs]))
    print("------------")
    print("\ntr_acc string: " + " & ".join([str(x) for x in tr_accs]))
    print("\nte_acc string: " + " & ".join([str(x) for x in te_accs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LR sparse features.')
    prepare_parser.add_feature_arguments(parser)
    args = parser.parse_args()

    split_id = args.split_id
    sel_features = \
        [features for features in ALL_FEATURES if vars(args)[features]]
    sel_features.sort()
    print("Selected features: ", sel_features)
    print("Cross-validation split: " + str(split_id))
    dataset = args.dataset
    print("Dataset name:", dataset)
    assert dataset in DATASET_PATH, "The specified dataset is not supported"

    # Load the model
    print("\nLoading model...")
    suffix = '_'.join(sel_features)
    model_path = "./artifacts/" + dataset + "/models/" + "s" + str(SEED) \
        + "-" + str(split_id) + "_" + suffix + ".pkl"
    with open(model_path, 'rb') as f:
        lr_model = pickle.load(f)
    print("Loaded model")

    # retrieve combined data and train/test-split
    data_dict = data_loader.load_preprocessed_data(dataset)
    X, y, split = \
        data_loader.get_combined_features_and_split(sel_features, split_id)

    # put evaluation code here
    eval_train_test_performance(data_dict, X, y, split, lr_model)
    print("\n----------------------------------------")

    eval_performance_over_time(data_dict, X, y, split, lr_model)
    print("\n----------------------------------------")

    # eval_pre_post_performance(data_dict, X, y, split, lr_model)
    # print("\n----------------------------------------")

    # Extract relevant rows from combined dataframe using a combination masks
    # store_results(metrics_train, metrics_test, selected_features,
    #               split_id, lr_model)
    print("\n----------------------------------------")
    print("Completed model analysis")
