import pickle
import argparse
import numpy as np
import pandas as pd
from config.constants import ALL_FEATURES, DATASET_PATH
import src.utils.data_loader as loader
from src.utils.metrics import compute_metrics
from src.training.compute_lr import train_func
import src.utils.prepare_parser as prepare_parser
from src.preprocessing.features.feature_util import phi_inv

COLS = {
    "i": "item_id",
    "s": "skill_id",
    "hashed_skill_id": "hashed_skill_id",
    "t": "topic_id",
    "c": "course_id",
    "sm": "s_module",
    "tea": "teacher_id",
    "sch": "school_id",
    "at": "app_type",
    "part_id": "part_id",
    "bundle_id": "bundle_id",
    "single": "SINGLE",  # NOTE: Marks that we train only one model
    "time": "TIME",  # NOTE: Marks that we train time specialized models
}

AVG_CORRECT = {
    "squirrel": 0.685249091469651,
    "ednet_kt3": 0.661876189869379,
    "eedi": 0.6429725856250825,
    "junyi_15": 0.8298862191292307,
}

BOUNDS = [(0, 10), (10, 50), (50, 100), (100, 250),
          (250, 500), (500, np.inf)]


def train_partitions(X, y, split, args, data_dict):

    interaction_df = data_dict["interaction_df"]
    col = COLS[args.col]
    ids = interaction_df[col].values

    s = 0
    N = len(interaction_df)
    y_pred_tr, y_pred_te = np.zeros(N), np.zeros(N)
    y_truth_tr, y_truth_te = np.zeros(N), np.zeros(N)

    for c in np.unique(ids):
        print("\n" + col + ": " + str(c))
        selector = (ids == c)
        s += np.sum(selector)  # check if we include all values once

        print("Preparing partition for training")
        X_train = X[split["selector_train"] & selector]
        y_train = y[split["selector_train"] & selector]
        X_test = X[split["selector_test"] & selector]
        y_test = y[split["selector_test"] & selector]

        if (len(y_train) == 0 or len(y_test) == 0 or (1 not in y_train) or
           (0 not in y_train) or (1 not in y_test) or (0 not in y_test)):
            pred_tr = np.ones(len(y_train)) * AVG_CORRECT[args.dataset]
            pred_te = np.ones(len(y_test)) * AVG_CORRECT[args.dataset]
        else:
            m_train, m_test, lr_model = \
                train_func(X_train, y_train, X_test, y_test, args)

            print("\nPerformance for " + col + " " + str(c))
            print(f"\nfeatures = {selected_features}, "
                f"\nacc_train = {m_train['acc']}, acc_test = {m_test['acc']}, "
                f"\nauc_train = {m_train['auc']}, auc_test = {m_test['auc']}, \n"
                f"rmse_train = {m_train['rmse']}, rmse_test = {m_test['rmse']},")
            pred_tr = lr_model.predict_proba(X_train)[:, 1]
            pred_te = lr_model.predict_proba(X_test)[:, 1]

        # Accumulate values for overall performance
        pointer = 0
        tr_sel = (split["selector_train"] & selector)
        assert len(tr_sel) == len(y), "Issue in length alignment"
        for i, sel in enumerate(tr_sel):
            if sel:
                y_pred_tr[i] = pred_tr[pointer]
                y_truth_tr[i] = y[i]
                pointer += 1

        pointer = 0
        te_sel = (split["selector_test"] & selector)
        for i, sel in enumerate(te_sel):
            if sel:
                y_pred_te[i] = pred_te[pointer]
                y_truth_te[i] = y[i]
                pointer += 1

    assert s == len(y), "Issue during filtering"
    print("sum", np.sum(y - y_truth_tr - y_truth_te))
    assert np.sum(y - (y_truth_tr + y_truth_te)) == 0, "Issue in alignment"
    print("Passed alignment")
    print("\n------------------------")
    print("Completed model computations")

    print("\nOverall performance:")
    acc_train, auc_train, _, mse_train, _ = \
        compute_metrics(y_pred_tr, y_truth_tr)
    acc_test, auc_test, _, mse_test, _ = \
        compute_metrics(y_pred_te, y_truth_te)

    print(f"\nfeatures = {selected_features}, "
          f"\nacc_train = {acc_train}, acc_test = {acc_test}, "
          f"\nauc_train = {auc_train}, auc_test = {auc_test}, "
          f"\nrmse_train = {mse_train}, rmse_test = {mse_test}, ")

    res_dict = {
        "y_pred_train": y_pred_tr,
        "y_pred_test": y_pred_te,
        "y_truth_train": y_truth_tr,
        "y_truth_teest": y_truth_te,
    }

    return res_dict


def train_single(X, y, split, args, data_dict):

    interaction_df = data_dict["interaction_df"]
    N = len(interaction_df)
    y_pred_tr, y_pred_te = np.zeros(N), np.zeros(N)
    y_truth_tr, y_truth_te = np.zeros(N), np.zeros(N)

    print("Preparing partition for training")
    X_train = X[split["selector_train"]]
    y_train = y[split["selector_train"]]
    X_test = X[split["selector_test"]]
    y_test = y[split["selector_test"]]

    m_train, m_test, lr_model = \
        train_func(X_train, y_train, X_test, y_test, args)

    print(f"\nfeatures = {selected_features}, "
        f"\nacc_train = {m_train['acc']}, acc_test = {m_test['acc']}, "
        f"\nauc_train = {m_train['auc']}, auc_test = {m_test['auc']}, \n"
        f"rmse_train = {m_train['rmse']}, rmse_test = {m_test['rmse']},")
    pred_tr = lr_model.predict_proba(X_train)[:, 1]
    pred_te = lr_model.predict_proba(X_test)[:, 1]

    # Accumulate values for overall performance
    pointer = 0
    tr_sel = split["selector_train"]
    assert len(tr_sel) == len(y), "Issue in length alignment"
    for i, sel in enumerate(tr_sel):
        if sel:
            y_pred_tr[i] = pred_tr[pointer]
            y_truth_tr[i] = y[i]
            pointer += 1

    pointer = 0
    te_sel = split["selector_test"]
    for i, sel in enumerate(te_sel):
        if sel:
            y_pred_te[i] = pred_te[pointer]
            y_truth_te[i] = y[i]
            pointer += 1

    print("sum", np.sum(y - y_truth_tr - y_truth_te))
    assert np.sum(y - (y_truth_tr + y_truth_te)) == 0, "Issue in alignment"
    print("Passed alignment")
    print("\n------------------------")
    print("Completed model computations")

    print("\nOverall performance:")
    acc_train, auc_train, _, mse_train, _ = \
        compute_metrics(y_pred_tr, y_truth_tr)
    acc_test, auc_test, _, mse_test, _ = \
        compute_metrics(y_pred_te, y_truth_te)

    print(f"\nfeatures = {selected_features}, "
          f"\nacc_train = {acc_train}, acc_test = {acc_test}, "
          f"\nauc_train = {auc_train}, auc_test = {auc_test}, "
          f"\nrmse_train = {mse_train}, rmse_test = {mse_test}, ")

    res_dict = {
        "y_pred_train": y_pred_tr,
        "y_pred_test": y_pred_te,
        "y_truth_train": y_truth_tr,
        "y_truth_teest": y_truth_te,
    }

    return res_dict


def train_time_specialized(X, y, split, args, data_dict):

    print("Loading tcA for filtering")
    interaction_df = data_dict["interaction_df"]
    tcA = pd.read_pickle(DATASET_PATH[args.dataset] + "features/tcA.pkl")
    tcA = phi_inv(tcA["tcA"].values)  # undo prior transformation

    s = 0
    N = len(interaction_df)
    y_pred_tr, y_pred_te = np.zeros(N), np.zeros(N)
    y_truth_tr, y_truth_te = np.zeros(N), np.zeros(N)

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
        pointer = 0
        tr_sel = np.array(split["selector_train"] & selector).astype(int)
        assert len(tr_sel) == len(y), "Issue in length alignment"
        for i, sel in enumerate(tr_sel):
            if i % 1000000 == 0:
                print("train", i, pointer)
            if sel:
                y_pred_tr[i] = pred_tr[pointer]
                y_truth_tr[i] = y[i]
                pointer += 1

        pointer = 0
        te_sel = np.array(split["selector_test"] & selector).astype(int)
        for i, sel in enumerate(te_sel):
            if i % 1000000 == 0:
                print("test", i, pointer)
            if sel:
                y_pred_te[i] = pred_te[pointer]
                y_truth_te[i] = y[i]
                pointer += 1

    assert s == len(y), "Issue during filtering"
    print("sum", np.sum(y - y_truth_tr - y_truth_te))
    assert np.sum(y - (y_truth_tr + y_truth_te)) == 0, "Issue in alignment"
    print("Passed alignment")
    print("\n------------------------")
    print("Completed model computations")

    print("\nOverall performance:")
    acc_train, auc_train, _, mse_train, _ = \
        compute_metrics(y_pred_tr, y_truth_tr)
    acc_test, auc_test, _, mse_test, _ = \
        compute_metrics(y_pred_te, y_truth_te)

    print(f"\nfeatures = {selected_features}, "
          f"\nacc_train = {acc_train}, acc_test = {acc_test}, "
          f"\nauc_train = {auc_train}, auc_test = {auc_test}, "
          f"\nrmse_train = {mse_train}, rmse_test = {mse_test}, ")

    res_dict = {
        "y_pred_train": y_pred_tr,
        "y_pred_test": y_pred_te,
        "y_truth_train": y_truth_tr,
        "y_truth_teest": y_truth_te,
    }

    return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train regression model.')
    prepare_parser.add_feature_arguments(parser)
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads for feature preparation.')
    parser.add_argument('--num_iterations', type=int, default=5000)
    parser.add_argument('--col', type=str, help="Column to split with.")

    args = parser.parse_args()
    split_id = args.split_id
    selected_features = [features for features in ALL_FEATURES
                         if vars(args)[features]]
    selected_features.sort()

    print("Selected features: ", selected_features)
    print("Num Threads: " + str(args.num_threads))
    print("Cross-validation split: " + str(split_id))
    print("Iterations: " + str(args.num_iterations))
    dataset = args.dataset
    print("Dataset name:", dataset)
    print("Partition with:", args.col)
    assert dataset in DATASET_PATH, "The specified dataset is not supported"
    assert args.col in COLS, "Specified column is not supported"
    suf = '_'.join(selected_features)

    X, y, split = loader.get_combined_features_and_split(selected_features,
                                                         split_id, dataset)

    # Models for different parts of the data
    data_dict = loader.load_preprocessed_data(dataset)

    if args.col == "single":
        res_dict = train_single(X, y, split, args, data_dict)
    elif args.col == "time":
        res_dict = train_time_specialized(X, y, split, args, data_dict)
    else:
        res_dict = train_partitions(X, y, split, args, data_dict)

    name = dataset + "_" + args.col + "_s" + str(split_id) + "_" + suf
    with open('notebooks/plotting/partitioning/' + name + '.pkl', 'wb') as f:
        pickle.dump(res_dict, f)

    print("\n----------------------------------------")
    print("Completed logistic regression")
