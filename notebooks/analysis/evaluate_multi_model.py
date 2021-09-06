# This is a helper script to determine the best combination of multiple models

import pickle
import numpy as np
from itertools import chain, combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


###############################################################################
# Set configuration
###############################################################################

# "elemmath_2021", "ednet_kt3", "eedi", "junyi_15"
DATASET = "elemmath_2021"
SEED = 888

# represents experimental setting
# SUFFIX = "i_s_scA_scW_tcA_tcW"

# represents relevant partitions
# i s t c sm tea sch at part part_id bundle bundle_id single

# elemmath_2021
PARTITIONS = ["single", "time", "i", "s", "sm", "tea", "sch", "c", "t", "at"]
SUFFIX = "i_icA_TW_icW_TW_lag_time_cat_n_gram_postcA_postcW_precA_precW_prev_"\
    + "resp_time_cat_rc_s_scA_TW_scW_TW_sm_t_tcA_TW_tcW_TW_user_avg_correct_vw"

# ednet
# PARTITIONS = ["single", "time", "i", "hashed_skill_id", "sm", "bundle_id",
#               "part_id", "at"]
# SUFFIX = "i_icA_TW_icW_TW_lag_time_cat_n_gram_partcA_partcW_" \
#     "prev_resp_time_cat_s_scA_TW_scW_TW_sm_tcA_TW_tcW_TW_user_avg_correct_vw"


# eedi
# SUFFIX = "bundle_i_icA_TW_icW_TW_n_gram_precA_precW_s_scA_TW_scW_TW_sm" + \
#    "_tcA_TW_tcW_TW_tea_user_avg_correct"
# PARTITIONS = ["single", "time", "i", "hashed_skill_id", "sm", "tea",
#               "bundle_id"]


# junyi
# SUFFIX = "hour_i_icA_TW_icW_TW_lag_time_cat_n_gram_postcA_postcW_precA_" \
#    + "precW_prev_resp_time_cat_rc_s_scA_TW_scW_TW_sm_tcA_TW_tcW_TW_" \
#    + "user_avg_correct"
# PARTITIONS = ["single", "time", "i", "s", "sm", "part_id"]

splits = []
for i in range(5):
    path = "./../../data/" + DATASET + "/preparation/split_s" + str(SEED) + \
        "_" + str(i) + ".pkl"
    with open(path, "rb") as file_object:
        s = pickle.load(file_object)
    splits.append(s)


###############################################################################
# Helper functions
###############################################################################

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def compute_metrics(y_pred, y, protection=1e-8):
    """
    Compute accuracy, AUC score, Negative Log Loss, MSE and F1 score
    """
    # print(y_pred.min(), y_pred.max(), y_pred.shape)
    y_pred = np.array([i if np.isfinite(i) else 0.5 for i in y_pred])
    acc = accuracy_score(y, y_pred >= 0.5)
    auc = roc_auc_score(y, y_pred)
    return acc, auc


class Metrics:
    """
    Keep track of metrics over time in a dictionary.
    """
    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def store(self, new_metrics):
        for key in new_metrics:
            if key in self.metrics:
                self.metrics[key] += new_metrics[key]
                self.counts[key] += 1
            else:
                self.metrics[key] = new_metrics[key]
                self.counts[key] = 1

    def average(self):
        average = {k: v / self.counts[k] for k, v in self.metrics.items()}
        self.metrics, self.counts = {}, {}
        return average


def comb_lr_performance(ps, dataset, suf, splits, split_id):
    # combine predictions
    train_selector = splits[split_id]["selector_train"]
    test_selector = splits[split_id]["selector_test"]

    pred_tr, pred_te = [], []
    for p in ps:
        path = "./partitioning/" + dataset + "_" + p + \
            "_s" + str(split_id) + "_" + suf + ".pkl"
        with open(path, 'rb') as f:
            res_dict = pickle.load(f)

        y_pred_tr = res_dict["y_pred_train"][train_selector]
        y_truth_tr = res_dict["y_truth_train"][train_selector]
        y_pred_te = res_dict["y_pred_test"][test_selector]
        y_truth_te = res_dict["y_truth_teest"][test_selector]

        pred_tr.append(y_pred_tr)
        pred_te.append(y_pred_te)

    X_train = np.array(pred_tr).T
    y_train = y_truth_tr
    X_test = np.array(pred_te).T
    y_test = y_truth_te

    lr_model = LogisticRegression(solver="liblinear",
                                  max_iter=5000,
                                  n_jobs=8,
                                  verbose=0)
    lr_model.fit(X_train, y_train)

    pred_tr = lr_model.predict_proba(X_train)[:, 1]
    pred_te = lr_model.predict_proba(X_test)[:, 1]

    acc_train, auc_train = \
        compute_metrics(pred_tr, y_train)
    acc_test, auc_test = \
        compute_metrics(pred_te, y_test)

    return acc_train, auc_train, acc_test, auc_test


###############################################################################
# Extraction code
###############################################################################

print("\nSearching combination model: " + DATASET)
print("-----------------------------")
print("Available splits:")
print(SUFFIX)
print(PARTITIONS)

best_combination = ""
best_acc_avg = 0
best_auc_avg = 0

# evaluate each combination
index = 0
for ps in powerset(PARTITIONS):
    index += 1
    ps = list(ps)
    if not (len(ps) >= 1):  # need at least one partitions
        continue
    print("")
    print(DATASET)
    print("Evaluating:", index, ps)

    acc_vals, auc_vals = [], []
    for split_id in range(1):
        print("Split", split_id)
        acc_train, auc_train, acc_test, auc_test = \
            comb_lr_performance(ps, DATASET, SUFFIX, splits, split_id)
        acc_vals.append(acc_test)
        auc_vals.append(auc_test)

    acc_vals = np.array(acc_vals)
    auc_vals = np.array(auc_vals)
    print(acc_vals)
    print(auc_vals)

    acc_avg = np.round(np.mean(acc_vals), decimals=6)
    auc_avg = np.round(np.mean(auc_vals), decimals=6)

    print("ACC", acc_avg)
    print("AUC", auc_avg)

    if auc_avg > best_auc_avg:
        best_combination = ps
        best_acc_avg = acc_avg
        best_auc_avg = auc_avg


print("")
print("Completed evaluation")
print("---------------------------------------\n")
print("Dataset:", DATASET)
print("Partitions", PARTITIONS)
print("Suffix", SUFFIX)
print("")
print("Best combination:", best_combination)
print("ACC:", best_acc_avg)
print("AUC:", best_auc_avg)
print("\n---------------------------------------\n")
