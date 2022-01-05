import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss, f1_score


def compute_metrics(y_pred, y, protection=1e-8):
    """
    Compute accuracy, AUC score, Negative Log Loss, MSE and F1 score
    """
    # print(y_pred.min(), y_pred.max(), y_pred.shape)
    y_pred = np.array([i if np.isfinite(i) else 0.5 for i in y_pred])
    acc = accuracy_score(y, y_pred >= 0.5)
    mse = brier_score_loss(y, y_pred)
    # certain metrics can only be computed if both classes are present
    if (0 in y) and (1 in y):
        auc = roc_auc_score(y, y_pred)
        nll = log_loss(y, y_pred)
        f1 = f1_score(y, y_pred >= 0.5)
    else:
        auc = -1
        nll = -1
        f1 = -1
    return acc, auc, nll, mse, f1


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
