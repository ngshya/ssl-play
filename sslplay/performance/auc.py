import numpy as np
from sklearn import metrics

def auc(y_real, y_pred):
    y_real = np.array(y_real)
    if np.max(y_real) == 1:
        y_real = y_real + 1
    y_pred = np.array(y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_real, y_pred, pos_label=2)
    return metrics.auc(fpr, tpr)