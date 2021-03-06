import numpy as np
from sklearn.metrics import roc_auc_score
import logging

def auc(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    if len(np.unique(y_real)) == 2:
        y_pred = y_pred[:, 1]
    out = roc_auc_score(y_real, y_pred, multi_class="ovr", average="macro")
    logging.debug("AUC MACRO: " + str(out))
    return out


def aucW(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    if len(np.unique(y_real)) == 2:
        y_pred = y_pred[:, 1]
    out = roc_auc_score(y_real, y_pred, multi_class="ovr", average="weighted")
    logging.debug("AUC WEIGHTED: " + str(out))
    return out 