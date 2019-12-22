from sklearn.metrics import f1_score
import numpy as np
import logging

def f1(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = y_pred.argmax(axis=1)
    out = f1_score(y_real, y_pred, average="macro")
    logging.debug("F1 MACRO: " + str(out))
    return out


def f1W(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = y_pred.argmax(axis=1)
    out = f1_score(y_real, y_pred, average="weighted")
    logging.debug("F1 WEIGHTED: " + str(out))
    return out