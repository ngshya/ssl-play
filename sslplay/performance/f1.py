from sklearn.metrics import f1_score
import numpy as np
import logging

def f1(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = y_pred.argmax(axis=1)
    out = f1_score(y_real, y_pred, average="macro")
    logging.debug("F1: " + str(out))
    return out