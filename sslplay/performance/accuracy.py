import numpy as np
import logging

def accuracy(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = y_pred.argmax(axis=1)
    out = sum(y_real == y_pred) / len(y_real)
    baseline = sum(y_real == np.random.choice(y_real, size=len(y_real), replace=True)) / len(y_real)
    logging.debug("ACCURACY: " + str(round(out, 4)) + " | Baseline: " + str(round(baseline, 4))) 
    return out