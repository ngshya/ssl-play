import numpy as np
import random
import logging
from sslplay.utils.ssplit import ssplit

def s3split(X, y, percentage_1, percentage_2, percentage_3, seed_1=1102, seed_2=1102):

    assert percentage_1 >= 0
    assert percentage_2 >= 0
    assert percentage_3 >= 0
    assert percentage_1 + percentage_2 + percentage_3 > 0

    y = np.array(y)
    X = np.array(X)

    tmp_percentage_sum = percentage_1 + percentage_2 + percentage_3 + 0.0
    percentage_1 = percentage_1 / tmp_percentage_sum * 100
    percentage_2 = percentage_2 / tmp_percentage_sum * 100
    percentage_3 = percentage_3 / tmp_percentage_sum * 100

    int_n = len(y)
    assert X.shape[0] == int_n
    
    tmp_y_counts = np.unique(y, return_counts=True)
    assert np.min(tmp_y_counts[1]) >= 3

    array_classes = sorted(tmp_y_counts[0])
    int_n_classes = len(array_classes)
    assert np.max(array_classes) + 1 == int_n_classes

    X1, y1, Xtmp, ytmp = ssplit(
        X=X, y=y, 
        percentage_1=percentage_1, 
        percentage_2=100.0-percentage_1, 
        min_el_1=1, 
        min_el_2=2, 
        seed=seed_1
    )

    X2, y2, X3, y3 = ssplit(
        X=Xtmp, y=ytmp, 
        percentage_1=percentage_2, 
        percentage_2=percentage_3, 
        min_el_1=1, 
        min_el_2=1, 
        seed=seed_2
    )

    logging.debug("Set 1 expected percentage: " + str(round(percentage_1, 4)) + " | real percentage: " + str(round(len(y1) / int_n * 100, 4)))
    logging.debug("Set 2 expected percentage: " + str(round(percentage_2, 4)) + " | real percentage: " + str(round(len(y2) / int_n * 100, 4)))
    logging.debug("Set 3 expected percentage: " + str(round(percentage_3, 4)) + " | real percentage: " + str(round(len(y3) / int_n * 100, 4)))

    return X1, y1, X2, y2, X3, y3
