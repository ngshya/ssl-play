import numpy as np
import random
import logging

def ssplit(X, y, percentage_1, percentage_2, seed=1102):

    assert percentage_1 > 0
    assert percentage_2 > 0

    np.random.seed(seed)
    random.seed(seed)

    y = np.array(y)
    X = np.array(X)

    tmp_percentage_sum = percentage_1 + percentage_2 + 0.0
    percentage_1 = percentage_1 / tmp_percentage_sum
    percentage_2 = percentage_2 / tmp_percentage_sum

    int_n = len(y)
    assert X.shape[0] == int_n
    
    tmp_y_counts = np.unique(y, return_counts=True)
    assert np.min(tmp_y_counts[1]) >= 2

    array_classes = sorted(tmp_y_counts[0])
    int_n_classes = len(array_classes)
    assert np.max(array_classes) + 1 == int_n_classes

    array_sets = np.repeat(None, int_n)

    for c in array_classes:
        array_bool_c = (y == c)
        int_n_c = sum(array_bool_c)
        array_sets_c = np.random.choice(
            a=[1, 2], 
            size=int_n_c, 
            p=[percentage_1, percentage_2], 
            replace=True
        )
        if len(np.unique(array_sets_c)) < 2:
            array_sets_c[random.sample(range(len(array_sets_c)), 2)] \
            = np.array([1,2])[2]
            logging.debug("The class " + str(c)\
            + " did not have all sets. Issue solved by adding them!")
        array_sets[array_bool_c] = array_sets_c

    X1 = X[array_sets == 1, :]
    y1 = y[array_sets == 1]
    X2 = X[array_sets == 2, :]
    y2 = y[array_sets == 2]

    return X1, y1, X2, y2
