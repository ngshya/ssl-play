import numpy as np
import random
import logging

def lut_split(X, y, percentage_test=20, percentage_unlabelled=70, percentage_labelled=10, seed=1102):

    np.random.seed(seed)
    random.seed(seed)

    y = np.array(y)
    X = np.array(X)

    tmp_percentage_sum = percentage_test + percentage_unlabelled + percentage_labelled + 0.0
    percentage_test = percentage_test / tmp_percentage_sum
    percentage_unlabelled = percentage_unlabelled / tmp_percentage_sum
    percentage_labelled = percentage_labelled / tmp_percentage_sum
    array_bool_percentage = [percentage_labelled > 0, percentage_unlabelled > 0, percentage_test > 0]
    int_sets = sum(array_bool_percentage)

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
        array_sets_c = np.random.choice(a=["T", "U", "L"], size=int_n_c, p=[percentage_test, percentage_unlabelled, percentage_labelled], replace=True)
        if len(np.unique(array_sets_c)) < sum(array_bool_percentage):
            array_sets_c[random.sample(range(len(array_sets_c)), int_sets)] = np.array(["L", "U", "T"])[array_bool_percentage]
            logging.debug("The class " + str(c) + " did not have all sets. Issue solved!")
        array_sets[array_bool_c] = array_sets_c

    Xl = X[array_sets == "L", :]
    yl = y[array_sets == "L"]
    Xu = X[array_sets == "U", :]
    yu = y[array_sets == "U"]
    Xt = X[array_sets == "T", :]
    yt = y[array_sets == "T"]

    logging.debug("--- Labelled ---")
    logging.debug("Labelled X: " + str(Xl.shape))
    logging.debug("Labelled y: " + str(len(yl)))
    logging.debug("Labelled percentage: " + str(round(len(yl) / int_n * 100, 4)) + "%; expected: " + str(round(percentage_labelled * 100, 4))) 
    logging.debug("Possible targets: " + str(sorted((np.unique(yl)))))

    logging.debug("--- Unlabelled ---")
    logging.debug("Unlabelled X: " + str(Xu.shape))
    logging.debug("Unlabelled y: " + str(len(yu)))
    logging.debug("Unlabelled percentage: " + str(round(len(yu) / int_n * 100, 4)) + "%; expected: " + str(round(percentage_unlabelled * 100, 4))) 
    logging.debug("Possible targets: " + str(sorted((np.unique(yu)))))

    logging.debug("--- Test ---")
    logging.debug("Test X: " + str(Xt.shape))
    logging.debug("Test y: " + str(len(yt)))
    logging.debug("Test percentage: " + str(round(len(yt) / int_n * 100, 4)) + "%; expected: " + str(round(percentage_test * 100, 4))) 
    logging.debug("Possible targets: " + str(sorted((np.unique(yt)))))


    return Xl, yl, Xu, yu, Xt, yt
