import pandas as pd
import numpy as np
import tensorflow as tf
import random

from sslplay.utils.s3split import s3split

from sslplay.performance.f1 import f1, f1W
from sslplay.performance.auc import auc, aucW
from sslplay.performance.accuracy import accuracy

import logging

def data_model_run(
    class_data, 
    class_model, 
    percentage_test, 
    percentage_unlabeled, 
    percentage_labeled, 
    dict_perf={
        "ACCURACY": accuracy, 
        "AUC_MACRO": auc, 
        "AUC_WEIGHTED": aucW, 
        "F1_MACRO": f1, 
        "F1_WEIGHTED": f1W
    },
    cv_folds=5, 
    random_samples=10, 
    seed=1102
):

    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    array_seed_1 = np.random.randint(low=1, high=9999, size=cv_folds)
    array_seed_2 = np.random.randint(low=1, high=9999, size=random_samples)

    obj_data = class_data()
    obj_data.load()

    dtf_performance = pd.DataFrame(
        data=None, 
        columns=["METRIC", "VALUE"]
    )

    for fold in range(cv_folds): 
        for n_sample in range(random_samples):

            obj_model = class_model()

            logging.info(
                "MODEL " + obj_model.name + " " 
                + "DATA " + obj_data.name + " " 
                + "T" + str(percentage_test) + " " 
                + "U" + str(percentage_unlabeled) + " "
                + "L" + str(percentage_labeled) + " "
                + "FOLD " + str(fold+1) + " "
                + "SAMPLE " + str(n_sample+1) + " "
            )

            Xt, yt, Xu, yu, Xl, yl = s3split(
                X=obj_data.X, 
                y=obj_data.y, 
                percentage_1=percentage_test, 
                percentage_2=percentage_unlabeled, 
                percentage_3=percentage_labeled,
                seed_1=array_seed_1[fold], 
                seed_2=array_seed_2[n_sample]
            )

            obj_model.fit(Xl, yl, Xu)
            array_test_pred = obj_model.predict(Xt)
            array_test_real = yt

            for key_perf in dict_perf.keys():
                dtf_performance = dtf_performance.append(pd.DataFrame({
                    "METRIC": [key_perf], 
                    "VALUE": [dict_perf[key_perf](array_test_real, 
                                                  array_test_pred)]
                }), sort=False).reset_index(drop=True)
            
    dtf_performance = dtf_performance\
    .groupby(["METRIC"])\
    .agg({"VALUE": [np.mean, np.std]})\
    .reset_index(drop=False)

    dtf_performance.columns = ["METRIC", "VALUE_MEAN", "VALUE_STD"]
    dtf_performance["MODEL"] = obj_model.name 
    dtf_performance["DATASET"] = obj_data.name
    dtf_performance["PERC_T"] = percentage_test
    dtf_performance["PERC_U"] = percentage_unlabeled
    dtf_performance["PERC_L"] = percentage_labeled
    dtf_performance["CV"] = cv_folds
    dtf_performance["SAMPLES"] = random_samples

    dtf_performance = dtf_performance.loc[:, ["MODEL", "DATASET", "PERC_T", "PERC_U", "PERC_L", "CV", "SAMPLES", "METRIC", "VALUE_MEAN", "VALUE_STD"]]


    logging.info('\t\n'+ dtf_performance.to_string().replace('\n', '\n\t'))

    return dtf_performance