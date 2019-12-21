import numpy as np
import pandas as pd
from sslplay.data.spambase import DataSpambase
from sslplay.data.creditcard import DataCreditCard
from sslplay.data.mnist import DataMNIST
from sslplay.data.cifar import DataCIFAR
from sslplay.model.random_forest import ModelRF
from sslplay.model.kmeans_random_forest import ModelKMeansRF
from sslplay.model.label_spreading import ModelLabelSpreading
from sslplay.performance.auc import auc
from sslplay.performance.f1 import f1
import logging

def test_models(
    array_datasets=[DataSpambase, DataCreditCard, DataMNIST, DataCIFAR], 
    perc_test = 20,
    array_perc_unla=[0, 79, 79.5, 79.9], 
    array_models=[ModelRF, ModelKMeansRF, ModelLabelSpreading], 
    dict_perf={"AUC": auc, "F1": f1}
):
    dtf_performance = pd.DataFrame(
        data=None, 
        columns=["MODEL", "DATASET", "PERCENTAGE_UNLABELLED", "METRIC", "VALUE"]
    )
    for dataset_class in array_datasets:
        obj_data = dataset_class()
        obj_data.load()
        obj_data.parse()
        for percentage_unlabelled in array_perc_unla:
            logging.debug("= = = " + obj_data.name + " = = =") 
            Xl, yl, Xu, yu, Xt, yt = obj_data.split(percentage_test=perc_test, percentage_unlabelled=percentage_unlabelled, percentage_labelled=100-perc_test-percentage_unlabelled)
            for model_class in array_models:
                obj_model = model_class()
                logging.debug("MODEL " + obj_model.name + " | DATASET: " + obj_data.name + " | PERC_UNLA: " + str(percentage_unlabelled))
                obj_model.fit(Xl, yl, Xu)
                array_test_pred = obj_model.predict(Xt)
                array_test_real = yt
                for key_perf in dict_perf.keys():
                    dtf_performance = dtf_performance.append(pd.DataFrame({
                        "MODEL": [obj_model.name], 
                        "DATASET": [obj_data.name], 
                        "PERCENTAGE_UNLABELLED": [percentage_unlabelled], 
                        "METRIC": [key_perf], 
                        "VALUE" : [dict_perf[key_perf](array_test_real, array_test_pred)]
                    }))

    return dtf_performance.reset_index(drop=True)