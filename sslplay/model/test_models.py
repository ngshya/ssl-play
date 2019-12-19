import numpy as np
import pandas as pd
from sslplay.data.spambase import DataSpambase
from sslplay.data.creditcard import DataCreditCard
from sslplay.data.mnist import DataMNIST
from sslplay.data.cifar import DataCIFAR
from sslplay.model.random_forest import ModelRF
from sslplay.model.kmeans_random_forest import ModelKMeansRF
from sslplay.performance.auc import auc
from sslplay.performance.f1 import f1

def test_models(
    array_datasets=[DataSpambase, DataCreditCard, DataMNIST, DataCIFAR], 
    array_port_unla=[0.0, 0.5, 0.9, 0.99], 
    array_models=[ModelRF, ModelKMeansRF], 
    dict_perf={"AUC": auc, "F1": f1}
):
    dtf_performance = pd.DataFrame(data=None, columns=["MODEL", "DATASET", "PORTION_UNLABELLED_DATA", "METRIC", "VALUE"])
    for model_class in array_models:
        obj_model = model_class()
        for dataset_class in array_datasets:
            obj_data = dataset_class()
            obj_data.load()
            obj_data.parse()
            for port_unla in array_port_unla: 
                dict_split_out = obj_data.split(port_test=0.2, port_unla=port_unla)
                obj_model.fit(dict_split_out["train_l"], dict_split_out["target_train_l"], dict_split_out["train_u"])
                array_test_pred = obj_model.predict(dict_split_out["test"])
                array_test_real = dict_split_out["target_test"]
                
                for key_perf in dict_perf.keys():

                    dtf_performance = dtf_performance.append(pd.DataFrame({
                        "MODEL": [obj_model.name], 
                        "DATASET": [obj_data.name], 
                        "PORTION_UNLABELLED_DATA": [port_unla], 
                        "METRIC": [key_perf], 
                        "VALUE" : [dict_perf[key_perf](array_test_real, array_test_pred)]
                    }))

    return dtf_performance.reset_index(drop=True)