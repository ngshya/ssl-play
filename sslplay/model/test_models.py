import numpy as np
import pandas as pd
import tensorflow as tf

from sslplay.performance.f1 import f1, f1W
from sslplay.performance.auc import auc, aucW
from sslplay.performance.accuracy import accuracy

from sslplay.data.spambase import DataSpambase
from sslplay.data.creditcard import DataCreditCard
from sslplay.data.splice import DataSplice
from sslplay.data.landsat import DataLandsat
from sslplay.data.letter import DataLetter
from sslplay.data.mnist import DataMNIST
from sslplay.data.usps import DataUSPS
from sslplay.data.cifar import DataCIFAR

from sslplay.model.random_forest import ModelRF
from sslplay.model.neural_network import ModelNeuralNetwork
from sslplay.model.kmeans_random_forest import ModelKMeansRF
from sslplay.model.ladder_network import ModelLadderNetwork
from sslplay.model.label_spreading import ModelLabelSpreading

import logging

def test_models(
    file_out="out.csv",
    array_datasets=[DataSpambase, DataCreditCard, DataSplice, DataLandsat, DataLetter, DataMNIST, DataUSPS, DataCIFAR], 
    perc_test = 20,
    array_perc_unla=[79.9], 
    array_models=[ModelRF, ModelNeuralNetwork, ModelKMeansRF, ModelLadderNetwork, ModelLabelSpreading], 
    dict_perf={"ACCURACY": accuracy, "AUC_MACRO": auc, "AUC_WEIGHTED": aucW, "F1_MACRO": f1, "F1_WEIGHTED": f1W}
):
    tf.set_random_seed(1102)
    np.random.seed(1102)

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
                        "PERC_UNLABELLED": [percentage_unlabelled], 
                        "METRIC": [key_perf], 
                        "VALUE" : [dict_perf[key_perf](array_test_real, array_test_pred)],
                        "Y_TEST_DISTR": [str([round(x, 2) for x in sorted(np.unique(array_test_real, return_counts=True)[1] / len(array_test_real) * 100)])]
                    }), sort=False).reset_index(drop=True)
        dtf_performance.to_csv(file_out, sep=";", index=False)

    return dtf_performance.reset_index(drop=True)