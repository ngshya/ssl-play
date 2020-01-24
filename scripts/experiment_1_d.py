import sys
import os
sys.path.append('.')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import pandas as pd
import numpy as np
import time

from sslplay.performance.f1 import f1, f1W
from sslplay.performance.auc import auc, aucW
from sslplay.performance.accuracy import accuracy

from sslplay.data.spambase import DataSpambase
from sslplay.data.creditcard import DataCreditCard
from sslplay.data.splice import DataSplice
from sslplay.data.landsat import DataLandsat
from sslplay.data.letter import DataLetter
from sslplay.data.digits import DataDigits
from sslplay.data.cifar import DataCIFAR

from sslplay.model.data_model_run import data_model_run
from sslplay.model.random_forest import ModelRF
from sslplay.model.neural_network import ModelNeuralNetwork
from sslplay.model.kmeans_random_forest import ModelKMeansRF
from sslplay.model.ladder_network import ModelLadderNetwork
from sslplay.model.label_spreading import ModelLabelSpreading

from sslplay.utils.s3split import s3split


np.random.seed(1102)


str_run_id = time.strftime("%Y-%m-%d-%H-%M")

obj_data = DataDigits()
obj_data.load()
obj_data.parse()

dtf_performance = None
dict_perf={
    "ACCURACY": accuracy, 
    "AUC_MACRO": auc, 
    "AUC_WEIGHTED": aucW, 
    "F1_MACRO": f1, 
    "F1_WEIGHTED": f1W
}

for nfold in range(2):
    seed_1 = np.random.randint(100)

    for nsample in range(2):
        seed_2 = np.random.randint(100)
        Xt, yt, Xu, yu, Xl, yl = s3split(
            X=obj_data.X, 
            y=obj_data.y, 
            percentage_1=20, 
            percentage_2=77, 
            percentage_3=3,
            seed_1=seed_1, 
            seed_2=seed_2
        )

        for p in [10, 30, 50, 70, 77]:
            q = p/77.0
            tmp_len = len(yu)
            array_bool = np.random.choice(
                a=[False, True], 
                size=tmp_len, 
                p=[1-q, q], 
                replace=True
            )
            Xutmp = Xu[array_bool, :]
            yutmp = yu[array_bool]

            for class_model in \
            [ModelRF, ModelNeuralNetwork, \
            ModelKMeansRF, ModelLadderNetwork, ModelLabelSpreading]:
                obj_model = class_model()
                logging.info("FOLD " + str(nfold+1) + \
                " | SAMPLE " + str(nsample+1) + \
                " | p " + str(p) + \
                " | MODEL " + obj_model.name)
                obj_model.fit(Xl, yl, Xutmp)
                array_test_pred = obj_model.predict(Xt)
                array_test_real = yt

                for key_perf in dict_perf.keys():
                    dtf_performance_tmp = pd.DataFrame({
                        "FOLD": [nfold+1],
                        "SAMPLE": [nsample+1],
                        "PERC_UNLA": [p],
                        "MODEL": [obj_model.name],
                        "METRIC": [key_perf], 
                        "VALUE": [dict_perf[key_perf](array_test_real, 
                            array_test_pred)]
                    })
                
                    if dtf_performance is None:
                        dtf_performance = dtf_performance = dtf_performance_tmp
                    else: 
                        dtf_performance = dtf_performance\
                        .append(dtf_performance_tmp, sort=False)\
                        .reset_index(drop=True)



dtf_performance.to_csv(
    path_or_buf="outputs/" + str_run_id + "-experiment-1-d.csv", 
    sep=";", 
    index=False,
    decimal="."
)