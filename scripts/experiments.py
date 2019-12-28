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

from sslplay.performance.f1 import f1
from sslplay.performance.auc import auc
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


# Experiment 1

dtf_performance = None
str_run_id = time.strftime("%Y-%m-%d-%H-%M")
for percentage_unlabeled in \
[40, 45, 50, 55, 60, 65, 70, 75, 79, 79.5, 79.9]:
    for class_model in \
    [ModelRF, ModelNeuralNetwork, \
    ModelKMeansRF, ModelLadderNetwork, ModelLabelSpreading]:
        dtf_performance_tmp = data_model_run(
            class_data=DataDigits,
            class_model=class_model, 
            percentage_test=20,
            percentage_unlabeled=percentage_unlabeled, 
            percentage_labeled=80-percentage_unlabeled, 
            cv_folds=2, 
            random_samples=2,
            seed=1102
        )
        if dtf_performance is None:
            dtf_performance = dtf_performance_tmp
        else: 
            dtf_performance = dtf_performance.append(dtf_performance_tmp)\
            .reset_index(drop=True)
        dtf_performance.to_csv(
            path_or_buf="outputs/" + str_run_id + "-experiment-1.csv", 
            sep=";", 
            index=False,
            decimal="."
        )



# Experiment 2

dtf_performance = None
str_run_id = time.strftime("%Y-%m-%d-%H-%M")
for class_data in \
[DataSpambase, DataCreditCard, \
DataSplice, DataLandsat, \
DataLetter, DataDigits, DataCIFAR]:
    for class_model in \
    [ModelRF, ModelNeuralNetwork, \
    ModelKMeansRF, ModelLadderNetwork, ModelLabelSpreading]:
        dtf_performance_tmp = data_model_run(
            class_data=class_data,
            class_model=class_model, 
            percentage_test=20,
            percentage_unlabeled=0.1, 
            percentage_labeled=79.9,
            cv_folds=2, 
            random_samples=2,
            seed=1102
        )
        if dtf_performance is None:
            dtf_performance = dtf_performance_tmp
        else: 
            dtf_performance = dtf_performance.append(dtf_performance_tmp)\
            .reset_index(drop=True)
        dtf_performance.to_csv(
            path_or_buf="outputs/" + str_run_id + "-experiment-2.csv", 
            sep=";", 
            index=False,
            decimal="."
        )



# Experiment 3

dtf_performance = None
str_run_id = time.strftime("%Y-%m-%d-%H-%M")
for class_data in \
[DataSpambase, DataCreditCard, \
DataSplice, DataLandsat, \
DataLetter, DataDigits, DataCIFAR]:
    for class_model in \
    [ModelRF, ModelNeuralNetwork, \
    ModelKMeansRF, ModelLadderNetwork, ModelLabelSpreading]:
        dtf_performance_tmp = data_model_run(
            class_data=class_data,
            class_model=class_model, 
            percentage_test=20,
            percentage_unlabeled=5, 
            percentage_labeled=75, 
            cv_folds=2, 
            random_samples=2,
            seed=1102
        )
        if dtf_performance is None:
            dtf_performance = dtf_performance_tmp
        else: 
            dtf_performance = dtf_performance.append(dtf_performance_tmp)\
            .reset_index(drop=True)
        dtf_performance.to_csv(
            path_or_buf="outputs/" + str_run_id + "-experiment-3.csv", 
            sep=";", 
            index=False,
            decimal="."
        )
