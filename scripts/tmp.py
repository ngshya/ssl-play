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


dtf_performance_tmp = data_model_run(
    class_data=DataDigits,
    class_model=ModelRF, 
    percentage_test=20,
    percentage_unlabeled=40, 
    percentage_labeled=40, 
    cv_folds=2, 
    random_samples=2,
    seed=1102
)
        