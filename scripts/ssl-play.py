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
import pickle
from pathlib import Path
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

import argparse

dict_data = {
    "spambase": DataSpambase,
    "creditcard": DataCreditCard,
    "splice": DataSplice,
    "landsat": DataLandsat,
    "letter": DataLetter,
    "digits": DataDigits,
    "cifar": DataCIFAR
}

dict_model = {
    "rf": ModelRF,
    "nn": ModelNeuralNetwork,
    "krf": ModelKMeansRF,
    "ln": ModelLadderNetwork,
    "ls": ModelLabelSpreading,
}


parser=argparse.ArgumentParser(
    description='Testing semi-supervised models on different datasets.'
)
parser.add_argument('--data', help='dataset')
parser.add_argument('--model', help='model')
parser.add_argument(
    '--ptest', 
    help='percentage test set', 
    default=20
)
parser.add_argument(
    '--punla', 
    help='percentage unlabelled data',
    default=75
)
parser.add_argument(
    '--plabe', 
    help='percentage labelled data', 
    default=5    
)
parser.add_argument(
    '--folds', 
    help='number of folds for cross validation', 
    default=1
)
parser.add_argument(
    '--samples', 
    help='number of labelled and unlabelled samples', 
    default=1
)
parser.add_argument('--seed', help='random seed', default=1102)
parser.add_argument(
    '--outfolder', 
    help='output folder', 
    default="outputs/batch_run_output/"
)
args=parser.parse_args()


if __name__ == "__main__":

    if not os.path.isdir("outputs"):
        os.system("mkdir outputs")

    if not os.path.isdir(Path("outputs/batch_run_output")):
        os.system("mkdir outputs/batch_run_output")

    file_name = Path(
        str(args.outfolder) \
        + "d_" + args.data \
        + "_m_" + args.model \
        + "_t" + str(args.ptest) \
        + "_u" + str(args.punla) \
        + "_l" + str(args.plabe) \
        + "_f" + str(args.folds) \
        + "_s" + str(args.samples) \
        + "_r" + str(args.seed) \
        + ".pickle"
    )

    if os.path.isfile(file_name): 
        print(str(file_name) + " already exists!")
        exit()
    
    dtf_performance_tmp = data_model_run(
        class_data=dict_data[args.data],
        class_model=dict_model[args.model], 
        percentage_test=float(args.ptest),
        percentage_unlabeled=float(args.punla), 
        percentage_labeled=float(args.plabe),
        cv_folds=int(args.folds), 
        random_samples=int(args.samples),
        seed=int(args.seed)
    )
    
    pickle.dump(dtf_performance_tmp, open(file_name, "wb"))

    print(str(file_name) + " saved!")

