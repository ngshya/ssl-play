import sys
import os
sys.path.append('.') 

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import pandas as pd

from sslplay.model.random_forest import ModelRF
from sslplay.model.label_spreading import ModelLabelSpreading
from sslplay.model.ladder_network import ModelLadderNetwork
from sslplay.model.neural_network import ModelNeuralNetwork

from sslplay.data.spambase import DataSpambase
from sslplay.data.mnist import DataMNIST

from sslplay.performance.auc import auc
from sslplay.performance.f1 import f1

obj_data = DataSpambase()
obj_data.load()
obj_data.parse()
Xl, yl, Xu, yu, Xt, yt = obj_data.split(percentage_test=20, percentage_unlabelled=10, percentage_labelled=70)

obj_model = ModelLadderNetwork()
obj_model.fit(Xl, yl, Xu)
array_test_pred = obj_model.predict(Xt)

auc(yt, array_test_pred)
f1(yt, array_test_pred)