import pandas as pd
import numpy as np

import sys
import os
sys.path.append('.')


from sslplay.performance.f1 import f1
from sslplay.performance.auc import auc
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
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


pun = 79.9

obj_data = DataLandsat()
obj_data.load()
obj_data.parse()
Xl, yl, Xu, yu, Xt, yt = obj_data.split(
    percentage_test=20, percentage_unlabelled=pun, percentage_labelled=80-pun)

dtf_out = pd.DataFrame(None, columns=["AUC", "ACCURACY", "F1"])

for class_model in [ModelRF, ModelNeuralNetwork]:

    obj_model = class_model()
    obj_model.fit(Xl, yl, Xu)
    array_test_pred = obj_model.predict(Xt)
    array_test_real = yt

    dtf_out.loc[obj_model.name, :] = [auc(array_test_real, array_test_pred), accuracy(array_test_real, array_test_pred), f1(array_test_real, array_test_pred)]

print("")
print("Dataset: " + obj_data.name)
print("Percentage unlabelled: " + str(pun))
print(dtf_out)