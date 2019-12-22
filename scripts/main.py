#%%
import sys
import os
sys.path.append('.') 


import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


#%%
from sslplay.model.test_models import test_models
from sslplay.data.spambase import DataSpambase
from sslplay.data.creditcard import DataCreditCard
from sslplay.data.mnist import DataMNIST
from sslplay.data.cifar import DataCIFAR
from sslplay.model.random_forest import ModelRF
from sslplay.model.neural_network import ModelNeuralNetwork
from sslplay.model.kmeans_random_forest import ModelKMeansRF
from sslplay.model.label_spreading import ModelLabelSpreading
from sslplay.model.ladder_network import ModelLadderNetwork


dtf_performance = test_models(array_perc_unla=[79.0])