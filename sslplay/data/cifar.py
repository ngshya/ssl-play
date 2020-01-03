from pandas import read_csv
import numpy as np
import pickle
from sslplay.utils.iforest import iforest 

class DataCIFAR:


    def __init__(self): 
        self.name = "CIFAR"


    def load(self):
        with open("data/cifar/data_batch_1", 'rb') as file:
            dict_tmp = pickle.load(file, encoding='bytes')
        self.X = dict_tmp[b"data"]
        self.y = np.array(dict_tmp[b"labels"])

    
    def parse(self):
        array_bool_inliers = iforest(
            self.X, 
            num_estimators=100, 
            random_state=1102, 
            contamination=0.05
        )
        self.X = self.X[array_bool_inliers, :]
        self.y = self.y[array_bool_inliers]
    