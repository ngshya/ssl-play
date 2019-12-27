from pandas import read_csv
import numpy as np
import pickle

class DataCIFAR:


    def __init__(self): 
        self.name = "CIFAR"


    def load(self):
        with open("data/cifar/data_batch_1", 'rb') as file:
            dict_tmp = pickle.load(file, encoding='bytes')
        self.X = dict_tmp[b"data"]
        self.y = np.array(dict_tmp[b"labels"])

    
    def parse(self):
        pass
    