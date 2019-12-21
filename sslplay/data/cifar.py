from pandas import read_csv
import numpy as np
import pickle
from sslplay.utils.lut_split import lut_split

class DataCIFAR:


    def __init__(self): 
        self.name = "CIFAR"


    def load(self):
        dict_tmp = pickle.load(open("data/cifar/data_batch_1", 'rb'), encoding='bytes')
        self.data = dict_tmp[b"data"]
        self.target = np.array(dict_tmp[b"labels"])

    
    def parse(self):
        pass


    def split(self, percentage_test=20, percentage_unlabelled=70, percentage_labelled=10, seed=1102):
        return lut_split(
            X=self.data, 
            y=self.target, 
            percentage_test=percentage_test, 
            percentage_unlabelled=percentage_unlabelled, 
            percentage_labelled=percentage_labelled
        )
    