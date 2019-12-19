from pandas import read_csv
import numpy as np
import pickle

class DataCIFAR:


    def __init__(self): 
        self.name = "CIFAR"


    def load(self):
        dict_tmp = pickle.load(open("data/cifar/data_batch_1", 'rb'), encoding='bytes')
        self.data = dict_tmp[b"data"]
        self.target = np.array(dict_tmp[b"labels"])

    
    def parse(self):
        pass


    def split(self, port_test=0.2, port_unla=0.0, seed=1102):
        np.random.seed(seed)
        int_size = self.data.shape[0]
        array_sets = np.repeat("L", int_size)
        array_test = np.random.choice([True, False], size=int_size, replace=True, p=[port_test, 1-port_test])
        array_unla = np.random.choice([True, False], size=int_size, replace=True, p=[port_unla, 1-port_unla])
        array_sets[array_test] = "T"
        array_sets[(~array_test) & array_unla] = "U"
        return {
            "train_l": self.data[array_sets == "L", :], 
            "target_train_l": np.array(self.target[array_sets == "L"]), 
            "train_u": self.data[array_sets == "U", :], 
            "target_train_u": np.array(self.target[array_sets == "U"]),  
            "test": self.data[array_sets == "T", :], 
            "target_test": np.array(self.target[array_sets == "T"]),  
        }
    