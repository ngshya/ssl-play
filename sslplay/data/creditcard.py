from pandas import read_csv
import numpy as np

class DataCreditCard:


    def __init__(self): 
        pass


    def load(self, path="data/creditcard/default of credit card clients.csv"):
        self.data = read_csv(path, sep=";")

    
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
            "train_l": self.data.loc[array_sets == "L", self.data.columns != "default"], 
            "target_train_l": np.array(self.data.loc[array_sets == "L", :]["default"]), 
            "train_u": self.data.loc[array_sets == "U", self.data.columns != "default"], 
            "target_train_u": np.array(self.data.loc[array_sets == "U", :]["default"]),  
            "test": self.data.loc[array_sets == "T", self.data.columns != "default"], 
            "target_test": np.array(self.data.loc[array_sets == "T", :]["default"]),  
        }
    