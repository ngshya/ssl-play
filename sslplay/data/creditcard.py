from pandas import read_csv
import numpy as np

class DataCreditCard:


    def __init__(self): 
        self.name = "CREDIT_CARDS"


    def load(self, path="data/creditcard/default of credit card clients.csv"):
        dtf_data = read_csv(path, sep=";")
        self.X = dtf_data.loc[:, dtf_data.columns != "default"]
        self.y = np.array(dtf_data["default"])

    
    def parse(self):
        pass


    