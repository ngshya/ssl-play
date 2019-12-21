from pandas import read_csv
import numpy as np
from sslplay.utils.lut_split import lut_split

class DataCreditCard:


    def __init__(self): 
        self.name = "CREDIT_CARDS"


    def load(self, path="data/creditcard/default of credit card clients.csv"):
        self.data = read_csv(path, sep=";")

    
    def parse(self):
        pass

    def split(self, percentage_test=20, percentage_unlabelled=70, percentage_labelled=10, seed=1102):
        return lut_split(
            X=self.data.loc[:, self.data.columns != "default"], 
            y=np.array(self.data["default"]), 
            percentage_test=percentage_test, 
            percentage_unlabelled=percentage_unlabelled, 
            percentage_labelled=percentage_labelled
        )
    