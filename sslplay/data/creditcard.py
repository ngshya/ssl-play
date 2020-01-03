from pandas import read_csv
import numpy as np
from sslplay.utils.iforest import iforest 

class DataCreditCard:


    def __init__(self): 
        self.name = "CREDIT_CARDS"


    def load(self, path="data/creditcard/default of credit card clients.csv"):
        dtf_data = read_csv(path, sep=";")
        self.X = np.array(dtf_data.loc[:, dtf_data.columns != "default"])
        self.y = np.array(dtf_data["default"])

    
    def parse(self):
        array_bool_inliers = iforest(
            self.X, 
            num_estimators=100, 
            random_state=1102, 
            contamination=0.05
        )
        self.X = self.X[array_bool_inliers, :]
        self.y = self.y[array_bool_inliers]


    