from pandas import read_csv
import numpy as np
from pandas import get_dummies
from sslplay.utils.iforest import iforest 

class DataCreditCard:


    def __init__(self): 
        self.name = "CREDIT_CARDS"


    def load(self, path="data/creditcard/default of credit card clients.csv"):
        dtf_data = read_csv(path, sep=";")
        self.X = np.array(dtf_data.loc[:, dtf_data.columns != "default"])
        self.y = np.array(dtf_data["default"])

    
    def parse(self):
        
        # SEX
        self.X[:, 1] = self.X[:, 1] - 1
        # EDUCATION
        self.X = np.hstack((self.X, get_dummies(self.X[:, 2])))
        self.X = np.delete(self.X,[2], 1)
        #MARRIAGE
        self.X = np.hstack((self.X, get_dummies(self.X[:, 2])))
        self.X = np.delete(self.X,[2], 1)

        array_bool_inliers = iforest(
            self.X, 
            num_estimators=100, 
            random_state=1102, 
            contamination=0.05
        )
        self.X = self.X[array_bool_inliers, :]
        self.y = self.y[array_bool_inliers]


    