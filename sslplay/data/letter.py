from pandas import read_csv
import numpy as np
from sslplay.utils.iforest import iforest 

class DataLetter:


    def __init__(self): 
        self.name = "LETTER"


    def load(self, path="data/letter/letter-recognition.data"):
        self.X = read_csv(path, header=None, sep=",")
        self.y = np.array(self.X.iloc[:, 0].astype("category").cat.codes.values)
        self.X = np.array(self.X.iloc[:, 1:17])
        
    
    def parse(self):
        array_bool_inliers = iforest(
            self.X, 
            num_estimators=100, 
            random_state=1102, 
            contamination=0.05
        )
        self.X = self.X[array_bool_inliers, :]
        self.y = self.y[array_bool_inliers]