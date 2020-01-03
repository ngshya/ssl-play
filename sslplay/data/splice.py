from pandas import read_csv
import numpy as np
from pandas import get_dummies
from scipy import sparse
from sslplay.utils.iforest import iforest 

class DataSplice:


    def __init__(self): 
        self.name = "SPLICE"


    def load(self, path="data/splice/splice.data"):
        self.X = read_csv(path, header=None)
        self.y = np.array(self.X[0].astype("category").cat.codes.values)
        self.X = np.array([list(x.strip()) for x in self.X[2]])
        self.X = map(lambda x: np.array(get_dummies(x)), np.transpose(self.X))
        self.X = [x for x in self.X]
        self.X = np.hstack(self.X)

    
    def parse(self):
        array_bool_inliers = iforest(
            self.X, 
            num_estimators=100, 
            random_state=1102, 
            contamination=0.05
        )
        self.X = self.X[array_bool_inliers, :]
        self.y = self.y[array_bool_inliers]

