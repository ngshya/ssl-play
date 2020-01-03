from pandas import read_csv, concat
import numpy as np
import glob
from sslplay.utils.iforest import iforest 

class DataDigits:


    def __init__(self): 
        self.name = "DIGITS"


    def load(self, path="data/digits/optdigits.*"):
        self.lst_all_files = glob.glob(path)
        self.X = (read_csv(f, header=None, sep=",") \
        for f in self.lst_all_files)
        self.X = concat(self.X, ignore_index=True)
        self.y = np.array(self.X.iloc[:, 64]\
        .astype("category").cat.codes.values)
        self.X = np.array(self.X.iloc[:, 0:64])
        
    
    def parse(self):
        array_bool_inliers = iforest(
            self.X, 
            num_estimators=100, 
            random_state=1102, 
            contamination=0.05
        )
        self.X = self.X[array_bool_inliers, :]
        self.y = self.y[array_bool_inliers]