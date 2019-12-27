from pandas import read_csv, concat
import numpy as np
import glob

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
        self.X = self.X.iloc[:, 0:64]
        
    
    def parse(self):
        pass