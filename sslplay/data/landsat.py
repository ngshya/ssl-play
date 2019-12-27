from pandas import read_csv, concat
import numpy as np
import glob

class DataLandsat:


    def __init__(self): 
        self.name = "LANDSAT"


    def load(self, path="data/landsat/sat.*"):
        self.lst_all_files = glob.glob(path)
        self.X = (read_csv(f, header=None, sep=" ") \
        for f in self.lst_all_files)
        self.X = concat(self.X, ignore_index=True)
        self.y = np.array(self.X.iloc[:, 36]\
        .astype("category").cat.codes.values)
        self.X = self.X.iloc[:, 0:36]
        
    
    def parse(self):
        pass
