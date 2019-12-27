from pandas import read_csv
import numpy as np

class DataSplice:


    def __init__(self): 
        self.name = "SPLICE"


    def load(self, path="data/splice/splice.data"):
        self.X = read_csv(path, header=None)
        self.y = np.array(self.X[0].astype("category").cat.codes.values)
        self.X = np.array([list(x.strip()) for x in self.X[2]])
        def f_tmp(x):
            dict_tmp = {'A': 0, 'C': 1, 'D': 2, 'G': 3, 'N': 4, 'R': 5, 'S': 6, 'T': 7}
            return dict_tmp[x]
        self.X = np.vectorize(f_tmp)(self.X)

    
    def parse(self):
        pass

