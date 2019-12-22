from pandas import read_csv
import numpy as np
from sslplay.utils.lut_split import lut_split

class DataSplice:


    def __init__(self): 
        self.name = "SPLICE"


    def load(self, path="data/splice/splice.data"):
        self.data = read_csv(path, header=None)
        self.target = np.array(self.data[0].astype("category").cat.codes.values)
        self.data = np.array([list(x.strip()) for x in self.data[2]])
        def f_tmp(x):
            dict_tmp = {'A': 0, 'C': 1, 'D': 2, 'G': 3, 'N': 4, 'R': 5, 'S': 6, 'T': 7}
            return dict_tmp[x]
        self.data = np.vectorize(f_tmp)(self.data)

    
    def parse(self):
        pass


    def split(self, percentage_test=20, percentage_unlabelled=70, percentage_labelled=10, seed=1102):
        return lut_split(
            X=self.data, 
            y=self.target, 
            percentage_test=percentage_test, 
            percentage_unlabelled=percentage_unlabelled, 
            percentage_labelled=percentage_labelled
        )