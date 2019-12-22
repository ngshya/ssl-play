from pandas import read_csv
import numpy as np
from sslplay.utils.lut_split import lut_split

class DataLetter:


    def __init__(self): 
        self.name = "LETTER"


    def load(self, path="data/letter/letter-recognition.data"):
        self.data = read_csv(path, header=None, sep=",")
        self.target = np.array(self.data.iloc[:, 0].astype("category").cat.codes.values)
        self.data = self.data.iloc[:, 1:17]
        
    
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