from pandas import read_csv
import numpy as np

class DataLetter:


    def __init__(self): 
        self.name = "LETTER"


    def load(self, path="data/letter/letter-recognition.data"):
        self.X = read_csv(path, header=None, sep=",")
        self.y = np.array(self.X.iloc[:, 0].astype("category").cat.codes.values)
        self.X = self.X.iloc[:, 1:17]
        
    
    def parse(self):
        pass