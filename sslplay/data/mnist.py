from pandas import read_csv
import numpy as np
from sklearn.datasets import load_digits

class DataMNIST:


    def __init__(self): 
        self.name = "MNIST"


    def load(self):
        digits = load_digits()
        n_samples = len(digits.images)
        self.X = digits.images.reshape((n_samples, -1))
        self.y = np.array(digits.target)

    
    def parse(self):
        pass



    