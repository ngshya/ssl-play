from pandas import read_csv
import numpy as np
from sklearn.datasets import load_digits
from sslplay.utils.lut_split import lut_split

class DataMNIST:


    def __init__(self): 
        self.name = "MNIST"


    def load(self):
        digits = load_digits()
        n_samples = len(digits.images)
        self.data = digits.images.reshape((n_samples, -1))
        self.target = np.array(digits.target)

    
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
    