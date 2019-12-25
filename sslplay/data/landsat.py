from pandas import read_csv, concat
import numpy as np
import glob
from sslplay.utils.lut_split import lut_split

class DataLandsat:


    def __init__(self): 
        self.name = "LANDSAT"


    def load(self, path="data/landsat/sat.*"):
        self.lst_all_files = glob.glob(path)
        self.data = (read_csv(f, header=None, sep=" ") for f in self.lst_all_files)
        self.data = concat(self.data, ignore_index=True)
        self.target = np.array(self.data.iloc[:, 36].astype("category").cat.codes.values)
        self.data = self.data.iloc[:, 0:36]
        
    
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