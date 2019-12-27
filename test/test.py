# To run tests:
# python -m unittest discover test

import sys
import unittest
sys.path.append('.') 
sys.path.append('../') 

class TestSample(unittest.TestCase):

    def test_sample(self):
        self.assertTrue(True)
        PING = "PONG"
        self.assertEqual(PING, "PONG")

###############################################################################

import numpy as np
from sslplay.utils.ssplit import ssplit
from sslplay.utils.s3split import s3split

class TestUtils(unittest.TestCase):


    def test_ssplit(self):
        
        X = np.random.rand(1000, 10)
        y = np.random.choice([0, 1], size=X.shape[0], replace=True)
        
        X1, y1, X2, y2 = ssplit(X=X, y=y, 
            percentage_1=50, percentage_2=50, seed=1102)
        self.assertTrue(len(y1) + len(y2) == len(y))
        self.assertTrue(X1.shape[0] == len(y1))
        self.assertTrue(X2.shape[0] == len(y2))
        self.assertTrue(len(np.unique(y1)) == len(np.unique(y2)))
        
        X3, y3, X4, y4 = ssplit(X=X, y=y, 
            percentage_1=50, percentage_2=50, seed=1102)
        self.assertTrue(np.sum(y1 == y3) == len(y1))
        self.assertTrue(np.sum(y2 == y4) == len(y2))
        self.assertTrue(np.abs(len(y1) / len(y) - 0.5) < 0.1)
        
        X1, y1, X2, y2 = ssplit(X=X, y=y, 
            percentage_1=99, percentage_2=1, seed=1102)
        self.assertTrue(len(np.unique(y1)) == len(np.unique(y2)))
        self.assertTrue(np.abs(len(y2) / len(y) - 0.01) < 0.01)

        X1, y1, X2, y2 = ssplit(X=X, y=y, 
            percentage_1=100, percentage_2=0, seed=1102)
        self.assertTrue(len(np.unique(y1)) == len(np.unique(y)))

        X1, y1, X2, y2 = ssplit(X=X, y=y, 
            percentage_1=99.9, percentage_2=0.1, 
            min_el_1=1, min_el_2=10, seed=1102)
        self.assertTrue(sum(y2 == 0) >= 10 )


    def test_s3split(self):

        X = np.random.rand(1000, 10)
        y = np.random.choice([0, 1, 2], size=X.shape[0], replace=True)
        
        X1, y1, X2, y2, X3, y3 = s3split(
            X=X, y=y, 
            percentage_1=20, 
            percentage_2=40, 
            percentage_3=40,
            seed_1=1102, 
            seed_2=1102
        )

        X4, y4, X5, y5, X6, y6 = s3split(
            X=X, y=y, 
            percentage_1=20, 
            percentage_2=40, 
            percentage_3=40,
            seed_1=1102, 
            seed_2=1991
        )

        self.assertTrue(sum(y1==y4) == len(y1))
        self.assertTrue((len(y2) != len(y5)) or (sum(y2==y5) < len(y2)))

###############################################################################

from sslplay.data.spambase import DataSpambase
from sslplay.data.creditcard import DataCreditCard
from sslplay.data.splice import DataSplice
from sslplay.data.landsat import DataLandsat
from sslplay.data.digits import DataDigits
from sslplay.data.letter import DataLetter
from sslplay.data.cifar import DataCIFAR

class TestData(unittest.TestCase):

    def test_spambase(self):
        obj_data = DataSpambase()
        obj_data.load()
        self.assertEqual(obj_data.X.shape[0], len(obj_data.y))
        self.assertEqual(obj_data.X.shape[0], 4601)

    def test_creditcard(self):
        obj_data = DataCreditCard()
        obj_data.load()
        self.assertEqual(obj_data.X.shape[0], len(obj_data.y))
        self.assertEqual(obj_data.X.shape[0], 30000)

    def test_splice(self):
        obj_data = DataSplice()
        obj_data.load()
        self.assertEqual(obj_data.X.shape[0], len(obj_data.y))
        self.assertEqual(obj_data.X.shape[0], 3190)

    def test_landsat(self):
        obj_data = DataLandsat()
        obj_data.load()
        self.assertEqual(obj_data.X.shape[0], len(obj_data.y))
        self.assertEqual(obj_data.X.shape[0], 6435)

    def test_digits(self):
        obj_data = DataDigits()
        obj_data.load()
        self.assertEqual(obj_data.X.shape[0], len(obj_data.y))
        self.assertEqual(obj_data.X.shape[0], 5620)

    def test_letter(self):
        obj_data = DataLetter()
        obj_data.load()
        self.assertEqual(obj_data.X.shape[0], len(obj_data.y))
        self.assertEqual(obj_data.X.shape[0], 20000)

    def test_cifar(self):
        obj_data = DataCIFAR()
        obj_data.load()
        self.assertEqual(obj_data.X.shape[0], len(obj_data.y))
        self.assertEqual(obj_data.X.shape[0], 10000)
###############################################################################

if __name__ == '__main__':
    unittest.main()